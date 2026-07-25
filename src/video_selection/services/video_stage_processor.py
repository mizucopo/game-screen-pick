"""Video Sourceのscanをpipeliningし3つのVideo Stageを組み立てる。"""

import os
import shutil
import time
from _thread import LockType
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, TimeoutError
from contextlib import suppress
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from threading import Event, Lock

from ..models.candidate_moment import CandidateMoment
from ..models.effective_configuration import EffectiveConfiguration
from ..models.frame_candidate_extraction import FrameCandidateExtraction
from ..models.frame_candidate_extraction_metrics import (
    FrameCandidateExtractionMetrics,
)
from ..models.media_probe import MediaProbe
from ..models.media_runtime_identity import MediaRuntimeIdentity
from ..models.media_stream import MediaStream
from ..models.processing_stage import VIDEO_STAGE_ORDER, ProcessingStage
from ..models.video_scan_result import VideoScanResult
from ..models.video_set import VideoSet
from ..models.video_source import VideoSource
from ..models.video_stage_result import VideoStageResult
from ..protocols.run_observer import RunObserver
from ..protocols.speech_runtime import SpeechRuntime
from ..protocols.video_stage_media_runtime import VideoStageMediaRuntime
from .analyze_neutral_images import (
    BLUR_REJECT_VARIANCE_MIN,
    NEUTRAL_ANALYSIS_ALGORITHM_VERSION,
)
from .build_refinement_pts_ranges import build_refinement_pts_ranges
from .build_stage_fingerprint import build_stage_fingerprint
from .build_video_scan_result import build_video_scan_result
from .completed_stage_writer import CompletedStageWriter
from .context_stage_processor import ContextStageProcessor
from .discover_candidate_moments import discover_candidate_moments
from .processing_stage_runner import ProcessingStageRunner
from .refine_candidate_moments import (
    combine_refined_candidate_groups,
    iter_refined_candidate_groups,
)
from .run_progress_tracker import RunProgressTracker
from .select_primary_video_stream import select_primary_video_stream
from .validate_video_set_snapshot import (
    validate_video_set_snapshot_metadata,
    validate_video_source_snapshot,
)
from .video_stage_artifacts import (
    restore_frame_candidate_extraction,
    restore_video_scan,
    serialize_frame_candidate_extraction,
    serialize_video_scan,
)

_SCAN_ALGORITHM_VERSION = "video-scan-v2"
_TIMELINE_ALGORITHM_VERSION = "exact-timeline-v1"
_SCAN_PROXY_ANALYSIS_VERSION = "scan-proxy-analysis-v1"
_HEARTBEAT_PROXY_CONTRACT = "ffmpeg-mjpeg-960-q3-no-metadata-v1"
_CANDIDATE_EXTRACTION_VERSION = "frame-candidate-extraction-v3"
_CONTENT_REJECT_VERSION = "content-reject-v2"
_DEDUPE_VERSION = "grayscale-64x36-mad-2-v1"
_ENTITY_ID_VERSION = "video-entity-id-v1"
_CANDIDATE_PROXY_CONTRACT = "ffmpeg-mjpeg-960-q3-no-metadata-v1"
_MAX_VIDEO_SCAN_WORKERS = 3
_LOGICAL_CPUS_PER_VIDEO_SCAN_WORKER = 8
_SCAN_PROGRESS_HEARTBEAT_SECONDS = 30.0

PreparedVideoScan = tuple[bool, float]
ProbedVideoSource = tuple[VideoSource, MediaProbe, MediaStream]


class VideoStageProcessor:
    """scanをbounded並列化しdownstreamを順序付きでpipeliningする。"""

    def __init__(
        self,
        media_runtime: VideoStageMediaRuntime,
        speech_runtime: SpeechRuntime,
        observer: RunObserver,
        *,
        progress: RunProgressTracker | None = None,
    ) -> None:
        self._media_runtime = media_runtime
        self._context_processor = ContextStageProcessor(
            media_runtime,
            speech_runtime,
            observer,
            progress=progress,
        )
        self._observer = observer
        self._progress = progress

    def process(
        self,
        video_set: VideoSet,
        configuration: EffectiveConfiguration,
    ) -> tuple[VideoStageResult, ...]:
        """scanを先行確定し各Video SourceをVideo Order順に組み立てる。"""
        validate_video_set_snapshot_metadata(video_set)
        runtime_identity = self._media_runtime.preflight()
        probed_sources: list[ProbedVideoSource] = []
        for source in video_set.sources:
            validate_video_set_snapshot_metadata(video_set)
            probe = self._media_runtime.probe(source.path)
            probed_sources.append((source, probe, select_primary_video_stream(probe)))
        worker_count = _video_scan_worker_count(len(probed_sources))
        results: list[VideoStageResult] = []
        prepared_scans: list[Future[PreparedVideoScan]] = []
        scan_cancellation = Event()
        scan_cancellation_lock = Lock()
        primary_scan_failure: list[BaseException] = []
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="video-scan",
        ) as executor:
            try:
                for source, _probe, primary_stream in probed_sources:
                    prepared_scans.append(
                        executor.submit(
                            self._prepare_scan,
                            scan_cancellation,
                            scan_cancellation_lock,
                            primary_scan_failure,
                            video_set,
                            source,
                            primary_stream,
                            configuration,
                            runtime_identity,
                        )
                    )
                for video_order, (probed, prepared_scan) in enumerate(
                    zip(probed_sources, prepared_scans, strict=True),
                    start=1,
                ):
                    source, probe, primary_stream = probed
                    progress_started = self._start_scan_wait_progress(
                        prepared_scan,
                        source,
                        video_order,
                        len(probed_sources),
                    )
                    results.append(
                        self._process_source(
                            video_set,
                            source,
                            probe,
                            primary_stream,
                            self._await_prepared_scan(
                                prepared_scan,
                                emit_heartbeat=progress_started,
                            ),
                            video_order,
                            configuration,
                            runtime_identity,
                            scan_progress_started=progress_started,
                        )
                    )
            except (Exception, KeyboardInterrupt) as error:
                for prepared_scan in prepared_scans:
                    prepared_scan.cancel()
                self._request_scan_cancellation(
                    scan_cancellation,
                    scan_cancellation_lock,
                    primary_scan_failure,
                )
                if primary_scan_failure and error is not primary_scan_failure[0]:
                    raise primary_scan_failure[0] from error
                raise
        validate_video_set_snapshot_metadata(video_set)
        return tuple(results)

    def _process_source(
        self,
        video_set: VideoSet,
        source: VideoSource,
        probe: MediaProbe,
        primary_stream: MediaStream,
        prepared_scan: PreparedVideoScan,
        video_order: int,
        configuration: EffectiveConfiguration,
        runtime_identity: MediaRuntimeIdentity,
        *,
        scan_progress_started: bool,
    ) -> VideoStageResult:
        """一つのVideo Sourceの3 Stageを確定または再利用する。"""
        validate_video_set_snapshot_metadata(video_set)
        runner = ProcessingStageRunner(
            configuration.processing_cache_folder,
            self._observer,
            subject_namespace="videos",
            subject_fingerprint=source.fingerprint,
            before_stage=lambda: validate_video_set_snapshot_metadata(video_set),
            stage_order=VIDEO_STAGE_ORDER,
            progress=self._progress,
            video_order=video_order,
            video_count=len(video_set.sources),
            video_relative_path=source.relative_path,
            work_unit_kind="video",
        )
        scan_input = _scan_semantic_input(
            source,
            primary_stream,
            runtime_identity,
            configuration,
        )
        scan_bundle = runner.adopt_prepared_bundle(
            ProcessingStage.SCAN_VIDEO,
            scan_input,
            reused=prepared_scan[0],
            duration_seconds=prepared_scan[1],
            progress_started_externally=scan_progress_started,
        )
        scan = restore_video_scan(scan_bundle.artifact, scan_bundle.root)
        discovery = discover_candidate_moments(
            video_fingerprint=source.fingerprint,
            timeline=scan.timeline,
            heartbeats=scan.heartbeats,
            scene_signals=scan.scene_signals,
            density_per_minute=configuration.candidate_density_per_minute,
            refinement_radius_seconds=configuration.refinement_radius_seconds,
        )
        scan_fingerprint = runner.completed_stages[0].fingerprint.value
        extraction_input = _extraction_semantic_input(
            source,
            scan_fingerprint,
            configuration,
        )
        extraction_bundle = runner.reuse_bundle(
            ProcessingStage.EXTRACT_FRAME_CANDIDATES,
            extraction_input,
        )
        if extraction_bundle is None:
            extraction_bundle = runner.complete_artifacts(
                ProcessingStage.EXTRACT_FRAME_CANDIDATES,
                extraction_input,
                lambda stage_root: self._produce_extraction_artifact(
                    source,
                    scan,
                    discovery.density_cap,
                    discovery.moments,
                    configuration,
                    stage_root,
                ),
            )
        extraction, extraction_metrics = restore_frame_candidate_extraction(
            extraction_bundle.artifact,
            extraction_bundle.root,
        )
        context = self._context_processor.process(
            video_set=video_set,
            source=source,
            probe=probe,
            scan=scan,
            configuration=configuration,
            media_runtime_identity=runtime_identity,
        )
        if context.completed_stage is None:
            msg = "Context Collection Stageが完了していません"
            raise RuntimeError(msg)
        return VideoStageResult(
            source=source,
            scan=scan,
            extraction=extraction,
            extraction_metrics=extraction_metrics,
            context=context,
            completed_stages=(*runner.completed_stages, context.completed_stage),
        )

    def _prepare_scan(
        self,
        scan_cancellation: Event,
        scan_cancellation_lock: LockType,
        primary_scan_failure: list[BaseException],
        video_set: VideoSet,
        source: VideoSource,
        primary_stream: MediaStream,
        configuration: EffectiveConfiguration,
        runtime_identity: MediaRuntimeIdentity,
    ) -> PreparedVideoScan:
        """一つのscan cacheを再利用またはatomic確定してdispositionを返す。"""
        if scan_cancellation.is_set():
            raise CancelledError
        try:
            semantic_input = _scan_semantic_input(
                source,
                primary_stream,
                runtime_identity,
                configuration,
            )
            fingerprint = build_stage_fingerprint(
                ProcessingStage.SCAN_VIDEO,
                (),
                semantic_input,
            )
            writer = CompletedStageWriter(
                configuration.processing_cache_folder,
                subject_namespace="videos",
                subject_fingerprint=source.fingerprint,
            )
            started_at = time.monotonic()
            bundle = writer.read_bundle(
                ProcessingStage.SCAN_VIDEO,
                fingerprint,
                (),
                semantic_input,
            )
            reused = bundle is not None
            if bundle is None:
                writer.write_artifacts(
                    ProcessingStage.SCAN_VIDEO,
                    fingerprint,
                    (),
                    semantic_input,
                    lambda stage_root: self._produce_scan_artifact(
                        scan_cancellation,
                        video_set,
                        source,
                        primary_stream,
                        configuration,
                        stage_root,
                    ),
                )
                bundle = writer.read_bundle(
                    ProcessingStage.SCAN_VIDEO,
                    fingerprint,
                    (),
                    semantic_input,
                )
            if bundle is None:
                msg = "先行確定したVideo Scan artifactを検証できませんでした"
                raise RuntimeError(msg)
            duration_seconds = max(time.monotonic() - started_at, 1e-9)
            return (reused, duration_seconds)
        except (Exception, KeyboardInterrupt) as error:
            self._request_scan_cancellation(
                scan_cancellation,
                scan_cancellation_lock,
                primary_scan_failure,
                cause=error,
            )
            raise

    def _produce_scan_artifact(
        self,
        scan_cancellation: Event,
        video_set: VideoSet,
        source: VideoSource,
        primary_stream: MediaStream,
        configuration: EffectiveConfiguration,
        stage_root: Path,
    ) -> dict[str, object]:
        """single-decode scanを実行しscene一時画像を除去する。"""
        if scan_cancellation.is_set():
            raise CancelledError
        thread_cpu_before = time.thread_time()
        started_at = time.monotonic()
        native_scan = self._media_runtime.scan_video(
            source.path,
            primary_stream,
            stage_root,
            heartbeat_interval_seconds=configuration.heartbeat_interval_seconds,
            scene_change_threshold=configuration.scene_change_threshold,
            scene_min_interval_seconds=configuration.scene_min_interval_seconds,
            decode_backend=configuration.decode_backend,
        )
        try:
            scan = build_video_scan_result(
                native_scan=native_scan,
                primary_stream=primary_stream,
                video_fingerprint=source.fingerprint,
                decode_backend=configuration.decode_backend,
            )
            artifact = serialize_video_scan(scan, stage_root)
        finally:
            shutil.rmtree(stage_root / ".scene-proxies", ignore_errors=True)
        wall_seconds = time.monotonic() - started_at
        cpu_seconds = native_scan.cpu_seconds + (time.thread_time() - thread_cpu_before)
        metrics = _artifact_metrics(artifact)
        metrics["wall_seconds"] = wall_seconds
        metrics["cpu_seconds"] = cpu_seconds
        metrics["input_seconds_per_wall_second"] = (
            float(scan.timeline.duration.seconds) / wall_seconds
            if wall_seconds > 0
            else 0.0
        )
        validate_video_source_snapshot(video_set, source)
        return artifact

    def _start_scan_wait_progress(
        self,
        prepared_scan: Future[PreparedVideoScan],
        source: VideoSource,
        video_order: int,
        video_count: int,
    ) -> bool:
        """未完了のbackground scanをactive Stageとして可視化する。"""
        if self._progress is None or prepared_scan.done():
            return False
        self._progress.start_stage(
            ProcessingStage.SCAN_VIDEO,
            video_order=video_order,
            video_count=video_count,
            video_relative_path=source.relative_path,
            work_unit_kind="video",
        )
        return True

    def _await_prepared_scan(
        self,
        prepared_scan: Future[PreparedVideoScan],
        *,
        emit_heartbeat: bool,
    ) -> PreparedVideoScan:
        """background scan完了を待ちactive Stageへ定期heartbeatを出す。"""
        if not emit_heartbeat or self._progress is None:
            return prepared_scan.result()
        while True:
            try:
                return prepared_scan.result(timeout=_SCAN_PROGRESS_HEARTBEAT_SECONDS)
            except TimeoutError:
                self._progress.heartbeat()

    def _request_scan_cancellation(
        self,
        cancellation: Event,
        cancellation_lock: LockType,
        primary_scan_failure: list[BaseException],
        *,
        cause: BaseException | None = None,
    ) -> None:
        """兄弟scanの開始を止めactive subprocess cancellationを一度だけ要求する。"""
        with cancellation_lock:
            first_request = not cancellation.is_set()
            if first_request and cause is not None:
                primary_scan_failure.append(cause)
            cancellation.set()
        if first_request:
            with suppress(Exception):
                self._media_runtime.cancel_video_scans()

    def _produce_extraction_artifact(
        self,
        source: VideoSource,
        scan: VideoScanResult,
        density_cap: int,
        moments: tuple[CandidateMoment, ...],
        configuration: EffectiveConfiguration,
        stage_root: Path,
    ) -> dict[str, object]:
        """native refinementとcandidate proxy確定を実行する。"""
        thread_cpu_before = time.thread_time()
        child_cpu_seconds = 0.0
        child_cpu_lock = Lock()

        def record_child_cpu_seconds(value: float) -> None:
            nonlocal child_cpu_seconds
            with child_cpu_lock:
                child_cpu_seconds += value

        started_at = time.monotonic()
        pts_ranges = build_refinement_pts_ranges(
            scan.timeline,
            moments,
            configuration.refinement_radius_seconds,
        )
        frames = (
            self._media_runtime.scan_video_frame_ranges(
                source.path,
                scan.primary_stream.index,
                pts_ranges,
                960,
                cpu_seconds_recorder=record_child_cpu_seconds,
            )
            if pts_ranges
            else iter(())
        )
        groups = iter_refined_candidate_groups(
            video_fingerprint=source.fingerprint,
            timeline=scan.timeline,
            moments=moments,
            frames=frames,
            refinement_radius_seconds=configuration.refinement_radius_seconds,
            max_frame_candidates=configuration.max_frame_candidates,
        )
        encoded_groups: list[FrameCandidateExtraction] = []
        for group in groups:
            encoded_groups.append(
                self._encode_candidate_group(
                    group,
                    stage_root,
                    record_child_cpu_seconds,
                )
            )
            # 次groupのdecode前に選抜前RGBへの最後の参照を解放する。
            del group
        extraction = combine_refined_candidate_groups(moments, tuple(encoded_groups))
        metrics = FrameCandidateExtractionMetrics(
            wall_seconds=0.0,
            cpu_seconds=0.0,
            density_cap=density_cap,
            actual_moment_count=len(extraction.moments),
            native_frame_count=extraction.native_frame_count,
            reject_breakdown=extraction.reject_breakdown,
            deduplicated_frame_count=extraction.deduplicated_frame_count,
            zero_frame_moment_count=extraction.zero_frame_moment_count,
            frame_candidate_count=len(extraction.candidates),
            frame_candidate_bytes=sum(
                len(candidate.image_bytes) for candidate in extraction.candidates
            ),
        )
        artifact = serialize_frame_candidate_extraction(extraction, metrics, stage_root)
        wall_seconds = time.monotonic() - started_at
        cpu_seconds = time.thread_time() - thread_cpu_before + child_cpu_seconds
        artifact_metrics = _artifact_metrics(artifact)
        artifact_metrics["wall_seconds"] = wall_seconds
        artifact_metrics["cpu_seconds"] = cpu_seconds
        return artifact

    def _encode_candidate_group(
        self,
        group: FrameCandidateExtraction,
        stage_root: Path,
        child_cpu_recorder: Callable[[float], None],
    ) -> FrameCandidateExtraction:
        """一つのrefinement groupのproxyを書きRGB artifactを解放する。"""
        encoded_candidates = []
        for candidate in group.candidates:
            if candidate.decoded_frame is None:
                msg = "Frame Candidate Proxy用のnative frameがありません"
                raise ValueError(msg)
            proxy_path = stage_root / "candidates" / f"{candidate.identifier}.jpg"
            child_cpu_recorder(
                self._media_runtime.write_mjpeg_proxy(
                    candidate.decoded_frame,
                    proxy_path,
                    quality=3,
                )
            )
            encoded_candidates.append(
                replace(
                    candidate,
                    image_bytes=proxy_path.read_bytes(),
                    proxy_path=proxy_path,
                    decoded_frame=None,
                )
            )
        return replace(group, candidates=tuple(encoded_candidates))


def _scan_semantic_input(
    source: VideoSource,
    stream: MediaStream,
    runtime_identity: MediaRuntimeIdentity,
    configuration: EffectiveConfiguration,
) -> dict[str, object]:
    return {
        "video_fingerprint": source.fingerprint,
        "primary_video_stream": {
            "index": stream.index,
            "codec_name": stream.codec_name,
            "time_base": _fraction_value(stream.time_base),
            "width": stream.width,
            "height": stream.height,
        },
        "media_runtime_identity": {
            "ffmpeg_version": runtime_identity.ffmpeg_version,
            "ffprobe_version": runtime_identity.ffprobe_version,
            "build_capability_sha256": runtime_identity.build_capability_sha256,
        },
        "decode_backend": configuration.decode_backend,
        "heartbeat_interval_seconds": configuration.heartbeat_interval_seconds,
        "scene_change_threshold": configuration.scene_change_threshold,
        "scene_min_interval_seconds": configuration.scene_min_interval_seconds,
        "heartbeat_proxy_contract": _HEARTBEAT_PROXY_CONTRACT,
        "scan_algorithm": _SCAN_ALGORITHM_VERSION,
        "timeline_algorithm": _TIMELINE_ALGORITHM_VERSION,
        "scan_proxy_analysis": _SCAN_PROXY_ANALYSIS_VERSION,
    }


def _extraction_semantic_input(
    source: VideoSource,
    scan_fingerprint: str,
    configuration: EffectiveConfiguration,
) -> dict[str, object]:
    return {
        "video_fingerprint": source.fingerprint,
        "upstream_scan_fingerprint": scan_fingerprint,
        "density_per_minute": configuration.candidate_density_per_minute,
        "refinement_radius_seconds": configuration.refinement_radius_seconds,
        "max_frame_candidates": configuration.max_frame_candidates,
        "candidate_extraction_algorithm": _CANDIDATE_EXTRACTION_VERSION,
        "neutral_analysis_algorithm": NEUTRAL_ANALYSIS_ALGORITHM_VERSION,
        "blur_reject_variance_min": BLUR_REJECT_VARIANCE_MIN,
        "content_reject_algorithm": _CONTENT_REJECT_VERSION,
        "source_local_dedupe_algorithm": _DEDUPE_VERSION,
        "entity_id_algorithm": _ENTITY_ID_VERSION,
        "candidate_proxy_contract": _CANDIDATE_PROXY_CONTRACT,
    }


def _fraction_value(value: Fraction | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {"numerator": value.numerator, "denominator": value.denominator}


def _video_scan_worker_count(video_count: int) -> int:
    """CPU scanを過剰subscribeしないbounded worker数を返す。"""
    logical_cpus = os.cpu_count() or 1
    cpu_workers = max(1, logical_cpus // _LOGICAL_CPUS_PER_VIDEO_SCAN_WORKER)
    return min(video_count, _MAX_VIDEO_SCAN_WORKERS, cpu_workers)


def _artifact_metrics(artifact: dict[str, object]) -> dict[str, object]:
    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict):
        msg = "Video Stage artifactにmetric objectがありません"
        raise ValueError(msg)
    return metrics
