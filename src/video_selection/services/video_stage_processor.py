"""Video Sourceのscanをpipeliningし3つのVideo Stageを組み立てる。"""

import os
import shutil
import time
from _thread import LockType
from collections.abc import Callable
from concurrent.futures import (
    CancelledError,
    Future,
    ThreadPoolExecutor,
    TimeoutError,
)
from contextlib import suppress
from dataclasses import replace
from fractions import Fraction
from functools import partial
from pathlib import Path
from threading import Event, Lock
from typing import cast

from ..models.candidate_moment import CandidateMoment
from ..models.checkpoint_operation import CheckpointOperation
from ..models.durable_work_unit_bundle import DurableWorkUnitBundle
from ..models.effective_configuration import EffectiveConfiguration
from ..models.empty_video_scan_partition import EmptyVideoScanPartition
from ..models.frame_candidate_extraction import FrameCandidateExtraction
from ..models.frame_candidate_extraction_metrics import (
    FrameCandidateExtractionMetrics,
)
from ..models.media_probe import MediaProbe
from ..models.media_runtime_identity import MediaRuntimeIdentity
from ..models.media_stream import MediaStream
from ..models.native_video_scan import NativeVideoScan
from ..models.prepared_video_scan import PreparedVideoScan
from ..models.processing_stage import VIDEO_STAGE_ORDER, ProcessingStage
from ..models.scanned_video_frame import ScannedVideoFrame
from ..models.video_scan_resource_sample import VideoScanResourceSample
from ..models.video_scan_result import VideoScanResult
from ..models.video_set import VideoSet
from ..models.video_source import VideoSource
from ..models.video_stage_result import VideoStageResult
from ..protocols.run_observer import RunObserver
from ..protocols.speech_runtime import SpeechRuntime
from ..protocols.video_stage_media_runtime import VideoStageMediaRuntime
from .adaptive_video_scan_controller import AdaptiveVideoScanController
from .adaptive_video_scan_scheduler import AdaptiveVideoScanScheduler
from .analyze_neutral_images import (
    BLUR_REJECT_VARIANCE_MIN,
    NEUTRAL_ANALYSIS_ALGORITHM_VERSION,
)
from .build_refinement_pts_ranges import build_refinement_pts_ranges
from .build_stage_fingerprint import build_stage_fingerprint
from .build_video_scan_result import build_video_scan_result
from .checkpoint_version import checkpoint_version
from .completed_stage_writer import CompletedStageWriter
from .context_stage_processor import ContextStageProcessor
from .discover_candidate_moments import discover_candidate_moments
from .durable_work_unit_cache import DurableWorkUnitCache
from .processing_stage_runner import ProcessingStageRunner
from .refine_candidate_moments import (
    combine_refined_candidate_groups,
    iter_refined_candidate_groups,
)
from .refinement_group_scheduler import RefinementGroupScheduler
from .resolve_frame_range_worker_count import resolve_frame_range_worker_count
from .run_progress_tracker import RunProgressTracker
from .sample_video_scan_resources_safely import sample_video_scan_resources_safely
from .select_primary_video_stream import select_primary_video_stream
from .select_scene_signal_frames import select_scene_signal_frames
from .validate_video_set_snapshot import (
    validate_video_set_snapshot_metadata,
    validate_video_source_snapshot,
)
from .video_scan_partition_artifacts import (
    restore_video_scan_partition,
    serialize_video_scan_partition,
)
from .video_scan_resource_sampler import VideoScanResourceSampler
from .video_stage_artifacts import (
    restore_frame_candidate_extraction,
    restore_video_scan,
    serialize_frame_candidate_extraction,
    serialize_video_scan,
)

_SCAN_ALGORITHM_VERSION = "video-scan-v6"
_SCAN_PARTITION_CHECKPOINT_VERSION = checkpoint_version(
    CheckpointOperation.VIDEO_SCAN_PARTITION
)
_SCAN_PARTITION_SECONDS = 900.0
_TIMELINE_ALGORITHM_VERSION = "exact-timeline-v1"
_SCAN_PROXY_ANALYSIS_VERSION = "scan-proxy-analysis-v1"
_HEARTBEAT_PROXY_CONTRACT = "ffmpeg-mjpeg-960-q3-no-metadata-v1"
_CANDIDATE_EXTRACTION_VERSION = "frame-candidate-extraction-v4"
_REFINEMENT_GROUP_CHECKPOINT_VERSION = checkpoint_version(
    CheckpointOperation.FRAME_REFINEMENT_GROUP
)
_CONTENT_REJECT_VERSION = "content-reject-v2"
_DEDUPE_VERSION = "grayscale-64x36-mad-2-v1"
_ENTITY_ID_VERSION = "video-entity-id-v1"
_CANDIDATE_PROXY_CONTRACT = "ffmpeg-mjpeg-960-q3-no-metadata-v1"
_SCAN_PROGRESS_HEARTBEAT_SECONDS = 30.0

ScanPartitionDuration = tuple[str, int]
VideoScanPartition = NativeVideoScan | EmptyVideoScanPartition
ProbedVideoSource = tuple[
    VideoSource,
    MediaProbe,
    MediaStream,
    Fraction,
    ScanPartitionDuration,
]


class VideoStageProcessor:
    """scanをbounded並列化しdownstreamを順序付きでpipeliningする。"""

    def __init__(
        self,
        media_runtime: VideoStageMediaRuntime,
        speech_runtime: SpeechRuntime,
        observer: RunObserver,
        *,
        progress: RunProgressTracker | None = None,
        resource_sampler: Callable[[], VideoScanResourceSample | None] | None = None,
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
        if resource_sampler is None:
            system_sampler = VideoScanResourceSampler()
            self._resource_sampler = system_sampler.sample
        else:
            self._resource_sampler = resource_sampler
        self._parallelism_controller: AdaptiveVideoScanController | None = None

    @property
    def parallelism_diagnostics(self) -> dict[str, object]:
        """cache identityと分離されたVideo Scan worker診断を返す。"""
        if self._parallelism_controller is None:
            return {}
        return self._parallelism_controller.diagnostics

    def process(
        self,
        video_set: VideoSet,
        configuration: EffectiveConfiguration,
        *,
        runtime_identity: MediaRuntimeIdentity | None = None,
    ) -> tuple[VideoStageResult, ...]:
        """scanを先行確定し各Video SourceをVideo Order順に組み立てる。"""
        validate_video_set_snapshot_metadata(video_set)
        resolved_runtime_identity = runtime_identity or self._media_runtime.preflight()
        automatic_workers = configuration.video_scan_workers == "auto"
        if automatic_workers:
            # 初回procfs counterをprobe前に確定し、後続sampleが直前のhash負荷ではなく
            # probe以降の差分CPU・disk値を観測できるようにする。
            self._safe_resource_sample()
        probed_sources: list[ProbedVideoSource] = []
        for source in video_set.sources:
            validate_video_set_snapshot_metadata(video_set)
            probe = self._media_runtime.probe(source.path)
            primary_stream = select_primary_video_stream(probe)
            probed_sources.append(
                (
                    source,
                    probe,
                    primary_stream,
                    _media_origin(probe),
                    _resolve_scan_partition_duration(
                        primary_stream,
                        probe.duration,
                    ),
                )
            )
        controller = AdaptiveVideoScanController(
            video_count=len(probed_sources),
            configured_workers=configuration.video_scan_workers,
            auto_max_workers=configuration.video_scan_auto_max_workers,
            decode_backend=configuration.decode_backend,
            logical_cpu_count=os.cpu_count() or 1,
            initial_resource_sample=(
                self._safe_resource_sample() if automatic_workers else None
            ),
        )
        self._parallelism_controller = controller
        results: list[VideoStageResult] = []
        scan_cancellation = Event()
        scan_cancellation_lock = Lock()
        primary_scan_failure: list[BaseException] = []
        try:
            with ThreadPoolExecutor(
                max_workers=controller.executor_capacity,
                thread_name_prefix="video-scan",
            ) as executor:
                scheduler = AdaptiveVideoScanScheduler(
                    executor,
                    controller,
                    lambda index: self._prepare_scan(
                        scan_cancellation,
                        scan_cancellation_lock,
                        primary_scan_failure,
                        video_set,
                        probed_sources[index][0],
                        probed_sources[index][2],
                        probed_sources[index][3],
                        probed_sources[index][4],
                        configuration,
                        resolved_runtime_identity,
                    ),
                    self._resource_sampler,
                )
                prepared_scans = scheduler.start(len(probed_sources))
                try:
                    for video_order, (probed, prepared_scan) in enumerate(
                        zip(probed_sources, prepared_scans, strict=True),
                        start=1,
                    ):
                        (
                            source,
                            probe,
                            primary_stream,
                            media_origin,
                            scan_partition_duration,
                        ) = probed
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
                                media_origin,
                                scan_partition_duration,
                                self._await_prepared_scan(
                                    prepared_scan,
                                    emit_heartbeat=progress_started,
                                ),
                                video_order,
                                configuration,
                                resolved_runtime_identity,
                                scan_progress_started=progress_started,
                            )
                        )
                except (Exception, KeyboardInterrupt) as error:
                    self._request_scan_cancellation(
                        scan_cancellation,
                        scan_cancellation_lock,
                        primary_scan_failure,
                    )
                    scheduler.cancel_pending()
                    if primary_scan_failure and error is not primary_scan_failure[0]:
                        raise primary_scan_failure[0] from error
                    raise
        except BaseException:
            controller.finish_incomplete_attempt()
            raise
        validate_video_set_snapshot_metadata(video_set)
        return tuple(results)

    def _process_source(
        self,
        video_set: VideoSet,
        source: VideoSource,
        probe: MediaProbe,
        primary_stream: MediaStream,
        media_origin: Fraction,
        scan_partition_duration: ScanPartitionDuration,
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
            media_origin,
            runtime_identity,
            configuration,
            scan_partition_duration,
        )
        scan_bundle = runner.adopt_prepared_bundle(
            ProcessingStage.SCAN_VIDEO,
            scan_input,
            reused=prepared_scan.reused,
            duration_seconds=prepared_scan.duration_seconds,
            progress_started_externally=scan_progress_started,
        )
        scan = _restore_scan_for_source(
            scan_bundle.artifact,
            scan_bundle.root,
            primary_stream,
            configuration.decode_backend,
        )
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
            validate_bundle=lambda value: _restore_extraction_for_source(
                value.artifact,
                value.root,
                source,
                scan,
                discovery.moments,
                discovery.density_cap,
            ),
        )
        if extraction_bundle is None:
            extraction_bundle = runner.complete_artifacts(
                ProcessingStage.EXTRACT_FRAME_CANDIDATES,
                extraction_input,
                lambda stage_root: self._produce_extraction_artifact(
                    video_set,
                    source,
                    scan,
                    discovery.density_cap,
                    discovery.moments,
                    configuration,
                    extraction_input,
                    stage_root,
                ),
                validate_bundle=lambda value: _restore_extraction_for_source(
                    value.artifact,
                    value.root,
                    source,
                    scan,
                    discovery.moments,
                    discovery.density_cap,
                ),
            )
        extraction, extraction_metrics = _restore_extraction_for_source(
            extraction_bundle.artifact,
            extraction_bundle.root,
            source,
            scan,
            discovery.moments,
            discovery.density_cap,
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
        media_origin: Fraction,
        scan_partition_duration: ScanPartitionDuration,
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
                media_origin,
                runtime_identity,
                configuration,
                scan_partition_duration,
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
            if bundle is not None:
                try:
                    _restore_scan_for_source(
                        bundle.artifact,
                        bundle.root,
                        primary_stream,
                        configuration.decode_backend,
                    )
                except (
                    FileNotFoundError,
                    IsADirectoryError,
                    NotADirectoryError,
                    TypeError,
                    ValueError,
                ):
                    writer.discard(ProcessingStage.SCAN_VIDEO, fingerprint)
                    bundle = None
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
                        media_origin,
                        scan_partition_duration[1],
                        configuration,
                        semantic_input,
                        stage_root,
                    ),
                    validate_bundle=lambda value: _restore_scan_for_source(
                        value.artifact,
                        value.root,
                        primary_stream,
                        configuration.decode_backend,
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
            return PreparedVideoScan(
                reused=reused,
                duration_seconds=duration_seconds,
                input_seconds_per_wall_second=(
                    None
                    if reused
                    else _metric_number(
                        bundle.artifact,
                        "input_seconds_per_wall_second",
                    )
                ),
            )
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
        media_origin: Fraction,
        scan_partition_duration_ts: int,
        configuration: EffectiveConfiguration,
        scan_input: dict[str, object],
        stage_root: Path,
    ) -> dict[str, object]:
        """固定partitionを個別確定しVideo Scanへ安定順で集約する。"""
        if scan_cancellation.is_set():
            raise CancelledError
        thread_cpu_before = time.thread_time()
        started_at = time.monotonic()
        checkpoint_cache = DurableWorkUnitCache(
            configuration.processing_cache_folder,
            subject_fingerprint=source.fingerprint,
            operation=CheckpointOperation.VIDEO_SCAN_PARTITION,
            observer=self._observer,
        )
        partitions: list[VideoScanPartition] = []
        for partition_index, (start_pts, end_pts) in enumerate(
            _build_scan_partitions(
                primary_stream,
                scan_partition_duration_ts,
            ),
            start=1,
        ):
            if scan_cancellation.is_set():
                raise CancelledError
            partition = self._resolve_scan_partition_checkpoint(
                checkpoint_cache,
                video_set,
                source,
                primary_stream,
                media_origin,
                configuration,
                scan_input,
                partition_index,
                start_pts,
                end_pts,
            )
            partitions.append(partition)
            if isinstance(partition, EmptyVideoScanPartition):
                if end_pts is not None:
                    if scan_cancellation.is_set():
                        raise CancelledError
                    partitions.append(
                        self._resolve_scan_partition_checkpoint(
                            checkpoint_cache,
                            video_set,
                            source,
                            primary_stream,
                            media_origin,
                            configuration,
                            scan_input,
                            partition_index,
                            start_pts,
                            None,
                        )
                    )
                break
        native_scan = _materialize_video_scan_partitions(
            tuple(partitions),
            stage_root,
            configuration.scene_min_interval_seconds,
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

    def _resolve_scan_partition_checkpoint(
        self,
        checkpoint_cache: DurableWorkUnitCache,
        video_set: VideoSet,
        source: VideoSource,
        stream: MediaStream,
        media_origin: Fraction,
        configuration: EffectiveConfiguration,
        scan_input: dict[str, object],
        partition_index: int,
        start_pts: int,
        end_pts: int | None,
    ) -> VideoScanPartition:
        """一つのframe有無を含むscan partition checkpointを解決する。"""
        partition_input = {
            "parent_stage_semantic_input": scan_input,
            "partition_index": partition_index,
            "start_pts": start_pts,
            "end_pts": end_pts,
        }

        def produce_partition(checkpoint_root: Path) -> dict[str, object]:
            return self._produce_scan_partition(
                video_set,
                source,
                stream,
                media_origin,
                configuration,
                checkpoint_root,
                start_pts,
                end_pts,
            )

        def validate_partition(value: DurableWorkUnitBundle) -> None:
            _restore_scan_partition_for_range(
                value.artifact,
                value.root,
                stream,
                start_pts,
                end_pts,
            )

        bundle, _reused = checkpoint_cache.resolve(
            (
                f"pts-{start_pts}-eof"
                if end_pts is None
                else f"pts-{start_pts}-{end_pts}"
            ),
            partition_input,
            produce_partition,
            validate_bundle=validate_partition,
        )
        partition = _restore_scan_partition_for_range(
            bundle.artifact,
            bundle.root,
            stream,
            start_pts,
            end_pts,
        )
        validate_video_source_snapshot(video_set, source)
        return partition

    def _produce_scan_partition(
        self,
        video_set: VideoSet,
        source: VideoSource,
        stream: MediaStream,
        media_origin: Fraction,
        configuration: EffectiveConfiguration,
        checkpoint_root: Path,
        start_pts: int,
        end_pts: int | None,
    ) -> dict[str, object]:
        """一つのscan partitionとscene proxyをcheckpointへ確定する。"""
        scan = self._media_runtime.scan_video_partition(
            source.path,
            stream,
            checkpoint_root,
            media_origin=media_origin,
            start_pts=start_pts,
            end_pts=end_pts,
            heartbeat_interval_seconds=configuration.heartbeat_interval_seconds,
            scene_change_threshold=configuration.scene_change_threshold,
            scene_min_interval_seconds=configuration.scene_min_interval_seconds,
            decode_backend=configuration.decode_backend,
        )
        if isinstance(scan, EmptyVideoScanPartition):
            shutil.rmtree(checkpoint_root / "heartbeats", ignore_errors=True)
            shutil.rmtree(checkpoint_root / ".scene-proxies", ignore_errors=True)
            validate_video_source_snapshot(video_set, source)
            return serialize_video_scan_partition(scan, checkpoint_root)
        temporary_scene_folder = checkpoint_root / ".scene-proxies"
        scene_folder = checkpoint_root / "scene-proxies"
        temporary_scene_folder.replace(scene_folder)
        persisted_scan = replace(
            scan,
            scene_frames=tuple(
                replace(
                    frame,
                    image_path=scene_folder / frame.image_path.name,
                )
                for frame in scan.scene_frames
            ),
        )
        validate_video_source_snapshot(video_set, source)
        return serialize_video_scan_partition(
            persisted_scan,
            checkpoint_root,
        )

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

    def _safe_resource_sample(self) -> VideoScanResourceSample | None:
        """resource取得失敗を安全側のsample欠落へ変換する。"""
        return sample_video_scan_resources_safely(self._resource_sampler)

    def _produce_extraction_artifact(
        self,
        video_set: VideoSet,
        source: VideoSource,
        scan: VideoScanResult,
        density_cap: int,
        moments: tuple[CandidateMoment, ...],
        configuration: EffectiveConfiguration,
        extraction_input: dict[str, object],
        stage_root: Path,
    ) -> dict[str, object]:
        """Refinement Window Groupごとに確定して安定順に集約する。"""
        thread_cpu_before = time.thread_time()
        child_cpu_seconds = 0.0
        worker_cpu_seconds = 0.0
        cpu_seconds_lock = Lock()

        def record_child_cpu_seconds(value: float) -> None:
            nonlocal child_cpu_seconds
            with cpu_seconds_lock:
                child_cpu_seconds += value

        def record_worker_cpu_seconds(value: float) -> None:
            nonlocal worker_cpu_seconds
            with cpu_seconds_lock:
                worker_cpu_seconds += value

        started_at = time.monotonic()
        pts_ranges = build_refinement_pts_ranges(
            scan.timeline,
            moments,
            configuration.refinement_radius_seconds,
        )
        checkpoint_cache = DurableWorkUnitCache(
            configuration.processing_cache_folder,
            subject_fingerprint=source.fingerprint,
            operation=CheckpointOperation.FRAME_REFINEMENT_GROUP,
            observer=self._observer,
        )

        def resolve_refinement_group(
            start_pts: int,
            end_pts: int,
            group_moments: tuple[CandidateMoment, ...],
            unit_input: dict[str, object],
        ) -> FrameCandidateExtraction:
            worker_cpu_before = time.thread_time()
            try:

                def produce_refinement_group(
                    checkpoint_root: Path,
                ) -> dict[str, object]:
                    return self._produce_refinement_group(
                        video_set,
                        source,
                        scan,
                        group_moments,
                        start_pts,
                        end_pts,
                        configuration,
                        checkpoint_root,
                        record_child_cpu_seconds,
                    )

                def validate_refinement_group(
                    value: DurableWorkUnitBundle,
                ) -> None:
                    _restore_refinement_group(
                        value.artifact,
                        value.root,
                        source,
                        scan,
                        group_moments,
                        start_pts,
                        end_pts,
                    )

                bundle, _reused = checkpoint_cache.resolve(
                    f"pts-{start_pts}-{end_pts}",
                    unit_input,
                    produce_refinement_group,
                    validate_bundle=validate_refinement_group,
                )
                validate_video_source_snapshot(video_set, source)
                return _restore_refinement_group(
                    bundle.artifact,
                    bundle.root,
                    source,
                    scan,
                    group_moments,
                    start_pts,
                    end_pts,
                )
            finally:
                record_worker_cpu_seconds(time.thread_time() - worker_cpu_before)

        refinement_tasks: list[Callable[[], FrameCandidateExtraction]] = []
        for start_pts, end_pts in pts_ranges:
            group_moments = tuple(
                moment for moment in moments if start_pts <= moment.source_pts < end_pts
            )
            unit_input: dict[str, object] = {
                "parent_stage_semantic_input": extraction_input,
                "pts_range": [start_pts, end_pts],
                "moment_ids": [moment.identifier for moment in group_moments],
            }
            refinement_tasks.append(
                partial(
                    resolve_refinement_group,
                    start_pts,
                    end_pts,
                    group_moments,
                    unit_input,
                )
            )
        tasks = tuple(refinement_tasks)
        resolved_groups = (
            RefinementGroupScheduler(
                max_workers=resolve_frame_range_worker_count(
                    len(tasks),
                    logical_cpu_count=os.cpu_count() or 1,
                )
            ).resolve(tasks)
            if tasks
            else ()
        )
        encoded_groups = tuple(
            _materialize_candidate_group(group, stage_root) for group in resolved_groups
        )
        extraction = combine_refined_candidate_groups(moments, encoded_groups)
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
        cpu_seconds = (
            time.thread_time()
            - thread_cpu_before
            + worker_cpu_seconds
            + child_cpu_seconds
        )
        artifact_metrics = _artifact_metrics(artifact)
        artifact_metrics["wall_seconds"] = wall_seconds
        artifact_metrics["cpu_seconds"] = cpu_seconds
        return artifact

    def _produce_refinement_group(
        self,
        video_set: VideoSet,
        source: VideoSource,
        scan: VideoScanResult,
        moments: tuple[CandidateMoment, ...],
        start_pts: int,
        end_pts: int,
        configuration: EffectiveConfiguration,
        checkpoint_root: Path,
        child_cpu_recorder: Callable[[float], None],
    ) -> dict[str, object]:
        """一つのRefinement Window Groupをdecode、解析、encodeする。"""
        frames = self._media_runtime.scan_video_frame_ranges(
            source.path,
            scan.primary_stream.index,
            ((start_pts, end_pts),),
            960,
            cpu_seconds_recorder=child_cpu_recorder,
        )
        groups = tuple(
            iter_refined_candidate_groups(
                video_fingerprint=source.fingerprint,
                timeline=scan.timeline,
                moments=moments,
                frames=frames,
                refinement_radius_seconds=configuration.refinement_radius_seconds,
                max_frame_candidates=configuration.max_frame_candidates,
            )
        )
        if len(groups) != 1:
            msg = "Refinement Window Groupは一つの結果を生成する必要があります"
            raise RuntimeError(msg)
        encoded = self._encode_candidate_group(
            groups[0],
            checkpoint_root,
            child_cpu_recorder,
        )
        validate_video_source_snapshot(video_set, source)
        metrics = _extraction_metrics(encoded, density_cap=len(moments))
        return serialize_frame_candidate_extraction(
            encoded,
            metrics,
            checkpoint_root,
        )

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


def _build_scan_partitions(
    stream: MediaStream,
    duration_ts: int,
) -> tuple[tuple[int, int | None], ...]:
    """probe durationを固定区間へ分け、最後だけEOFまで開く。"""
    if (
        stream.kind != "video"
        or stream.time_base is None
        or stream.start_pts is None
        or duration_ts <= 0
    ):
        msg = "再開可能なVideo Scanにはstart PTSと正のdurationが必要です"
        raise ValueError(msg)
    step_value = Fraction(str(_SCAN_PARTITION_SECONDS)) / stream.time_base
    step_pts = max(1, step_value.numerator // step_value.denominator)
    hinted_end = stream.start_pts + duration_ts
    starts = tuple(range(stream.start_pts, hinted_end, step_pts))
    if len(starts) > 1 and duration_ts % step_pts != 0:
        starts = starts[:-1]
    if not starts:
        msg = "Video Scan partitionを構築できませんでした"
        raise ValueError(msg)
    return tuple(
        (
            start,
            None if index == len(starts) - 1 else starts[index + 1],
        )
        for index, start in enumerate(starts)
    )


def _materialize_video_scan_partitions(
    partitions: tuple[VideoScanPartition, ...],
    stage_root: Path,
    scene_min_interval_seconds: float,
) -> NativeVideoScan:
    """partition checkpointをstable順に親Stageへmaterializeする。"""
    if not partitions:
        msg = "Video Scanには1件以上のpartitionが必要です"
        raise ValueError(msg)
    framed_partitions = tuple(
        partition for partition in partitions if isinstance(partition, NativeVideoScan)
    )
    if not framed_partitions:
        msg = "Video Scan partition全体に表示可能frameがありません"
        raise ValueError(msg)
    first = framed_partitions[0]
    if any(
        partition.stream_index != first.stream_index
        or partition.time_base != first.time_base
        for partition in partitions
    ):
        msg = "Video Scan partitionのstream timingが不正です"
        raise ValueError(msg)
    previous_last_pts: int | None = None
    for partition in framed_partitions:
        if previous_last_pts is not None and partition.origin_pts <= previous_last_pts:
            msg = "Video Scan partitionの順序またはstream timingが不正です"
            raise ValueError(msg)
        previous_last_pts = partition.last_frame_pts
    heartbeats = _materialize_scanned_frames(
        tuple(
            frame for partition in framed_partitions for frame in partition.heartbeats
        ),
        stage_root / "heartbeats",
    )
    if not heartbeats:
        msg = "Video Scan partition全体にheartbeatがありません"
        raise ValueError(msg)
    scene_candidates = tuple(
        frame for partition in framed_partitions for frame in partition.scene_frames
    )
    scene_frames = _materialize_scanned_frames(
        select_scene_signal_frames(
            scene_candidates,
            scene_min_interval_seconds,
        ),
        stage_root / ".scene-proxies",
    )
    last = framed_partitions[-1]
    return NativeVideoScan(
        stream_index=first.stream_index,
        origin_pts=first.origin_pts,
        last_frame_pts=last.last_frame_pts,
        last_frame_duration_ts=last.last_frame_duration_ts,
        time_base=first.time_base,
        heartbeats=heartbeats,
        scene_frames=scene_frames,
        wall_seconds=sum(partition.wall_seconds for partition in partitions),
        cpu_seconds=sum(partition.cpu_seconds for partition in partitions),
        decode_pass_count=sum(partition.decode_pass_count for partition in partitions),
    )


def _restore_scan_partition_for_range(
    artifact: dict[str, object],
    checkpoint_root: Path,
    stream: MediaStream,
    start_pts: int,
    end_pts: int | None,
) -> VideoScanPartition:
    """partition artifactが要求したstreamと半開区間に属するか検証する。"""
    partition = restore_video_scan_partition(artifact, checkpoint_root)
    if isinstance(partition, EmptyVideoScanPartition):
        if (
            stream.time_base is None
            or partition.stream_index != stream.index
            or partition.time_base != stream.time_base
            or partition.start_pts != start_pts
            or partition.end_pts != end_pts
        ):
            raise ValueError("空Video Scan partitionの要求PTS rangeが不正です")
        return partition
    if (
        stream.time_base is None
        or partition.stream_index != stream.index
        or partition.time_base != stream.time_base
        or partition.origin_pts < start_pts
        or (
            end_pts is not None
            and (partition.origin_pts >= end_pts or partition.last_frame_pts >= end_pts)
        )
    ):
        raise ValueError("Video Scan partitionの要求PTS rangeが不正です")
    return partition


def _restore_scan_for_source(
    artifact: dict[str, object],
    stage_root: Path,
    expected_stream: MediaStream,
    expected_decode_backend: str,
) -> VideoScanResult:
    """親Scan artifactを現在probeのstreamとdecode契約へ照合する。"""
    scan = restore_video_scan(artifact, stage_root)
    if (
        scan.primary_stream != expected_stream
        or scan.metrics.decode_backend != expected_decode_backend
    ):
        raise ValueError("Video Scan artifactのsource streamが不正です")
    return scan


def _restore_refinement_group(
    artifact: dict[str, object],
    checkpoint_root: Path,
    source: VideoSource,
    scan: VideoScanResult,
    expected_moments: tuple[CandidateMoment, ...],
    start_pts: int,
    end_pts: int,
) -> FrameCandidateExtraction:
    """Refinement artifactが要求MomentとPTS rangeだけを所有するか検証する。"""
    extraction, metrics = restore_frame_candidate_extraction(
        artifact,
        checkpoint_root,
    )
    expected_moment_values = tuple(
        (
            moment.identifier,
            moment.source_pts,
            moment.anchor_time,
            moment.timeline_segment_id,
            moment.evidence,
            moment.proxy_quality_score,
        )
        for moment in expected_moments
    )
    actual_moment_values = tuple(
        (
            moment.identifier,
            moment.source_pts,
            moment.anchor_time,
            moment.timeline_segment_id,
            moment.evidence,
            moment.proxy_quality_score,
        )
        for moment in extraction.moments
    )
    if actual_moment_values != expected_moment_values or metrics.density_cap != len(
        expected_moments
    ):
        raise ValueError("Refinement GroupのCandidate Momentが不正です")
    for candidate in extraction.candidates:
        if (
            candidate.video_fingerprint != source.fingerprint
            or candidate.stream_index != scan.primary_stream.index
            or candidate.origin_pts != scan.timeline.origin_pts
            or candidate.time_base != scan.timeline.time_base
            or candidate.source_pts is None
            or not start_pts <= candidate.source_pts < end_pts
        ):
            raise ValueError("Refinement GroupのFrame Candidateが不正です")
    return extraction


def _restore_extraction_for_source(
    artifact: dict[str, object],
    stage_root: Path,
    source: VideoSource,
    scan: VideoScanResult,
    expected_moments: tuple[CandidateMoment, ...],
    expected_density_cap: int,
) -> tuple[FrameCandidateExtraction, FrameCandidateExtractionMetrics]:
    """親Extraction artifactを現在のMomentとsourceへ照合する。"""
    extraction, metrics = restore_frame_candidate_extraction(
        artifact,
        stage_root,
    )
    expected_moment_values = tuple(
        (
            moment.identifier,
            moment.source_pts,
            moment.anchor_time,
            moment.timeline_segment_id,
            moment.evidence,
            moment.proxy_quality_score,
        )
        for moment in expected_moments
    )
    actual_moment_values = tuple(
        (
            moment.identifier,
            moment.source_pts,
            moment.anchor_time,
            moment.timeline_segment_id,
            moment.evidence,
            moment.proxy_quality_score,
        )
        for moment in extraction.moments
    )
    if (
        actual_moment_values != expected_moment_values
        or metrics.density_cap != expected_density_cap
    ):
        raise ValueError("Frame Candidate Extraction artifactのMomentが不正です")
    for candidate in extraction.candidates:
        if (
            candidate.video_fingerprint != source.fingerprint
            or candidate.stream_index != scan.primary_stream.index
            or candidate.origin_pts != scan.timeline.origin_pts
            or candidate.time_base != scan.timeline.time_base
        ):
            raise ValueError("Frame Candidate Extraction artifactのsourceが不正です")
    return extraction, metrics


def _materialize_scanned_frames(
    frames: tuple[ScannedVideoFrame, ...],
    output_folder: Path,
) -> tuple[ScannedVideoFrame, ...]:
    """checkpoint proxy列を親Stageのstable index pathへ複製する。"""
    output_folder.mkdir(parents=True, exist_ok=True)
    materialized: list[ScannedVideoFrame] = []
    previous_pts: int | None = None
    for index, frame in enumerate(frames, start=1):
        if previous_pts is not None and frame.source_pts <= previous_pts:
            msg = "Video Scan partition proxyのPTS順序が不正です"
            raise ValueError(msg)
        output_path = output_folder / f"{index:012d}.jpg"
        output_path.write_bytes(frame.image_path.read_bytes())
        materialized.append(replace(frame, image_path=output_path))
        previous_pts = frame.source_pts
    return tuple(materialized)


def _scan_semantic_input(
    source: VideoSource,
    stream: MediaStream,
    media_origin: Fraction,
    runtime_identity: MediaRuntimeIdentity,
    configuration: EffectiveConfiguration,
    scan_partition_duration: ScanPartitionDuration,
) -> dict[str, object]:
    return {
        "video_fingerprint": source.fingerprint,
        "primary_video_stream": {
            "index": stream.index,
            "codec_name": stream.codec_name,
            "time_base": _fraction_value(stream.time_base),
            "start_pts": stream.start_pts,
            "duration_ts": stream.duration_ts,
            "width": stream.width,
            "height": stream.height,
        },
        "media_origin": _fraction_value(media_origin),
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
        "scan_partition_contract": {
            "version": _SCAN_PARTITION_CHECKPOINT_VERSION,
            "seconds": _SCAN_PARTITION_SECONDS,
            "last_partition": "open-ended-eof",
            "duration_hint": {
                "source": scan_partition_duration[0],
                "duration_ts": scan_partition_duration[1],
            },
        },
        "timeline_algorithm": _TIMELINE_ALGORITHM_VERSION,
        "scan_proxy_analysis": _SCAN_PROXY_ANALYSIS_VERSION,
    }


def _resolve_scan_partition_duration(
    stream: MediaStream,
    container_duration: Fraction | None,
) -> ScanPartitionDuration:
    """partition開始点だけに使う正のduration hintをstream tickで返す。"""
    if stream.kind != "video" or stream.time_base is None or stream.start_pts is None:
        msg = "再開可能なVideo Scanにはvideo streamのstart PTSとtime baseが必要です"
        raise ValueError(msg)
    if stream.duration_ts is not None and stream.duration_ts > 0:
        return ("stream", stream.duration_ts)
    if container_duration is None or container_duration <= 0:
        msg = "再開可能なVideo Scanにはstreamまたはcontainerの正のdurationが必要です"
        raise ValueError(msg)
    duration_ticks = container_duration / stream.time_base
    duration_ts = (
        duration_ticks.numerator + duration_ticks.denominator - 1
    ) // duration_ticks.denominator
    if duration_ts <= 0:
        msg = "Video Scanのcontainer durationをstream tickへ変換できませんでした"
        raise ValueError(msg)
    return ("container", duration_ts)


def _media_origin(probe: MediaProbe) -> Fraction:
    """全streamのうち最も早いexact開始timestampを返す。"""
    origins = tuple(
        stream.start_pts * stream.time_base
        for stream in probe.streams
        if stream.start_pts is not None and stream.time_base is not None
    )
    if not origins:
        msg = "Video Scanにはmedia streamの開始PTSが必要です"
        raise ValueError(msg)
    return min(origins)


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
        "refinement_group_contract": {
            "version": _REFINEMENT_GROUP_CHECKPOINT_VERSION,
        },
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


def _artifact_metrics(artifact: dict[str, object]) -> dict[str, object]:
    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict):
        msg = "Video Stage artifactにmetric objectがありません"
        raise ValueError(msg)
    return metrics


def _metric_number(artifact: dict[str, object], key: str) -> float:
    value = _artifact_metrics(artifact).get(key)
    if type(value) not in {int, float}:
        msg = f"Video Stage artifactの{key} metricが不正です"
        raise ValueError(msg)
    return float(cast(int | float, value))


def _materialize_candidate_group(
    group: FrameCandidateExtraction,
    stage_root: Path,
) -> FrameCandidateExtraction:
    """checkpoint proxyを親Stageへstable pathでmaterializeする。"""
    materialized = []
    for candidate in group.candidates:
        proxy_path = stage_root / "candidates" / f"{candidate.identifier}.jpg"
        proxy_path.parent.mkdir(parents=True, exist_ok=True)
        proxy_path.write_bytes(candidate.image_bytes)
        materialized.append(replace(candidate, proxy_path=proxy_path))
    return replace(group, candidates=tuple(materialized))


def _extraction_metrics(
    extraction: FrameCandidateExtraction,
    *,
    density_cap: int,
) -> FrameCandidateExtractionMetrics:
    """Work Unit artifact用の意味的件数を構築する。"""
    return FrameCandidateExtractionMetrics(
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
