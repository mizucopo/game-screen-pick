"""Video Sourceを2つのVideo Stageへ直列で通す。"""

import resource
import shutil
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from ..models.candidate_moment import CandidateMoment
from ..models.effective_configuration import EffectiveConfiguration
from ..models.frame_candidate_extraction import FrameCandidateExtraction
from ..models.frame_candidate_extraction_metrics import (
    FrameCandidateExtractionMetrics,
)
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
from .build_video_scan_result import build_video_scan_result
from .context_stage_processor import ContextStageProcessor
from .discover_candidate_moments import discover_candidate_moments
from .processing_stage_runner import ProcessingStageRunner
from .refine_candidate_moments import (
    combine_refined_candidate_groups,
    iter_refined_candidate_groups,
)
from .select_primary_video_stream import select_primary_video_stream
from .validate_video_set_snapshot import (
    validate_video_set_snapshot,
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
_CANDIDATE_EXTRACTION_VERSION = "frame-candidate-extraction-v2"
_CONTENT_REJECT_VERSION = "content-reject-v2"
_DEDUPE_VERSION = "grayscale-64x36-mad-2-v1"
_ENTITY_ID_VERSION = "video-entity-id-v1"
_CANDIDATE_PROXY_CONTRACT = "ffmpeg-mjpeg-960-q3-no-metadata-v1"


class VideoStageProcessor:
    """Video Order順にsource-localな3つのVideo Stageを確定する。"""

    def __init__(
        self,
        media_runtime: VideoStageMediaRuntime,
        speech_runtime: SpeechRuntime,
        observer: RunObserver,
    ) -> None:
        self._media_runtime = media_runtime
        self._context_processor = ContextStageProcessor(
            media_runtime,
            speech_runtime,
            observer,
        )
        self._observer = observer

    def process(
        self,
        video_set: VideoSet,
        configuration: EffectiveConfiguration,
    ) -> tuple[VideoStageResult, ...]:
        """各Video Sourceをscanからcontext収集まで直列処理する。"""
        validate_video_set_snapshot(video_set)
        runtime_identity = self._media_runtime.preflight()
        results: list[VideoStageResult] = []
        for source in video_set.sources:
            results.append(
                self._process_source(
                    video_set,
                    source,
                    configuration,
                    runtime_identity,
                )
            )
        return tuple(results)

    def _process_source(
        self,
        video_set: VideoSet,
        source: VideoSource,
        configuration: EffectiveConfiguration,
        runtime_identity: MediaRuntimeIdentity,
    ) -> VideoStageResult:
        """一つのVideo Sourceの3 Stageを確定または再利用する。"""
        validate_video_source_snapshot(video_set, source)
        probe = self._media_runtime.probe(source.path)
        primary_stream = select_primary_video_stream(probe)
        runner = ProcessingStageRunner(
            configuration.processing_cache_folder,
            self._observer,
            subject_namespace="videos",
            subject_fingerprint=source.fingerprint,
            before_stage=lambda: validate_video_source_snapshot(video_set, source),
            stage_order=VIDEO_STAGE_ORDER,
        )
        scan_input = _scan_semantic_input(
            source,
            primary_stream,
            runtime_identity,
            configuration,
        )
        scan_bundle = runner.reuse_bundle(ProcessingStage.SCAN_VIDEO, scan_input)
        if scan_bundle is None:
            scan_bundle = runner.complete_artifacts(
                ProcessingStage.SCAN_VIDEO,
                scan_input,
                lambda stage_root: self._produce_scan_artifact(
                    source,
                    primary_stream,
                    configuration,
                    stage_root,
                ),
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

    def _produce_scan_artifact(
        self,
        source: VideoSource,
        primary_stream: MediaStream,
        configuration: EffectiveConfiguration,
        stage_root: Path,
    ) -> dict[str, object]:
        """single-decode scanを実行しscene一時画像を除去する。"""
        cpu_before = _stage_cpu_seconds()
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
        cpu_seconds = _stage_cpu_seconds() - cpu_before
        metrics = _artifact_metrics(artifact)
        metrics["wall_seconds"] = wall_seconds
        metrics["cpu_seconds"] = cpu_seconds
        metrics["input_seconds_per_wall_second"] = (
            float(scan.timeline.duration.seconds) / wall_seconds
            if wall_seconds > 0
            else 0.0
        )
        return artifact

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
        cpu_before = _stage_cpu_seconds()
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
            encoded_groups.append(self._encode_candidate_group(group, stage_root))
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
        cpu_seconds = _stage_cpu_seconds() - cpu_before
        artifact_metrics = _artifact_metrics(artifact)
        artifact_metrics["wall_seconds"] = wall_seconds
        artifact_metrics["cpu_seconds"] = cpu_seconds
        return artifact

    def _encode_candidate_group(
        self,
        group: FrameCandidateExtraction,
        stage_root: Path,
    ) -> FrameCandidateExtraction:
        """一つのrefinement groupのproxyを書きRGB artifactを解放する。"""
        encoded_candidates = []
        for candidate in group.candidates:
            if candidate.decoded_frame is None:
                msg = "Frame Candidate Proxy用のnative frameがありません"
                raise ValueError(msg)
            proxy_path = stage_root / "candidates" / f"{candidate.identifier}.jpg"
            self._media_runtime.write_mjpeg_proxy(
                candidate.decoded_frame,
                proxy_path,
                quality=3,
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


def _stage_cpu_seconds() -> float:
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (
        self_usage.ru_utime
        + self_usage.ru_stime
        + child_usage.ru_utime
        + child_usage.ru_stime
    )


def _artifact_metrics(artifact: dict[str, object]) -> dict[str, object]:
    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict):
        msg = "Video Stage artifactにmetric objectがありません"
        raise ValueError(msg)
    return metrics
