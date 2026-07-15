"""Video Stage domain resultとCompleted Stage JSONを相互変換する。"""

from collections.abc import Mapping
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import cast

from ..models.candidate_moment import CandidateMoment, MomentEvidence
from ..models.content_reject_reason import ContentRejectReason
from ..models.frame_candidate import FrameCandidate
from ..models.frame_candidate_extraction import FrameCandidateExtraction
from ..models.frame_candidate_extraction_metrics import (
    FrameCandidateExtractionMetrics,
)
from ..models.heartbeat_proxy import HeartbeatProxy
from ..models.media_stream import MediaStream, MediaStreamKind
from ..models.neutral_image_analysis import NeutralImageAnalysis
from ..models.neutral_image_metrics import NeutralImageMetrics
from ..models.scene_signal import SceneSignal
from ..models.timeline_segment import TimelineSegment
from ..models.video_duration import VideoDuration
from ..models.video_scan_metrics import VideoScanMetrics
from ..models.video_scan_result import VideoScanResult
from ..models.video_timeline import VideoTimeline

_SCAN_SCHEMA = "game-screen-pick/video-scan@1.0.0"
_EXTRACTION_SCHEMA = "game-screen-pick/frame-candidate-extraction@1.0.0"


def serialize_video_scan(scan: VideoScanResult, stage_root: Path) -> dict[str, object]:
    """Video Scan Resultをpath非依存のJSON artifactへ変換する。"""
    return {
        "schema": _SCAN_SCHEMA,
        "primary_stream": _serialize_stream(scan.primary_stream),
        "timeline": {
            "origin_pts": scan.timeline.origin_pts,
            "time_base": _serialize_fraction(scan.timeline.time_base),
            "duration": _serialize_fraction(scan.timeline.duration.seconds),
            "segments": [
                {
                    "id": segment.identifier,
                    "start": _serialize_fraction(segment.start),
                    "end": _serialize_fraction(segment.end),
                }
                for segment in scan.timeline.segments
            ],
        },
        "heartbeats": [
            {
                "source_pts": heartbeat.source_pts,
                "video_time": _serialize_fraction(heartbeat.video_time),
                "proxy_path": _relative_artifact_path(
                    heartbeat.proxy_path,
                    stage_root,
                ),
                "quality_score": heartbeat.quality_score,
                "eligible": heartbeat.eligible,
            }
            for heartbeat in scan.heartbeats
        ],
        "scene_signals": [
            {
                "source_pts": scene.source_pts,
                "video_time": _serialize_fraction(scene.video_time),
                "quality_score": scene.quality_score,
                "eligible": scene.eligible,
            }
            for scene in scan.scene_signals
        ],
        "metrics": {
            **asdict(scan.metrics),
            "input_duration": _serialize_fraction(scan.metrics.input_duration),
        },
    }


def restore_video_scan(
    artifact: Mapping[str, object],
    stage_root: Path,
) -> VideoScanResult:
    """検証済みCompleted Stage JSONからVideo Scan Resultを復元する。"""
    if artifact.get("schema") != _SCAN_SCHEMA:
        msg = "Video Scan artifact schemaが不正です"
        raise ValueError(msg)
    timeline_value = _mapping(artifact.get("timeline"))
    metrics_value = _mapping(artifact.get("metrics"))
    timeline = VideoTimeline(
        origin_pts=_integer(timeline_value.get("origin_pts")),
        time_base=_fraction(timeline_value.get("time_base")),
        duration=VideoDuration(_fraction(timeline_value.get("duration"))),
        segments=tuple(
            TimelineSegment(
                identifier=_string(item.get("id")),
                start=_fraction(item.get("start")),
                end=_fraction(item.get("end")),
            )
            for item in _mapping_list(timeline_value.get("segments"))
        ),
    )
    heartbeats = tuple(
        HeartbeatProxy(
            source_pts=_integer(item.get("source_pts")),
            video_time=_fraction(item.get("video_time")),
            proxy_path=_artifact_path(stage_root, item.get("proxy_path")),
            quality_score=_number(item.get("quality_score")),
            eligible=_boolean(item.get("eligible")),
        )
        for item in _mapping_list(artifact.get("heartbeats"))
    )
    scene_signals = tuple(
        SceneSignal(
            source_pts=_integer(item.get("source_pts")),
            video_time=_fraction(item.get("video_time")),
            quality_score=_number(item.get("quality_score")),
            eligible=_boolean(item.get("eligible")),
        )
        for item in _mapping_list(artifact.get("scene_signals"))
    )
    return VideoScanResult(
        primary_stream=_restore_stream(_mapping(artifact.get("primary_stream"))),
        timeline=timeline,
        heartbeats=heartbeats,
        scene_signals=scene_signals,
        metrics=VideoScanMetrics(
            input_duration=_fraction(metrics_value.get("input_duration")),
            wall_seconds=_number(metrics_value.get("wall_seconds")),
            cpu_seconds=_number(metrics_value.get("cpu_seconds")),
            input_seconds_per_wall_second=_number(
                metrics_value.get("input_seconds_per_wall_second")
            ),
            decode_backend=_string(metrics_value.get("decode_backend")),
            decode_pass_count=_integer(metrics_value.get("decode_pass_count")),
            heartbeat_count=_integer(metrics_value.get("heartbeat_count")),
            heartbeat_bytes=_integer(metrics_value.get("heartbeat_bytes")),
            heartbeat_max_gap_seconds=_number(
                metrics_value.get("heartbeat_max_gap_seconds")
            ),
            heartbeat_p95_gap_seconds=_number(
                metrics_value.get("heartbeat_p95_gap_seconds")
            ),
            scene_signal_count=_integer(metrics_value.get("scene_signal_count")),
            timeline_segment_count=_integer(
                metrics_value.get("timeline_segment_count")
            ),
        ),
    )


def serialize_frame_candidate_extraction(
    extraction: FrameCandidateExtraction,
    metrics: FrameCandidateExtractionMetrics,
    stage_root: Path,
) -> dict[str, object]:
    """Frame Candidate抽出結果をCompleted Stage JSONへ変換する。"""
    return {
        "schema": _EXTRACTION_SCHEMA,
        "moments": [_serialize_moment(moment) for moment in extraction.moments],
        "candidates": [
            {
                "id": candidate.identifier,
                "video_fingerprint": candidate.video_fingerprint,
                "stream_index": candidate.stream_index,
                "source_pts": candidate.source_pts,
                "origin_pts": candidate.origin_pts,
                "time_base": _serialize_fraction(
                    _required_fraction(candidate.time_base)
                ),
                "video_time": _serialize_fraction(
                    _required_fraction(candidate.video_time)
                ),
                "proxy_path": _relative_artifact_path(
                    _required_path(candidate.proxy_path),
                    stage_root,
                ),
                "analysis": _serialize_analysis(_required_analysis(candidate.analysis)),
            }
            for candidate in extraction.candidates
        ],
        "diagnostics": {
            "native_frame_count": extraction.native_frame_count,
            "reject_breakdown": extraction.reject_breakdown,
            "deduplicated_frame_count": extraction.deduplicated_frame_count,
            "zero_frame_moment_count": extraction.zero_frame_moment_count,
        },
        "metrics": asdict(metrics),
    }


def restore_frame_candidate_extraction(
    artifact: Mapping[str, object],
    stage_root: Path,
) -> tuple[FrameCandidateExtraction, FrameCandidateExtractionMetrics]:
    """Completed Stage JSONからFrame Candidate抽出結果とmetricを復元する。"""
    if artifact.get("schema") != _EXTRACTION_SCHEMA:
        msg = "Frame Candidate Extraction artifact schemaが不正です"
        raise ValueError(msg)
    diagnostics = _mapping(artifact.get("diagnostics"))
    metrics_value = _mapping(artifact.get("metrics"))
    candidates = tuple(
        _restore_candidate(item, stage_root)
        for item in _mapping_list(artifact.get("candidates"))
    )
    extraction = FrameCandidateExtraction(
        moments=tuple(
            _restore_moment(item) for item in _mapping_list(artifact.get("moments"))
        ),
        candidates=candidates,
        native_frame_count=_integer(diagnostics.get("native_frame_count")),
        reject_breakdown=_integer_mapping(diagnostics.get("reject_breakdown")),
        deduplicated_frame_count=_integer(diagnostics.get("deduplicated_frame_count")),
        zero_frame_moment_count=_integer(diagnostics.get("zero_frame_moment_count")),
    )
    metrics = FrameCandidateExtractionMetrics(
        wall_seconds=_number(metrics_value.get("wall_seconds")),
        cpu_seconds=_number(metrics_value.get("cpu_seconds")),
        density_cap=_integer(metrics_value.get("density_cap")),
        actual_moment_count=_integer(metrics_value.get("actual_moment_count")),
        native_frame_count=_integer(metrics_value.get("native_frame_count")),
        reject_breakdown=_integer_mapping(metrics_value.get("reject_breakdown")),
        deduplicated_frame_count=_integer(
            metrics_value.get("deduplicated_frame_count")
        ),
        zero_frame_moment_count=_integer(metrics_value.get("zero_frame_moment_count")),
        frame_candidate_count=_integer(metrics_value.get("frame_candidate_count")),
        frame_candidate_bytes=_integer(metrics_value.get("frame_candidate_bytes")),
    )
    return extraction, metrics


def _serialize_stream(stream: MediaStream) -> dict[str, object]:
    return {
        "index": stream.index,
        "kind": stream.kind,
        "codec_name": stream.codec_name,
        "time_base": (
            None if stream.time_base is None else _serialize_fraction(stream.time_base)
        ),
        "start_pts": stream.start_pts,
        "duration_ts": stream.duration_ts,
        "width": stream.width,
        "height": stream.height,
        "sample_rate": stream.sample_rate,
        "channels": stream.channels,
        "language": stream.language,
        "is_default": stream.is_default,
        "is_forced": stream.is_forced,
        "is_attached_picture": stream.is_attached_picture,
    }


def _restore_stream(value: Mapping[str, object]) -> MediaStream:
    kind = _string(value.get("kind"))
    if kind not in {"video", "audio", "subtitle", "data", "attachment"}:
        msg = "Media Stream kindが不正です"
        raise ValueError(msg)
    time_base_value = value.get("time_base")
    return MediaStream(
        index=_integer(value.get("index")),
        kind=cast(MediaStreamKind, kind),
        codec_name=_string(value.get("codec_name")),
        time_base=None if time_base_value is None else _fraction(time_base_value),
        start_pts=_optional_integer(value.get("start_pts")),
        duration_ts=_optional_integer(value.get("duration_ts")),
        width=_optional_integer(value.get("width")),
        height=_optional_integer(value.get("height")),
        sample_rate=_optional_integer(value.get("sample_rate")),
        channels=_optional_integer(value.get("channels")),
        language=_optional_string(value.get("language")),
        is_default=_boolean(value.get("is_default")),
        is_forced=_boolean(value.get("is_forced")),
        is_attached_picture=_boolean(value.get("is_attached_picture")),
    )


def _serialize_moment(moment: CandidateMoment) -> dict[str, object]:
    return {
        "id": moment.identifier,
        "source_pts": moment.source_pts,
        "anchor_time": _serialize_fraction(moment.anchor_time),
        "timeline_segment_id": moment.timeline_segment_id,
        "evidence": list(moment.evidence),
        "proxy_quality_score": moment.proxy_quality_score,
        "frame_candidate_ids": list(moment.frame_candidate_ids),
    }


def _restore_moment(value: Mapping[str, object]) -> CandidateMoment:
    evidence_values = _string_list(value.get("evidence"))
    if any(item not in {"heartbeat", "scene"} for item in evidence_values):
        msg = "Candidate Moment evidenceが不正です"
        raise ValueError(msg)
    return CandidateMoment(
        identifier=_string(value.get("id")),
        source_pts=_integer(value.get("source_pts")),
        anchor_time=_fraction(value.get("anchor_time")),
        timeline_segment_id=_string(value.get("timeline_segment_id")),
        evidence=cast(tuple[MomentEvidence, ...], evidence_values),
        proxy_quality_score=_number(value.get("proxy_quality_score")),
        frame_candidate_ids=_string_list(value.get("frame_candidate_ids")),
    )


def _serialize_analysis(analysis: NeutralImageAnalysis) -> dict[str, object]:
    return {
        "source_pts": analysis.source_pts,
        "metrics": asdict(analysis.metrics),
        "quality_score": analysis.quality_score,
        "visual_feature": list(analysis.visual_feature),
        "grayscale_signature_hex": analysis.grayscale_signature.hex(),
        "reject_reason": (
            None if analysis.reject_reason is None else analysis.reject_reason.value
        ),
    }


def _restore_analysis(value: Mapping[str, object]) -> NeutralImageAnalysis:
    metrics = _mapping(value.get("metrics"))
    reason_value = value.get("reject_reason")
    try:
        reason = (
            None if reason_value is None else ContentRejectReason(_string(reason_value))
        )
        signature = bytes.fromhex(_string(value.get("grayscale_signature_hex")))
    except ValueError as error:
        msg = "Neutral Image Analysis artifactが不正です"
        raise ValueError(msg) from error
    return NeutralImageAnalysis(
        source_pts=_integer(value.get("source_pts")),
        metrics=NeutralImageMetrics(
            **{
                field_name: _number(metrics.get(field_name))
                for field_name in NeutralImageMetrics.__dataclass_fields__
            }
        ),
        quality_score=_number(value.get("quality_score")),
        visual_feature=tuple(
            _number(item) for item in _list(value.get("visual_feature"))
        ),
        grayscale_signature=signature,
        reject_reason=reason,
    )


def _restore_candidate(
    value: Mapping[str, object],
    stage_root: Path,
) -> FrameCandidate:
    proxy_path = _artifact_path(stage_root, value.get("proxy_path"))
    return FrameCandidate(
        identifier=_string(value.get("id")),
        image_bytes=proxy_path.read_bytes(),
        video_fingerprint=_string(value.get("video_fingerprint")),
        stream_index=_integer(value.get("stream_index")),
        source_pts=_integer(value.get("source_pts")),
        origin_pts=_integer(value.get("origin_pts")),
        time_base=_fraction(value.get("time_base")),
        video_time=_fraction(value.get("video_time")),
        analysis=_restore_analysis(_mapping(value.get("analysis"))),
        proxy_path=proxy_path,
    )


def _serialize_fraction(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction(value: object) -> Fraction:
    mapping = _mapping(value)
    denominator = _integer(mapping.get("denominator"))
    if denominator == 0:
        msg = "exact time denominatorは0にできません"
        raise ValueError(msg)
    return Fraction(_integer(mapping.get("numerator")), denominator)


def _relative_artifact_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        msg = "Video Stage proxyはStage root配下にある必要があります"
        raise ValueError(msg) from error


def _artifact_path(root: Path, value: object) -> Path:
    relative = PurePosixPath(_string(value))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        msg = "Video Stage artifact pathが不正です"
        raise ValueError(msg)
    path = root.joinpath(*relative.parts)
    if not path.is_file() or path.is_symlink():
        msg = "Video Stage proxy artifactがありません"
        raise ValueError(msg)
    return path


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        msg = "Video Stage artifactにはobjectが必要です"
        raise ValueError(msg)
    return cast(Mapping[str, object], value)


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        msg = "Video Stage artifactにはarrayが必要です"
        raise ValueError(msg)
    return value


def _mapping_list(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(item) for item in _list(value))


def _string_list(value: object) -> tuple[str, ...]:
    return tuple(_string(item) for item in _list(value))


def _integer_mapping(value: object) -> dict[str, int]:
    return {key: _integer(item) for key, item in _mapping(value).items()}


def _string(value: object) -> str:
    if type(value) is not str:
        msg = "Video Stage artifactにはstringが必要です"
        raise ValueError(msg)
    return value


def _optional_string(value: object) -> str | None:
    return None if value is None else _string(value)


def _integer(value: object) -> int:
    if type(value) is not int:
        msg = "Video Stage artifactにはintegerが必要です"
        raise ValueError(msg)
    return value


def _optional_integer(value: object) -> int | None:
    return None if value is None else _integer(value)


def _number(value: object) -> float:
    if type(value) not in {int, float}:
        msg = "Video Stage artifactにはnumberが必要です"
        raise ValueError(msg)
    return float(cast(int | float, value))


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        msg = "Video Stage artifactにはbooleanが必要です"
        raise ValueError(msg)
    return value


def _required_fraction(value: Fraction | None) -> Fraction:
    if value is None:
        msg = "Frame Candidateにexact timeがありません"
        raise ValueError(msg)
    return value


def _required_path(value: Path | None) -> Path:
    if value is None:
        msg = "Frame Candidate Proxy pathがありません"
        raise ValueError(msg)
    return value


def _required_analysis(
    value: NeutralImageAnalysis | None,
) -> NeutralImageAnalysis:
    if value is None:
        msg = "Frame CandidateにNeutral Image Analysisがありません"
        raise ValueError(msg)
    return value
