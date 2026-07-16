"""Context Collection Stage Fingerprintのsemantic input構築。"""

import hashlib
import json
from fractions import Fraction

from ..models.effective_configuration import EffectiveConfiguration
from ..models.media_runtime_identity import MediaRuntimeIdentity
from ..models.media_stream import MediaStream
from ..models.video_scan_result import VideoScanResult
from ..models.video_source import VideoSource
from ..protocols.speech_runtime import SpeechRuntime
from .normalize_context_language import normalize_context_language

_CONTEXT_POLICY_VERSION = "context-collection-v1"
_SUBTITLE_EXTRACTION_VERSION = "embedded-subtitle-v1"


def build_context_stage_semantic_input(
    source: VideoSource,
    scan: VideoScanResult,
    subtitle_stream: MediaStream | None,
    audio_stream: MediaStream | None,
    use_stt: bool,
    configuration: EffectiveConfiguration,
    media_runtime_identity: MediaRuntimeIdentity,
    speech_runtime: SpeechRuntime,
) -> dict[str, object]:
    """結果へ影響するContext Stage入力だけを正規化する。"""
    timeline_payload = {
        "origin_pts": scan.timeline.origin_pts,
        "time_base": _fraction_value(scan.timeline.time_base),
        "duration": _fraction_value(scan.timeline.duration.seconds),
    }
    timeline_digest = hashlib.sha256(
        json.dumps(
            timeline_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "policy_version": _CONTEXT_POLICY_VERSION,
        "subtitle_extraction_version": _SUBTITLE_EXTRACTION_VERSION,
        "video_fingerprint": source.fingerprint,
        "timeline_contract": "exact-video-time-v1",
        "timeline_digest": timeline_digest,
        "language": normalize_context_language(configuration.language),
        "subtitle_stream_index": configuration.subtitle_stream_index,
        "selected_subtitle_stream": _subtitle_stream_value(subtitle_stream),
        "selected_audio_stream": _audio_stream_value(audio_stream),
        "media_runtime_identity": {
            "ffmpeg_version": media_runtime_identity.ffmpeg_version,
            "ffprobe_version": media_runtime_identity.ffprobe_version,
            "build_capability_sha256": (media_runtime_identity.build_capability_sha256),
        },
        **(
            {
                "audio_stream_index": configuration.audio_stream_index,
                "speech_runtime_identity": speech_runtime.runtime_identity,
                "resolved_model_identity": speech_runtime.resolved_model_identity,
                "speech_device": configuration.speech_to_text_device,
                "speech_compute_type": configuration.speech_to_text_compute_type,
                "speech_beam_size": configuration.speech_to_text_beam_size,
                "speech_vad_filter": configuration.speech_vad_filter,
                "speech_chunk_seconds": configuration.speech_chunk_seconds,
                "speech_overlap_seconds": configuration.speech_overlap_seconds,
                "word_group_policy": "word-gap-1.5s-v1",
                "reliability_policy": "avg-logprob--0.8-min-chars-3-v1",
            }
            if use_stt
            else {}
        ),
    }


def _subtitle_stream_value(stream: MediaStream | None) -> dict[str, object] | None:
    if stream is None:
        return None
    return {
        "index": stream.index,
        "codec_name": stream.codec_name,
        "time_base": (
            None if stream.time_base is None else _fraction_value(stream.time_base)
        ),
        "start_pts": stream.start_pts,
        "duration_ts": stream.duration_ts,
        "language": stream.language,
        "is_default": stream.is_default,
        "is_forced": stream.is_forced,
    }


def _audio_stream_value(stream: MediaStream | None) -> dict[str, object] | None:
    if stream is None:
        return None
    return {
        "index": stream.index,
        "codec_name": stream.codec_name,
        "time_base": (
            None if stream.time_base is None else _fraction_value(stream.time_base)
        ),
        "start_pts": stream.start_pts,
        "duration_ts": stream.duration_ts,
        "language": stream.language,
        "is_default": stream.is_default,
    }


def _fraction_value(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]
