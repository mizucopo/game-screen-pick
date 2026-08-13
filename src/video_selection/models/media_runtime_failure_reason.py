"""MediaRuntimeのstable failure reason。"""

from enum import Enum


class MediaRuntimeFailureReason(str, Enum):
    """external tool detailから独立したmedia failure分類。"""

    FFMPEG_NOT_FOUND = "ffmpeg_not_found"
    FFPROBE_NOT_FOUND = "ffprobe_not_found"
    UNSUPPORTED_FFMPEG_VERSION = "unsupported_ffmpeg_version"
    FFMPEG_FFPROBE_VERSION_MISMATCH = "ffmpeg_ffprobe_version_mismatch"
    MISSING_REQUIRED_DEMUXER_OR_DECODER = "missing_required_demuxer_or_decoder"
    MEDIA_PROBE_FAILED = "media_probe_failed"
    DECODER_FAILURE = "decoder_failure"
    DECODER_STALLED = "decoder_stalled"
    FRAME_EXTRACTION_FAILED = "frame_extraction_failed"
    AUDIO_EXTRACTION_FAILED = "audio_extraction_failed"
    SUBTITLE_EXTRACTION_FAILED = "subtitle_extraction_failed"
