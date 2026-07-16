"""Context Collection Stageのstable failure reason。"""

from enum import StrEnum


class ContextStageFailureReason(StrEnum):
    """runtime detailやraw textを含まないfatal分類。"""

    AMBIGUOUS_AUDIO_STREAM = "ambiguous_audio_stream"
    AMBIGUOUS_SUBTITLE_STREAM = "ambiguous_subtitle_stream"
    CHUNK_FAILED = "chunk_failed"
    INVALID_AUDIO_STREAM = "invalid_audio_stream"
    INVALID_SUBTITLE_STREAM = "invalid_subtitle_stream"
    STT_ANALYSIS_FAILED = "stt_analysis_failed"
    TIMESTAMP_DRIFT = "timestamp_drift"
    UNSUPPORTED_BITMAP_SUBTITLE = "unsupported_bitmap_subtitle"
