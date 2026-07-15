"""Video Stageのstable Content Reject Reason。"""

from enum import StrEnum


class ContentRejectReason(StrEnum):
    """Neutral Image Analysisがframeを除外する理由。"""

    BLACKOUT = "blackout"
    WHITEOUT = "whiteout"
    SINGLE_TONE = "single_tone"
    BLUR = "blur"
    FADE_TRANSITION = "fade_transition"
    TEMPORAL_TRANSITION = "temporal_transition"

    @classmethod
    def empty_breakdown(cls) -> dict[str, int]:
        """全stable reasonを0件で初期化する。"""
        return {reason.value: 0 for reason in cls}
