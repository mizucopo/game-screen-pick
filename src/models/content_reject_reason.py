"""content filter の hard reject 理由."""

from enum import StrEnum


class ContentRejectReason(StrEnum):
    """content filter が画像を除外した理由."""

    BLACKOUT = "blackout"
    WHITEOUT = "whiteout"
    SINGLE_TONE = "single_tone"
    FADE_TRANSITION = "fade_transition"
    TEMPORAL_TRANSITION = "temporal_transition"

    @classmethod
    def empty_breakdown(cls) -> dict[str, int]:
        """全理由を0件で初期化したbreakdownを返す."""
        return {reason.value: 0 for reason in cls}
