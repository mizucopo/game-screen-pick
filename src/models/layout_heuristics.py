"""画像構造から導出されるレイアウトヒューリスティクス。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutHeuristics:
    """補助的なレイアウトヒューリスティクス."""

    dialogue_overlay_score: float
    menu_layout_score: float
    title_layout_score: float
    game_over_layout_score: float
