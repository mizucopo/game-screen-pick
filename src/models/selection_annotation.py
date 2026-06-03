"""候補ごとの選定注釈。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionAnnotation:
    """scene mix 選定で候補へ付与される注釈."""

    score_band: str | None = None
    outlier_rejected: bool = False
