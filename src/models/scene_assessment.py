"""Scene assessment derived from CLIP and heuristics."""

from dataclasses import dataclass

from ..constants.scene_label import SceneLabel


@dataclass(frozen=True)
class SceneAssessment:
    """画面種別の評価結果."""

    play_score: float
    event_score: float
    density_score: float
    scene_label: SceneLabel
    scene_confidence: float
