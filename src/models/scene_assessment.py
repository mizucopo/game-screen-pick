"""Scene assessment derived from CLIP and heuristics."""

from dataclasses import dataclass

from ..constants.scene_label import SceneLabel


@dataclass(frozen=True)
class SceneAssessment:
    """画面種別の評価結果."""

    gameplay_score: float
    event_score: float
    other_score: float
    scene_label: SceneLabel
    scene_confidence: float
    transition_risk_score: float
    bright_washout_score: float
    veiled_transition_score: float
    transition_suppressed_event: bool
