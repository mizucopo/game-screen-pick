"""組み込み選定プロファイルとデフォルト値。"""

from ..models.scene_mix import SceneMix
from ..models.selection_profile import SelectionProfile

DEFAULT_SCENE_MIX = SceneMix(play=0.7, event=0.3)

ACTIVE_PROFILE = SelectionProfile(
    name="active",
    quality_weights={
        "blur_score": 0.17,
        "contrast": 0.14,
        "color_richness": 0.07,
        "visual_balance": 0.09,
        "edge_density": 0.16,
        "action_intensity": 0.18,
        "ui_density": 0.14,
        "dramatic_score": 0.05,
    },
)

STATIC_PROFILE = SelectionProfile(
    name="static",
    quality_weights={
        "blur_score": 0.16,
        "contrast": 0.15,
        "color_richness": 0.05,
        "visual_balance": 0.11,
        "edge_density": 0.10,
        "action_intensity": 0.07,
        "ui_density": 0.31,
        "dramatic_score": 0.05,
    },
)

PROFILE_REGISTRY = {
    ACTIVE_PROFILE.name: ACTIVE_PROFILE,
    STATIC_PROFILE.name: STATIC_PROFILE,
}
