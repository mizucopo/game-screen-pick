"""Built-in selection profiles and defaults."""

from ..models.scene_mix import SceneMix
from ..models.selection_profile import SelectionProfile

DEFAULT_SCENE_MIX = SceneMix(gameplay=0.5, event=0.4, other=0.1)

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
    activity_weights={
        "action_intensity": 0.45,
        "edge_density": 0.25,
        "ui_density": 0.15,
        "gameplay_score": 0.15,
    },
    activity_mix_ratio=(0.20, 0.45, 0.35),
    selection_scene_weight=0.60,
    selection_quality_weight=0.40,
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
    activity_weights={
        "action_intensity": 0.25,
        "edge_density": 0.20,
        "ui_density": 0.35,
        "gameplay_score": 0.20,
    },
    activity_mix_ratio=(0.40, 0.40, 0.20),
    selection_scene_weight=0.60,
    selection_quality_weight=0.40,
)

PROFILE_REGISTRY = {
    ACTIVE_PROFILE.name: ACTIVE_PROFILE,
    STATIC_PROFILE.name: STATIC_PROFILE,
}
