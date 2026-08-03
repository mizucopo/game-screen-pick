"""VisionRuntime v1のschemaとversion定数。"""

from ..models.candidate_annotation import (
    CONTEXT_CUE_RELEVANCES,
    EXPLANATION_VALUES,
    SCREEN_TEXT_KINDS,
    SPOILER_RISKS,
)
from ..models.candidate_frame_observation import (
    CANDIDATE_FRAME_CONTENT_KINDS,
    CANDIDATE_INTERFACE_KINDS,
    CHARACTER_BODY_VISIBILITIES,
    DIALOGUE_TEXT_PRESENTATIONS,
    PRIMARY_SUBJECT_VISIBILITIES,
    TRANSIENT_OBSTRUCTIONS,
)
from ..models.combat_encounter_basis import COMBAT_ENCOUNTER_BASES
from ..models.combat_encounter_kind import COMBAT_ENCOUNTER_KINDS
from ..models.scene_catalog_entry import SCENE_SELECTION_ROLES
from ..models.scene_kind import SCENE_KINDS

SCENE_CATALOG_PROMPT_VERSION = "scene-catalog-prompt-v5"
SCENE_CATALOG_SCHEMA_VERSION = "scene-catalog-schema-v2"
SCENE_CATALOG_STAGE_CONTRACT_VERSION = "scene-catalog-stage-v7"
CANDIDATE_ANNOTATION_PROMPT_VERSION = "candidate-annotation-prompt-v17"
CANDIDATE_ANNOTATION_SCHEMA_VERSION = "candidate-annotation-schema-v12"
CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION = "candidate-annotation-stage-v32"
CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_PROMPT_VERSION = (
    "candidate-annotation-relationship-repair-prompt-v1"
)
CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_SCHEMA_VERSION = (
    "candidate-annotation-relationship-repair-schema-v1"
)
CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_STAGE_CONTRACT_VERSION = (
    "candidate-annotation-relationship-repair-stage-v1"
)
CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_NUM_PREDICT = 1024
CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_EVIDENCE_MAX_LENGTH = 160
COMBAT_ENCOUNTER_VERIFICATION_PROMPT_VERSION = "combat-encounter-verification-prompt-v3"
COMBAT_ENCOUNTER_VERIFICATION_SCHEMA_VERSION = "combat-encounter-verification-schema-v3"
COMBAT_ENCOUNTER_VERIFICATION_STAGE_CONTRACT_VERSION = (
    "combat-encounter-verification-stage-v3"
)
COMBAT_ENCOUNTER_CONFIRMATION_PROMPT_VERSION = "combat-encounter-confirmation-prompt-v3"
COMBAT_ENCOUNTER_CONFIRMATION_STAGE_CONTRACT_VERSION = (
    "combat-encounter-confirmation-stage-v3"
)
COMBAT_VISIBILITY_VERIFICATION_PROMPT_VERSION = (
    "combat-visibility-verification-prompt-v2"
)
COMBAT_VISIBILITY_VERIFICATION_SCHEMA_VERSION = (
    "combat-visibility-verification-schema-v2"
)
COMBAT_VISIBILITY_VERIFICATION_STAGE_CONTRACT_VERSION = (
    "combat-visibility-verification-stage-v2"
)
COMBAT_VISIBILITY_CONFIRMATION_PROMPT_VERSION = (
    "combat-visibility-confirmation-prompt-v1"
)
COMBAT_VISIBILITY_CONFIRMATION_STAGE_CONTRACT_VERSION = (
    "combat-visibility-confirmation-stage-v1"
)
COMBAT_VISIBILITY_EDGE_AUDIT_PROMPT_VERSION = "combat-visibility-edge-audit-prompt-v2"
COMBAT_VISIBILITY_EDGE_AUDIT_SCHEMA_VERSION = "combat-visibility-edge-audit-schema-v1"
COMBAT_VISIBILITY_EDGE_AUDIT_STAGE_CONTRACT_VERSION = (
    "combat-visibility-edge-audit-stage-v2"
)
COMBAT_VISIBILITY_EDGE_STRIP_VERSION = "combat-visibility-edge-strips-v1"
PUBLICATION_BOUNDARY_VERIFICATION_PROMPT_VERSION = (
    "publication-boundary-verification-prompt-v1"
)
PUBLICATION_BOUNDARY_VERIFICATION_SCHEMA_VERSION = (
    "publication-boundary-verification-schema-v1"
)
PUBLICATION_BOUNDARY_VERIFICATION_STAGE_CONTRACT_VERSION = (
    "publication-boundary-verification-stage-v1"
)
RETRY_POLICY_VERSION = "ollama-retry-v10"
VISION_GENERATION_SEED = 0

SCENE_CATALOG_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scenes": {
            "type": "array",
            "minItems": 3,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "slug": {
                        "type": "string",
                        "pattern": "^[a-z0-9]+(?:-[a-z0-9]+)*$",
                    },
                    "display_name": {"type": "string", "minLength": 1},
                    "description": {"type": "string", "minLength": 1},
                    "scene_kind": {
                        "type": "string",
                        "enum": list(SCENE_KINDS),
                    },
                    "selection_role": {
                        "type": "string",
                        "enum": list(SCENE_SELECTION_ROLES),
                    },
                },
                "required": [
                    "slug",
                    "display_name",
                    "description",
                    "scene_kind",
                    "selection_role",
                ],
            },
        }
    },
    "required": ["scenes"],
}

CANDIDATE_ANNOTATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "frame_observations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "frame_id": {"type": "string"},
                    "scene_slug": {"type": "string"},
                    "scene_catalog_match": {
                        "type": "boolean",
                        "description": (
                            "画像だけで選択Sceneの表示名と説明にある具体的な場所・"
                            "人物・出来事まで確認できる場合だけtrue。Scene Kindだけが"
                            "一致する場合やContext Cueによる推測はfalse。"
                        ),
                    },
                    "content_kind": {
                        "type": "string",
                        "enum": list(CANDIDATE_FRAME_CONTENT_KINDS),
                    },
                    "interface_kind": {
                        "type": "string",
                        "enum": list(CANDIDATE_INTERFACE_KINDS),
                    },
                    "prominent_event_portrait": {"type": "boolean"},
                    "cinematic_event_presentation": {"type": "boolean"},
                    "on_screen_dialogue_text_visible": {
                        "type": "boolean",
                        "description": (
                            "画像内で登場人物の台詞文字を実際に読める場合だけtrue。"
                            "音声、Context Cue、人物portrait、空欄、HUD、目的表示は"
                            "false。"
                        ),
                    },
                    "dialogue_text_presentation": {
                        "type": "string",
                        "enum": list(DIALOGUE_TEXT_PRESENTATIONS),
                        "description": (
                            "画像内で読める台詞文字の視覚的な表示形式。"
                            "音声やContext Cueしかない場合はnone。"
                        ),
                    },
                    "visible_action": {"type": "boolean"},
                    "visible_character_or_enemy": {"type": "boolean"},
                    "combat_encounter_kind": {
                        "type": "string",
                        "enum": list(COMBAT_ENCOUNTER_KINDS),
                    },
                    "combat_encounter_basis": {
                        "type": "string",
                        "enum": list(COMBAT_ENCOUNTER_BASES),
                        "description": (
                            "Combat Encounter Kindを支持する積極的な画像内根拠。"
                            "主要戦闘の根拠がないことだけではordinary_*にしない。"
                        ),
                    },
                    "player_body_visibility": {
                        "type": "string",
                        "enum": list(CHARACTER_BODY_VISIBILITIES),
                    },
                    "opponent_body_visibility": {
                        "type": "string",
                        "enum": list(CHARACTER_BODY_VISIBILITIES),
                    },
                    "effect_only_frame": {
                        "type": "boolean",
                        "description": (
                            "主内容が一時的な光・爆発・煙だけで、人物・敵・物体の"
                            "本体を主対象として一つも明瞭に判別できない場合だけtrue。"
                        ),
                    },
                    "explanation_value": {
                        "type": "string",
                        "enum": list(EXPLANATION_VALUES),
                    },
                    "screen_text_kind": {
                        "type": "string",
                        "enum": list(SCREEN_TEXT_KINDS),
                    },
                    "primary_subject_visibility": {
                        "type": "string",
                        "enum": list(PRIMARY_SUBJECT_VISIBILITIES),
                    },
                    "transient_obstruction": {
                        "type": "string",
                        "enum": list(TRANSIENT_OBSTRUCTIONS),
                    },
                    "spoiler_risk": {
                        "type": "string",
                        "enum": list(SPOILER_RISKS),
                    },
                    "spoiler_evidence": {"type": "string"},
                },
                "required": [
                    "frame_id",
                    "scene_slug",
                    "scene_catalog_match",
                    "content_kind",
                    "interface_kind",
                    "prominent_event_portrait",
                    "cinematic_event_presentation",
                    "on_screen_dialogue_text_visible",
                    "dialogue_text_presentation",
                    "visible_action",
                    "visible_character_or_enemy",
                    "combat_encounter_kind",
                    "combat_encounter_basis",
                    "player_body_visibility",
                    "opponent_body_visibility",
                    "effect_only_frame",
                    "explanation_value",
                    "screen_text_kind",
                    "primary_subject_visibility",
                    "transient_obstruction",
                    "spoiler_risk",
                    "spoiler_evidence",
                ],
            },
        },
        "context_relevance": {
            "type": "string",
            "enum": list(CONTEXT_CUE_RELEVANCES),
        },
        "supporting_context_cue_ids": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
        },
    },
    "required": [
        "frame_observations",
        "context_relevance",
        "supporting_context_cue_ids",
    ],
}

COMBAT_VISIBILITY_VERIFICATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "effect_screen_coverage": {
            "type": "string",
            "enum": ["none", "under_quarter", "quarter_to_half", "over_half"],
        },
        "largest_foreground_element": {
            "type": "string",
            "enum": [
                "player_body",
                "opponent_body",
                "other_character_body",
                "environment",
                "interface",
                "visual_effect",
                "unclear",
            ],
        },
        "player_body_visibility": {
            "type": "string",
            "enum": list(CHARACTER_BODY_VISIBILITIES),
        },
        "opponent_body_visibility": {
            "type": "string",
            "enum": list(CHARACTER_BODY_VISIBILITIES),
        },
        "opponent_body_framing": {
            "type": "string",
            "enum": ["complete", "edge_cropped", "occluded", "absent"],
        },
        "effect_overlaps_combatant_body": {
            "type": "string",
            "enum": ["none", "partial", "severe"],
        },
        "effect_only_frame": {"type": "boolean"},
    },
    "required": [
        "effect_screen_coverage",
        "largest_foreground_element",
        "player_body_visibility",
        "opponent_body_visibility",
        "opponent_body_framing",
        "effect_overlaps_combatant_body",
        "effect_only_frame",
    ],
}

COMBAT_VISIBILITY_EDGE_AUDIT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "edges": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "edge": {
                        "type": "string",
                        "enum": ["top", "bottom", "left", "right"],
                    },
                    "opponent_body_present": {"type": "boolean"},
                    "opponent_body_reaches_outer_edge": {"type": "boolean"},
                },
                "required": [
                    "edge",
                    "opponent_body_present",
                    "opponent_body_reaches_outer_edge",
                ],
            },
        }
    },
    "required": ["edges"],
}

COMBAT_ENCOUNTER_VERIFICATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "combat_encounter_kind": {
            "type": "string",
            "enum": list(COMBAT_ENCOUNTER_KINDS),
        },
        "combat_encounter_basis": {
            "type": "string",
            "enum": list(COMBAT_ENCOUNTER_BASES),
        },
        "combat_encounter_evidence": {
            "type": "string",
            "enum": ["none", "enemy_status_ui", "opposing_bodies", "both"],
        },
    },
    "required": [
        "combat_encounter_kind",
        "combat_encounter_basis",
        "combat_encounter_evidence",
    ],
}

PUBLICATION_BOUNDARY_VERIFICATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "transient_transition_effect": {"type": "boolean"},
        "transition_effect_kind": {
            "type": "string",
            "enum": [
                "none",
                "white_wipe",
                "motion_blur_or_streak",
                "fade",
                "other",
            ],
        },
        "transition_effect_coverage": {
            "type": "string",
            "enum": ["none", "under_quarter", "quarter_to_half", "over_half"],
        },
        "cinematic_letterbox": {"type": "boolean"},
        "event_staging": {"type": "boolean"},
        "on_screen_dialogue_text_visible": {"type": "boolean"},
        "visible_character_action": {"type": "boolean"},
        "primary_content_readability": {
            "type": "string",
            "enum": ["clear", "partial", "obscured"],
        },
    },
    "required": [
        "transient_transition_effect",
        "transition_effect_kind",
        "transition_effect_coverage",
        "cinematic_letterbox",
        "event_staging",
        "on_screen_dialogue_text_visible",
        "visible_character_action",
        "primary_content_readability",
    ],
}
