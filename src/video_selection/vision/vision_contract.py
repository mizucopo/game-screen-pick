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
from ..models.scene_catalog_entry import SCENE_SELECTION_ROLES

SCENE_CATALOG_PROMPT_VERSION = "scene-catalog-prompt-v2"
SCENE_CATALOG_SCHEMA_VERSION = "scene-catalog-schema-v1"
SCENE_CATALOG_STAGE_CONTRACT_VERSION = "scene-catalog-stage-v1"
CANDIDATE_ANNOTATION_PROMPT_VERSION = "candidate-annotation-prompt-v14"
CANDIDATE_ANNOTATION_SCHEMA_VERSION = "candidate-annotation-schema-v9"
CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION = "candidate-annotation-stage-v14"
COMBAT_VISIBILITY_VERIFICATION_PROMPT_VERSION = (
    "combat-visibility-verification-prompt-v2"
)
COMBAT_VISIBILITY_VERIFICATION_SCHEMA_VERSION = (
    "combat-visibility-verification-schema-v2"
)
COMBAT_VISIBILITY_VERIFICATION_STAGE_CONTRACT_VERSION = (
    "combat-visibility-verification-stage-v2"
)
RETRY_POLICY_VERSION = "ollama-retry-v7"

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
                    "selection_role": {
                        "type": "string",
                        "enum": list(SCENE_SELECTION_ROLES),
                    },
                },
                "required": [
                    "slug",
                    "display_name",
                    "description",
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
                    "combat_action": {"type": "boolean"},
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
                    "content_kind",
                    "interface_kind",
                    "prominent_event_portrait",
                    "cinematic_event_presentation",
                    "on_screen_dialogue_text_visible",
                    "dialogue_text_presentation",
                    "visible_action",
                    "visible_character_or_enemy",
                    "combat_action",
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
