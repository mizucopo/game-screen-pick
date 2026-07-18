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
    PRIMARY_SUBJECT_VISIBILITIES,
    TRANSIENT_OBSTRUCTIONS,
)
from ..models.scene_catalog_entry import SCENE_SELECTION_ROLES

SCENE_CATALOG_PROMPT_VERSION = "scene-catalog-prompt-v2"
SCENE_CATALOG_SCHEMA_VERSION = "scene-catalog-schema-v1"
SCENE_CATALOG_STAGE_CONTRACT_VERSION = "scene-catalog-stage-v1"
CANDIDATE_ANNOTATION_PROMPT_VERSION = "candidate-annotation-prompt-v7"
CANDIDATE_ANNOTATION_SCHEMA_VERSION = "candidate-annotation-schema-v5"
CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION = "candidate-annotation-stage-v4"
RETRY_POLICY_VERSION = "ollama-retry-v4"

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
                    "visible_dialogue_text": {"type": "boolean"},
                    "visible_action": {"type": "boolean"},
                    "visible_character_or_enemy": {"type": "boolean"},
                    "combat_action": {"type": "boolean"},
                    "visible_player_character": {"type": "boolean"},
                    "visible_opponent": {"type": "boolean"},
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
                    "visible_dialogue_text",
                    "visible_action",
                    "visible_character_or_enemy",
                    "combat_action",
                    "visible_player_character",
                    "visible_opponent",
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
