"""VisionRuntime v1のschemaとversion定数。"""

from ..models.candidate_annotation import (
    BLOG_IMAGE_TYPES,
    CONTEXT_CUE_RELEVANCES,
    EXPLANATION_VALUES,
    SCREEN_TEXT_KINDS,
    SPOILER_RISKS,
)
from ..models.scene_catalog_entry import SCENE_SELECTION_ROLES

SCENE_CATALOG_PROMPT_VERSION = "scene-catalog-prompt-v1"
SCENE_CATALOG_SCHEMA_VERSION = "scene-catalog-schema-v1"
SCENE_CATALOG_STAGE_CONTRACT_VERSION = "scene-catalog-stage-v1"
CANDIDATE_ANNOTATION_PROMPT_VERSION = "candidate-annotation-prompt-v2"
CANDIDATE_ANNOTATION_SCHEMA_VERSION = "candidate-annotation-schema-v2"
CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION = "candidate-annotation-stage-v1"
RETRY_POLICY_VERSION = "ollama-retry-v1"

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
        "representative_frame_id": {"type": "string"},
        "scene_slug": {"type": "string"},
        "blog_image_type": {
            "type": "string",
            "enum": list(BLOG_IMAGE_TYPES),
        },
        "explanation_value": {
            "type": "string",
            "enum": list(EXPLANATION_VALUES),
        },
        "annotation_summary": {"type": "string", "minLength": 1},
        "frame_choice_reason": {"type": "string", "minLength": 1},
        "screen_text_kind": {
            "type": "string",
            "enum": list(SCREEN_TEXT_KINDS),
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
        "spoiler_risk": {
            "type": "string",
            "enum": list(SPOILER_RISKS),
        },
        "spoiler_evidence": {"type": "string"},
    },
    "required": [
        "representative_frame_id",
        "scene_slug",
        "blog_image_type",
        "explanation_value",
        "annotation_summary",
        "frame_choice_reason",
        "screen_text_kind",
        "context_relevance",
        "supporting_context_cue_ids",
        "spoiler_risk",
        "spoiler_evidence",
    ],
}
