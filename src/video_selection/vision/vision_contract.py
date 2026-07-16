"""VisionRuntime v1のschemaとversion定数。"""

SCENE_CATALOG_PROMPT_VERSION = "scene-catalog-prompt-v1"
SCENE_CATALOG_SCHEMA_VERSION = "scene-catalog-schema-v1"
SCENE_CATALOG_STAGE_CONTRACT_VERSION = "scene-catalog-stage-v1"
CANDIDATE_ANNOTATION_PROMPT_VERSION = "candidate-annotation-prompt-v1"
CANDIDATE_ANNOTATION_SCHEMA_VERSION = "candidate-annotation-schema-v1"
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
                        "enum": ["ordinary", "cinematic", "recurring_gameplay"],
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
            "enum": ["normal_gameplay", "event", "menu", "title", "other"],
        },
        "explanation_value": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
        },
        "annotation_summary": {"type": "string", "minLength": 1},
        "frame_choice_reason": {"type": "string", "minLength": 1},
        "screen_text_kind": {
            "type": "string",
            "enum": ["none", "dialogue", "menu", "title", "hud", "other"],
        },
        "context_relevance": {
            "type": "string",
            "enum": ["unavailable", "none", "weak", "strong"],
        },
        "supporting_context_cue_ids": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
        },
        "spoiler_risk": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
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
