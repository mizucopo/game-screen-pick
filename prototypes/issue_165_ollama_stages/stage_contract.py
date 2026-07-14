"""段階別Ollama評価contractのlogic prototype。"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

Payload = dict[str, Any]

TOTAL_VIDEO_DURATION_SECONDS = Fraction(182_426_481, 1_000)
DEFAULT_CANDIDATE_MOMENT_DENSITY = Fraction(2, 1)
MAX_FRAMES_PER_CANDIDATE_MOMENT = 3
REFERENCE_SHORTLIST_SIZE = 500
OBSERVED_WARM_SECONDS = (2.176, 2.366)


def architecture_case() -> Payload:
    """独立3段階案と集約案の比較を返す。"""
    return {
        "case": "Ollama stage architecture",
        "question": (
            "segment_labeling、candidate_scoring、frame_refinementを"
            "独立したOllama段階にするべきか"
        ),
        "rejected": {
            "name": "three independent Ollama decisions",
            "ollama_stages": [
                "segment_labeling",
                "frame_refinement",
                "candidate_scoring",
            ],
            "problems": [
                (
                    "segment_labelingがshared Scene Catalogを使うと、Video Stageから"
                    "後段のVideo Set Stageへの逆依存になる"
                ),
                (
                    "frame_refinementで画質を再判断するとNeutral Image Analysisと"
                    "責務が重複する"
                ),
                (
                    "candidate_scoringをモデルへ任せると、同じsemantic判断を再度行い、"
                    "soft coverageやspoiler penaltyを決定的に調整できない"
                ),
                (
                    "shortlist 1件あたり最低2回の候補向けmodel callに加え、"
                    "shortlist前のsegment callが必要になる"
                ),
            ],
        },
        "adopted": {
            "name": "local Video Stage + two Video Set Ollama operations",
            "flow": [
                "Video Stage: local Candidate Moment discovery",
                "Video Stage: local Frame Refinement and Neutral Image Analysis",
                (
                    "Video Set Stage: local Scene Catalog Representatives and "
                    "Selection Shortlist"
                ),
                "Video Set Stage: Ollama Scene Catalog once per Video Set",
                (
                    "Video Set Stage: Ollama Candidate Annotation once per "
                    "shortlisted moment"
                ),
                "Video Set Stage: deterministic scoring, diversity and final selection",
            ],
            "ollama_operations": [
                {
                    "name": "Scene Catalog",
                    "cardinality": "one Completed Stage per Video Set",
                },
                {
                    "name": "Candidate Annotation",
                    "cardinality": (
                        "one independently cacheable Completed Stage per shortlisted "
                        "Candidate Moment"
                    ),
                },
            ],
            "candidate_model_calls": "N for N shortlisted Candidate Moments",
            "semantic_model_split": (
                "Scene CatalogとCandidate Annotationは同じvision model能力で足りる。"
                "Context Cueは同じcallへtextとして渡し、別text modelを設けない"
            ),
        },
        "audit": ownership_audit(),
    }


def responsibility_rows() -> list[Payload]:
    """各判断の唯一のownerを返す。"""
    return [
        {
            "decision": "Timeline Segment and Candidate Moment discovery",
            "owner": "Video Stage",
            "mechanism": "local heartbeat, scene signal and temporal diversity",
            "must_not_do": "Scene Catalog classification",
        },
        {
            "decision": "Candidate Moment Density",
            "owner": "Video Stage",
            "mechanism": "local upper bound; default 2 moments/minute",
            "must_not_do": "force a quota or depend on requested output count",
        },
        {
            "decision": "Frame Refinement",
            "owner": "Video Stage",
            "mechanism": (
                "local native frames, invalid-frame rejection and deduplication"
            ),
            "must_not_do": "call Ollama or use Video Order",
        },
        {
            "decision": "Image Quality",
            "owner": "Neutral Image Analysis in Video Stage",
            "mechanism": "local metrics and hard Blog Candidate gate",
            "must_not_do": "appear in Candidate Annotation output",
        },
        {
            "decision": "Context Cue extraction",
            "owner": "Video Stage",
            "mechanism": "embedded text subtitle or faster-whisper",
            "must_not_do": "generate Candidate Moments or accept/reject frames",
        },
        {
            "decision": "Scene vocabulary and Scene Selection Role",
            "owner": "Ollama Scene Catalog in Video Set Stage",
            "mechanism": "one shared catalog from cross-video representatives",
            "must_not_do": "create a per-video catalog",
        },
        {
            "decision": "Representative Frame",
            "owner": "Ollama Candidate Annotation in Video Set Stage",
            "mechanism": "choose semantic representative from 1-3 locally valid frames",
            "must_not_do": "re-score objective image quality",
        },
        {
            "decision": "Scene",
            "owner": "Ollama Candidate Annotation in Video Set Stage",
            "mechanism": "choose exactly one Scene Catalog slug",
            "must_not_do": "emit a free-form scene or confidence score",
        },
        {
            "decision": "Blog Image Type",
            "owner": "Ollama Candidate Annotation in Video Set Stage",
            "mechanism": "normal_gameplay, event, menu, title or other",
            "must_not_do": "decide final type coverage",
        },
        {
            "decision": "Explanation Value",
            "owner": "Ollama Candidate Annotation in Video Set Stage",
            "mechanism": "none, low, medium or high semantic assessment",
            "must_not_do": "emit a final numeric score",
        },
        {
            "decision": "On-screen Text",
            "owner": "Ollama Candidate Annotation in Video Set Stage",
            "mechanism": (
                "classify only its role as dialogue, menu, title, hud or other"
            ),
            "must_not_do": "publish generated text as an exact quotation",
        },
        {
            "decision": "Context Cue Relevance",
            "owner": "Ollama Candidate Annotation in Video Set Stage",
            "mechanism": "relevance plus IDs of supporting usable cues",
            "must_not_do": "let cue-only evidence make an invalid frame eligible",
        },
        {
            "decision": "Spoiler Risk",
            "owner": "Ollama Candidate Annotation in Video Set Stage",
            "mechanism": "none, low, medium or high semantic risk",
            "must_not_do": "apply the configured spoiler penalty",
        },
        {
            "decision": "Final Score, soft coverage and diversity",
            "owner": "deterministic selector in Video Set Stage",
            "mechanism": "policy to be resolved by Issue #167",
            "must_not_do": "ask Ollama to rank the full shortlist",
        },
    ]


def ownership_audit() -> Payload:
    """同じ判断に複数ownerがないことを確認する。"""
    rows = responsibility_rows()
    decisions = [str(row["decision"]) for row in rows]
    duplicate_decisions = sorted(
        decision for decision in set(decisions) if decisions.count(decision) > 1
    )
    return {
        "unique_owner_per_decision": not duplicate_decisions,
        "duplicate_decisions": duplicate_decisions,
        "backward_dependency": False,
        "ollama_after_local_shortlist_only": True,
        "ollama_emits_final_numeric_score": False,
    }


def scene_catalog_schema() -> Payload:
    """Scene Catalogのstrict schemaを返す。"""
    scene = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "slug": {"type": "string", "pattern": "^[a-z0-9]+(?:-[a-z0-9]+)*$"},
            "display_name": {"type": "string", "minLength": 1},
            "description": {"type": "string", "minLength": 1},
            "selection_role": {
                "type": "string",
                "enum": ["ordinary", "cinematic", "recurring_gameplay"],
            },
        },
        "required": ["slug", "display_name", "description", "selection_role"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scenes": {
                "type": "array",
                "minItems": 3,
                "maxItems": 8,
                "items": scene,
            }
        },
        "required": ["scenes"],
        "local_domain_validation": [
            "scene slug is unique",
            "other exists exactly once",
            "other selection_role is ordinary",
        ],
    }


def candidate_annotation_schema() -> Payload:
    """Candidate Annotation v1のstrict schemaを返す。"""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "representative_frame_id": {
                "type": "string",
                "description": "入力された1-3件のFrame Candidate IDのいずれか",
            },
            "scene_slug": {
                "type": "string",
                "description": "入力されたScene Catalogのslugのいずれか",
            },
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
        "local_domain_validation": [
            "representative_frame_id belongs to request frame IDs",
            "scene_slug belongs to request Scene Catalog",
            "supporting_context_cue_ids are supplied usable cue IDs",
            "unavailable or none context has no supporting cue IDs",
            "other scene is a valid classification, never a failure fallback",
        ],
        "intentionally_absent": [
            "quality_score",
            "scene_confidence",
            "final_selection_score",
            "eligible or selected flag",
            "generated verbatim screen text",
            "model reasoning trace",
        ],
    }


def candidate_annotation_case() -> Payload:
    """実frameで確認したCandidate Annotation例を返す。"""
    return {
        "case": "three-frame dialogue Candidate Moment",
        "request": {
            "candidate_moment_id": "moment-dialogue-611",
            "frame_candidate_ids": [
                "frame-604_5",
                "frame-611_5",
                "frame-618_5",
            ],
            "scene_catalog_slugs": [
                "conversation",
                "exploration",
                "battle",
                "menu",
                "title",
                "other",
            ],
            "context_cue_ids": ["cue-604_5", "cue-611_5", "cue-618_5"],
            "video_set_progress": "0.20",
            "quality_metrics_in_prompt": False,
        },
        "contract_output_example": {
            "representative_frame_id": "frame-618_5",
            "scene_slug": "conversation",
            "blog_image_type": "event",
            "explanation_value": "high",
            "annotation_summary": (
                "怪我で探しに行けない人物を助ける依頼の背景が説明される会話"
            ),
            "frame_choice_reason": (
                "依頼の事情が画面内の台詞として最も具体的に示される"
            ),
            "screen_text_kind": "dialogue",
            "context_relevance": "strong",
            "supporting_context_cue_ids": ["cue-618_5"],
            "spoiler_risk": "none",
            "spoiler_evidence": "",
        },
        "schema": candidate_annotation_schema(),
    }


def retry_and_failure_case() -> Payload:
    """失敗境界と再試行contractを返す。"""
    return {
        "case": "retry and failure boundary",
        "request_policy": {
            "stream": False,
            "format": "full JSON Schema object",
            "think": False,
            "temperature": 0,
            "max_attempts": 2,
            "retry_policy_version": "ollama-retry/v1",
        },
        "retryable_once": [
            "connection reset or timeout",
            "HTTP 408, 429 or 5xx",
            "empty or truncated response",
            "JSON Schema or local domain validation failure",
        ],
        "not_retryable": [
            "model missing or vision capability absent",
            "invalid configuration or unreadable input artifact",
            "HTTP 4xx other than 408 or 429",
            "cache or manifest corruption that cannot be isolated",
        ],
        "retry_rules": [
            "retry keeps the same images, cues, catalog and semantic request",
            "validation retry adds only stable validation codes, not raw model output",
            "429 honors Retry-After up to 30 seconds; other retry waits 1 second",
            "never drop images or cues, halve the catalog input, or fallback to other",
        ],
        "atomicity": [
            "Scene Catalog is one atomic Completed Stage",
            (
                "each Candidate Annotation is its own atomic Completed Stage so an "
                "interrupted run can reuse already completed annotations"
            ),
            "a failed annotation prevents final selection and output publication",
        ],
        "diagnostics": [
            "model name, model digest and Ollama version",
            "prompt, schema, stage contract and retry policy versions",
            "request fingerprint, cache hit, attempt count and validation code",
            "image and cue counts, durations, prompt/eval token counts and done reason",
            "no absolute paths, raw reasoning trace or generated screen-text quotation",
        ],
    }


def invalidation_case() -> Payload:
    """入力変更ごとのcache invalidationを返す。"""
    return {
        "case": "stage fingerprints and cache invalidation",
        "scene_catalog_stage_fingerprint": [
            "Video Set Fingerprint and upstream Neutral Image Analysis fingerprints",
            "ordered representative Frame Candidate IDs and content hashes",
            "Selection Intent and Scene Hint",
            "model name and digest, Ollama version and generation options",
            "catalog prompt, schema, stage contract and retry policy versions",
        ],
        "candidate_annotation_stage_fingerprint": [
            "Video Set Fingerprint and Candidate Moment ID",
            "ordered Frame Candidate IDs and content hashes",
            "supplied Context Cue IDs, text, exact ranges and input policy version",
            "Scene Catalog fingerprint",
            "Video Order-derived progress and Selection Intent",
            "model name and digest, Ollama version and generation options",
            "annotation prompt, schema, stage contract and retry policy versions",
        ],
        "does_not_invalidate_candidate_annotation": [
            "requested output count",
            "final score weights",
            "spoiler sensitivity and penalty",
            "blog image type soft-coverage targets",
            "output path or report formatting",
        ],
        "examples": [
            {
                "change": "requested output count",
                "recompute": (
                    "Selection Shortlist and annotations newly entering it; existing "
                    "Scene Catalog and Candidate Annotations remain reusable"
                ),
            },
            {
                "change": "annotation prompt or schema",
                "recompute": "Candidate Annotation only",
            },
            {
                "change": "Scene Catalog content",
                "recompute": "Scene Catalog and every dependent Candidate Annotation",
            },
            {
                "change": "Video Order",
                "recompute": "all Video Set Stages; reusable Video Stages remain",
            },
            {
                "change": "final spoiler penalty",
                "recompute": "deterministic selection only",
            },
        ],
    }


def density_comparison() -> Payload:
    """提供Video Setに対する候補密度の上限とmodel costを返す。"""
    rows = []
    total_minutes = TOTAL_VIDEO_DURATION_SECONDS / 60
    for density in (Fraction(1), Fraction(2), Fraction(4)):
        moment_cap = math.ceil(total_minutes * density)
        rows.append(
            {
                "density_per_minute": int(density),
                "candidate_moment_cap": moment_cap,
                "maximum_persisted_frame_candidates": (
                    moment_cap * MAX_FRAMES_PER_CANDIDATE_MOMENT
                ),
                "if_every_moment_were_annotated_serially": elapsed_range(moment_cap),
            }
        )
    return {
        "case": "Candidate Moment Density",
        "provided_video_set_duration": "50:40:26.481",
        "comparison": rows,
        "adopted_default": {
            "density_per_minute": int(DEFAULT_CANDIDATE_MOMENT_DENSITY),
            "meaning": (
                "upper bound of one retained hypothesis per 30 seconds on average"
            ),
            "not_a_quota": True,
        },
        "cost_boundary": {
            "ollama_runs_on": "local Selection Shortlist only",
            "reference_shortlist_size": REFERENCE_SHORTLIST_SIZE,
            "observed_serial_warm_time": elapsed_range(REFERENCE_SHORTLIST_SIZE),
            "consequence": (
                "density controls local coverage/refinement and cache size; "
                "it does not "
                "directly multiply Ollama calls beyond the shortlist"
            ),
        },
    }


def elapsed_range(count: int) -> str:
    """実測warm latencyからserial所要時間範囲を返す。"""
    low_seconds = count * OBSERVED_WARM_SECONDS[0]
    high_seconds = count * OBSERVED_WARM_SECONDS[1]
    return f"{format_duration(low_seconds)}-{format_duration(high_seconds)}"


def format_duration(seconds: float) -> str:
    """秒を短い時間表記へ変換する。"""
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} min"
    return f"{minutes / 60:.1f} h"


def capability_probe_case() -> Payload:
    """target実機で行ったstructured-output probeを返す。"""
    return {
        "case": "target-machine Ollama capability probe",
        "environment": {
            "target": "Windows 11 Pro / WSL2 Ubuntu / RTX 5090",
            "ollama_version": "0.31.2",
            "model": "qwen3-vl:8b-instruct",
            "model_digest": (
                "0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319"
            ),
            "advertised_capabilities": ["vision", "completion", "tools"],
        },
        "three_image_probe": {
            "source": "provided video frames at 604.5, 611.5 and 618.5 seconds",
            "schema_accepted": True,
            "representative_frame": "frame-618_5",
            "stable_semantic_result_runs": "3/3",
            "warm_elapsed_seconds": [2.176, 2.366],
            "prompt_eval_count": 4747,
            "eval_count": 291,
        },
        "no_context_probe": {
            "source": "provided video frame at 10015 seconds",
            "context_relevance": "unavailable",
            "elapsed_seconds": 2.340,
        },
        "observation": (
            "generated screen-text summaries mixed nearby frame/cue meaning, so the v1 "
            "contract retains only screen_text_kind and supporting cue IDs; "
            "it does not "
            "publish model-generated text as an exact quotation"
        ),
    }


def prototype_cases() -> list[Payload]:
    """人間が設計判断を確認するcase一覧を返す。"""
    return [
        architecture_case(),
        {"case": "unique responsibility matrix", "rows": responsibility_rows()},
        candidate_annotation_case(),
        retry_and_failure_case(),
        invalidation_case(),
        density_comparison(),
        capability_probe_case(),
    ]
