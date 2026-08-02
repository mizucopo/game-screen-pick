"""Scene CatalogとCandidate Annotationのcache artifact変換。"""

from dataclasses import replace
from typing import Mapping, cast

from ..models.candidate_annotation import (
    BLOG_IMAGE_TYPES,
    CONTEXT_CUE_RELEVANCES,
    EXPLANATION_VALUES,
    SCREEN_TEXT_KINDS,
    SPOILER_RISKS,
    CandidateAnnotation,
    candidate_annotation_context_is_valid,
    candidate_annotation_free_text_is_safe,
    candidate_annotation_relationships_are_valid,
)
from ..models.candidate_annotation_request import CandidateAnnotationRequest
from ..models.combat_encounter_kind import COMBAT_ENCOUNTER_KINDS
from ..models.scene_catalog import SceneCatalog
from ..models.scene_catalog_entry import (
    SCENE_SELECTION_ROLES,
    SceneCatalogEntry,
    SceneSelectionRole,
)
from ..models.scene_kind import SCENE_KINDS, SceneKind
from ..models.vision_inference_diagnostics import VisionInferenceDiagnostics

_CATALOG_SCHEMA = "game-screen-pick/scene-catalog@2.0.0"
_ANNOTATION_SCHEMA = "game-screen-pick/candidate-annotation@3.0.0"


def serialize_scene_catalog(
    catalog: SceneCatalog,
    diagnostics: VisionInferenceDiagnostics,
) -> dict[str, object]:
    """domain検証済みCatalogとsafe diagnosticsだけを保存する。"""
    return {
        "schema": _CATALOG_SCHEMA,
        "scenes": [_scene_value(scene) for scene in catalog.scenes],
        "diagnostics": _diagnostics_value(diagnostics),
    }


def restore_scene_catalog(
    artifact: Mapping[str, object],
) -> tuple[SceneCatalog, VisionInferenceDiagnostics]:
    """Catalog artifactをstrictに復元してcache hitを記録する。"""
    if artifact.get("schema") != _CATALOG_SCHEMA:
        raise ValueError("Scene Catalog artifact schemaが不正です")
    raw_scenes = artifact.get("scenes")
    if not isinstance(raw_scenes, list):
        raise ValueError("Scene Catalog artifact scenesが不正です")
    scenes = tuple(_restore_scene(item) for item in raw_scenes)
    diagnostics = _restore_diagnostics(artifact.get("diagnostics"))
    return SceneCatalog(scenes), replace(diagnostics, cache_hit=True)


def serialize_candidate_annotation(
    annotation: CandidateAnnotation,
    diagnostics: VisionInferenceDiagnostics,
) -> dict[str, object]:
    """model応答ではなく検証済みannotationとsafe diagnosticsを保存する。"""
    return {
        "schema": _ANNOTATION_SCHEMA,
        "annotation": {
            "candidate_moment_id": annotation.candidate_moment_id,
            "representative_frame_id": annotation.candidate.identifier,
            "scene_slug": annotation.scene_slug,
            "blog_image_type": annotation.blog_image_type,
            "explanation_value": annotation.explanation_value,
            "annotation_summary": annotation.summary,
            "frame_choice_reason": annotation.frame_choice_reason,
            "screen_text_kind": annotation.screen_text_kind,
            "context_relevance": annotation.context_relevance,
            "supporting_context_cue_ids": list(annotation.supporting_context_cue_ids),
            "spoiler_risk": annotation.spoiler_risk,
            "spoiler_evidence": annotation.spoiler_evidence,
            "combat_encounter_kind": annotation.combat_encounter_kind,
        },
        "diagnostics": _diagnostics_value(diagnostics),
    }


def restore_candidate_annotation(
    artifact: Mapping[str, object],
    request: CandidateAnnotationRequest,
    catalog: SceneCatalog,
) -> tuple[CandidateAnnotation, VisionInferenceDiagnostics]:
    """Annotation artifactを現在のFrame/Cue/Catalogへ所属検証して復元する。"""
    if artifact.get("schema") != _ANNOTATION_SCHEMA:
        raise ValueError("Candidate Annotation artifact schemaが不正です")
    raw_annotation = artifact.get("annotation")
    if not isinstance(raw_annotation, dict) or not all(
        isinstance(key, str) for key in raw_annotation
    ):
        raise ValueError("Candidate Annotation artifact fieldが不正です")
    annotation = cast(dict[str, object], raw_annotation)
    candidate_moment_id = annotation.get("candidate_moment_id")
    representative_frame_id = annotation.get("representative_frame_id")
    scene_slug = annotation.get("scene_slug")
    blog_image_type = annotation.get("blog_image_type")
    explanation_value = annotation.get("explanation_value")
    summary = annotation.get("annotation_summary")
    frame_choice_reason = annotation.get("frame_choice_reason")
    screen_text_kind = annotation.get("screen_text_kind")
    context_relevance = annotation.get("context_relevance")
    raw_cue_ids = annotation.get("supporting_context_cue_ids")
    spoiler_risk = annotation.get("spoiler_risk")
    spoiler_evidence = annotation.get("spoiler_evidence")
    combat_encounter_kind = annotation.get("combat_encounter_kind")
    frames = {item.identifier: item for item in request.frame_candidates}
    available_cue_ids = tuple(item.identifier for item in request.context_cues)
    if (
        candidate_moment_id != request.moment.identifier
        or not isinstance(representative_frame_id, str)
        or representative_frame_id not in frames
        or not isinstance(scene_slug, str)
        or scene_slug not in catalog.slugs
        or blog_image_type not in BLOG_IMAGE_TYPES
        or explanation_value not in EXPLANATION_VALUES
        or not isinstance(summary, str)
        or not isinstance(frame_choice_reason, str)
        or screen_text_kind not in SCREEN_TEXT_KINDS
        or context_relevance not in CONTEXT_CUE_RELEVANCES
        or not isinstance(raw_cue_ids, list)
        or not all(isinstance(item, str) for item in raw_cue_ids)
        or spoiler_risk not in SPOILER_RISKS
        or not isinstance(spoiler_evidence, str)
        or combat_encounter_kind not in COMBAT_ENCOUNTER_KINDS
    ):
        raise ValueError("Candidate Annotation artifact domainが不正です")
    typed_context_relevance = context_relevance
    typed_cue_ids = tuple(cast(list[str], raw_cue_ids))
    typed_spoiler_risk = spoiler_risk
    typed_combat_encounter_kind = combat_encounter_kind
    if (
        not candidate_annotation_relationships_are_valid(
            typed_context_relevance,
            typed_cue_ids,
            typed_spoiler_risk,
            spoiler_evidence,
        )
        or not candidate_annotation_context_is_valid(
            typed_context_relevance,
            typed_cue_ids,
            available_cue_ids,
        )
        or not candidate_annotation_free_text_is_safe(
            (summary, frame_choice_reason, spoiler_evidence),
            tuple(item.text for item in request.context_cues),
        )
    ):
        raise ValueError("Candidate Annotation artifact domainが不正です")
    restored = CandidateAnnotation(
        candidate=frames[representative_frame_id],
        summary=summary,
        candidate_moment_id=candidate_moment_id,
        scene_slug=scene_slug,
        blog_image_type=blog_image_type,
        explanation_value=explanation_value,
        frame_choice_reason=frame_choice_reason,
        screen_text_kind=screen_text_kind,
        context_relevance=typed_context_relevance,
        supporting_context_cue_ids=typed_cue_ids,
        spoiler_risk=typed_spoiler_risk,
        spoiler_evidence=spoiler_evidence,
        combat_encounter_kind=typed_combat_encounter_kind,
    )
    diagnostics = _restore_diagnostics(artifact.get("diagnostics"))
    return restored, replace(diagnostics, cache_hit=True)


def _scene_value(scene: SceneCatalogEntry) -> dict[str, str]:
    return {
        "slug": scene.slug,
        "display_name": scene.display_name,
        "description": scene.description,
        "scene_kind": scene.scene_kind,
        "selection_role": scene.selection_role,
    }


def _restore_scene(value: object) -> SceneCatalogEntry:
    if not isinstance(value, dict):
        raise ValueError("Scene Catalog artifact entryが不正です")
    slug = value.get("slug")
    display_name = value.get("display_name")
    description = value.get("description")
    scene_kind = value.get("scene_kind")
    selection_role = value.get("selection_role")
    if (
        not isinstance(slug, str)
        or not isinstance(display_name, str)
        or not isinstance(description, str)
        or scene_kind not in SCENE_KINDS
        or selection_role not in SCENE_SELECTION_ROLES
    ):
        raise ValueError("Scene Catalog artifact entry fieldが不正です")
    return SceneCatalogEntry(
        slug,
        display_name,
        description,
        cast(SceneKind, scene_kind),
        cast(SceneSelectionRole, selection_role),
    )


def _diagnostics_value(
    diagnostics: VisionInferenceDiagnostics,
) -> dict[str, object]:
    return {
        "request_fingerprint": diagnostics.request_fingerprint,
        "model_name": diagnostics.model_name,
        "model_identity": diagnostics.model_identity,
        "runtime_identity": diagnostics.runtime_identity,
        "prompt_version": diagnostics.prompt_version,
        "schema_version": diagnostics.schema_version,
        "stage_contract_version": diagnostics.stage_contract_version,
        "retry_policy_version": diagnostics.retry_policy_version,
        "cache_hit": diagnostics.cache_hit,
        "attempt_count": diagnostics.attempt_count,
        "validation_code": diagnostics.validation_code,
        "image_count": diagnostics.image_count,
        "context_cue_count": diagnostics.context_cue_count,
        "duration_seconds": diagnostics.duration_seconds,
        "prompt_eval_count": diagnostics.prompt_eval_count,
        "eval_count": diagnostics.eval_count,
        "done_reason": diagnostics.done_reason,
    }


def _restore_diagnostics(value: object) -> VisionInferenceDiagnostics:
    if not isinstance(value, dict):
        raise ValueError("Vision diagnostics artifactが不正です")
    required_strings = (
        "request_fingerprint",
        "model_name",
        "model_identity",
        "runtime_identity",
        "prompt_version",
        "schema_version",
        "stage_contract_version",
        "retry_policy_version",
    )
    if not all(isinstance(value.get(key), str) for key in required_strings):
        raise ValueError("Vision diagnostics artifact fieldが不正です")
    attempt_count = value.get("attempt_count")
    image_count = value.get("image_count")
    context_cue_count = value.get("context_cue_count")
    duration_seconds = value.get("duration_seconds")
    cache_hit = value.get("cache_hit")
    if (
        not isinstance(attempt_count, int)
        or isinstance(attempt_count, bool)
        or not isinstance(image_count, int)
        or isinstance(image_count, bool)
        or not isinstance(context_cue_count, int)
        or isinstance(context_cue_count, bool)
        or not isinstance(duration_seconds, int | float)
        or isinstance(duration_seconds, bool)
        or not isinstance(cache_hit, bool)
    ):
        raise ValueError("Vision diagnostics artifact numeric fieldが不正です")
    validation_code = value.get("validation_code")
    done_reason = value.get("done_reason")
    prompt_eval_count = value.get("prompt_eval_count")
    eval_count = value.get("eval_count")
    if (
        validation_code is not None
        and not isinstance(validation_code, str)
        or done_reason is not None
        and not isinstance(done_reason, str)
        or prompt_eval_count is not None
        and (
            not isinstance(prompt_eval_count, int)
            or isinstance(prompt_eval_count, bool)
        )
        or eval_count is not None
        and (not isinstance(eval_count, int) or isinstance(eval_count, bool))
    ):
        raise ValueError("Vision diagnostics artifact optional fieldが不正です")
    return VisionInferenceDiagnostics(
        request_fingerprint=cast(str, value["request_fingerprint"]),
        model_name=cast(str, value["model_name"]),
        model_identity=cast(str, value["model_identity"]),
        runtime_identity=cast(str, value["runtime_identity"]),
        prompt_version=cast(str, value["prompt_version"]),
        schema_version=cast(str, value["schema_version"]),
        stage_contract_version=cast(str, value["stage_contract_version"]),
        retry_policy_version=cast(str, value["retry_policy_version"]),
        cache_hit=cache_hit,
        attempt_count=attempt_count,
        validation_code=validation_code,
        image_count=image_count,
        context_cue_count=context_cue_count,
        duration_seconds=float(duration_seconds),
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
        done_reason=done_reason,
    )
