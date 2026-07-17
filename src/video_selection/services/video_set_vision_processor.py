"""Video Set単位のScene CatalogとCandidate Annotation Stage。"""

import hashlib
from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import TypeVar

from ..models.candidate_annotation import (
    CandidateAnnotation,
    candidate_annotation_context_is_valid,
    candidate_annotation_free_text_is_safe,
    candidate_annotation_relationships_are_valid,
)
from ..models.candidate_annotation_request import CandidateAnnotationRequest
from ..models.completed_stage import CompletedStage
from ..models.effective_configuration import EffectiveConfiguration
from ..models.frame_candidate import FrameCandidate
from ..models.model_role import ModelRole
from ..models.processing_stage import ProcessingStage
from ..models.resolved_model import ResolvedModel
from ..models.resolved_models import ResolvedModels
from ..models.scene_catalog import SceneCatalog
from ..models.scene_catalog_request import SceneCatalogRequest
from ..models.stage_fingerprint import StageFingerprint
from ..models.video_set import VideoSet
from ..models.vision_inference_diagnostics import VisionInferenceDiagnostics
from ..models.vision_stage_result import VisionStageResult
from ..protocols.run_observer import RunObserver
from ..protocols.vision_runtime import VisionRuntime
from ..vision.vision_contract import (
    CANDIDATE_ANNOTATION_PROMPT_VERSION,
    CANDIDATE_ANNOTATION_SCHEMA_VERSION,
    CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION,
    RETRY_POLICY_VERSION,
    SCENE_CATALOG_PROMPT_VERSION,
    SCENE_CATALOG_SCHEMA_VERSION,
    SCENE_CATALOG_STAGE_CONTRACT_VERSION,
)
from .build_stage_fingerprint import build_stage_fingerprint
from .completed_stage_writer import CompletedStageWriter
from .external_work_monitor import ExternalWorkMonitor
from .run_progress_tracker import RunProgressTracker
from .snapshot_frame_candidates import snapshot_frame_candidates
from .validate_video_set_snapshot import validate_video_set_snapshot
from .vision_stage_artifacts import (
    restore_candidate_annotation,
    restore_scene_catalog,
    serialize_candidate_annotation,
    serialize_scene_catalog,
)

VisionStageValue = TypeVar("VisionStageValue")


class VideoSetVisionProcessor:
    """VisionRuntimeとMoment単位atomic cacheを一つの深いmoduleに保つ。"""

    def __init__(
        self,
        runtime: VisionRuntime,
        observer: RunObserver,
        *,
        progress: RunProgressTracker | None = None,
    ) -> None:
        self._runtime = runtime
        self._observer = observer
        self._progress = progress
        self._external_work = (
            ExternalWorkMonitor(progress) if progress is not None else None
        )

    def process(
        self,
        *,
        video_set: VideoSet,
        representatives: tuple[FrameCandidate, ...],
        representative_source_fingerprints: tuple[StageFingerprint, ...],
        annotation_requests: tuple[CandidateAnnotationRequest, ...],
        configuration: EffectiveConfiguration,
        resolved_models: ResolvedModels,
    ) -> VisionStageResult:
        """共有Catalogと指定ShortlistのAnnotationを確定または再利用する。"""
        selection_intent = _validate_inputs(
            representatives,
            representative_source_fingerprints,
            annotation_requests,
        )
        writer = CompletedStageWriter(
            configuration.processing_cache_folder,
            subject_namespace="video-sets",
            subject_fingerprint=video_set.fingerprint,
        )
        catalog_model = resolved_models.for_role(ModelRole.SCENE_CATALOG)
        annotation_model = resolved_models.for_role(ModelRole.CANDIDATE_ANNOTATION)
        catalog_request = SceneCatalogRequest(
            representatives=representatives,
            selection_intent=selection_intent,
            scene_hint=configuration.scene_hint,
        )
        validate_video_set_snapshot(video_set)
        catalog, catalog_diagnostics, catalog_stage = self._catalog_stage(
            writer,
            video_set,
            catalog_request,
            representative_source_fingerprints,
            catalog_model,
            configuration.scene_catalog_num_ctx,
        )
        annotations: list[CandidateAnnotation] = []
        annotation_diagnostics: list[VisionInferenceDiagnostics] = []
        completed_stages = [catalog_stage]
        for request in annotation_requests:
            validate_video_set_snapshot(video_set)
            annotation, diagnostics, completed = self._annotation_stage(
                writer,
                video_set,
                request,
                catalog,
                catalog_stage.fingerprint,
                annotation_model,
                configuration.candidate_annotation_num_ctx,
            )
            annotations.append(annotation)
            annotation_diagnostics.append(diagnostics)
            completed_stages.append(completed)
        return VisionStageResult(
            catalog=catalog,
            annotations=tuple(annotations),
            catalog_diagnostics=catalog_diagnostics,
            annotation_diagnostics=tuple(annotation_diagnostics),
            completed_stages=tuple(completed_stages),
        )

    def _catalog_stage(
        self,
        writer: CompletedStageWriter,
        video_set: VideoSet,
        request: SceneCatalogRequest,
        upstream_fingerprints: tuple[StageFingerprint, ...],
        model: ResolvedModel,
        num_ctx: int,
    ) -> tuple[SceneCatalog, VisionInferenceDiagnostics, CompletedStage]:
        """Video Setごとに一つのScene Catalog Stageを扱う。"""
        semantic_input = _catalog_semantic_input(video_set, request, model, num_ctx)
        fingerprint = build_stage_fingerprint(
            ProcessingStage.BUILD_SCENE_CATALOG,
            upstream_fingerprints,
            semantic_input,
        )
        self._start_progress_stage(
            ProcessingStage.BUILD_SCENE_CATALOG,
            "scene_catalog",
        )
        (catalog, diagnostics), completed, reused = _execute_cached_vision_stage(
            writer=writer,
            stage=ProcessingStage.BUILD_SCENE_CATALOG,
            fingerprint=fingerprint,
            upstream_fingerprints=upstream_fingerprints,
            semantic_input=semantic_input,
            generate=lambda: self._run_external(
                lambda: self._runtime.create_scene_catalog(
                    request,
                    model,
                    num_ctx=num_ctx,
                ),
                reason_code="scene_catalog_inference_started",
            ),
            serialize=lambda value: serialize_scene_catalog(*value),
            restore=restore_scene_catalog,
            artifact_label="Scene Catalog",
        )
        self._complete_progress_stage(reused)
        self._observer.stage_completed(completed)
        return catalog, diagnostics, completed

    def _annotation_stage(
        self,
        writer: CompletedStageWriter,
        video_set: VideoSet,
        request: CandidateAnnotationRequest,
        catalog: SceneCatalog,
        catalog_fingerprint: StageFingerprint,
        model: ResolvedModel,
        num_ctx: int,
    ) -> tuple[CandidateAnnotation, VisionInferenceDiagnostics, CompletedStage]:
        """一つのCandidate Momentだけを独立Stageとして扱う。"""
        semantic_input = _annotation_semantic_input(
            video_set,
            request,
            catalog_fingerprint,
            model,
            num_ctx,
        )
        upstream = (catalog_fingerprint,)
        fingerprint = build_stage_fingerprint(
            ProcessingStage.ANNOTATE_CANDIDATE,
            upstream,
            semantic_input,
        )

        def generate() -> tuple[CandidateAnnotation, VisionInferenceDiagnostics]:
            generated = self._run_external(
                lambda: self._runtime.annotate_candidate(
                    request,
                    catalog,
                    model,
                    num_ctx=num_ctx,
                ),
                reason_code="candidate_annotation_inference_started",
            )
            annotation, diagnostics = generated
            _validate_runtime_annotation(annotation, request, catalog)
            return generated

        self._start_progress_stage(
            ProcessingStage.ANNOTATE_CANDIDATE,
            "candidate",
        )
        (annotation, diagnostics), completed, reused = _execute_cached_vision_stage(
            writer=writer,
            stage=ProcessingStage.ANNOTATE_CANDIDATE,
            fingerprint=fingerprint,
            upstream_fingerprints=upstream,
            semantic_input=semantic_input,
            generate=generate,
            serialize=lambda value: serialize_candidate_annotation(*value),
            restore=lambda artifact: restore_candidate_annotation(
                artifact,
                request,
                catalog,
            ),
            artifact_label="Candidate Annotation",
        )
        self._complete_progress_stage(reused)
        self._observer.stage_completed(completed)
        return annotation, diagnostics, completed

    def _start_progress_stage(
        self,
        stage: ProcessingStage,
        work_unit_kind: str,
    ) -> None:
        if self._progress is not None:
            self._progress.start_stage(stage, work_unit_kind=work_unit_kind)

    def _complete_progress_stage(self, reused: bool) -> None:
        if self._progress is None:
            return
        self._progress.record_work_sample("reuse" if reused else "recompute")
        self._progress.cache_observed(
            cache_hit_count=1 if reused else 0,
            cache_miss_count=0 if reused else 1,
            reuse_count=1 if reused else 0,
            recompute_count=0 if reused else 1,
            reason_code="cache_reused" if reused else "stage_recomputed",
        )
        self._progress.complete_stage()

    def _run_external(
        self,
        operation: Callable[[], VisionStageValue],
        *,
        reason_code: str,
    ) -> VisionStageValue:
        if self._external_work is None:
            return operation()
        return self._external_work.run(operation, reason_code=reason_code)


def _execute_cached_vision_stage(
    *,
    writer: CompletedStageWriter,
    stage: ProcessingStage,
    fingerprint: StageFingerprint,
    upstream_fingerprints: tuple[StageFingerprint, ...],
    semantic_input: Mapping[str, object],
    generate: Callable[[], VisionStageValue],
    serialize: Callable[[VisionStageValue], dict[str, object]],
    restore: Callable[[Mapping[str, object]], VisionStageValue],
    artifact_label: str,
) -> tuple[VisionStageValue, CompletedStage, bool]:
    """同じfingerprintの生成と復元を一つのlock lifecycleで確定する。"""
    generated: VisionStageValue | None = None

    def produce(_stage_folder: Path) -> dict[str, object]:
        nonlocal generated
        generated = generate()
        return serialize(generated)

    completed = writer.write_artifacts(
        stage,
        fingerprint,
        upstream_fingerprints,
        semantic_input,
        produce,
    )
    if generated is not None:
        return generated, completed, False
    artifact = writer.read(
        stage,
        fingerprint,
        upstream_fingerprints,
        semantic_input,
    )
    if artifact is None:
        raise RuntimeError(f"確定した{artifact_label} artifactを復元できません")
    return restore(artifact), completed, True


def _validate_inputs(
    representatives: tuple[FrameCandidate, ...],
    representative_source_fingerprints: tuple[StageFingerprint, ...],
    requests: tuple[CandidateAnnotationRequest, ...],
) -> str:
    if not requests:
        raise ValueError("Selection Shortlistには1件以上のCandidate Momentが必要です")
    selection_intents = {request.selection_intent for request in requests}
    moment_ids = tuple(request.moment.identifier for request in requests)
    fingerprint_values = tuple(
        fingerprint.value for fingerprint in representative_source_fingerprints
    )
    if (
        not representative_source_fingerprints
        or len(fingerprint_values) != len(set(fingerprint_values))
        or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in fingerprint_values
        )
        or len(selection_intents) != 1
        or len(moment_ids) != len(set(moment_ids))
    ):
        raise ValueError("Vision Stage inputのSelection IntentまたはMomentが不正です")
    selection_intent = next(iter(selection_intents))
    SceneCatalogRequest(representatives, selection_intent)
    return selection_intent


def _validate_runtime_annotation(
    annotation: CandidateAnnotation,
    request: CandidateAnnotationRequest,
    catalog: SceneCatalog,
) -> None:
    frames_by_id = {item.identifier: item for item in request.frame_candidates}
    expected_candidate = frames_by_id.get(annotation.candidate.identifier)
    cue_ids = tuple(item.identifier for item in request.context_cues)
    if (
        annotation.candidate_moment_id != request.moment.identifier
        or expected_candidate != annotation.candidate
        or annotation.scene_slug not in catalog.slugs
        or not candidate_annotation_relationships_are_valid(
            annotation.context_relevance,
            annotation.supporting_context_cue_ids,
            annotation.spoiler_risk,
            annotation.spoiler_evidence,
        )
        or not candidate_annotation_context_is_valid(
            annotation.context_relevance,
            annotation.supporting_context_cue_ids,
            cue_ids,
        )
        or not candidate_annotation_free_text_is_safe(
            (
                annotation.summary,
                annotation.frame_choice_reason or "",
                annotation.spoiler_evidence,
            ),
            tuple(item.text for item in request.context_cues),
        )
    ):
        raise ValueError("VisionRuntimeがrequest外のCandidate Annotationを返しました")


def _catalog_semantic_input(
    video_set: VideoSet,
    request: SceneCatalogRequest,
    model: ResolvedModel,
    num_ctx: int,
) -> dict[str, object]:
    return {
        "video_set_fingerprint": video_set.fingerprint,
        "representatives": list(snapshot_frame_candidates(request.representatives)),
        "selection_intent": request.selection_intent,
        "scene_hint": request.scene_hint,
        "model": {**model.semantic_input(), "num_ctx": num_ctx},
        "generation_options": {"temperature": 0, "stream": False, "think": False},
        "prompt_version": SCENE_CATALOG_PROMPT_VERSION,
        "schema_version": SCENE_CATALOG_SCHEMA_VERSION,
        "stage_contract_version": SCENE_CATALOG_STAGE_CONTRACT_VERSION,
        "retry_policy_version": RETRY_POLICY_VERSION,
    }


def _annotation_semantic_input(
    video_set: VideoSet,
    request: CandidateAnnotationRequest,
    catalog_fingerprint: StageFingerprint,
    model: ResolvedModel,
    num_ctx: int,
) -> dict[str, object]:
    return {
        "video_set_fingerprint": video_set.fingerprint,
        "candidate_moment_id": request.moment.identifier,
        "frame_candidates": list(snapshot_frame_candidates(request.frame_candidates)),
        "context_cues": [
            {
                "id": cue.identifier,
                "source_kind": cue.source_kind,
                "start": _fraction_value(cue.start),
                "end": _fraction_value(cue.end),
                "text_sha256": hashlib.sha256(cue.text.encode()).hexdigest(),
            }
            for cue in request.context_cues
        ],
        "cue_selection_policy_version": request.cue_selection_policy_version,
        "scene_catalog_fingerprint": catalog_fingerprint.value,
        "video_set_progress": _fraction_value(request.video_set_progress),
        "selection_intent": request.selection_intent,
        "model": {**model.semantic_input(), "num_ctx": num_ctx},
        "generation_options": {"temperature": 0, "stream": False, "think": False},
        "prompt_version": CANDIDATE_ANNOTATION_PROMPT_VERSION,
        "schema_version": CANDIDATE_ANNOTATION_SCHEMA_VERSION,
        "stage_contract_version": CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION,
        "retry_policy_version": RETRY_POLICY_VERSION,
    }


def _fraction_value(value: Fraction) -> Mapping[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}
