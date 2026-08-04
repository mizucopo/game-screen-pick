"""Video Set単位のScene CatalogとCandidate Annotation Stage。"""

import hashlib
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from threading import BoundedSemaphore
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
from ..vision.detect_cinematic_letterbox import (
    CINEMATIC_LETTERBOX_DETECTION_VERSION,
)
from ..vision.vision_contract import (
    CANDIDATE_ANNOTATION_PROMPT_VERSION,
    CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_EVIDENCE_MAX_LENGTH,
    CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_NUM_PREDICT,
    CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_PROMPT_VERSION,
    CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_SCHEMA_VERSION,
    CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_STAGE_CONTRACT_VERSION,
    CANDIDATE_ANNOTATION_SCHEMA_VERSION,
    CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION,
    COMBAT_ENCOUNTER_CONFIRMATION_PROMPT_VERSION,
    COMBAT_ENCOUNTER_CONFIRMATION_STAGE_CONTRACT_VERSION,
    COMBAT_ENCOUNTER_VERIFICATION_PROMPT_VERSION,
    COMBAT_ENCOUNTER_VERIFICATION_SCHEMA_VERSION,
    COMBAT_ENCOUNTER_VERIFICATION_STAGE_CONTRACT_VERSION,
    COMBAT_VISIBILITY_CONFIRMATION_PROMPT_VERSION,
    COMBAT_VISIBILITY_CONFIRMATION_STAGE_CONTRACT_VERSION,
    COMBAT_VISIBILITY_EDGE_AUDIT_PROMPT_VERSION,
    COMBAT_VISIBILITY_EDGE_AUDIT_SCHEMA_VERSION,
    COMBAT_VISIBILITY_EDGE_AUDIT_STAGE_CONTRACT_VERSION,
    COMBAT_VISIBILITY_EDGE_STRIP_VERSION,
    COMBAT_VISIBILITY_VERIFICATION_PROMPT_VERSION,
    COMBAT_VISIBILITY_VERIFICATION_SCHEMA_VERSION,
    COMBAT_VISIBILITY_VERIFICATION_STAGE_CONTRACT_VERSION,
    PUBLICATION_BOUNDARY_VERIFICATION_PROMPT_VERSION,
    PUBLICATION_BOUNDARY_VERIFICATION_SCHEMA_VERSION,
    PUBLICATION_BOUNDARY_VERIFICATION_STAGE_CONTRACT_VERSION,
    RETRY_POLICY_VERSION,
    SCENE_CATALOG_PROMPT_VERSION,
    SCENE_CATALOG_SCHEMA_VERSION,
    SCENE_CATALOG_STAGE_CONTRACT_VERSION,
    VISION_GENERATION_SEED,
)
from .build_stage_fingerprint import build_stage_fingerprint
from .completed_stage_writer import CompletedStageWriter
from .external_work_monitor import ExternalWorkMonitor
from .run_progress_tracker import RunProgressTracker
from .snapshot_frame_candidates import snapshot_frame_candidates
from .validate_video_set_snapshot import validate_video_set_snapshot_metadata
from .vision_stage_artifacts import (
    restore_candidate_annotation,
    restore_scene_catalog,
    serialize_candidate_annotation,
    serialize_scene_catalog,
)

VisionStageValue = TypeVar("VisionStageValue")
type _CachedAnnotationStageResult = tuple[
    CandidateAnnotation,
    VisionInferenceDiagnostics,
    CompletedStage,
    bool,
]
type _CandidateMomentPlan = tuple[
    tuple[CandidateAnnotationRequest, ...],
    tuple[_CachedAnnotationStageResult | None, ...],
]
type _CandidateMomentExecution = tuple[
    CandidateAnnotation | None,
    tuple[_CachedAnnotationStageResult, ...],
    Exception | None,
]
_EXPLANATION_PRIORITY = {"none": 0, "low": 1, "medium": 2, "high": 3}
_CONTENT_PRIORITY = {
    "document": 0,
    "tutorial_help": 0,
    "event_setup": 0,
    "gameplay_idle": 1,
    "save": 2,
    "map": 3,
    "other_interface": 3,
    "other": 4,
    "shop": 4,
    "title": 4,
    "event_action": 5,
    "gameplay_action": 5,
    "event_dialogue": 6,
}
_VISIBILITY_PRIORITY = {"absent": 0, "partial": 1, "clear": 2}
_OBSTRUCTION_PRIORITY = {"severe": 0, "partial": 1, "none": 2}


def plan_vision_stage_fingerprints(
    *,
    video_set: VideoSet,
    representatives: tuple[FrameCandidate, ...],
    representative_source_fingerprints: tuple[StageFingerprint, ...],
    annotation_requests: tuple[CandidateAnnotationRequest, ...],
    configuration: EffectiveConfiguration,
    resolved_models: ResolvedModels,
) -> tuple[StageFingerprint, ...]:
    """推論せずCatalogと全Annotationの入力fingerprintを計画する。"""
    selection_intent = _validate_inputs(
        representatives,
        representative_source_fingerprints,
        annotation_requests,
    )
    catalog_request = SceneCatalogRequest(
        representatives=representatives,
        selection_intent=selection_intent,
        scene_hint=configuration.scene_hint,
    )
    catalog_fingerprint = build_stage_fingerprint(
        ProcessingStage.BUILD_SCENE_CATALOG,
        representative_source_fingerprints,
        _catalog_semantic_input(
            video_set,
            catalog_request,
            resolved_models.for_role(ModelRole.SCENE_CATALOG),
            configuration.scene_catalog_num_ctx,
        ),
    )
    annotation_model = resolved_models.for_role(ModelRole.CANDIDATE_ANNOTATION)
    return (
        catalog_fingerprint,
        *(
            build_stage_fingerprint(
                ProcessingStage.ANNOTATE_CANDIDATE,
                (catalog_fingerprint,),
                _annotation_semantic_input(
                    video_set,
                    frame_request,
                    catalog_fingerprint,
                    annotation_model,
                    configuration.candidate_annotation_num_ctx,
                ),
            )
            for request in annotation_requests
            for frame_request in _single_frame_annotation_requests(request)
        ),
    )


class VideoSetVisionProcessor:
    """VisionRuntimeとFrame単位atomic cacheを一つの深いmoduleに保つ。"""

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
        validate_video_set_snapshot_metadata(video_set)
        catalog, catalog_diagnostics, catalog_stage = self._catalog_stage(
            writer,
            video_set,
            catalog_request,
            representative_source_fingerprints,
            catalog_model,
            configuration.scene_catalog_num_ctx,
        )
        annotations, annotation_diagnostics, annotation_stages = (
            self._annotation_stages(
                writer=writer,
                video_set=video_set,
                requests=annotation_requests,
                catalog=catalog,
                catalog_fingerprint=catalog_stage.fingerprint,
                model=annotation_model,
                num_ctx=configuration.candidate_annotation_num_ctx,
                max_parallel_requests=configuration.ollama_max_parallel_requests,
            )
        )
        completed_stages = [catalog_stage, *annotation_stages]
        result = VisionStageResult(
            catalog=catalog,
            annotations=annotations,
            catalog_diagnostics=catalog_diagnostics,
            annotation_diagnostics=annotation_diagnostics,
            completed_stages=tuple(completed_stages),
        )
        validate_video_set_snapshot_metadata(video_set)
        return result

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
        self._complete_progress_stage(reused, completed.fingerprint)
        self._observer.stage_completed(completed)
        return catalog, diagnostics, completed

    def _annotation_stages(
        self,
        *,
        writer: CompletedStageWriter,
        video_set: VideoSet,
        requests: tuple[CandidateAnnotationRequest, ...],
        catalog: SceneCatalog,
        catalog_fingerprint: StageFingerprint,
        model: ResolvedModel,
        num_ctx: int,
        max_parallel_requests: int,
    ) -> tuple[
        tuple[CandidateAnnotation, ...],
        tuple[VisionInferenceDiagnostics, ...],
        tuple[CompletedStage, ...],
    ]:
        """Candidate Momentを独立cacheのままbounded並列評価する。"""
        plans = tuple(
            self._plan_candidate_moment(
                writer=writer,
                video_set=video_set,
                request=request,
                catalog=catalog,
                catalog_fingerprint=catalog_fingerprint,
                model=model,
                num_ctx=num_ctx,
            )
            for request in requests
        )
        inference_limiter = BoundedSemaphore(max_parallel_requests)
        return self._execute_candidate_moments(
            writer=writer,
            video_set=video_set,
            plans=plans,
            catalog=catalog,
            catalog_fingerprint=catalog_fingerprint,
            model=model,
            num_ctx=num_ctx,
            max_parallel_requests=max_parallel_requests,
            inference_limiter=inference_limiter,
        )

    def _plan_candidate_moment(
        self,
        *,
        writer: CompletedStageWriter,
        video_set: VideoSet,
        request: CandidateAnnotationRequest,
        catalog: SceneCatalog,
        catalog_fingerprint: StageFingerprint,
        model: ResolvedModel,
        num_ctx: int,
    ) -> _CandidateMomentPlan:
        """一Momentの既存frame cacheと条件付きfallback範囲を確定する。"""
        validate_video_set_snapshot_metadata(video_set)
        frame_requests = _single_frame_annotation_requests(request)
        primary = _restore_cached_annotation_stage(
            writer=writer,
            video_set=video_set,
            request=frame_requests[0],
            catalog=catalog,
            catalog_fingerprint=catalog_fingerprint,
            model=model,
            num_ctx=num_ctx,
        )
        restored: tuple[_CachedAnnotationStageResult | None, ...] = (primary,)
        if primary is not None and _combat_fallback_is_required(
            primary[0],
            frame_requests,
        ):
            restored = (
                primary,
                *(
                    _restore_cached_annotation_stage(
                        writer=writer,
                        video_set=video_set,
                        request=fallback,
                        catalog=catalog,
                        catalog_fingerprint=catalog_fingerprint,
                        model=model,
                        num_ctx=num_ctx,
                    )
                    for fallback in frame_requests[1:]
                ),
            )
        return frame_requests, restored

    def _execute_candidate_moments(
        self,
        *,
        writer: CompletedStageWriter,
        video_set: VideoSet,
        plans: tuple[_CandidateMomentPlan, ...],
        catalog: SceneCatalog,
        catalog_fingerprint: StageFingerprint,
        model: ResolvedModel,
        num_ctx: int,
        max_parallel_requests: int,
        inference_limiter: BoundedSemaphore,
    ) -> tuple[
        tuple[CandidateAnnotation, ...],
        tuple[VisionInferenceDiagnostics, ...],
        tuple[CompletedStage, ...],
    ]:
        """全Momentを連続並列実行し結果とprogressを入力順で確定する。"""
        self._start_progress_stage(
            ProcessingStage.ANNOTATE_CANDIDATE,
            "candidate",
        )
        pending = tuple(
            (index, plan)
            for index, plan in enumerate(plans)
            if _candidate_moment_plan_requires_inference(plan)
        )
        pending_indexes = {index for index, _plan in pending}

        def execute_plan(plan: _CandidateMomentPlan) -> _CandidateMomentExecution:
            return self._execute_candidate_moment(
                writer=writer,
                video_set=video_set,
                plan=plan,
                catalog=catalog,
                catalog_fingerprint=catalog_fingerprint,
                model=model,
                num_ctx=num_ctx,
                max_parallel_requests=max_parallel_requests,
                inference_limiter=inference_limiter,
            )

        def execute_plans() -> tuple[_CandidateMomentExecution, ...]:
            results: list[_CandidateMomentExecution | None] = [None] * len(plans)
            if pending:
                executor: ThreadPoolExecutor | None = None
                futures: list[tuple[int, Future[_CandidateMomentExecution]]] = []
                try:
                    executor = ThreadPoolExecutor(
                        max_workers=min(max_parallel_requests, len(pending)),
                        thread_name_prefix="candidate-moment",
                    )
                    for index, plan in pending:
                        futures.append((index, executor.submit(execute_plan, plan)))
                    for index, plan in enumerate(plans):
                        if index not in pending_indexes:
                            results[index] = execute_plan(plan)
                    for index, future in futures:
                        results[index] = future.result()
                except BaseException:
                    for _index, future in futures:
                        future.cancel()
                    with suppress(Exception):
                        self._runtime.cancel_candidate_annotations()
                    if executor is not None:
                        executor.shutdown(wait=False, cancel_futures=True)
                    raise
                else:
                    if executor is None:
                        raise AssertionError("Candidate executorが確定していません")
                    executor.shutdown()
            else:
                results = [execute_plan(plan) for plan in plans]
            completed: list[_CandidateMomentExecution] = []
            for result in results:
                if result is None:
                    raise AssertionError("Candidate Momentの結果が確定していません")
                completed.append(result)
            return tuple(completed)

        executions = (
            self._run_external(
                execute_plans,
                reason_code="candidate_annotation_inference_started",
            )
            if pending
            else execute_plans()
        )

        completed_results: list[_CachedAnnotationStageResult] = []
        failures: list[Exception] = []
        annotations: list[CandidateAnnotation] = []
        for annotation, frame_results, failure in executions:
            completed_results.extend(frame_results)
            if failure is not None:
                failures.append(failure)
            elif annotation is None:
                raise AssertionError("Candidate Momentのannotationが確定していません")
            else:
                annotations.append(annotation)
        for index, (_annotation, diagnostics, completed, reused) in enumerate(
            completed_results
        ):
            if index:
                self._start_progress_stage(
                    ProcessingStage.ANNOTATE_CANDIDATE,
                    "candidate",
                )
            duration_seconds = 0.0 if reused else diagnostics.duration_seconds
            self._complete_progress_stage(
                reused,
                completed.fingerprint,
                duration_seconds=duration_seconds,
            )
            self._observer.stage_completed(completed)
        if failures:
            if completed_results:
                self._start_progress_stage(
                    ProcessingStage.ANNOTATE_CANDIDATE,
                    "candidate",
                )
            raise failures[0]
        return (
            tuple(annotations),
            tuple(result[1] for result in completed_results),
            tuple(result[2] for result in completed_results),
        )

    def _execute_candidate_moment(
        self,
        *,
        writer: CompletedStageWriter,
        video_set: VideoSet,
        plan: _CandidateMomentPlan,
        catalog: SceneCatalog,
        catalog_fingerprint: StageFingerprint,
        model: ResolvedModel,
        num_ctx: int,
        max_parallel_requests: int,
        inference_limiter: BoundedSemaphore,
    ) -> _CandidateMomentExecution:
        """一Momentを独立評価し部分成功と失敗を値として返す。"""
        validate_video_set_snapshot_metadata(video_set)
        frame_requests, restored = plan
        primary = restored[0]
        if primary is None:
            try:
                primary = self._execute_annotation_stage(
                    writer=writer,
                    video_set=video_set,
                    request=frame_requests[0],
                    catalog=catalog,
                    catalog_fingerprint=catalog_fingerprint,
                    model=model,
                    num_ctx=num_ctx,
                    inference_limiter=inference_limiter,
                )
            except Exception as error:  # noqa: BLE001
                return None, (), error
        completed_results = [primary]
        primary_annotation = primary[0]
        if not _combat_fallback_is_required(primary_annotation, frame_requests):
            return primary_annotation, tuple(completed_results), None

        fallback_restored = (
            restored[1:]
            if len(restored) > 1
            else tuple(
                _restore_cached_annotation_stage(
                    writer=writer,
                    video_set=video_set,
                    request=fallback,
                    catalog=catalog,
                    catalog_fingerprint=catalog_fingerprint,
                    model=model,
                    num_ctx=num_ctx,
                )
                for fallback in frame_requests[1:]
            )
        )
        fallback_results = self._execute_fallback_annotation_stages(
            writer=writer,
            video_set=video_set,
            requests=frame_requests[1:],
            restored=fallback_restored,
            catalog=catalog,
            catalog_fingerprint=catalog_fingerprint,
            model=model,
            num_ctx=num_ctx,
            max_parallel_requests=max_parallel_requests,
            inference_limiter=inference_limiter,
        )
        failures: list[Exception] = []
        for result in fallback_results:
            if isinstance(result, Exception):
                failures.append(result)
            else:
                completed_results.append(result)
        if failures:
            return None, tuple(completed_results), failures[0]
        frame_annotations = tuple(result[0] for result in completed_results)
        return (
            _select_representative_annotation(frame_annotations),
            tuple(completed_results),
            None,
        )

    def _execute_fallback_annotation_stages(
        self,
        *,
        writer: CompletedStageWriter,
        video_set: VideoSet,
        requests: tuple[CandidateAnnotationRequest, ...],
        restored: tuple[_CachedAnnotationStageResult | None, ...],
        catalog: SceneCatalog,
        catalog_fingerprint: StageFingerprint,
        model: ResolvedModel,
        num_ctx: int,
        max_parallel_requests: int,
        inference_limiter: BoundedSemaphore,
    ) -> tuple[_CachedAnnotationStageResult | Exception, ...]:
        """同一Momentの不足fallbackだけをbounded並列評価する。"""
        results: list[_CachedAnnotationStageResult | Exception | None] = list(restored)
        pending = tuple(
            (index, request)
            for index, (request, cached) in enumerate(
                zip(requests, restored, strict=True)
            )
            if cached is None
        )
        if pending:
            with ThreadPoolExecutor(
                max_workers=min(max_parallel_requests, len(pending)),
                thread_name_prefix="candidate-fallback",
            ) as executor:
                futures = tuple(
                    (
                        index,
                        executor.submit(
                            self._execute_annotation_stage,
                            writer=writer,
                            video_set=video_set,
                            request=request,
                            catalog=catalog,
                            catalog_fingerprint=catalog_fingerprint,
                            model=model,
                            num_ctx=num_ctx,
                            inference_limiter=inference_limiter,
                        ),
                    )
                    for index, request in pending
                )
                for index, future in futures:
                    try:
                        results[index] = future.result()
                    except Exception as error:  # noqa: BLE001
                        results[index] = error
        final_results: list[_CachedAnnotationStageResult | Exception] = []
        for result in results:
            if result is None:
                raise AssertionError("fallback annotation結果が確定していません")
            final_results.append(result)
        return tuple(final_results)

    def _execute_annotation_stage(
        self,
        *,
        writer: CompletedStageWriter,
        video_set: VideoSet,
        request: CandidateAnnotationRequest,
        catalog: SceneCatalog,
        catalog_fingerprint: StageFingerprint,
        model: ResolvedModel,
        num_ctx: int,
        inference_limiter: BoundedSemaphore,
    ) -> _CachedAnnotationStageResult:
        """一枚のannotationを生成または復元しprogressには触れず返す。"""
        semantic_input, upstream, fingerprint = _annotation_stage_definition(
            video_set,
            request,
            catalog_fingerprint,
            model,
            num_ctx,
        )

        def generate() -> tuple[CandidateAnnotation, VisionInferenceDiagnostics]:
            with inference_limiter:
                generated = self._runtime.annotate_candidate(
                    request,
                    catalog,
                    model,
                    num_ctx=num_ctx,
                )
            annotation, _diagnostics = generated
            _validate_runtime_annotation(annotation, request, catalog)
            return generated

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
        return annotation, diagnostics, completed, reused

    def _start_progress_stage(
        self,
        stage: ProcessingStage,
        work_unit_kind: str,
    ) -> None:
        if self._progress is not None:
            self._progress.start_stage(stage, work_unit_kind=work_unit_kind)

    def _complete_progress_stage(
        self,
        reused: bool,
        fingerprint: StageFingerprint,
        *,
        duration_seconds: float | None = None,
    ) -> None:
        if self._progress is None:
            return
        self._record_progress_stage_result(
            reused,
            fingerprint,
            duration_seconds=duration_seconds,
        )
        self._progress.complete_stage(
            duration_seconds,
            stage_fingerprint=fingerprint,
        )

    def _record_progress_stage_result(
        self,
        reused: bool,
        fingerprint: StageFingerprint,
        *,
        duration_seconds: float | None = None,
    ) -> None:
        """active progress Stageへ一つのcache結果を記録する。"""
        if self._progress is None:
            return
        self._progress.record_work_sample(
            "reuse" if reused else "recompute",
            duration_seconds=duration_seconds,
        )
        self._progress.cache_observed(
            cache_hit_count=1 if reused else 0,
            cache_miss_count=0 if reused else 1,
            reuse_count=1 if reused else 0,
            recompute_count=0 if reused else 1,
            reason_code="cache_reused" if reused else "stage_recomputed",
            stage_fingerprint=fingerprint,
        )

    def _run_external(
        self,
        operation: Callable[[], VisionStageValue],
        *,
        reason_code: str,
    ) -> VisionStageValue:
        if self._external_work is None:
            return operation()
        return self._external_work.run(operation, reason_code=reason_code)


def _candidate_moment_plan_requires_inference(plan: _CandidateMomentPlan) -> bool:
    """計画済みMomentに未確定frameがあるかを返す。"""
    _requests, restored = plan
    return any(result is None for result in restored)


def _combat_fallback_is_required(
    primary: CandidateAnnotation,
    frame_requests: tuple[CandidateAnnotationRequest, ...],
) -> bool:
    """成功したPrimaryからCombat Representative Fallback要否を返す。"""
    return (
        primary.combat_action
        and primary.explanation_value == "none"
        and len(frame_requests) > 1
    )


def _annotation_stage_definition(
    video_set: VideoSet,
    request: CandidateAnnotationRequest,
    catalog_fingerprint: StageFingerprint,
    model: ResolvedModel,
    num_ctx: int,
) -> tuple[dict[str, object], tuple[StageFingerprint, ...], StageFingerprint]:
    """一枚のCandidate Annotation Stage identityを返す。"""
    semantic_input = _annotation_semantic_input(
        video_set,
        request,
        catalog_fingerprint,
        model,
        num_ctx,
    )
    upstream = (catalog_fingerprint,)
    return (
        semantic_input,
        upstream,
        build_stage_fingerprint(
            ProcessingStage.ANNOTATE_CANDIDATE,
            upstream,
            semantic_input,
        ),
    )


def _restore_cached_annotation_stage(
    *,
    writer: CompletedStageWriter,
    video_set: VideoSet,
    request: CandidateAnnotationRequest,
    catalog: SceneCatalog,
    catalog_fingerprint: StageFingerprint,
    model: ResolvedModel,
    num_ctx: int,
) -> _CachedAnnotationStageResult | None:
    """有効な一枚annotation cacheを並列処理前に復元する。"""
    semantic_input, upstream, fingerprint = _annotation_stage_definition(
        video_set,
        request,
        catalog_fingerprint,
        model,
        num_ctx,
    )
    artifact = writer.read(
        ProcessingStage.ANNOTATE_CANDIDATE,
        fingerprint,
        upstream,
        semantic_input,
    )
    if artifact is None:
        return None
    try:
        annotation, diagnostics = restore_candidate_annotation(
            artifact,
            request,
            catalog,
        )
    except (TypeError, ValueError):
        return None
    return (
        annotation,
        diagnostics,
        CompletedStage(
            ProcessingStage.ANNOTATE_CANDIDATE,
            fingerprint,
            upstream,
            semantic_input,
        ),
        True,
    )


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
        validate_bundle=lambda value: restore(value.artifact),
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


def _single_frame_annotation_requests(
    request: CandidateAnnotationRequest,
) -> tuple[CandidateAnnotationRequest, ...]:
    """Moment共通入力を保ちframeごとの独立requestへ分ける。"""
    return tuple(
        replace(
            request,
            moment=replace(
                request.moment,
                frame_candidate_ids=(frame.identifier,),
            ),
            frame_candidates=(frame,),
        )
        for frame in request.frame_candidates
    )


def _select_representative_annotation(
    annotations: tuple[CandidateAnnotation, ...],
) -> CandidateAnnotation:
    """戦闘fallback結果から説明価値と中立画質でRepresentativeを確定する。"""
    primary = annotations[0]
    eligible = tuple(
        annotation
        for annotation in annotations
        if annotation.combat_action and annotation.explanation_value != "none"
    )
    if not eligible:
        return primary
    return min(eligible, key=_representative_annotation_key)


def _representative_annotation_key(
    annotation: CandidateAnnotation,
) -> tuple[int, int, int, int, int, float, str]:
    """説明価値、内容、可視性、遮蔽、Neutral品質の比較keyを返す。"""
    evidence = annotation.representative_frame_evidence
    content_priority = (
        0 if evidence is None else _CONTENT_PRIORITY[evidence.content_kind]
    )
    opponent_priority = (
        0
        if evidence is None
        else _VISIBILITY_PRIORITY[evidence.opponent_body_visibility]
    )
    subject_priority = (
        0
        if evidence is None
        else _VISIBILITY_PRIORITY[evidence.primary_subject_visibility]
    )
    obstruction_priority = (
        0 if evidence is None else _OBSTRUCTION_PRIORITY[evidence.transient_obstruction]
    )
    analysis = annotation.candidate.analysis
    quality_score = 0.0 if analysis is None else analysis.quality_score
    return (
        -_EXPLANATION_PRIORITY[annotation.explanation_value],
        -content_priority,
        -opponent_priority,
        -subject_priority,
        -obstruction_priority,
        -quality_score,
        annotation.candidate.identifier,
    )


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
        "generation_options": {
            "temperature": 0,
            "stream": False,
            "think": False,
            "seed": VISION_GENERATION_SEED,
        },
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
        "generation_options": {
            "temperature": 0,
            "stream": False,
            "think": False,
            "seed": VISION_GENERATION_SEED,
        },
        "prompt_version": CANDIDATE_ANNOTATION_PROMPT_VERSION,
        "schema_version": CANDIDATE_ANNOTATION_SCHEMA_VERSION,
        "stage_contract_version": CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION,
        "candidate_annotation_relationship_repair_prompt_version": (
            CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_PROMPT_VERSION
        ),
        "candidate_annotation_relationship_repair_schema_version": (
            CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_SCHEMA_VERSION
        ),
        "candidate_annotation_relationship_repair_stage_contract_version": (
            CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_STAGE_CONTRACT_VERSION
        ),
        "candidate_annotation_relationship_repair_num_predict": (
            CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_NUM_PREDICT
        ),
        "candidate_annotation_relationship_repair_evidence_max_length": (
            CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_EVIDENCE_MAX_LENGTH
        ),
        "combat_encounter_verification_prompt_version": (
            COMBAT_ENCOUNTER_VERIFICATION_PROMPT_VERSION
        ),
        "combat_encounter_verification_schema_version": (
            COMBAT_ENCOUNTER_VERIFICATION_SCHEMA_VERSION
        ),
        "combat_encounter_verification_stage_contract_version": (
            COMBAT_ENCOUNTER_VERIFICATION_STAGE_CONTRACT_VERSION
        ),
        "combat_encounter_confirmation_prompt_version": (
            COMBAT_ENCOUNTER_CONFIRMATION_PROMPT_VERSION
        ),
        "combat_encounter_confirmation_schema_version": (
            COMBAT_ENCOUNTER_VERIFICATION_SCHEMA_VERSION
        ),
        "combat_encounter_confirmation_stage_contract_version": (
            COMBAT_ENCOUNTER_CONFIRMATION_STAGE_CONTRACT_VERSION
        ),
        "combat_visibility_verification_prompt_version": (
            COMBAT_VISIBILITY_VERIFICATION_PROMPT_VERSION
        ),
        "combat_visibility_verification_schema_version": (
            COMBAT_VISIBILITY_VERIFICATION_SCHEMA_VERSION
        ),
        "combat_visibility_verification_stage_contract_version": (
            COMBAT_VISIBILITY_VERIFICATION_STAGE_CONTRACT_VERSION
        ),
        "combat_visibility_confirmation_prompt_version": (
            COMBAT_VISIBILITY_CONFIRMATION_PROMPT_VERSION
        ),
        "combat_visibility_confirmation_schema_version": (
            COMBAT_VISIBILITY_VERIFICATION_SCHEMA_VERSION
        ),
        "combat_visibility_confirmation_stage_contract_version": (
            COMBAT_VISIBILITY_CONFIRMATION_STAGE_CONTRACT_VERSION
        ),
        "combat_visibility_edge_audit_prompt_version": (
            COMBAT_VISIBILITY_EDGE_AUDIT_PROMPT_VERSION
        ),
        "combat_visibility_edge_audit_schema_version": (
            COMBAT_VISIBILITY_EDGE_AUDIT_SCHEMA_VERSION
        ),
        "combat_visibility_edge_audit_stage_contract_version": (
            COMBAT_VISIBILITY_EDGE_AUDIT_STAGE_CONTRACT_VERSION
        ),
        "combat_visibility_edge_strip_version": (COMBAT_VISIBILITY_EDGE_STRIP_VERSION),
        "publication_boundary_verification_prompt_version": (
            PUBLICATION_BOUNDARY_VERIFICATION_PROMPT_VERSION
        ),
        "publication_boundary_verification_schema_version": (
            PUBLICATION_BOUNDARY_VERIFICATION_SCHEMA_VERSION
        ),
        "publication_boundary_verification_stage_contract_version": (
            PUBLICATION_BOUNDARY_VERIFICATION_STAGE_CONTRACT_VERSION
        ),
        "cinematic_letterbox_detection_version": (
            CINEMATIC_LETTERBOX_DETECTION_VERSION
        ),
        "retry_policy_version": RETRY_POLICY_VERSION,
    }


def _fraction_value(value: Fraction) -> Mapping[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}
