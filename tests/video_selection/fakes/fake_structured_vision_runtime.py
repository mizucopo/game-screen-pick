from collections.abc import Callable
from dataclasses import replace
from threading import Event

from src.video_selection.models.candidate_annotation import CandidateAnnotation
from src.video_selection.models.candidate_annotation_request import (
    CandidateAnnotationRequest,
)
from src.video_selection.models.resolved_model import ResolvedModel
from src.video_selection.models.scene_catalog import SceneCatalog
from src.video_selection.models.scene_catalog_request import SceneCatalogRequest
from src.video_selection.models.vision_inference_diagnostics import (
    VisionInferenceDiagnostics,
)


class FakeStructuredVisionRuntime:
    """固定CatalogとMoment別Annotationを返す記録用fake。"""

    def __init__(
        self,
        catalog: SceneCatalog,
        annotations: tuple[CandidateAnnotation, ...],
        *,
        fail_scene_catalog: bool = False,
        failure_moment_id: str | None = None,
        failure_moment_ids: frozenset[str] | None = None,
        failure_frame_id: str | None = None,
        reject_all_calls: bool = False,
        scene_catalog_call_started: Event | None = None,
        release_scene_catalog_call: Event | None = None,
        on_candidate_annotation: Callable[[], None] | None = None,
        on_candidate_annotation_request: (
            Callable[[CandidateAnnotationRequest], None] | None
        ) = None,
        on_cancel_candidate_annotations: Callable[[], None] | None = None,
    ) -> None:
        self._catalog = catalog
        self._annotations = {
            (
                annotation.candidate_moment_id,
                annotation.candidate.identifier,
            ): annotation
            for annotation in annotations
        }
        self._fail_scene_catalog = fail_scene_catalog
        single_failure_moment = (
            frozenset()
            if failure_moment_id is None
            else frozenset((failure_moment_id,))
        )
        self._failure_moment_ids = (
            failure_moment_ids or frozenset()
        ) | single_failure_moment
        self._failure_frame_id = failure_frame_id
        self._reject_all_calls = reject_all_calls
        self._scene_catalog_call_started = scene_catalog_call_started
        self._release_scene_catalog_call = release_scene_catalog_call
        self._on_candidate_annotation = on_candidate_annotation
        self._on_candidate_annotation_request = on_candidate_annotation_request
        self._on_cancel_candidate_annotations = on_cancel_candidate_annotations
        self.scene_catalog_calls: list[SceneCatalogRequest] = []
        self.candidate_annotation_calls: list[CandidateAnnotationRequest] = []
        self.cancel_candidate_annotations_call_count = 0

    def create_scene_catalog(
        self,
        request: SceneCatalogRequest,
        model: ResolvedModel,
        *,
        num_ctx: int,
    ) -> tuple[SceneCatalog, VisionInferenceDiagnostics]:
        """固定Catalogと診断を返す。"""
        del num_ctx
        if self._reject_all_calls:
            raise AssertionError("Scene Catalogが再生成されました")
        is_first_call = not self.scene_catalog_calls
        self.scene_catalog_calls.append(request)
        if self._fail_scene_catalog:
            raise RuntimeError("fake raw catalog response: chain of thought")
        if is_first_call and self._scene_catalog_call_started is not None:
            self._scene_catalog_call_started.set()
            if (
                self._release_scene_catalog_call is not None
                and not self._release_scene_catalog_call.wait(timeout=5)
            ):
                raise TimeoutError("Scene Catalog callが解放されませんでした")
        return self._catalog, _diagnostics(model, len(request.representatives), 0)

    def annotate_candidate(
        self,
        request: CandidateAnnotationRequest,
        catalog: SceneCatalog,
        model: ResolvedModel,
        *,
        num_ctx: int,
    ) -> tuple[CandidateAnnotation, VisionInferenceDiagnostics]:
        """Moment IDに対応する固定Annotationと診断を返す。"""
        del catalog, num_ctx
        if self._reject_all_calls:
            raise AssertionError("Candidate Annotationが再生成されました")
        self.candidate_annotation_calls.append(request)
        if self._on_candidate_annotation is not None:
            self._on_candidate_annotation()
        if self._on_candidate_annotation_request is not None:
            self._on_candidate_annotation_request(request)
        if (
            request.moment.identifier in self._failure_moment_ids
            or request.frame_candidates[0].identifier == self._failure_frame_id
        ):
            raise RuntimeError("fake raw response: chain of thought")
        annotation = self._annotations[
            (
                request.moment.identifier,
                request.frame_candidates[0].identifier,
            )
        ]
        return annotation, replace(
            _diagnostics(
                model,
                len(request.frame_candidates),
                len(request.context_cues),
            ),
            request_fingerprint=request.moment.identifier[4:],
        )

    def cancel_candidate_annotations(self) -> None:
        """Candidate Annotationの中止要求を記録する。"""
        self.cancel_candidate_annotations_call_count += 1
        if self._on_cancel_candidate_annotations is not None:
            self._on_cancel_candidate_annotations()


def _diagnostics(
    model: ResolvedModel,
    image_count: int,
    context_cue_count: int,
) -> VisionInferenceDiagnostics:
    return VisionInferenceDiagnostics(
        request_fingerprint="a" * 64,
        model_name=model.configured_name,
        model_identity=model.execution_identity.identifier,
        runtime_identity=model.runtime_identity.identifier,
        prompt_version="fake-prompt-v1",
        schema_version="fake-schema-v1",
        stage_contract_version="fake-stage-v1",
        retry_policy_version="fake-retry-v1",
        cache_hit=False,
        attempt_count=1,
        validation_code=None,
        image_count=image_count,
        context_cue_count=context_cue_count,
        duration_seconds=0.25,
        prompt_eval_count=100,
        eval_count=20,
        done_reason="stop",
    )
