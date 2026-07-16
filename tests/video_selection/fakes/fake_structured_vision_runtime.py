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
        failure_moment_id: str | None = None,
        reject_all_calls: bool = False,
        scene_catalog_call_started: Event | None = None,
        release_scene_catalog_call: Event | None = None,
    ) -> None:
        self._catalog = catalog
        self._annotations = {
            annotation.candidate_moment_id: annotation for annotation in annotations
        }
        self._failure_moment_id = failure_moment_id
        self._reject_all_calls = reject_all_calls
        self._scene_catalog_call_started = scene_catalog_call_started
        self._release_scene_catalog_call = release_scene_catalog_call
        self.scene_catalog_calls: list[SceneCatalogRequest] = []
        self.candidate_annotation_calls: list[CandidateAnnotationRequest] = []

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
        if request.moment.identifier == self._failure_moment_id:
            raise RuntimeError("fake raw response: chain of thought")
        annotation = self._annotations[request.moment.identifier]
        return annotation, replace(
            _diagnostics(
                model,
                len(request.frame_candidates),
                len(request.context_cues),
            ),
            request_fingerprint=request.moment.identifier[4:],
        )


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
