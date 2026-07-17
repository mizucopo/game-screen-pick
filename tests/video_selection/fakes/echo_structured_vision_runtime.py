"""requestから決定的な有効結果を生成するVisionRuntime fake。"""

from src.video_selection.models.candidate_annotation import CandidateAnnotation
from src.video_selection.models.candidate_annotation_request import (
    CandidateAnnotationRequest,
)
from src.video_selection.models.resolved_model import ResolvedModel
from src.video_selection.models.scene_catalog import SceneCatalog
from src.video_selection.models.scene_catalog_entry import SceneCatalogEntry
from src.video_selection.models.scene_catalog_request import SceneCatalogRequest
from src.video_selection.models.vision_inference_diagnostics import (
    VisionInferenceDiagnostics,
)


class EchoStructuredVisionRuntime:
    """request先頭frameをordinary sceneとして注釈し呼び出しを記録する。"""

    def __init__(self) -> None:
        self.scene_catalog_calls: list[SceneCatalogRequest] = []
        self.candidate_annotation_calls: list[CandidateAnnotationRequest] = []

    def create_scene_catalog(
        self,
        request: SceneCatalogRequest,
        model: ResolvedModel,
        *,
        num_ctx: int,
    ) -> tuple[SceneCatalog, VisionInferenceDiagnostics]:
        """固定された有効Scene Catalogを返す。"""
        del num_ctx
        self.scene_catalog_calls.append(request)
        return (
            SceneCatalog(
                (
                    SceneCatalogEntry(
                        "gameplay",
                        "ゲームプレイ",
                        "通常のゲームプレイ場面",
                        "ordinary",
                    ),
                    SceneCatalogEntry(
                        "event",
                        "イベント",
                        "イベント場面",
                        "cinematic",
                    ),
                    SceneCatalogEntry(
                        "other",
                        "その他",
                        "分類不能な場面",
                        "ordinary",
                    ),
                )
            ),
            _diagnostics(model, len(request.representatives), 0),
        )

    def annotate_candidate(
        self,
        request: CandidateAnnotationRequest,
        catalog: SceneCatalog,
        model: ResolvedModel,
        *,
        num_ctx: int,
    ) -> tuple[CandidateAnnotation, VisionInferenceDiagnostics]:
        """request先頭frameに有効な固定annotationを付与する。"""
        del catalog, num_ctx
        self.candidate_annotation_calls.append(request)
        return (
            CandidateAnnotation(
                candidate=request.frame_candidates[0],
                candidate_moment_id=request.moment.identifier,
                summary="通常のゲームプレイ場面",
                scene_slug="gameplay",
                blog_image_type="normal_gameplay",
                explanation_value="high",
                frame_choice_reason="画面内容が明確に写る",
                screen_text_kind="hud",
                context_relevance=("none" if request.context_cues else "unavailable"),
                spoiler_risk="none",
            ),
            _diagnostics(
                model,
                len(request.frame_candidates),
                len(request.context_cues),
                request_fingerprint=request.moment.identifier[4:],
            ),
        )


def _diagnostics(
    model: ResolvedModel,
    image_count: int,
    context_cue_count: int,
    *,
    request_fingerprint: str = "a" * 64,
) -> VisionInferenceDiagnostics:
    """固定されたprivacy-safeな推論診断を返す。"""
    return VisionInferenceDiagnostics(
        request_fingerprint=request_fingerprint,
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
        duration_seconds=0.01,
        prompt_eval_count=10,
        eval_count=5,
        done_reason="stop",
    )
