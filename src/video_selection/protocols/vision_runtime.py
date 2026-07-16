"""strict structured outputを生成するVisionRuntime port。"""

from typing import Protocol

from ..models.candidate_annotation import CandidateAnnotation
from ..models.candidate_annotation_request import CandidateAnnotationRequest
from ..models.resolved_model import ResolvedModel
from ..models.scene_catalog import SceneCatalog
from ..models.scene_catalog_request import SceneCatalogRequest
from ..models.vision_inference_diagnostics import VisionInferenceDiagnostics


class VisionRuntime(Protocol):
    """model transport、strict validation、retryを閉じ込めるseam。"""

    def create_scene_catalog(
        self,
        request: SceneCatalogRequest,
        model: ResolvedModel,
        *,
        num_ctx: int,
    ) -> tuple[SceneCatalog, VisionInferenceDiagnostics]:
        """共有Scene Catalogとprivacy-safe診断を返す。"""

    def annotate_candidate(
        self,
        request: CandidateAnnotationRequest,
        catalog: SceneCatalog,
        model: ResolvedModel,
        *,
        num_ctx: int,
    ) -> tuple[CandidateAnnotation, VisionInferenceDiagnostics]:
        """一つのCandidate Momentのannotationと診断を返す。"""
