"""Video Set Vision Stageの完了結果。"""

from dataclasses import dataclass

from .candidate_annotation import CandidateAnnotation
from .completed_stage import CompletedStage
from .scene_catalog import SceneCatalog
from .vision_inference_diagnostics import VisionInferenceDiagnostics


@dataclass(frozen=True)
class VisionStageResult:
    """共有Catalog、Moment別Annotation、診断、完了Stageを保持する。"""

    catalog: SceneCatalog
    annotations: tuple[CandidateAnnotation, ...]
    catalog_diagnostics: VisionInferenceDiagnostics
    annotation_diagnostics: tuple[VisionInferenceDiagnostics, ...]
    completed_stages: tuple[CompletedStage, ...]

    def __post_init__(self) -> None:
        """Annotation、診断、Completed Stageの件数を検証する。"""
        if (
            len(self.annotations) > len(self.annotation_diagnostics)
            or len(self.completed_stages) != len(self.annotation_diagnostics) + 1
        ):
            msg = "Vision Stage resultの件数が一致しません"
            raise ValueError(msg)
