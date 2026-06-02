"""出力adapter向けの候補record。"""

from dataclasses import dataclass, replace
from pathlib import Path

from .scored_candidate import ScoredCandidate
from .selection_annotation import SelectionAnnotation


@dataclass(frozen=True)
class OutputCandidateRecord:
    """出力adapterが利用する候補1件分の安定した値."""

    source_path: str
    filename: str
    suffix: str
    scene_label: str
    play_score: float
    event_score: float
    density_score: float
    scene_confidence: float
    quality_score: float
    selection_score: float
    score_band: str | None
    outlier_rejected: bool
    output_path: str | None = None

    @classmethod
    def from_scored_candidate(
        cls,
        candidate: ScoredCandidate,
        selection_annotation: SelectionAnnotation | None = None,
    ) -> "OutputCandidateRecord":
        """選定内部候補を出力用recordへ射影する."""
        annotation = selection_annotation or SelectionAnnotation()
        path = Path(candidate.path)
        return cls(
            source_path=candidate.path,
            filename=path.name,
            suffix=path.suffix,
            scene_label=candidate.scene_assessment.scene_label.value,
            play_score=round(candidate.scene_assessment.play_score, 4),
            event_score=round(candidate.scene_assessment.event_score, 4),
            density_score=round(candidate.scene_assessment.density_score, 4),
            scene_confidence=round(candidate.scene_assessment.scene_confidence, 4),
            quality_score=round(candidate.quality_score, 4),
            selection_score=round(candidate.selection_score, 4),
            score_band=annotation.score_band,
            outlier_rejected=annotation.outlier_rejected,
        )

    def with_output_path(self, output_path: str | None) -> "OutputCandidateRecord":
        """出力先パスを反映したrecordを返す."""
        return replace(self, output_path=output_path)
