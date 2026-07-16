"""最終選定へ渡す注釈済みBlog Candidate。"""

import math
from dataclasses import dataclass
from fractions import Fraction

from .candidate_annotation import CandidateAnnotation
from .scene_catalog_entry import SCENE_SELECTION_ROLES, SceneSelectionRole


@dataclass(frozen=True)
class BlogCandidate:
    """Annotationと決定的selectorに必要なlocal値を保持する。"""

    annotation: CandidateAnnotation
    scene_selection_role: SceneSelectionRole
    video_order: int
    video_set_progress: Fraction
    shortlist_rank: int

    def __post_init__(self) -> None:
        """有効なRepresentative Frameと安定した順序値だけを受理する。"""
        frame = self.annotation.candidate
        analysis = frame.analysis
        if (
            analysis is None
            or not analysis.eligible
            or frame.video_time is None
            or self.scene_selection_role not in SCENE_SELECTION_ROLES
            or self.video_order < 0
            or not 0 <= self.video_set_progress < 1
            or self.shortlist_rank < 0
            or not 0 <= analysis.quality_score <= 1
            or not math.isfinite(analysis.quality_score)
            or not analysis.visual_feature
            or any(not math.isfinite(value) for value in analysis.visual_feature)
        ):
            msg = "Blog Candidateには有効なframe、進行位置、local順が必要です"
            raise ValueError(msg)

    @property
    def identifier(self) -> str:
        """Representative Frameの安定IDを返す。"""
        return self.annotation.candidate.identifier

    @property
    def quality_score(self) -> float:
        """Neutral Image AnalysisのQuality Scoreを返す。"""
        analysis = self.annotation.candidate.analysis
        if analysis is None:  # pragma: no cover - __post_init__で保証される
            raise AssertionError
        return analysis.quality_score

    @property
    def visual_feature(self) -> tuple[float, ...]:
        """Neutral Image Analysisの視覚特徴を返す。"""
        analysis = self.annotation.candidate.analysis
        if analysis is None:  # pragma: no cover - __post_init__で保証される
            raise AssertionError
        return analysis.visual_feature
