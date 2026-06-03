"""解析済み画像から最終候補を選定する domain module."""

from ..analyzers.metric_calculator import MetricCalculator
from ..constants.scene_label import SceneLabel
from ..constants.selection_profiles import PROFILE_REGISTRY
from ..models.analyzed_image import AnalyzedImage
from ..models.content_filter_result import ContentFilterResult
from ..models.picker_statistics import PickerStatistics
from ..models.scored_candidate import ScoredCandidate
from ..models.selection_config import SelectionConfig
from ..models.selection_result import SelectionResult
from .candidate_scorer import CandidateScorer
from .content_filter import ContentFilter
from .profile_resolver import ProfileResolver
from .scene_mix_selector import SceneMixSelector
from .scene_scorer import SceneScorer
from .whole_input_profiler import WholeInputProfiler


class AnalyzedImageSelector:
    """解析済み画像から play / event の最終選定結果を作る."""

    def __init__(
        self,
        config: SelectionConfig,
        metric_calculator: MetricCalculator,
    ) -> None:
        """selectorを初期化する."""
        self.config = config
        self._scene_scorer = SceneScorer()
        self._profile_resolver = ProfileResolver()
        self._candidate_scorer = CandidateScorer(metric_calculator)
        self._scene_mix_selector = SceneMixSelector(config)
        self._content_filter = ContentFilter(WholeInputProfiler())

    def select(
        self,
        analyzed_images: list[AnalyzedImage],
        num: int,
        total_files: int | None = None,
        analyzed_fail: int = 0,
    ) -> tuple[list[ScoredCandidate], list[ScoredCandidate], PickerStatistics]:
        """解析済み画像から候補を選択する."""
        content_filter_result = self._content_filter.filter(analyzed_images)
        candidates, resolved_profile, scene_distribution = self._score_candidates(
            content_filter_result.kept_images
        )
        selection_result = self._scene_mix_selector.select(candidates, num)
        selected = selection_result.selected
        rejected = self._build_rejected_candidates(candidates, selected)

        stats = self._build_statistics(
            analyzed_images=analyzed_images,
            total_files=total_files,
            analyzed_fail=analyzed_fail,
            content_filter_result=content_filter_result,
            selection_result=selection_result,
            selected_count=len(selected),
            resolved_profile=resolved_profile,
            scene_distribution=scene_distribution,
        )
        return selected, rejected, stats

    @staticmethod
    def _build_rejected_candidates(
        candidates: list[ScoredCandidate],
        selected: list[ScoredCandidate],
    ) -> list[ScoredCandidate]:
        """非選択候補を選定スコア順に並べる."""
        selected_paths = {candidate.path for candidate in selected}
        return sorted(
            [
                candidate
                for candidate in candidates
                if candidate.path not in selected_paths
            ],
            key=lambda item: item.selection_score,
            reverse=True,
        )

    def _build_statistics(
        self,
        analyzed_images: list[AnalyzedImage],
        total_files: int | None,
        analyzed_fail: int,
        content_filter_result: ContentFilterResult,
        selection_result: SelectionResult[ScoredCandidate],
        selected_count: int,
        resolved_profile: str,
        scene_distribution: dict[str, int],
    ) -> PickerStatistics:
        """選定処理で得た中間結果から統計を組み立てる."""
        input_total = total_files if total_files is not None else len(analyzed_images)
        return PickerStatistics(
            total_files=input_total,
            analyzed_ok=len(analyzed_images),
            analyzed_fail=analyzed_fail,
            rejected_by_similarity=selection_result.rejected_by_similarity,
            rejected_by_content_filter=content_filter_result.rejected_by_content_filter,
            selected_count=selected_count,
            resolved_profile=resolved_profile,
            scene_distribution=scene_distribution,
            scene_mix_target=selection_result.target_counts,
            scene_mix_actual=selection_result.actual_counts,
            threshold_relaxation_steps=self.config.compute_threshold_steps(
                self.config.similarity_threshold
            ),
            content_filter_breakdown=content_filter_result.content_filter_breakdown,
            whole_input_profile=content_filter_result.whole_input_profile,
            selection_annotations_by_path=selection_result.annotations_by_path,
        )

    def _score_candidates(
        self,
        analyzed_images: list[AnalyzedImage],
    ) -> tuple[list[ScoredCandidate], str, dict[str, int]]:
        """scene評価とprofile解決を行い、最終候補を作る."""
        assessments = self._scene_scorer.assess_batch(
            analyzed_images,
            self.config.scene_mix,
        )
        resolved_profile, _profile_scores = self._profile_resolver.resolve(
            self.config.profile,
            analyzed_images,
        )
        profile = PROFILE_REGISTRY[resolved_profile]

        candidates = [
            self._candidate_scorer.score(
                image,
                assessment,
                profile,
            )
            for image, assessment in zip(analyzed_images, assessments, strict=True)
        ]
        scene_distribution: dict[str, int] = {
            SceneLabel.PLAY.value: sum(
                1
                for candidate in candidates
                if candidate.scene_assessment.scene_label == SceneLabel.PLAY
            ),
            SceneLabel.EVENT.value: sum(
                1
                for candidate in candidates
                if candidate.scene_assessment.scene_label == SceneLabel.EVENT
            ),
        }
        return candidates, resolved_profile, scene_distribution
