"""解析済み画像から最終候補を選定する domain module."""

from ..analyzers.metric_calculator import MetricCalculator
from ..constants.selection_profiles import PROFILE_REGISTRY
from ..models.analyzed_image import AnalyzedImage
from ..models.content_filter_result import ContentFilterResult
from ..models.picker_statistics import PickerStatistics
from ..models.scene_assessment import SceneAssessment
from ..models.scene_catalog_entry import SceneCatalogEntry
from ..models.scored_candidate import ScoredCandidate
from ..models.selection_config import SelectionConfig
from ..models.selection_result import SelectionResult
from ..protocols.scene_analyzer_like import SceneAnalyzerLike
from .candidate_scorer import CandidateScorer
from .content_filter import ContentFilter
from .dynamic_scene_selector import DynamicSceneSelector
from .profile_resolver import ProfileResolver
from .whole_input_profiler import WholeInputProfiler


class AnalyzedImageSelector:
    """解析済み画像からblog用sceneの最終選定結果を作る."""

    CATALOG_SAMPLE_LIMIT = 24

    def __init__(
        self,
        config: SelectionConfig,
        metric_calculator: MetricCalculator,
        scene_analyzer: SceneAnalyzerLike,
    ) -> None:
        """selectorを初期化する."""
        self.config = config
        self._scene_analyzer = scene_analyzer
        self._profile_resolver = ProfileResolver()
        self._candidate_scorer = CandidateScorer(metric_calculator)
        self._scene_selector = DynamicSceneSelector(
            similarity_threshold=config.similarity_threshold,
            threshold_steps=config.compute_threshold_steps(config.similarity_threshold),
        )
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
        (
            candidates,
            resolved_profile,
            scene_distribution,
            scene_catalog,
            classification_failed,
            classification_failure_rate,
        ) = self._score_candidates(content_filter_result.kept_images)
        selection_result = self._scene_selector.select(candidates, num)
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
            scene_catalog=scene_catalog,
            ollama_classification_failed=classification_failed,
            ollama_classification_failure_rate=classification_failure_rate,
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
        scene_catalog: list[SceneCatalogEntry],
        ollama_classification_failed: int,
        ollama_classification_failure_rate: float,
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
            scene_catalog=scene_catalog,
            ollama_classification_failed=ollama_classification_failed,
            ollama_classification_failure_rate=ollama_classification_failure_rate,
        )

    def _score_candidates(
        self,
        analyzed_images: list[AnalyzedImage],
    ) -> tuple[
        list[ScoredCandidate],
        str,
        dict[str, int],
        list[SceneCatalogEntry],
        int,
        float,
    ]:
        """Ollama scene評価とprofile解決を行い、最終候補を作る."""
        if not analyzed_images:
            resolved_profile, _profile_scores = self._profile_resolver.resolve(
                self.config.profile,
                analyzed_images,
            )
            return [], resolved_profile, {}, [], 0, 0.0

        representative_paths = [
            image.path
            for image in sorted(
                analyzed_images,
                key=lambda image: image.raw_metrics.blur_score,
                reverse=True,
            )[: self.CATALOG_SAMPLE_LIMIT]
        ]
        scene_catalog = self._scene_analyzer.generate_scene_catalog(
            representative_paths,
            self.config.scene_hint,
        )
        resolved_profile, _profile_scores = self._profile_resolver.resolve(
            self.config.profile,
            analyzed_images,
        )
        profile = PROFILE_REGISTRY[resolved_profile]

        candidates: list[ScoredCandidate] = []
        classification_failed = 0
        for image in analyzed_images:
            classification = self._scene_analyzer.classify_image(
                image.path,
                scene_catalog,
            )
            if classification is None:
                classification_failed += 1
                continue
            assessment = SceneAssessment(
                scene_slug=classification.scene_slug,
                scene_display_name=classification.scene_display_name,
                scene_description=classification.scene_description,
                scene_confidence=classification.confidence,
            )
            candidates.append(
                self._candidate_scorer.score(
                    image,
                    assessment,
                    profile,
                )
            )
        scene_distribution = self._build_scene_distribution(candidates)
        failure_rate = classification_failed / len(analyzed_images)
        return (
            candidates,
            resolved_profile,
            scene_distribution,
            scene_catalog,
            classification_failed,
            failure_rate,
        )

    @staticmethod
    def _build_scene_distribution(
        candidates: list[ScoredCandidate],
    ) -> dict[str, int]:
        """分類済み候補のscene分布を返す."""
        distribution: dict[str, int] = {}
        for candidate in candidates:
            distribution[candidate.scene_slug] = (
                distribution.get(candidate.scene_slug, 0) + 1
            )
        return distribution
