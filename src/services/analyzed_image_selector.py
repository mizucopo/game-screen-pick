"""解析済み画像から最終候補を選定する domain module."""

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..analyzers.metric_calculator import MetricCalculator
from ..constants.selection_quality_weights import DEFAULT_QUALITY_WEIGHTS
from ..models.analyzed_image import AnalyzedImage
from ..models.content_filter_result import ContentFilterResult
from ..models.picker_statistics import PickerStatistics
from ..models.scene_assessment import SceneAssessment
from ..models.scene_catalog_entry import SceneCatalogEntry
from ..models.scene_classification import SceneClassification
from ..models.scored_candidate import ScoredCandidate
from ..models.scored_scene_candidates import ScoredSceneCandidates
from ..models.selection_config import SelectionConfig
from ..models.selection_result import SelectionResult
from ..protocols.scene_analyzer_like import SceneAnalyzerLike
from ..utils.vector_utils import VectorUtils
from .candidate_scorer import CandidateScorer
from .content_filter import ContentFilter
from .dynamic_scene_selector import DynamicSceneSelector
from .whole_input_profiler import WholeInputProfiler

logger = logging.getLogger(__name__)


class AnalyzedImageSelector:
    """解析済み画像からblog用sceneの最終選定結果を作る."""

    CATALOG_SAMPLE_LIMIT = 24
    SELECTION_SHORTLIST_SIZE_MULTIPLIER = 10
    SELECTION_SHORTLIST_MIN_SIZE = 500
    SELECTION_SHORTLIST_MAX_SIZE = 2000
    SELECTION_SHORTLIST_SIMILARITY_THRESHOLD = 0.95

    def __init__(
        self,
        config: SelectionConfig,
        metric_calculator: MetricCalculator,
        scene_analyzer: SceneAnalyzerLike,
    ) -> None:
        """selectorを初期化する."""
        self.config = config
        self._scene_analyzer = scene_analyzer
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
        scored = self._score_candidates(content_filter_result.kept_images, num)
        selection_result = self._scene_selector.select(scored.candidates, num)
        selected = selection_result.selected
        rejected = self._build_rejected_candidates(scored.candidates, selected)

        stats = self._build_statistics(
            analyzed_images=analyzed_images,
            total_files=total_files,
            analyzed_fail=analyzed_fail,
            content_filter_result=content_filter_result,
            selection_result=selection_result,
            selected_count=len(selected),
            scored=scored,
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
        scored: ScoredSceneCandidates,
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
            scene_distribution=scored.scene_distribution,
            scene_mix_target=selection_result.target_counts,
            scene_mix_actual=selection_result.actual_counts,
            threshold_relaxation_steps=self.config.compute_threshold_steps(
                self.config.similarity_threshold
            ),
            content_filter_breakdown=content_filter_result.content_filter_breakdown,
            whole_input_profile=content_filter_result.whole_input_profile,
            selection_annotations_by_path=selection_result.annotations_by_path,
            scene_catalog=scored.scene_catalog,
            ollama_catalog_fallback_used=scored.ollama_catalog_fallback_used,
            ollama_catalog_fallback_reason=scored.ollama_catalog_fallback_reason,
            ollama_classification_failed=scored.classification_failed,
            ollama_classification_failure_rate=scored.classification_failure_rate,
        )

    def _score_candidates(
        self,
        analyzed_images: list[AnalyzedImage],
        num: int,
    ) -> ScoredSceneCandidates:
        """Ollama scene評価を行い、最終候補を作る."""
        if not analyzed_images:
            return self._empty_scored_scene_candidates()

        selection_shortlist = self._build_selection_shortlist(analyzed_images, num)
        if not selection_shortlist:
            return self._empty_scored_scene_candidates()
        if len(selection_shortlist) < len(analyzed_images):
            logger.info(
                "Selection Shortlist作成: "
                f"{len(selection_shortlist)}/{len(analyzed_images)}件"
            )

        representative_paths = self._build_representative_paths(selection_shortlist)
        try:
            logger.info(
                f"Ollama scene catalog作成中: 代表画像 {len(representative_paths)} 件"
            )
            scene_catalog = self._scene_analyzer.generate_scene_catalog(
                representative_paths,
                self.config.scene_hint,
            )
            logger.info(f"Ollama scene catalog作成完了: scene {len(scene_catalog)} 件")
        except (OSError, ValueError) as error:
            fallback_reason = f"{type(error).__name__}: {error}"
            logger.debug(
                "Ollama scene catalog作成に失敗したためfallback sceneで選定します: "
                f"{fallback_reason}"
            )
            scene_catalog = self._fallback_scene_catalog()
            classifications: Sequence[SceneClassification | None] = (
                self._fallback_classifications(selection_shortlist, scene_catalog[0])
            )
            return self._score_classifications(
                selection_shortlist,
                scene_catalog,
                classifications,
                catalog_fallback_used=True,
                catalog_fallback_reason=fallback_reason,
            )

        classifications = self._classify_images(selection_shortlist, scene_catalog)
        failed_count = sum(
            1 for classification in classifications if classification is None
        )
        logger.info(
            "Ollama画像分類完了: "
            f"成功 {len(classifications) - failed_count} 件, 失敗 {failed_count} 件"
        )
        return self._score_classifications(
            selection_shortlist,
            scene_catalog,
            classifications,
        )

    def _build_selection_shortlist(
        self,
        analyzed_images: list[AnalyzedImage],
        num: int,
    ) -> list[AnalyzedImage]:
        """Ollama分類へ進めるSelection Shortlistを作る."""
        shortlist_size = self._selection_shortlist_size(num, len(analyzed_images))
        if shortlist_size <= 0:
            return []
        if len(analyzed_images) <= shortlist_size:
            return analyzed_images

        ordered_images = sorted(
            analyzed_images,
            key=lambda image: (
                -self._neutral_quality_score(image),
                image.path,
            ),
        )
        selected_indices, _rejected_indices = VectorUtils.filter_by_similarity(
            candidates=[image.combined_features for image in ordered_images],
            num=shortlist_size,
            similarity_threshold=self.SELECTION_SHORTLIST_SIMILARITY_THRESHOLD,
            compute_threshold_steps=lambda threshold: [threshold],
        )
        selected_index_set = set(selected_indices)
        shortlist = [ordered_images[index] for index in selected_indices]
        for index, image in enumerate(ordered_images):
            if len(shortlist) >= shortlist_size:
                break
            if index not in selected_index_set:
                shortlist.append(image)
        return shortlist

    @classmethod
    def _selection_shortlist_size(cls, num: int, total_count: int) -> int:
        """選定枚数からSelection Shortlistの上限件数を返す."""
        requested_count = max(num, 0)
        target_size = max(
            requested_count * cls.SELECTION_SHORTLIST_SIZE_MULTIPLIER,
            cls.SELECTION_SHORTLIST_MIN_SIZE,
        )
        capped_size = min(target_size, cls.SELECTION_SHORTLIST_MAX_SIZE)
        return min(total_count, max(requested_count, capped_size))

    def _neutral_quality_score(self, image: AnalyzedImage) -> float:
        """scene分類前に使える品質スコアを返す."""
        return self._candidate_scorer.metric_calculator.calculate_quality_score(
            image.normalized_metrics,
            DEFAULT_QUALITY_WEIGHTS,
        )

    @staticmethod
    def _empty_scored_scene_candidates() -> ScoredSceneCandidates:
        """分類対象がない場合の空のscore結果を返す."""
        return ScoredSceneCandidates(
            candidates=[],
            scene_distribution={},
            scene_catalog=[],
            classification_failed=0,
            classification_failure_rate=0.0,
        )

    def _score_classifications(
        self,
        analyzed_images: list[AnalyzedImage],
        scene_catalog: list[SceneCatalogEntry],
        classifications: Sequence[SceneClassification | None],
        catalog_fallback_used: bool = False,
        catalog_fallback_reason: str | None = None,
    ) -> ScoredSceneCandidates:
        """scene分類結果から候補scoreと統計を作る."""
        candidates: list[ScoredCandidate] = []
        classification_failed = 0
        for image, classification in zip(analyzed_images, classifications, strict=True):
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
                )
            )
        scene_distribution = self._build_scene_distribution(candidates)
        failure_rate = classification_failed / len(analyzed_images)
        return ScoredSceneCandidates(
            candidates=candidates,
            scene_distribution=scene_distribution,
            scene_catalog=scene_catalog,
            classification_failed=classification_failed,
            classification_failure_rate=failure_rate,
            ollama_catalog_fallback_used=catalog_fallback_used,
            ollama_catalog_fallback_reason=catalog_fallback_reason,
        )

    @staticmethod
    def _fallback_scene_catalog() -> list[SceneCatalogEntry]:
        """Ollama不可時に使うfallback scene catalogを返す."""
        return [
            SceneCatalogEntry(
                slug="fallback",
                display_name="未分類",
                description="Ollamaで分類できない場合の代替scene",
            )
        ]

    @staticmethod
    def _fallback_classifications(
        analyzed_images: list[AnalyzedImage],
        fallback_scene: SceneCatalogEntry,
    ) -> list[SceneClassification]:
        """全候補をfallback sceneへ分類する."""
        return [
            SceneClassification(
                scene_slug=fallback_scene.slug,
                scene_display_name=fallback_scene.display_name,
                scene_description=fallback_scene.description,
                confidence=0.0,
            )
            for _image in analyzed_images
        ]

    def _classify_images(
        self,
        analyzed_images: list[AnalyzedImage],
        scene_catalog: list[SceneCatalogEntry],
    ) -> list[SceneClassification | None]:
        """画像ごとのscene分類を実行する."""
        max_workers = self.config.ollama.max_workers if self.config.ollama else 1
        image_paths = [image.path for image in analyzed_images]
        total_count = len(image_paths)
        logger.info(f"Ollama画像分類開始: 対象 {total_count} 件, worker {max_workers}")
        if max_workers == 1 or total_count <= 1:
            return self._classify_images_sequentially(image_paths, scene_catalog)
        return self._classify_images_concurrently(
            image_paths,
            scene_catalog,
            max_workers,
        )

    def _classify_images_sequentially(
        self,
        image_paths: list[str],
        scene_catalog: list[SceneCatalogEntry],
    ) -> list[SceneClassification | None]:
        """画像を1件ずつscene分類する."""
        classifications: list[SceneClassification | None] = []
        for completed_count, image_path in enumerate(image_paths, start=1):
            classifications.append(self._classify_image_path(image_path, scene_catalog))
            self._log_classification_progress(completed_count, len(image_paths))
        return classifications

    def _classify_images_concurrently(
        self,
        image_paths: list[str],
        scene_catalog: list[SceneCatalogEntry],
        max_workers: int,
    ) -> list[SceneClassification | None]:
        """画像を並列にscene分類する."""
        classifications: list[SceneClassification | None] = [None] * len(image_paths)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._classify_image_path,
                    image_path,
                    scene_catalog,
                ): index
                for index, image_path in enumerate(image_paths)
            }
            for completed_count, future in enumerate(as_completed(futures), start=1):
                classifications[futures[future]] = future.result()
                self._log_classification_progress(completed_count, len(image_paths))
        return classifications

    @staticmethod
    def _log_classification_progress(completed_count: int, total_count: int) -> None:
        """Ollama画像分類の進捗を出力する."""
        logger.info(f"Ollama画像分類進捗: {completed_count}/{total_count}")

    def _classify_image_path(
        self,
        image_path: str,
        scene_catalog: list[SceneCatalogEntry],
    ) -> SceneClassification | None:
        """1画像をscene分類する."""
        return self._scene_analyzer.classify_image(image_path, scene_catalog)

    @classmethod
    def _build_representative_paths(
        cls,
        analyzed_images: list[AnalyzedImage],
    ) -> list[str]:
        """scene catalog用の代表画像pathを返す."""
        ordered_images = sorted(
            analyzed_images,
            key=lambda image: image.raw_metrics.blur_score,
            reverse=True,
        )
        return [image.path for image in ordered_images[: cls.CATALOG_SAMPLE_LIMIT]]

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
