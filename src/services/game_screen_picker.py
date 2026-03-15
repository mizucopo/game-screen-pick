"""ゲーム画面ピッカーの統合オーケストレーション."""

import random
import re
from pathlib import Path

from ..constants.selection_profiles import PROFILE_REGISTRY
from ..models.adaptive_scores import AdaptiveScores
from ..models.analyzed_image import AnalyzedImage
from ..models.picker_statistics import PickerStatistics
from ..models.scored_candidate import ScoredCandidate
from ..models.selection_config import SelectionConfig
from ..protocols.analyzer_like import AnalyzerLike
from ..services.candidate_scorer import CandidateScorer
from ..services.content_filter import ContentFilter
from ..services.profile_resolver import ProfileResolver
from ..services.scene_mix_selector import SceneMixSelector
from ..services.scene_scorer import SceneScorer
from ..services.whole_input_profiler import WholeInputProfiler


class GameScreenPicker:
    """画像解析と選定を統合する.

    フォルダ走査、解析、scene評価、profile解決、候補採点、
    scene mix 選定、統計生成までをひとつの入り口として提供する。
    """

    def __init__(
        self,
        analyzer: AnalyzerLike,
        config: SelectionConfig | None = None,
        rng: random.Random | None = None,
    ):
        """ピッカーを初期化する.

        Args:
            analyzer: 中立解析結果を返すAnalyzer実装。
            config: scene mix、類似度しきい値、profile指定を含む選択設定。
            rng: 入力順のバイアスを避けるために使う乱数生成器。
                未指定時は内部で `random.Random()` を生成する。
        """
        self.analyzer = analyzer
        self.config = config or SelectionConfig()
        self._rng = rng or random.Random()
        self._scene_scorer = SceneScorer(self.analyzer.model_manager)
        self._profile_resolver = ProfileResolver()
        self._candidate_scorer = CandidateScorer(self.analyzer.metric_calculator)
        self._scene_mix_selector = SceneMixSelector(self.config)
        self._content_filter = ContentFilter(WholeInputProfiler())

    @staticmethod
    def load_image_files(folder: str, recursive: bool) -> list[Path]:
        """フォルダから画像ファイルを取得する.

        Args:
            folder: 入力フォルダのパス。
            recursive: サブフォルダも含めて探索するかどうか。

        Returns:
            対応拡張子を持つ画像パスの一覧。
        """
        path_obj = Path(folder)
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [
            path
            for path in (path_obj.rglob("*") if recursive else path_obj.glob("*"))
            if path.suffix.lower() in extensions
        ]
        return sorted(
            files,
            key=lambda path: [
                int(chunk) if chunk.isdigit() else chunk.lower()
                for chunk in re.split(r"(\d+)", path.relative_to(path_obj).as_posix())
            ],
        )

    def _analyze_images(
        self,
        files: list[Path],
        show_progress: bool = False,
    ) -> list[AnalyzedImage]:
        """画像を解析して中立特徴を取得する.

        AnalyzerのバッチAPIを使って画像群を一括解析し、
        読み込み失敗や解析失敗で `None` になった要素をここで除外する。
        返却値はまだscene判定や選定スコアを持たない中立データである。

        Args:
            files: 解析対象の画像パス一覧。
            show_progress: 解析進捗ログを出すかどうか。

        Returns:
            正常に解析できた `AnalyzedImage` のみを含むリスト。
        """
        paths = [str(file_path) for file_path in files]
        results = self.analyzer.analyze_batch(
            paths,
            batch_size=self.config.batch_size,
            show_progress=show_progress,
        )
        return [result for result in results if result is not None]

    def _score_candidates(
        self,
        analyzed_images: list[AnalyzedImage],
        adaptive_scores_by_image_id: dict[int, AdaptiveScores],
    ) -> tuple[list[ScoredCandidate], str, dict[str, int], dict[str, float]]:
        """scene評価とprofile解決を行い、最終候補を作る.

        各画像へ `SceneScorer` で画面種別スコアを付与し、
        `ProfileResolver` で `auto` を `active` / `static` へ解決したうえで、
        `CandidateScorer` により品質・活動量・最終選定スコアを計算する。
        あわせて、レポート用のscene分布も集計する。

        Args:
            analyzed_images: scene判定前の中立解析結果。
            adaptive_scores_by_image_id: 入力全体適応スコア。

        Returns:
            1. 最終スコア付き候補のリスト
            2. 解決済みプロファイル名
            3. scene labelごとの件数分布
            4. profile解決時に使ったスコア内訳
        """
        assessments = [
            self._scene_scorer.assess(
                image,
                distinctiveness_score=adaptive_scores_by_image_id[
                    id(image)
                ].distinctiveness_score,
            )
            for image in analyzed_images
        ]
        resolved_profile, profile_scores = self._profile_resolver.resolve(
            self.config.profile,
            analyzed_images,
            assessments,
        )
        profile = PROFILE_REGISTRY[resolved_profile]

        candidates = [
            self._candidate_scorer.score(
                image,
                assessment,
                profile,
                adaptive_scores_by_image_id[id(image)].information_score,
                adaptive_scores_by_image_id[id(image)].distinctiveness_score,
            )
            for image, assessment in zip(analyzed_images, assessments, strict=True)
        ]
        scene_distribution = {
            "gameplay": sum(
                1
                for candidate in candidates
                if candidate.scene_assessment.scene_label.value == "gameplay"
            ),
            "event": sum(
                1
                for candidate in candidates
                if candidate.scene_assessment.scene_label.value == "event"
            ),
            "other": sum(
                1
                for candidate in candidates
                if candidate.scene_assessment.scene_label.value == "other"
            ),
        }
        return candidates, resolved_profile, scene_distribution, profile_scores

    def select(
        self,
        folder: str,
        num: int,
        recursive: bool,
        show_progress: bool = True,
    ) -> tuple[list[ScoredCandidate], list[ScoredCandidate], PickerStatistics]:
        """フォルダから画像を選択する.

        入力フォルダから対象画像を自然順で集め、
        中立解析とcontent filterを通した後にscene mix選定を実行する。
        フォルダ単位のI/Oを伴う高水準APIとして使うことを想定している。

        Args:
            folder: 入力画像フォルダ。
            num: 選択したい画像枚数。
            recursive: サブフォルダも探索対象に含めるかどうか。
            show_progress: 解析進捗ログを出すかどうか。

        Returns:
            1. 選択された候補
            2. 非選択になった候補
            3. 実行統計をまとめた `PickerStatistics`
        """
        files = GameScreenPicker.load_image_files(folder, recursive)
        total_files = len(files)

        analyzed_images = self._analyze_images(files, show_progress)
        analyzed_ok = len(analyzed_images)
        analyzed_fail = total_files - analyzed_ok

        return self.select_from_analyzed(
            analyzed_images=analyzed_images,
            total_files=total_files,
            analyzed_fail=analyzed_fail,
            num=num,
        )

    def select_from_analyzed(
        self,
        analyzed_images: list[AnalyzedImage],
        num: int,
        total_files: int | None = None,
        analyzed_fail: int = 0,
    ) -> tuple[list[ScoredCandidate], list[ScoredCandidate], PickerStatistics]:
        """解析済み画像から候補を選択する.

        このメソッドはファイルI/Oを行わず、ドメインロジックだけを扱う。
        scene判定、profile解決、候補採点、scene mix選定、統計生成を
        純粋に組み合わせるため、単体テストや再選定処理の入り口として使える。

        Args:
            analyzed_images: 解析済みの中立画像データ。
            num: 選択したい画像枚数。
            total_files: 元の入力総数。 `None` の場合は解析済み件数を使う。
            analyzed_fail: 解析失敗件数。フォルダ起点の統計で使う。

        Returns:
            1. 選択された候補
            2. 非選択候補を選定スコア順に並べたリスト
            3. scene mix目標値と実績を含む `PickerStatistics`
        """
        content_filter_result = self._content_filter.filter(analyzed_images)
        filtered_images = content_filter_result.kept_images
        adaptive_scores_by_image_id = content_filter_result.adaptive_scores_by_image_id
        candidates, resolved_profile, scene_distribution, _profile_scores = (
            self._score_candidates(filtered_images, adaptive_scores_by_image_id)
        )
        profile = PROFILE_REGISTRY[resolved_profile]
        selected, rejected_by_similarity, scene_mix_target, scene_mix_actual = (
            self._scene_mix_selector.select(candidates, num, profile)
        )
        selected_ids = {id(candidate) for candidate in selected}
        rejected = sorted(
            [
                candidate
                for candidate in candidates
                if id(candidate) not in selected_ids
            ],
            key=lambda item: item.selection_score,
            reverse=True,
        )

        stats = PickerStatistics(
            total_files=total_files
            if total_files is not None
            else len(analyzed_images),
            analyzed_ok=len(analyzed_images),
            analyzed_fail=analyzed_fail,
            rejected_by_similarity=rejected_by_similarity,
            rejected_by_content_filter=content_filter_result.rejected_by_content_filter,
            selected_count=len(selected),
            resolved_profile=resolved_profile,
            scene_distribution=scene_distribution,
            scene_mix_target=scene_mix_target,
            scene_mix_actual=scene_mix_actual,
            threshold_relaxation_used=self.config.compute_threshold_steps(
                self.config.similarity_threshold
            ),
            content_filter_breakdown=content_filter_result.content_filter_breakdown,
        )
        return selected, rejected, stats
