"""Game screen picker for diverse image selection."""

import random
from pathlib import Path

from ..analyzers.image_quality_analyzer import ImageQualityAnalyzer
from ..constants.score_weights import ScoreWeights
from ..models.image_metrics import ImageMetrics
from ..models.picker_statistics import PickerStatistics
from ..models.selection_config import SelectionConfig
from .activity_mix_selector import ActivityMixSelector
from .diversity_selector import DiversitySelector


class GameScreenPicker:
    """ゲーム画面選択クラス."""

    def __init__(
        self,
        analyzer: ImageQualityAnalyzer,
        config: SelectionConfig | None = None,
        rng: random.Random | None = None,
    ):
        """ピッカーを初期化する.

        Args:
            analyzer: 画像品質アナライザー
            config: 選択設定（Noneの場合はデフォルト値を使用）
            rng: 乱数生成器（Noneの場合はデフォルトのRandomを使用）
        """
        self.analyzer = analyzer
        self.config = config or SelectionConfig()
        self._rng = rng or random.Random()
        self._activity_weights = ScoreWeights.get_activity_weights()

        # セレクターを初期化
        self._diversity_selector = DiversitySelector(config=self.config, rng=self._rng)
        self._activity_mix_selector = ActivityMixSelector(
            activity_weights=self._activity_weights,
            config=self.config,
        )

    @staticmethod
    def load_image_files(folder: str, recursive: bool) -> list[Path]:
        """フォルダから画像ファイルのパスを取得する.

        Args:
            folder: 画像フォルダのパス
            recursive: サブフォルダも再帰的に探索するかどうか

        Returns:
            画像ファイルのパスリスト
        """
        path_obj = Path(folder)
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return [
            p
            for p in (path_obj.rglob("*") if recursive else path_obj.glob("*"))
            if p.suffix.lower() in exts
        ]

    def _analyze_images(
        self, files: list[Path], show_progress: bool = False
    ) -> list[ImageMetrics]:
        """画像ファイルを解析して品質スコアを計算する（バッチ対応版）.

        Args:
            files: 画像ファイルのパスリスト
            show_progress: 進捗表示をするかどうか

        Returns:
            解析結果のリスト
        """
        paths = [str(f) for f in files]
        results = self.analyzer.analyze_batch(
            paths, batch_size=self.config.batch_size, show_progress=show_progress
        )
        return [r for r in results if r is not None]

    def select(
        self,
        folder: str,
        num: int,
        similarity_threshold: float,
        recursive: bool,
        show_progress: bool = True,
    ) -> tuple[list[ImageMetrics], PickerStatistics]:
        """フォルダから画像を選択する.

        Args:
            folder: 画像フォルダのパス
            num: 選択する画像数
            similarity_threshold: 類似度の閾値
            recursive: サブフォルダも探索するかどうか
            show_progress: 進捗表示をするかどうか

        Returns:
            (選択された画像メトリクスのリスト, 統計情報)
        """
        # ファイルを取得
        files = GameScreenPicker.load_image_files(folder, recursive)
        total_files = len(files)

        # ランダムにシャッフル（フォルダやファイル名のバイアスを排除）
        self._rng.shuffle(files)

        if show_progress:
            print(f"合計 {total_files} 枚を解析中...")

        # 画像を解析
        all_results = self._analyze_images(files, show_progress)
        analyzed_ok = len(all_results)
        analyzed_fail = total_files - analyzed_ok

        # スコア順にソート（最高品質が上にくる）
        all_results.sort(key=lambda x: x.total_score, reverse=True)

        # 多様性に基づいて選択
        if self.config.activity_mix_enabled:
            selected, rejected_by_similarity = self._activity_mix_selector.select(
                all_results,
                num,
                similarity_threshold,
            )
        else:
            selected, rejected_by_similarity = self._diversity_selector.select(
                all_results, num, similarity_threshold
            )

        stats = PickerStatistics(
            total_files=total_files,
            analyzed_ok=analyzed_ok,
            analyzed_fail=analyzed_fail,
            rejected_by_similarity=rejected_by_similarity,
            selected_count=len(selected),
        )

        return selected, stats

    def select_from_analyzed(
        self,
        analyzed_images: list[ImageMetrics],
        num: int,
        similarity_threshold: float,
    ) -> tuple[list[ImageMetrics], PickerStatistics]:
        """解析済みの画像リストから多様性を考慮して選択する.

        このメソッドはIO操作を行わず、純粋なドメインロジックのみを提供する。
        テストや既に解析済みの画像がある場合に使用する。

        Args:
            analyzed_images: 解析済みの画像メトリクスリスト
            num: 選択する画像数
            similarity_threshold: 類似度の閾値

        Returns:
            (選択された画像メトリクスのリスト, 統計情報)
        """
        # スコア順にソート（コピーを作成して元のリストを変更しない）
        sorted_results = sorted(
            analyzed_images, key=lambda x: x.total_score, reverse=True
        )

        if self.config.activity_mix_enabled:
            selected, rejected_by_similarity = self._activity_mix_selector.select(
                sorted_results, num, similarity_threshold
            )
        else:
            selected, rejected_by_similarity = self._diversity_selector.select(
                sorted_results, num, similarity_threshold
            )

        stats = PickerStatistics(
            total_files=len(analyzed_images),
            analyzed_ok=len(analyzed_images),
            analyzed_fail=0,
            rejected_by_similarity=rejected_by_similarity,
            selected_count=len(selected),
        )

        return selected, stats
