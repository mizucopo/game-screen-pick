"""Game screen picker for diverse image selection."""

from pathlib import Path
from typing import List, Optional
import random

import numpy as np

from ..analyzers.image_quality_analyzer import ImageQualityAnalyzer
from ..models.image_metrics import ImageMetrics
from ..models.picker_statistics import PickerStatistics


class GameScreenPicker:
    """ゲーム画面選択クラス."""

    def __init__(
        self, analyzer: ImageQualityAnalyzer, rng: Optional[random.Random] = None
    ):
        """ピッカーを初期化する.

        Args:
            analyzer: 画像品質アナライザー
            rng: 乱数生成器（Noneの場合はデフォルトのRandomを使用）
        """
        self.analyzer = analyzer
        self._rng = rng or random.Random()

    def _load_image_files(self, folder: str, recursive: bool) -> List[Path]:
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
        self, files: List[Path], show_progress: bool = False
    ) -> List[ImageMetrics]:
        """画像ファイルを解析して品質スコアを計算する（バッチ対応版）.

        Args:
            files: 画像ファイルのパスリスト
            show_progress: 進捗表示をするかどうか

        Returns:
            解析結果のリスト
        """
        paths = [str(f) for f in files]
        results = self.analyzer.analyze_batch(
            paths, batch_size=32, show_progress=show_progress
        )
        return [r for r in results if r is not None]

    @staticmethod
    def _select_diverse_images(
        all_results: List[ImageMetrics],
        num: int,
        similarity_threshold: float,
    ) -> tuple[List[ImageMetrics], int]:
        """多様性を考慮して画像を選択する.

        段階的しきい値緩和で指定数を確実に満たす：
        1. 全候補を対象に類似度判定を行う
        2. 指定数に満たない場合、しきい値を段階的に緩和
        3. 最終フォールバックとして類似度制約を外して高スコア順で埋める

        Args:
            all_results: 解析済みの画像メトリクスリスト（スコア降順ソート済み）
            num: 選択する画像数
            similarity_threshold: 類似度の閾値（これ以上は類似とみなす）

        Returns:
            選択された画像メトリクスのリスト（最大num件、
            有効画像数以下なら必ずnum件）と類似度で除外された数のタプル
        """
        if not all_results:
            return [], 0

        # 全候補を対象にする（固定上位M件の制限を廃止）
        candidates = all_results

        # 特徴量を事前にL2正規化（コサイン類似度 = 内積になる）
        # 正規化されたベクトル同士の内積はコサイン類似度と等価
        eps = 1e-8
        normalized_features = []
        for c in candidates:
            norm = np.linalg.norm(c.features)
            if norm < eps:
                # ゼロノルムの場合はゼロベクトルとして扱う
                normalized_features.append(np.zeros_like(c.features))
            else:
                normalized_features.append(c.features / norm)

        # 段階的しきい値緩和のステップ（上限0.98）
        threshold_steps = [
            similarity_threshold,
            min(similarity_threshold + 0.03, 0.98),
            min(similarity_threshold + 0.06, 0.98),
            min(similarity_threshold + 0.10, 0.98),
            min(similarity_threshold + 0.15, 0.98),
        ]

        selected: List[ImageMetrics] = []
        selected_indices: set[int] = set()
        rejected_count = 0
        processed_indices: set[int] = set()

        # 各しきい値で選択を試行
        for threshold in threshold_steps:
            for idx, candidate in enumerate(candidates):
                if idx in processed_indices:
                    continue

                if len(selected) >= num:
                    break

                candidate_feat = normalized_features[idx]

                # 既に選ばれた画像たちと「見た目」を比較
                is_similar = False
                for sel_idx in selected_indices:
                    sel_feat = normalized_features[sel_idx]
                    sim = np.dot(candidate_feat, sel_feat)
                    if sim > threshold:
                        is_similar = True
                        break

                if not is_similar:
                    selected.append(candidate)
                    selected_indices.add(idx)
                    processed_indices.add(idx)
                else:
                    rejected_count += 1
                    processed_indices.add(idx)

            if len(selected) >= num:
                break

        # 最終フォールバック：まだ不足する場合は未選択の高スコア順で埋める
        # （類似度制約を外す）
        if len(selected) < num:
            for idx, candidate in enumerate(candidates):
                if len(selected) >= num:
                    break
                if idx not in selected_indices:
                    selected.append(candidate)
                    selected_indices.add(idx)

        # スコア順でソートして返す
        selected.sort(key=lambda x: x.total_score, reverse=True)
        return selected, rejected_count

    def select(
        self,
        folder: str,
        num: int,
        similarity_threshold: float,
        recursive: bool,
        show_progress: bool = True,
    ) -> tuple[List[ImageMetrics], PickerStatistics]:
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
        files = self._load_image_files(folder, recursive)
        total_files = len(files)

        # ランダムにシャッフル（フォルダやファイル名のバイアスを破壊）
        self._rng.shuffle(files)

        if show_progress:
            print(f"合計 {total_files} 枚を解析中...")

        # 画像を解析
        all_results = self._analyze_images(files, show_progress)
        analyzed_ok = len(all_results)
        analyzed_fail = total_files - analyzed_ok

        # スコア順にソート（最高画質が上にくる）
        all_results.sort(key=lambda x: x.total_score, reverse=True)

        # 多様性に基づいて選択
        selected, rejected_by_similarity = self._select_diverse_images(
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

    @staticmethod
    def select_from_analyzed(
        analyzed_images: List[ImageMetrics],
        num: int,
        similarity_threshold: float,
    ) -> tuple[List[ImageMetrics], PickerStatistics]:
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
        selected, rejected_by_similarity = GameScreenPicker._select_diverse_images(
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
