"""Game screen picker for diverse image selection."""

from pathlib import Path
from typing import List
import random

import numpy as np

from ..analyzers.image_quality_analyzer import ImageQualityAnalyzer
from ..models.image_metrics import ImageMetrics


class GameScreenPicker:
    """ゲーム画面選択クラス."""

    def __init__(self, analyzer: ImageQualityAnalyzer):
        """ピッカーを初期化する.

        Args:
            analyzer: 画像品質アナライザー
        """
        self.analyzer = analyzer

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
    ) -> List[ImageMetrics]:
        """多様性を考慮して画像を選択する.

        2段階選抜方式で性能を最適化：
        1. まず品質スコア上位M件に絞る（M = max(num*5, 200)）
        2. その上位M件の中で類似度判定を行う

        Args:
            all_results: 解析済みの画像メトリクスリスト（スコア降順ソート済み）
            num: 選択する画像数
            similarity_threshold: 類似度の閾値（これ以上は類似とみなす）

        Returns:
            選択された画像メトリクスのリスト
        """
        # 2段階選抜：まず上位M件に絞る
        # Mの値は取得数の5倍か200の大きい方（多様性確保のため）
        m = max(num * 5, 200)
        candidates = all_results[:m]

        # 特徴量を事前にL2正規化（コサイン類似度 = 内積になる）
        # 正規化されたベクトル同士の内積はコサイン類似度と等価
        normalized_features = [
            c.features / np.linalg.norm(c.features) for c in candidates
        ]

        selected: List[ImageMetrics] = []
        selected_features: List[np.ndarray] = []

        for idx, candidate in enumerate(candidates):
            if len(selected) >= num:
                break

            candidate_feat = normalized_features[idx]

            # 既に選ばれた画像たちと「見た目」を比較
            # 事前正規化済みなので np.dot だけでコサイン類似度を計算可能
            is_similar = False
            for s_feat in selected_features:
                sim = np.dot(candidate_feat, s_feat)
                if sim > similarity_threshold:
                    is_similar = True
                    break

            if not is_similar:
                selected.append(candidate)
                selected_features.append(candidate_feat)

        return selected

    def select(
        self,
        folder: str,
        num: int,
        similarity_threshold: float,
        recursive: bool,
        show_progress: bool = True,
    ) -> List[ImageMetrics]:
        """フォルダから画像を選択する.

        Args:
            folder: 画像フォルダのパス
            num: 選択する画像数
            similarity_threshold: 類似度の閾値
            recursive: サブフォルダも探索するかどうか
            show_progress: 進捗表示をするかどうか

        Returns:
            選択された画像メトリクスのリスト
        """
        # ファイルを取得
        files = self._load_image_files(folder, recursive)

        # ランダムにシャッフル（フォルダやファイル名のバイアスを破壊）
        random.shuffle(files)

        if show_progress:
            print(f"合計 {len(files)} 枚を解析中...")

        # 画像を解析
        all_results = self._analyze_images(files, show_progress)

        # スコア順にソート（最高画質が上にくる）
        all_results.sort(key=lambda x: x.total_score, reverse=True)

        # 多様性に基づいて選択
        return self._select_diverse_images(all_results, num, similarity_threshold)

    @staticmethod
    def select_from_analyzed(
        analyzed_images: List[ImageMetrics],
        num: int,
        similarity_threshold: float,
    ) -> List[ImageMetrics]:
        """解析済みの画像リストから多様性を考慮して選択する.

        このメソッドはIO操作を行わず、純粋なドメインロジックのみを提供する。
        テストや既に解析済みの画像がある場合に使用する。

        Args:
            analyzed_images: 解析済みの画像メトリクスリスト
            num: 選択する画像数
            similarity_threshold: 類似度の閾値

        Returns:
            選択された画像メトリクスのリスト
        """
        # スコア順にソート（コピーを作成して元のリストを変更しない）
        sorted_results = sorted(
            analyzed_images, key=lambda x: x.total_score, reverse=True
        )
        return GameScreenPicker._select_diverse_images(
            sorted_results, num, similarity_threshold
        )
