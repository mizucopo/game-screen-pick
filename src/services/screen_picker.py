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

        段階的しきい値緩和で指定数を確実に満たす：
        1. 全候補を対象に類似度判定を行う
        2. 指定数に満たない場合、しきい値を段階的に緩和
        3. 最終フォールバックとして類似度制約を外して高スコア順で埋める

        Args:
            all_results: 解析済みの画像メトリクスリスト（スコア降順ソート済み）
            num: 選択する画像数
            similarity_threshold: 類似度の閾値（これ以上は類似とみなす）

        Returns:
            選択された画像メトリクスのリスト（最大num件、有効画像数以下なら必ずnum件）
        """
        if not all_results:
            return []

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

        # 各しきい値で選択を試行
        for threshold in threshold_steps:
            if len(selected) >= num:
                break

            for idx, candidate in enumerate(candidates):
                if len(selected) >= num:
                    break

                if idx in selected_indices:
                    continue

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
