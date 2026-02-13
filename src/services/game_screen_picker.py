"""Game screen picker for diverse image selection."""

import random
from pathlib import Path

import numpy as np

from ..analyzers.image_quality_analyzer import ImageQualityAnalyzer
from ..constants.score_weights import ScoreWeights
from ..models.activity_bucket import ActivityBucket
from ..models.bucketed_image import BucketedImage
from ..models.image_metrics import ImageMetrics
from ..models.picker_statistics import PickerStatistics
from ..models.selection_config import SelectionConfig


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

    @staticmethod
    def _calculate_activity_score(
        image: ImageMetrics, weights: dict[str, float]
    ) -> float:
        """画像の活動量スコアを計算する.

        Args:
            image: 画像メトリクス
            weights: 活動量計算用の重み

        Returns:
            活動量スコア（正規化済みメトリクスの加重平均）
        """
        norm = image.normalized_metrics
        return (
            weights.get("action_intensity", 0.55) * norm.get("action_intensity", 0)
            + weights.get("edge_density", 0.25) * norm.get("edge_density", 0)
            + weights.get("dramatic_score", 0.20) * norm.get("dramatic_score", 0)
        )

    @staticmethod
    def _assign_buckets(
        images: list[ImageMetrics],
        activity_weights: dict[str, float],
    ) -> list[BucketedImage]:
        """画像に活動量バケットを割り当てる.

        Args:
            images: 画像メトリクスのリスト
            activity_weights: 活動量計算用の重み

        Returns:
            バケット付けされた画像のリスト
        """
        # 活動量スコアを計算
        activity_scores = [
            GameScreenPicker._calculate_activity_score(img, activity_weights)
            for img in images
        ]

        # q30/q70で分位点分割
        q30 = np.percentile(activity_scores, 30, method="linear")
        q70 = np.percentile(activity_scores, 70, method="linear")

        bucketed: list[BucketedImage] = []
        for img, score in zip(images, activity_scores):
            if score < q30:
                bucket = ActivityBucket.LOW
            elif score < q70:
                bucket = ActivityBucket.MID
            else:
                bucket = ActivityBucket.HIGH
            bucketed.append(BucketedImage(img, bucket, score))

        return bucketed

    @staticmethod
    def _select_with_activity_mix(
        all_results: list[ImageMetrics],
        num: int,
        activity_weights: dict[str, float],
        similarity_threshold: float,
        config: SelectionConfig | None = None,
    ) -> tuple[list[ImageMetrics], int]:
        """活動量バケットを考慮して画像を選択する.

        Args:
            all_results: 解析済みの画像メトリクスリスト（総合スコア降順）
            num: 選択する画像数
            activity_weights: 活動量計算用の重み
            similarity_threshold: 類似度の閾値
            config: 選択設定（Noneの場合はデフォルト値を使用）

        Returns:
            (選択された画像メトリクスのリスト, 類似度で除外された数) のタプル
        """
        selection_config = config or SelectionConfig()

        # まず類似度フィルタリングを適用
        diverse_candidates, rejected_count = GameScreenPicker._select_diverse_images(
            all_results,
            num * 3,  # 活動量ミックスのために候補を多めに取得
            similarity_threshold,
            selection_config,
        )

        if not diverse_candidates:
            return [], rejected_count

        # バケット付け
        bucketed = GameScreenPicker._assign_buckets(
            diverse_candidates, activity_weights
        )

        # バケットごとにグループ化
        by_bucket: dict[ActivityBucket, list[BucketedImage]] = {
            ActivityBucket.LOW: [],
            ActivityBucket.MID: [],
            ActivityBucket.HIGH: [],
        }
        for b in bucketed:
            by_bucket[b.bucket].append(b)

        # num>=3かつ全バケット非空なら各バケット最低1枚を先取り
        selected: list[ImageMetrics] = []
        selected_ids: set[int] = set()

        if num >= 3 and all(len(v) > 0 for v in by_bucket.values()):
            # 各バケットから最高スコアの画像を1枚選択
            all_buckets = [ActivityBucket.LOW, ActivityBucket.MID, ActivityBucket.HIGH]
            for bucket_type in all_buckets:
                best = max(by_bucket[bucket_type], key=lambda x: x.image.total_score)
                selected.append(best.image)
                selected_ids.add(id(best.image))

        # 残り枠を計算
        remaining = num - len(selected)
        if remaining <= 0:
            selected.sort(key=lambda x: x.total_score, reverse=True)
            return selected[:num], rejected_count

        # 30/40/30の目標配分を計算
        ratio = selection_config.activity_mix_ratio
        target_counts = {
            ActivityBucket.LOW: max(1, round(num * ratio[0])),
            ActivityBucket.MID: max(1, round(num * ratio[1])),
            ActivityBucket.HIGH: max(1, round(num * ratio[2])),
        }

        # 既選択分を差し引く
        all_buckets = [ActivityBucket.LOW, ActivityBucket.MID, ActivityBucket.HIGH]
        for bucket_type in all_buckets:
            bucket_images = [b.image for b in by_bucket[bucket_type]]
            if any(id(s) in [id(img) for img in bucket_images] for s in selected):
                target_counts[bucket_type] -= 1

        # 残りを総合スコア順に選択
        # 優先順: mid > high > low（最大剰余法）
        priority = [ActivityBucket.MID, ActivityBucket.HIGH, ActivityBucket.LOW]

        for bucket_type in priority:
            need = target_counts.get(bucket_type, 0)
            if need <= 0:
                continue

            for b in by_bucket[bucket_type]:
                if need <= 0:
                    break
                if id(b.image) not in selected_ids:
                    selected.append(b.image)
                    selected_ids.add(id(b.image))
                    need -= 1

        # まだ不足する場合は未選択候補を総合スコア順でフォールバック
        if len(selected) < num:
            for b in bucketed:
                if len(selected) >= num:
                    break
                if id(b.image) not in selected_ids:
                    selected.append(b.image)
                    selected_ids.add(id(b.image))

        # 総合スコア順でソート
        selected.sort(key=lambda x: x.total_score, reverse=True)
        return selected[:num], rejected_count

    @staticmethod
    def _select_diverse_images(
        all_results: list[ImageMetrics],
        num: int,
        similarity_threshold: float,
        config: SelectionConfig | None = None,
    ) -> tuple[list[ImageMetrics], int]:
        """多様性を考慮して画像を選択する.

        Args:
            all_results: 解析済みの画像メトリクスリスト（スコア降順ソート済み）
            num: 選択する画像数
            similarity_threshold: 類似度の閾値（これ以上は類似とみなす）
            config: 選択設定（Noneの場合はデフォルト値を使用）

        Returns:
            選択された画像メトリクスのリスト（最大num件、
            有効画像数以下なら必ずnum件）と類似度で除外された数のタプル
        """
        selection_config = config or SelectionConfig()

        if not all_results:
            return [], 0

        # 全候補を対象にする（固定上位M件の制限を廃止）
        candidates = all_results

        # 特徴ベクトルを事前にL2正規化（コサイン類似度 = 内積になる）
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

        # 段階的しきい値緩和のステップ
        threshold_steps = selection_config.compute_threshold_steps(similarity_threshold)

        selected: list[ImageMetrics] = []
        selected_indices: set[int] = set()
        rejected_indices: set[int] = set()  # ユニークな拒否数を追跡

        # 各しきい値で選択を試行
        for threshold in threshold_steps:
            for idx, candidate in enumerate(candidates):
                # 既に選択または永続拒否された候補はスキップ
                if idx in selected_indices or idx in rejected_indices:
                    continue

                if len(selected) >= num:
                    break

                candidate_feat = normalized_features[idx]

                # 既に選ばれた画像たちと「見た目」を比較（ベクトル化）
                # selected_featuresを行列化して一括計算
                is_similar = False
                if selected_indices:
                    selected_features = np.array(
                        [normalized_features[i] for i in selected_indices]
                    )
                    sims = selected_features @ candidate_feat
                    if np.any(sims > threshold):
                        is_similar = True

                if not is_similar:
                    selected.append(candidate)
                    selected_indices.add(idx)
                else:
                    # 最終しきい値ラウンドで拒否された場合のみ記録
                    if threshold == threshold_steps[-1]:
                        rejected_indices.add(idx)

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
        return selected, len(rejected_indices)

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
            (
                selected,
                rejected_by_similarity,
            ) = GameScreenPicker._select_with_activity_mix(
                all_results,
                num,
                self._activity_weights,
                similarity_threshold,
                self.config,
            )
        else:
            selected, rejected_by_similarity = GameScreenPicker._select_diverse_images(
                all_results, num, similarity_threshold, self.config
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
        analyzed_images: list[ImageMetrics],
        num: int,
        similarity_threshold: float,
        config: SelectionConfig | None = None,
    ) -> tuple[list[ImageMetrics], PickerStatistics]:
        """解析済みの画像リストから多様性を考慮して選択する.

        このメソッドはIO操作を行わず、純粋なドメインロジックのみを提供する。
        テストや既に解析済みの画像がある場合に使用する。

        Args:
            analyzed_images: 解析済みの画像メトリクスリスト
            num: 選択する画像数
            similarity_threshold: 類似度の閾値
            config: 選択設定（Noneの場合はデフォルト値を使用）

        Returns:
            (選択された画像メトリクスのリスト, 統計情報)
        """
        # スコア順にソート（コピーを作成して元のリストを変更しない）
        sorted_results = sorted(
            analyzed_images, key=lambda x: x.total_score, reverse=True
        )
        selection_config = config or SelectionConfig()

        if selection_config.activity_mix_enabled:
            activity_weights = ScoreWeights.get_activity_weights()
            (
                selected,
                rejected_by_similarity,
            ) = GameScreenPicker._select_with_activity_mix(
                sorted_results,
                num,
                activity_weights,
                similarity_threshold,
                selection_config,
            )
        else:
            selected, rejected_by_similarity = GameScreenPicker._select_diverse_images(
                sorted_results, num, similarity_threshold, selection_config
            )

        stats = PickerStatistics(
            total_files=len(analyzed_images),
            analyzed_ok=len(analyzed_images),
            analyzed_fail=0,
            rejected_by_similarity=rejected_by_similarity,
            selected_count=len(selected),
        )

        return selected, stats
