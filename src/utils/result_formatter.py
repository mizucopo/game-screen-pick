"""結果フォーマットユーティリティクラス.

選択結果と統計情報をコンソールに出力する機能を提供する。
"""

import logging
from pathlib import Path

from ..models.picker_statistics import PickerStatistics
from ..models.scored_candidate import ScoredCandidate

logger = logging.getLogger(__name__)


class ResultFormatter:
    """結果フォーマットユーティリティクラス.

    選択結果と統計情報をコンソールに出力する機能を提供する。
    """

    @staticmethod
    def display_results(
        selected: list["ScoredCandidate"], stats: "PickerStatistics"
    ) -> None:
        """選択結果を表示する.

        Args:
            selected: 選択された画像メトリクスのリスト
            stats: 統計情報
        """
        logger.info("--- 選択された画像一覧 ---")
        for i, res in enumerate(selected):
            logger.info(
                f"[{i + 1}] {Path(res.path).name} "
                f"(Scene: {res.scene_assessment.scene_label.value}, "
                f"Score: {res.selection_score:.2f})"
            )

        logger.info("--- 統計情報 ---")
        logger.info(f"総ファイル数: {stats.total_files}")
        logger.info(f"解析成功: {stats.analyzed_ok}")
        logger.info(f"解析失敗: {stats.analyzed_fail}")
        logger.info(f"類似度で除外: {stats.rejected_by_similarity}")
        logger.info(f"選択数: {stats.selected_count}")
        logger.info(f"プロファイル: {stats.resolved_profile}")
        logger.info(f"画面分布(候補): {stats.scene_distribution}")
        logger.info(f"画面分布(目標): {stats.scene_mix_target}")
        logger.info(f"画面分布(実績): {stats.scene_mix_actual}")
