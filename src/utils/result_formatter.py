"""結果フォーマットユーティリティクラス.

選択結果と統計情報をコンソールに出力する機能を提供する。
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ResultFormatter:
    """結果フォーマットユーティリティクラス.

    選択結果と統計情報をコンソールに出力する機能を提供する。
    """

    @staticmethod
    def display_results(selected: list[Any], stats: Any) -> None:
        """選択結果を表示する.

        Args:
            selected: 選択された画像メトリクスのリスト
            stats: 統計情報
        """
        logger.info("--- 選択された画像一覧 ---")
        for i, res in enumerate(selected):
            logger.info(
                f"[{i + 1}] {Path(res.path).name} (Score: {res.total_score:.2f})"
            )

        logger.info("--- 統計情報 ---")
        logger.info(f"総ファイル数: {stats.total_files}")
        logger.info(f"解析成功: {stats.analyzed_ok}")
        logger.info(f"解析失敗: {stats.analyzed_fail}")
        logger.info(f"類似度で除外: {stats.rejected_by_similarity}")
        logger.info(f"選択数: {stats.selected_count}")
