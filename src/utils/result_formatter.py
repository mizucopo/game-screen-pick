"""結果フォーマットユーティリティ."""

from pathlib import Path
from typing import Any


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
        print("\n--- 選択された画像一覧 ---")
        for i, res in enumerate(selected):
            print(f"[{i + 1}] {Path(res.path).name} (Score: {res.total_score:.2f})")

        print("\n--- 統計情報 ---")
        print(f"総ファイル数: {stats.total_files}")
        print(f"解析成功: {stats.analyzed_ok}")
        print(f"解析失敗: {stats.analyzed_fail}")
        print(f"類似度で除外: {stats.rejected_by_similarity}")
        print(f"選択数: {stats.selected_count}")
