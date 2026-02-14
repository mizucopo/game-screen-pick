"""Game screen picker statistics."""

from dataclasses import dataclass


@dataclass
class PickerStatistics:
    """選択統計情報.

    Attributes:
        total_files: 処理対象の総ファイル数
        analyzed_ok: 画像解析に成功したファイル数
        analyzed_fail: 画像解析に失敗したファイル数
        rejected_by_similarity: 類似度フィルタで除外された数
        selected_count: 最終的に選択された画像数
    """

    total_files: int
    analyzed_ok: int
    analyzed_fail: int
    rejected_by_similarity: int
    selected_count: int
