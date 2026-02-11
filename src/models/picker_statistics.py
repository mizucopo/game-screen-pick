"""Game screen picker statistics."""

from dataclasses import dataclass


@dataclass
class PickerStatistics:
    """選択統計情報.

    Attributes:
        total_files: 総ファイル数
        analyzed_ok: 解析成功数
        analyzed_fail: 解析失敗数
        rejected_by_similarity: 類似度で除外された数
        selected_count: 最終選択数
    """

    total_files: int
    analyzed_ok: int
    analyzed_fail: int
    rejected_by_similarity: int
    selected_count: int
