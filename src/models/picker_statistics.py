"""ゲーム画面ピッカーの統計情報。"""

from dataclasses import dataclass

from .whole_input_profile import WholeInputProfile


@dataclass
class PickerStatistics:
    """選択統計情報.

    Attributes:
        total_files: 処理対象の総ファイル数
        analyzed_ok: 画像解析に成功したファイル数
        analyzed_fail: 画像解析に失敗したファイル数
        rejected_by_similarity: 類似度フィルタで除外された数
        rejected_by_content_filter: content filter で hard reject された数
        selected_count: 最終的に選択された画像数
        resolved_profile: 実行されたプロファイル
        scene_distribution: 解析済み候補全体の画面種別分布
        scene_mix_target: 目標の画面種別配分
        scene_mix_actual: 実際の画面種別配分
        threshold_relaxation_steps: 類似度しきい値緩和ステップ
        content_filter_breakdown: content filter 除外理由ごとの件数
        whole_input_profile: 入力全体の明暗傾向プロフィール
    """

    total_files: int
    analyzed_ok: int
    analyzed_fail: int
    rejected_by_similarity: int
    rejected_by_content_filter: int
    selected_count: int
    resolved_profile: str
    scene_distribution: dict[str, int]
    scene_mix_target: dict[str, int]
    scene_mix_actual: dict[str, int]
    threshold_relaxation_steps: list[float]
    content_filter_breakdown: dict[str, int]
    whole_input_profile: WholeInputProfile | None = None
