"""選定レポートをJSONへ出力する."""

import json
from pathlib import Path

from ..models.picker_statistics import PickerStatistics
from ..models.scored_candidate import ScoredCandidate


class ReportWriter:
    """選定結果をJSONで保存する.

    scene mix の目標値と実績、選択候補と非選択候補の理由を
    後から確認できる形でファイルへ書き出す。
    """

    @staticmethod
    def write(
        path: str,
        selected: list[ScoredCandidate],
        rejected: list[ScoredCandidate],
        stats: PickerStatistics,
    ) -> None:
        """JSONレポートを書き出す.

        レポートには解決済みプロファイル、scene分布、scene mix目標値、
        実績、しきい値緩和履歴、選択候補と非選択候補の説明情報を含める。

        Args:
            path: 出力先JSONファイルパス。
            selected: 最終的に選ばれた候補。
            rejected: 非選択となった候補。
            stats: scene mix実績を含む集計情報。
        """
        payload = {
            "resolved_profile": stats.resolved_profile,
            "scene_distribution": stats.scene_distribution,
            "scene_mix_target": stats.scene_mix_target,
            "scene_mix_actual": stats.scene_mix_actual,
            "threshold_relaxation_used": stats.threshold_relaxation_used,
            "selected": [
                ReportWriter._serialize_candidate(candidate)
                for candidate in selected
            ],
            "rejected": [
                ReportWriter._serialize_candidate(candidate)
                for candidate in rejected
            ],
        }
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _serialize_candidate(candidate: ScoredCandidate) -> dict[str, object]:
        """候補1件を辞書へ変換する.

        Args:
            candidate: JSONへ落とし込む候補。

        Returns:
            scene label と主要スコアだけを抜き出した辞書。
        """
        return {
            "path": candidate.path,
            "scene_label": candidate.scene_assessment.scene_label.value,
            "gameplay_score": round(candidate.scene_assessment.gameplay_score, 4),
            "event_score": round(candidate.scene_assessment.event_score, 4),
            "other_score": round(candidate.scene_assessment.other_score, 4),
            "quality_score": round(candidate.quality_score, 4),
            "selection_score": round(candidate.selection_score, 4),
        }
