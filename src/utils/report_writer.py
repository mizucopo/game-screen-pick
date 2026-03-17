"""選定レポートをJSONへ出力する."""

import json
from pathlib import Path

from ..models.metric_distribution import MetricDistribution
from ..models.picker_statistics import PickerStatistics
from ..models.scored_candidate import ScoredCandidate


class ReportWriter:
    """選定結果をJSONで保存する."""

    @staticmethod
    def write(
        path: str,
        selected: list[ScoredCandidate],
        rejected: list[ScoredCandidate],
        stats: PickerStatistics,
        output_paths_by_candidate_id: dict[int, str] | None = None,
    ) -> None:
        """JSONレポートを書き出す."""
        output_paths_by_candidate_id = output_paths_by_candidate_id or {}
        payload = {
            "resolved_profile": stats.resolved_profile,
            "scene_distribution": stats.scene_distribution,
            "scene_mix_target": stats.scene_mix_target,
            "scene_mix_actual": stats.scene_mix_actual,
            "threshold_relaxation_steps": stats.threshold_relaxation_steps,
            "rejected_by_content_filter": stats.rejected_by_content_filter,
            "content_filter_breakdown": stats.content_filter_breakdown,
            "whole_input_profile": ReportWriter._serialize_whole_input_profile(stats),
            "selected": [
                ReportWriter._serialize_candidate(
                    candidate,
                    output_paths_by_candidate_id.get(id(candidate)),
                )
                for candidate in selected
            ],
            "rejected": [
                ReportWriter._serialize_candidate(candidate) for candidate in rejected
            ],
        }
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _serialize_candidate(
        candidate: ScoredCandidate,
        output_path: str | None = None,
    ) -> dict[str, object]:
        """候補1件を辞書へ変換する."""
        payload: dict[str, object] = {
            "path": candidate.path,
            "scene_label": candidate.scene_assessment.scene_label.value,
            "play_score": round(candidate.scene_assessment.play_score, 4),
            "event_score": round(candidate.scene_assessment.event_score, 4),
            "density_score": round(candidate.scene_assessment.density_score, 4),
            "scene_confidence": round(candidate.scene_assessment.scene_confidence, 4),
            "quality_score": round(candidate.quality_score, 4),
            "selection_score": round(candidate.selection_score, 4),
            "score_band": candidate.score_band,
            "outlier_rejected": candidate.outlier_rejected,
        }
        if output_path is not None:
            payload["output_path"] = output_path
        return payload

    @staticmethod
    def _serialize_distribution(
        distribution: MetricDistribution,
    ) -> dict[str, float]:
        """分布プロフィールをJSON向け辞書へ変換する."""
        return {
            "p10": round(distribution.p10, 4),
            "p25": round(distribution.p25, 4),
            "p50": round(distribution.p50, 4),
            "p90": round(distribution.p90, 4),
        }

    @staticmethod
    def _serialize_whole_input_profile(
        stats: PickerStatistics,
    ) -> dict[str, dict[str, float]] | None:
        """入力全体分布プロフィールをJSON向けに直列化する."""
        profile = stats.whole_input_profile
        if profile is None:
            return None
        return {
            "brightness": ReportWriter._serialize_distribution(profile.brightness),
            "contrast": ReportWriter._serialize_distribution(profile.contrast),
            "edge_density": ReportWriter._serialize_distribution(profile.edge_density),
            "action_intensity": ReportWriter._serialize_distribution(
                profile.action_intensity
            ),
            "luminance_entropy": ReportWriter._serialize_distribution(
                profile.luminance_entropy
            ),
            "luminance_range": ReportWriter._serialize_distribution(
                profile.luminance_range
            ),
            "near_black_ratio": ReportWriter._serialize_distribution(
                profile.near_black_ratio
            ),
            "near_white_ratio": ReportWriter._serialize_distribution(
                profile.near_white_ratio
            ),
            "dominant_tone_ratio": ReportWriter._serialize_distribution(
                profile.dominant_tone_ratio
            ),
        }
