"""選定レポートをJSONへ出力する."""

import json
from pathlib import Path

from ..models.output_candidate_record import OutputCandidateRecord
from ..models.output_record import OutputRecord


class ReportWriter:
    """選定結果をJSONで保存する."""

    @staticmethod
    def write(
        path: str,
        output_record: OutputRecord,
    ) -> None:
        """JSONレポートを書き出す."""
        payload = {
            "scene_distribution": output_record.scene_distribution,
            "scene_mix_target": output_record.scene_mix_target,
            "scene_mix_actual": output_record.scene_mix_actual,
            "scene_catalog": output_record.scene_catalog,
            "ollama_catalog_fallback_used": output_record.ollama_catalog_fallback_used,
            "ollama_catalog_fallback_reason": (
                output_record.ollama_catalog_fallback_reason
            ),
            "ollama_classification_failed": output_record.ollama_classification_failed,
            "ollama_classification_failure_rate": (
                output_record.ollama_classification_failure_rate
            ),
            "threshold_relaxation_steps": output_record.threshold_relaxation_steps,
            "rejected_by_content_filter": output_record.rejected_by_content_filter,
            "content_filter_breakdown": output_record.content_filter_breakdown,
            "whole_input_profile": output_record.whole_input_profile,
            "selected": [
                ReportWriter._serialize_candidate(candidate)
                for candidate in output_record.selected
            ],
            "rejected": [
                ReportWriter._serialize_candidate(candidate)
                for candidate in output_record.rejected
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
        candidate: OutputCandidateRecord,
    ) -> dict[str, object]:
        """候補1件を辞書へ変換する."""
        payload: dict[str, object] = {
            "path": candidate.source_path,
            "scene_slug": candidate.scene_slug,
            "scene_display_name": candidate.scene_display_name,
            "scene_description": candidate.scene_description,
            "scene_confidence": candidate.scene_confidence,
            "quality_score": candidate.quality_score,
            "selection_score": candidate.selection_score,
            "score_band": candidate.score_band,
            "variant_group": candidate.variant_group,
            "outlier_rejected": candidate.outlier_rejected,
        }
        if candidate.output_path is not None:
            payload["output_path"] = candidate.output_path
        return payload
