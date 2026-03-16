"""report_writer.pyの単体テスト."""

import json
from pathlib import Path

from src.constants.scene_label import SceneLabel
from src.models.picker_statistics import PickerStatistics
from src.utils.report_writer import ReportWriter
from tests.conftest import create_scored_candidate


def _build_stats() -> PickerStatistics:
    """ReportWriter用の最小統計情報を作る."""
    return PickerStatistics(
        total_files=4,
        analyzed_ok=4,
        analyzed_fail=0,
        rejected_by_similarity=2,
        rejected_by_content_filter=2,
        selected_count=2,
        resolved_profile="static",
        scene_distribution={"gameplay": 2, "event": 2, "other": 0},
        scene_mix_target={"gameplay": 1, "event": 1, "other": 0},
        scene_mix_actual={"gameplay": 1, "event": 1, "other": 0},
        threshold_relaxation_used=[0.72],
        content_filter_breakdown={
            "blackout": 0,
            "whiteout": 0,
            "single_tone": 0,
            "fade_transition": 2,
            "temporal_transition": 0,
        },
    )


def test_report_writer_adds_scene_diagnostics_to_candidates(tmp_path: Path) -> None:
    """候補ごとに scene 診断情報が追記されること."""
    report_path = tmp_path / "report.json"
    selected = [
        create_scored_candidate(
            path="/tmp/promotion.jpg",
            scene_label=SceneLabel.EVENT,
            gameplay_score=0.42,
            event_score=0.41,
            other_score=0.39,
            scene_confidence=0.02,
            transition_risk_score=0.11,
        ),
        create_scored_candidate(
            path="/tmp/raw_event.jpg",
            scene_label=SceneLabel.EVENT,
            gameplay_score=0.35,
            event_score=0.44,
            other_score=0.41,
            scene_confidence=0.06,
            transition_risk_score=0.08,
        ),
    ]
    rejected = [
        create_scored_candidate(
            path="/tmp/fallback.jpg",
            scene_label=SceneLabel.OTHER,
            gameplay_score=0.43,
            event_score=0.30,
            other_score=0.41,
            scene_confidence=0.01,
            transition_risk_score=0.12,
        ),
        create_scored_candidate(
            path="/tmp/suppressed.jpg",
            scene_label=SceneLabel.OTHER,
            gameplay_score=0.38,
            event_score=0.40,
            other_score=0.39,
            scene_confidence=0.0,
            transition_risk_score=0.63,
            transition_suppressed_event=True,
        ),
    ]

    ReportWriter.write(
        path=str(report_path),
        selected=selected,
        rejected=rejected,
        stats=_build_stats(),
        output_paths_by_candidate_id={
            id(selected[0]): "/tmp/output/event0001.jpg",
            id(selected[1]): "/tmp/output/event0002.jpg",
        },
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    selected_entry = payload["selected"][0]
    fallback_entry = payload["rejected"][0]
    suppressed_entry = payload["rejected"][1]
    diagnostics_summary = payload["scene_diagnostics_summary"]

    assert selected_entry["scene_confidence"] == 0.02
    assert selected_entry["output_path"] == "/tmp/output/event0001.jpg"
    assert selected_entry["transition_risk_score"] == 0.11
    assert selected_entry["argmax_scene_label"] == "gameplay"
    assert selected_entry["argmax_score"] == 0.42
    assert selected_entry["argmax_margin"] == 0.01
    assert selected_entry["fallback_applied"] is False
    assert selected_entry["event_promotion_applied"] is True
    assert selected_entry["transition_suppressed_event"] is False
    assert selected_entry["event_gap_to_winner"] == 0.01

    assert fallback_entry["argmax_scene_label"] == "gameplay"
    assert fallback_entry["fallback_applied"] is True
    assert fallback_entry["transition_suppressed_event"] is False
    assert "output_path" not in fallback_entry

    assert suppressed_entry["argmax_scene_label"] == "event"
    assert suppressed_entry["fallback_applied"] is False
    assert suppressed_entry["event_promotion_applied"] is False
    assert suppressed_entry["transition_suppressed_event"] is True
    assert suppressed_entry["event_gap_to_winner"] == 0.0

    assert diagnostics_summary == {
        "argmax_distribution": {
            "gameplay": 2,
            "event": 2,
            "other": 0,
        },
        "selected_event_breakdown": {
            "raw_event": 1,
            "promoted_from_gameplay": 1,
            "avg_scene_confidence": 0.04,
            "low_confidence_count": 1,
        },
        "adjustment_counts": {
            "fallback_applied": 1,
            "event_promotion_applied": 1,
        },
        "transition_counts": {
            "fade_transition_rejected": 2,
            "event_suppressed": 1,
        },
    }


def test_report_writer_keeps_existing_top_level_payload(tmp_path: Path) -> None:
    """既存の top-level payload を維持しつつ候補詳細を拡張すること."""
    report_path = tmp_path / "report.json"
    selected = [create_scored_candidate(path="/tmp/selected.jpg")]

    ReportWriter.write(
        path=str(report_path),
        selected=selected,
        rejected=[],
        stats=_build_stats(),
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["resolved_profile"] == "static"
    assert payload["scene_distribution"] == {
        "gameplay": 2,
        "event": 2,
        "other": 0,
    }
    assert payload["scene_mix_target"] == {
        "gameplay": 1,
        "event": 1,
        "other": 0,
    }
    assert payload["scene_mix_actual"] == {
        "gameplay": 1,
        "event": 1,
        "other": 0,
    }
    assert payload["rejected_by_content_filter"] == 2
    assert payload["content_filter_breakdown"] == {
        "blackout": 0,
        "whiteout": 0,
        "single_tone": 0,
        "fade_transition": 2,
        "temporal_transition": 0,
    }
    assert "scene_confidence" in payload["selected"][0]
    assert "transition_risk_score" in payload["selected"][0]
    assert "transition_suppressed_event" in payload["selected"][0]
    assert "output_path" not in payload["selected"][0]
    assert "scene_diagnostics_summary" in payload
