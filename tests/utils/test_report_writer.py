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
        total_files=3,
        analyzed_ok=3,
        analyzed_fail=0,
        rejected_by_similarity=1,
        rejected_by_content_filter=0,
        selected_count=2,
        resolved_profile="static",
        scene_distribution={"gameplay": 2, "event": 1, "other": 0},
        scene_mix_target={"gameplay": 1, "event": 1, "other": 0},
        scene_mix_actual={"gameplay": 1, "event": 1, "other": 0},
        threshold_relaxation_used=[0.72],
        content_filter_breakdown={
            "blackout": 0,
            "whiteout": 0,
            "single_tone": 0,
            "fade_transition": 0,
            "temporal_transition": 0,
        },
    )


def test_report_writer_adds_scene_diagnostics_to_candidates(tmp_path: Path) -> None:
    """候補ごとに scene 診断情報が追記されること."""
    # Arrange
    report_path = tmp_path / "report.json"
    selected = [
        create_scored_candidate(
            path="/tmp/promotion.jpg",
            scene_label=SceneLabel.EVENT,
            gameplay_score=0.42,
            event_score=0.41,
            other_score=0.39,
            scene_confidence=0.02,
        ),
        create_scored_candidate(
            path="/tmp/raw_event.jpg",
            scene_label=SceneLabel.EVENT,
            gameplay_score=0.35,
            event_score=0.44,
            other_score=0.41,
            scene_confidence=0.06,
        )
    ]
    rejected = [
        create_scored_candidate(
            path="/tmp/fallback.jpg",
            scene_label=SceneLabel.OTHER,
            gameplay_score=0.43,
            event_score=0.30,
            other_score=0.41,
            scene_confidence=0.01,
        )
    ]

    # Act
    ReportWriter.write(
        path=str(report_path),
        selected=selected,
        rejected=rejected,
        stats=_build_stats(),
    )

    # Assert
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    selected_entry = payload["selected"][0]
    rejected_entry = payload["rejected"][0]
    diagnostics_summary = payload["scene_diagnostics_summary"]

    assert selected_entry["scene_confidence"] == 0.02
    assert selected_entry["argmax_scene_label"] == "gameplay"
    assert selected_entry["argmax_score"] == 0.42
    assert selected_entry["argmax_margin"] == 0.01
    assert selected_entry["fallback_applied"] is False
    assert selected_entry["event_promotion_applied"] is True
    assert selected_entry["event_gap_to_winner"] == 0.01

    assert rejected_entry["scene_confidence"] == 0.01
    assert rejected_entry["argmax_scene_label"] == "gameplay"
    assert rejected_entry["argmax_score"] == 0.43
    assert rejected_entry["argmax_margin"] == 0.02
    assert rejected_entry["fallback_applied"] is True
    assert rejected_entry["event_promotion_applied"] is False
    assert rejected_entry["event_gap_to_winner"] == 0.13
    assert diagnostics_summary == {
        "argmax_distribution": {
            "gameplay": 2,
            "event": 1,
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
    }


def test_report_writer_keeps_existing_top_level_payload(tmp_path: Path) -> None:
    """既存の top-level payload を維持しつつ候補詳細を拡張すること."""
    # Arrange
    report_path = tmp_path / "report.json"
    selected = [create_scored_candidate(path="/tmp/selected.jpg")]

    # Act
    ReportWriter.write(
        path=str(report_path),
        selected=selected,
        rejected=[],
        stats=_build_stats(),
    )

    # Assert
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["resolved_profile"] == "static"
    assert payload["scene_distribution"] == {
        "gameplay": 2,
        "event": 1,
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
    assert payload["rejected_by_content_filter"] == 0
    assert payload["content_filter_breakdown"] == {
        "blackout": 0,
        "whiteout": 0,
        "single_tone": 0,
        "fade_transition": 0,
        "temporal_transition": 0,
    }
    assert "scene_confidence" in payload["selected"][0]
    assert "scene_diagnostics_summary" in payload
