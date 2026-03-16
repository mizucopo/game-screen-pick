"""report_writer.pyの単体テスト."""

import json
from pathlib import Path

from src.constants.scene_label import SceneLabel
from src.models.picker_statistics import PickerStatistics
from src.utils.report_writer import ReportWriter
from tests.conftest import (
    build_whole_input_profile,
    create_analyzed_image,
    create_scored_candidate,
)


def _build_stats() -> PickerStatistics:
    whole_input_profile = build_whole_input_profile(
        create_analyzed_image(
            path="/tmp/profile_0.jpg",
            raw_metrics_dict={
                "brightness": 95.0,
                "contrast": 0.45,
                "edge_density": 0.12,
                "action_intensity": 0.08,
                "luminance_entropy": 5.2,
                "near_black_ratio": 0.08,
                "near_white_ratio": 0.04,
                "dominant_tone_ratio": 0.58,
                "luminance_range": 32.0,
            },
        ),
        create_analyzed_image(
            path="/tmp/profile_1.jpg",
            raw_metrics_dict={
                "brightness": 125.0,
                "contrast": 0.52,
                "edge_density": 0.18,
                "action_intensity": 0.15,
                "luminance_entropy": 6.1,
                "near_black_ratio": 0.02,
                "near_white_ratio": 0.10,
                "dominant_tone_ratio": 0.64,
                "luminance_range": 40.0,
            },
        ),
    )
    return PickerStatistics(
        total_files=4,
        analyzed_ok=4,
        analyzed_fail=0,
        rejected_by_similarity=1,
        rejected_by_content_filter=2,
        selected_count=2,
        resolved_profile="static",
        scene_distribution={"play": 2, "event": 2},
        scene_mix_target={"play": 1, "event": 1},
        scene_mix_actual={"play": 1, "event": 1},
        threshold_relaxation_used=[0.72],
        content_filter_breakdown={
            "blackout": 0,
            "whiteout": 0,
            "single_tone": 0,
            "fade_transition": 2,
            "temporal_transition": 0,
        },
        whole_input_profile=whole_input_profile,
    )


def test_report_writer_serializes_play_event_fields(tmp_path: Path) -> None:
    """play/eventフィールドが正しくシリアライズされること.

    Given:
        - play候補にplay_score/event_score/density_score/score_bandが設定されている
        - event候補にoutlier_rejectedがTrueで設定されている
        - 統計情報にscene_distribution/scene_mix_targetがある
    When:
        - ReportWriterでJSONレポートを出力する
    Then:
        - 各スコアフィールドが正しく出力されること
        - output_pathが正しく記録されること
        - outlier_rejectedがTrueで出力されること
    """
    # Arrange
    report_path = tmp_path / "report.json"
    selected = [
        create_scored_candidate(
            path="/tmp/play.jpg",
            scene_label=SceneLabel.PLAY,
            play_score=0.8,
            event_score=0.2,
            density_score=0.8,
            selection_score=0.8,
            score_band="high",
        )
    ]
    rejected = [
        create_scored_candidate(
            path="/tmp/event.jpg",
            scene_label=SceneLabel.EVENT,
            play_score=0.2,
            event_score=0.8,
            density_score=0.2,
            selection_score=0.8,
            score_band="low",
            outlier_rejected=True,
        )
    ]

    # Act
    ReportWriter.write(
        path=str(report_path),
        selected=selected,
        rejected=rejected,
        stats=_build_stats(),
        output_paths_by_candidate_id={
            id(selected[0]): "/tmp/output/play0001.jpg",
        },
    )

    # Assert
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["scene_distribution"] == {"play": 2, "event": 2}
    assert payload["scene_mix_target"] == {"play": 1, "event": 1}
    assert payload["selected"][0]["play_score"] == 0.8
    assert payload["selected"][0]["event_score"] == 0.2
    assert payload["selected"][0]["density_score"] == 0.8
    assert payload["selected"][0]["score_band"] == "high"
    assert payload["selected"][0]["output_path"] == "/tmp/output/play0001.jpg"
    assert payload["rejected"][0]["outlier_rejected"] is True


def test_report_writer_keeps_whole_input_profile(tmp_path: Path) -> None:
    """whole_input_profileが保持されること.

    Given:
        - 統計情報にwhole_input_profileが含まれている
        - content_filter_breakdownにfade_transition=2がある
        - rejected_by_content_filter=2がある
    When:
        - ReportWriterでJSONレポートを出力する
    Then:
        - whole_input_profileが出力されること
        - content_filter_breakdownが正しく出力されること
        - scene_diagnostics_summaryが含まれないこと
    """
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
    assert payload["rejected_by_content_filter"] == 2
    assert payload["content_filter_breakdown"]["fade_transition"] == 2
    assert payload["whole_input_profile"] is not None
    profile = payload["whole_input_profile"]
    assert "contrast" in profile
    assert "edge_density" in profile
    assert "action_intensity" in profile
    assert "luminance_entropy" in profile
    assert "scene_diagnostics_summary" not in payload
    assert "score_band" in payload["selected"][0]
