"""report_writer.pyの単体テスト."""

import json
from pathlib import Path

from src.models.output_candidate_record import OutputCandidateRecord
from src.models.output_record import OutputRecord
from src.utils.report_writer import ReportWriter


def _build_candidate(
    *,
    source_path: str = "/tmp/battle.jpg",
    scene_slug: str = "battle",
    scene_display_name: str = "戦闘",
    scene_description: str = "敵との戦闘場面",
    variant_group: str | None = "battle_001",
    score_band: str | None = "high",
    outlier_rejected: bool = False,
    output_path: str | None = None,
) -> OutputCandidateRecord:
    filename = Path(source_path).name
    return OutputCandidateRecord(
        source_path=source_path,
        filename=filename,
        suffix=Path(filename).suffix,
        scene_slug=scene_slug,
        scene_display_name=scene_display_name,
        scene_description=scene_description,
        scene_confidence=0.5,
        quality_score=0.6,
        selection_score=0.8,
        score_band=score_band,
        variant_group=variant_group,
        outlier_rejected=outlier_rejected,
        output_path=output_path,
    )


def _build_output_record(
    *,
    selected: list[OutputCandidateRecord] | None = None,
    rejected: list[OutputCandidateRecord] | None = None,
    whole_input_profile: dict[str, dict[str, float]] | None = None,
) -> OutputRecord:
    return OutputRecord(
        selected=selected if selected is not None else [_build_candidate()],
        rejected=rejected if rejected is not None else [],
        total_files=4,
        analyzed_ok=4,
        analyzed_fail=0,
        rejected_by_similarity=1,
        rejected_by_content_filter=2,
        selected_count=2,
        scene_distribution={"battle": 2, "conversation": 2},
        scene_mix_target={"battle": 1, "conversation": 1},
        scene_mix_actual={"battle": 1, "conversation": 1},
        threshold_relaxation_steps=[0.72],
        content_filter_breakdown={
            "blackout": 0,
            "whiteout": 0,
            "single_tone": 0,
            "fade_transition": 2,
            "temporal_transition": 0,
        },
        whole_input_profile=whole_input_profile,
        scene_catalog=[
            {
                "slug": "battle",
                "display_name": "戦闘",
                "description": "敵との戦闘場面",
            },
            {
                "slug": "conversation",
                "display_name": "会話",
                "description": "人物同士の会話場面",
            },
            {
                "slug": "other",
                "display_name": "その他",
                "description": "分類しにくい場面",
            },
        ],
        ollama_classification_failed=1,
        ollama_classification_failure_rate=0.25,
        ollama_catalog_fallback_used=True,
        ollama_catalog_fallback_reason="OSError: timed out",
    )


def test_report_writer_serializes_output_record_fields(tmp_path: Path) -> None:
    """output recordから既存JSONフィールドが出力されること.

    Arrange:
        - 出力候補と統計値を持つoutput recordがある
    Act:
        - ReportWriterでJSONレポートを出力する
    Assert:
        - 既存の候補フィールドと統計フィールドが維持されること
    """
    # Arrange
    report_path = tmp_path / "report.json"
    selected = [
        _build_candidate(
            source_path="/tmp/battle.jpg",
            output_path="/tmp/output/battle0001.jpg",
        )
    ]
    rejected = [
        _build_candidate(
            source_path="/tmp/conversation.jpg",
            scene_slug="conversation",
            scene_display_name="会話",
            scene_description="人物同士の会話場面",
            variant_group="conversation_001",
            score_band="low",
            outlier_rejected=True,
        )
    ]

    # Act
    ReportWriter.write(
        str(report_path),
        _build_output_record(selected=selected, rejected=rejected),
    )

    # Assert
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "resolved_profile" not in payload
    assert payload["scene_distribution"] == {"battle": 2, "conversation": 2}
    assert payload["scene_mix_target"] == {"battle": 1, "conversation": 1}
    assert payload["scene_catalog"][0]["display_name"] == "戦闘"
    assert payload["ollama_classification_failed"] == 1
    assert payload["ollama_classification_failure_rate"] == 0.25
    assert payload["ollama_catalog_fallback_used"] is True
    assert payload["ollama_catalog_fallback_reason"] == "OSError: timed out"
    assert payload["threshold_relaxation_steps"] == [0.72]
    assert payload["rejected_by_content_filter"] == 2
    assert payload["content_filter_breakdown"]["fade_transition"] == 2
    assert payload["selected"][0]["path"] == "/tmp/battle.jpg"
    assert payload["selected"][0]["scene_slug"] == "battle"
    assert payload["selected"][0]["scene_display_name"] == "戦闘"
    assert payload["selected"][0]["scene_description"] == "敵との戦闘場面"
    assert payload["selected"][0]["variant_group"] == "battle_001"
    assert payload["selected"][0]["score_band"] == "high"
    assert payload["selected"][0]["output_path"] == "/tmp/output/battle0001.jpg"
    assert payload["rejected"][0]["outlier_rejected"] is True


def test_report_writer_keeps_whole_input_profile(tmp_path: Path) -> None:
    """whole_input_profileが保持されること.

    Arrange:
        - output recordにwhole_input_profileが含まれている
    Act:
        - ReportWriterでJSONレポートを出力する
    Assert:
        - whole_input_profileが出力されること
        - scene_diagnostics_summaryが含まれないこと
    """
    # Arrange
    report_path = tmp_path / "report.json"
    profile = {
        "brightness": {"p10": 95.0, "p25": 100.0, "p50": 110.0, "p90": 125.0},
        "contrast": {"p10": 0.45, "p25": 0.5, "p50": 0.52, "p90": 0.55},
    }

    # Act
    ReportWriter.write(
        str(report_path),
        _build_output_record(whole_input_profile=profile),
    )

    # Assert
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["whole_input_profile"] == profile
    assert "scene_diagnostics_summary" not in payload
    assert "score_band" in payload["selected"][0]


def test_write_creates_json_file(tmp_path: Path) -> None:
    """JSONレポートファイルが作成されること."""
    # Arrange
    report_path = tmp_path / "report.json"

    # Act
    ReportWriter.write(str(report_path), _build_output_record())

    # Assert
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert "selected" in data
    assert len(data["selected"]) == 1


def test_write_handles_empty_selected(tmp_path: Path) -> None:
    """選択結果が空でもJSONが正しく出力されること."""
    # Arrange
    report_path = tmp_path / "report.json"

    # Act
    ReportWriter.write(
        str(report_path),
        _build_output_record(selected=[], rejected=[]),
    )

    # Assert
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["selected"] == []
    assert data["whole_input_profile"] is None
