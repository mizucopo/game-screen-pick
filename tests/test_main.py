"""main.py CLIの単体テスト."""

import json
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest

from src.constants.scene_label import SceneLabel
from src.main import Main
from src.models.picker_statistics import PickerStatistics
from src.services.game_screen_picker import GameScreenPicker
from tests.conftest import create_scored_candidate


@pytest.fixture
def mock_game_screen_picker() -> Generator[MagicMock, None, None]:
    picker = MagicMock(spec=GameScreenPicker)
    empty_stats = PickerStatistics(
        total_files=0,
        analyzed_ok=0,
        analyzed_fail=0,
        rejected_by_similarity=0,
        rejected_by_content_filter=0,
        selected_count=0,
        resolved_profile="active",
        scene_distribution={"play": 0, "event": 0},
        scene_mix_target={"play": 0, "event": 0},
        scene_mix_actual={"play": 0, "event": 0},
        threshold_relaxation_used=[0.72],
        content_filter_breakdown={
            "blackout": 0,
            "whiteout": 0,
            "single_tone": 0,
            "fade_transition": 0,
            "temporal_transition": 0,
        },
    )
    picker.select.return_value = ([], [], empty_stats)
    yield picker


@pytest.fixture
def setup_test_dirs(tmp_path: Path) -> tuple[Path, Path]:
    test_dir = tmp_path / "test_images"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return test_dir, output_dir


def test_cli_selects_and_copies_images(
    monkeypatch: pytest.MonkeyPatch,
    mock_game_screen_picker: MagicMock,
    setup_test_dirs: tuple[Path, Path],
) -> None:
    """CLIが画像を選択してコピーすること."""
    # Arrange
    test_dir, output_dir = setup_test_dirs
    for index in range(5):
        (test_dir / f"image{index}.jpg").write_bytes(b"fake_image_data")

    results = [
        create_scored_candidate(path=str(test_dir / f"image{index}.jpg"))
        for index in range(3)
    ]
    stats = PickerStatistics(
        total_files=5,
        analyzed_ok=5,
        analyzed_fail=0,
        rejected_by_similarity=2,
        rejected_by_content_filter=0,
        selected_count=3,
        resolved_profile="active",
        scene_distribution={"play": 3, "event": 2},
        scene_mix_target={"play": 2, "event": 1},
        scene_mix_actual={"play": 2, "event": 1},
        threshold_relaxation_used=[0.72, 0.75],
        content_filter_breakdown={
            "blackout": 0,
            "whiteout": 0,
            "single_tone": 0,
            "fade_transition": 0,
            "temporal_transition": 0,
        },
    )
    mock_game_screen_picker.select.return_value = (results, [], stats)

    monkeypatch.setattr(
        "src.main.GameScreenPicker",
        lambda *_args, **_kwargs: mock_game_screen_picker,
    )
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer",
        lambda *_args, **_kwargs: MagicMock(),
    )

    # Act
    Main(args=["-n", "3", str(test_dir), str(output_dir)]).run()

    # Assert
    assert (output_dir / "image0.jpg").exists()
    assert (output_dir / "image1.jpg").exists()
    assert (output_dir / "image2.jpg").exists()


def test_cli_writes_report_json_with_new_fields(
    monkeypatch: pytest.MonkeyPatch,
    mock_game_screen_picker: MagicMock,
    setup_test_dirs: tuple[Path, Path],
) -> None:
    """CLIが新しいフィールドを含むJSONレポートを出力すること."""
    # Arrange
    test_dir, output_dir = setup_test_dirs
    report_path = output_dir / "report.json"
    source = test_dir / "image0.jpg"
    source.write_bytes(b"fake_image_data")

    results = [
        create_scored_candidate(
            path=str(source),
            scene_label=SceneLabel.PLAY,
            play_score=0.8,
            event_score=0.2,
            density_score=0.8,
            selection_score=0.8,
            score_band="high",
        )
    ]
    stats = PickerStatistics(
        total_files=1,
        analyzed_ok=1,
        analyzed_fail=0,
        rejected_by_similarity=0,
        rejected_by_content_filter=0,
        selected_count=1,
        resolved_profile="static",
        scene_distribution={"play": 1, "event": 0},
        scene_mix_target={"play": 1, "event": 0},
        scene_mix_actual={"play": 1, "event": 0},
        threshold_relaxation_used=[0.72],
        content_filter_breakdown={
            "blackout": 0,
            "whiteout": 0,
            "single_tone": 0,
            "fade_transition": 0,
            "temporal_transition": 0,
        },
    )
    mock_game_screen_picker.select.return_value = (results, [], stats)
    monkeypatch.setattr(
        "src.main.GameScreenPicker",
        lambda *_args, **_kwargs: mock_game_screen_picker,
    )
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer",
        lambda *_args, **_kwargs: MagicMock(),
    )

    # Act
    Main(
        args=[
            "--report-json",
            str(report_path),
            str(test_dir),
            str(output_dir),
        ]
    ).run()

    # Assert
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["selected"][0]["path"] == str(source)
    assert payload["selected"][0]["output_path"] == str((output_dir / "image0.jpg").resolve())
    assert payload["selected"][0]["play_score"] == 0.8
    assert payload["selected"][0]["event_score"] == 0.2
    assert payload["selected"][0]["score_band"] == "high"


def test_cli_renames_outputs_by_scene(
    monkeypatch: pytest.MonkeyPatch,
    mock_game_screen_picker: MagicMock,
    setup_test_dirs: tuple[Path, Path],
) -> None:
    """CLIがscene別にファイル名を変更すること."""
    # Arrange
    test_dir, output_dir = setup_test_dirs
    sources = {
        "play_png": test_dir / "source_play.png",
        "event_jpg": test_dir / "source_event.jpg",
        "play_jpg": test_dir / "source_play2.jpg",
    }
    for path in sources.values():
        path.write_bytes(b"fake_image_data")

    results = [
        create_scored_candidate(
            path=str(sources["play_png"]),
            scene_label=SceneLabel.PLAY,
        ),
        create_scored_candidate(
            path=str(sources["event_jpg"]),
            scene_label=SceneLabel.EVENT,
        ),
        create_scored_candidate(
            path=str(sources["play_jpg"]),
            scene_label=SceneLabel.PLAY,
        ),
    ]
    stats = PickerStatistics(
        total_files=3,
        analyzed_ok=3,
        analyzed_fail=0,
        rejected_by_similarity=0,
        rejected_by_content_filter=0,
        selected_count=3,
        resolved_profile="active",
        scene_distribution={"play": 2, "event": 1},
        scene_mix_target={"play": 2, "event": 1},
        scene_mix_actual={"play": 2, "event": 1},
        threshold_relaxation_used=[0.72],
        content_filter_breakdown={
            "blackout": 0,
            "whiteout": 0,
            "single_tone": 0,
            "fade_transition": 0,
            "temporal_transition": 0,
        },
    )
    mock_game_screen_picker.select.return_value = (results, [], stats)
    monkeypatch.setattr(
        "src.main.GameScreenPicker",
        lambda *_args, **_kwargs: mock_game_screen_picker,
    )
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer",
        lambda *_args, **_kwargs: MagicMock(),
    )

    # Act
    Main(args=["-n", "3", "--rename", str(test_dir), str(output_dir)]).run()

    # Assert
    assert (output_dir / "play0001.png").exists()
    assert (output_dir / "event0001.jpg").exists()
    assert (output_dir / "play0002.jpg").exists()


def test_build_selection_config_prefers_cli_over_config(tmp_path: Path) -> None:
    """CLIオプションが設定ファイルより優先されること."""
    # Arrange
    config_path = tmp_path / "picker.toml"
    config_path.write_text(
        '[selection]\nprofile = "static"\n'
        "[scene_mix]\nplay = 0.6\nevent = 0.4\n"
        "[thresholds]\nsimilarity = 0.66\n",
        encoding="utf-8",
    )

    # Act
    config = Main.build_selection_config(
        config_path=str(config_path),
        profile="active",
        scene_mix=None,
        similarity=0.8,
        batch_size=None,
    )

    # Assert
    assert config.profile == "active"
    assert config.similarity_threshold == 0.8
    assert config.scene_mix.play == 0.6


@pytest.mark.parametrize(
    "args,error_pattern",
    [
        (["-n", "-1"], "正の整数"),
        (["--similarity", "1.5"], "0.0~1.0"),
        (["--scene-mix", "play=0.7,event=0.4"], "scene_mixの合計"),
    ],
)
def test_cli_validates_inputs(
    monkeypatch: pytest.MonkeyPatch,
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
    args: list[str],
    error_pattern: str,
) -> None:
    """CLIが無効な入力をバリデーションすること."""
    # Arrange
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()
    output_path.mkdir()
    monkeypatch.setattr(
        "src.main.GameScreenPicker",
        lambda *_args, **_kwargs: mock_game_screen_picker,
    )
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer",
        lambda *_args, **_kwargs: MagicMock(),
    )

    # Act / Assert
    with pytest.raises(Exception, match=error_pattern):
        Main(args=[*args, str(input_path), str(output_path)]).run()
