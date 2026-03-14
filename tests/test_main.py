"""main.py CLIの単体テスト.

scene mix 対応後のCLIについて、
画像コピー、JSONレポート出力、設定優先順位、入力バリデーションを確認する。
"""

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import click
import pytest

from src.main import Main
from src.models.picker_statistics import PickerStatistics
from src.services.game_screen_picker import GameScreenPicker
from tests.conftest import create_scored_candidate


@pytest.fixture
def mock_game_screen_picker() -> Generator[MagicMock, None, None]:
    """GameScreenPickerをモック.

    現在の `picker.select` 戻り値である
    `(selected, rejected, stats)` を返すように初期化する。
    """
    picker = MagicMock(spec=GameScreenPicker)
    empty_stats = PickerStatistics(
        total_files=0,
        analyzed_ok=0,
        analyzed_fail=0,
        rejected_by_similarity=0,
        selected_count=0,
        resolved_profile="active",
        scene_distribution={"gameplay": 0, "event": 0, "other": 0},
        scene_mix_target={"gameplay": 0, "event": 0, "other": 0},
        scene_mix_actual={"gameplay": 0, "event": 0, "other": 0},
        threshold_relaxation_used=[0.72],
    )
    picker.select.return_value = ([], [], empty_stats)
    yield picker


@pytest.fixture
def setup_test_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """テスト用の入出力ディレクトリを作成する共通fixture."""
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
    """画像が選択されてコピーされること.

    Given:
        - 有効な入力ディレクトリが存在する
        - picker は3件の選択結果を返す
    When:
        - CLIを実行する
    Then:
        - 選択された画像が出力ディレクトリへコピーされること
    """
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
        selected_count=3,
        resolved_profile="active",
        scene_distribution={"gameplay": 3, "event": 2, "other": 0},
        scene_mix_target={"gameplay": 2, "event": 1, "other": 0},
        scene_mix_actual={"gameplay": 2, "event": 1, "other": 0},
        threshold_relaxation_used=[0.72, 0.75],
    )
    mock_game_screen_picker.select.return_value = (results, [], stats)

    args = ["-n", "3", str(test_dir), str(output_dir)]
    monkeypatch.setattr(
        "src.main.GameScreenPicker",
        lambda *_args, **_kwargs: mock_game_screen_picker,
    )
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer",
        lambda *_args, **_kwargs: MagicMock(),
    )

    # Act
    Main(args=args).run()

    # Assert
    assert (output_dir / "image0.jpg").exists()
    assert (output_dir / "image1.jpg").exists()
    assert (output_dir / "image2.jpg").exists()


def test_cli_writes_report_json(
    monkeypatch: pytest.MonkeyPatch,
    mock_game_screen_picker: MagicMock,
    setup_test_dirs: tuple[Path, Path],
) -> None:
    """report-json指定時にJSONが出力されること.

    Given:
        - picker が選択結果と統計情報を返す
        - `--report-json` の出力先が指定されている
    When:
        - CLIを実行する
    Then:
        - JSONレポートファイルが作成されること
    """
    # Arrange
    test_dir, output_dir = setup_test_dirs
    report_path = output_dir / "report.json"
    source = test_dir / "image0.jpg"
    source.write_bytes(b"fake_image_data")

    results = [create_scored_candidate(path=str(source))]
    stats = PickerStatistics(
        total_files=1,
        analyzed_ok=1,
        analyzed_fail=0,
        rejected_by_similarity=0,
        selected_count=1,
        resolved_profile="static",
        scene_distribution={"gameplay": 1, "event": 0, "other": 0},
        scene_mix_target={"gameplay": 1, "event": 0, "other": 0},
        scene_mix_actual={"gameplay": 1, "event": 0, "other": 0},
        threshold_relaxation_used=[0.72],
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
    assert report_path.exists()


def test_build_selection_config_prefers_cli_over_config(tmp_path: Path) -> None:
    """CLI overrideが設定ファイルより優先されること.

    Given:
        - profile、scene mix、similarity を含む設定ファイルがある
        - その一部をCLI引数で上書きする
    When:
        - `build_selection_config` を呼ぶ
    Then:
        - 優先順位 `CLI > config file > built-in default` で解決されること
    """
    # Arrange
    config_path = tmp_path / "picker.toml"
    config_path.write_text(
        '[selection]\nprofile = "static"\n'
        "[scene_mix]\ngameplay = 0.6\nevent = 0.3\nother = 0.1\n"
        "[thresholds]\nsimilarity = 0.66\n",
        encoding="utf-8",
    )

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
    assert config.scene_mix.gameplay == 0.6


@pytest.mark.parametrize(
    "args,input_path_setup,error_pattern",
    [
        ([], "nonexistent", "does not exist"),
        ([], "file_path", "is a file"),
        (["-n", "-1"], None, "正の整数"),
        (["--similarity", "1.5"], None, "0.0~1.0"),
        (
            ["--scene-mix", "gameplay=0.7,event=0.4,other=0.1"],
            None,
            "scene_mixの合計",
        ),
    ],
)
def test_cli_validates_inputs(
    monkeypatch: pytest.MonkeyPatch,
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
    args: list[str],
    input_path_setup: str | None,
    error_pattern: str,
) -> None:
    """無効な入力に対して適切なエラーが発生すること.

    Given:
        - 無効な入力値または無効なパスがある
    When:
        - CLIを実行する
    Then:
        - Click またはモデルバリデーション由来の例外が発生すること
    """
    # Arrange
    if input_path_setup == "nonexistent":
        input_path = "/nonexistent/directory"
    elif input_path_setup == "file_path":
        input_path = str(tmp_path / "file.jpg")
        Path(input_path).touch()
    else:
        input_path = str(tmp_path / "valid_dir")
        Path(input_path).mkdir()

    output_path = str(tmp_path / "output")
    full_args = args + [input_path, output_path]
    monkeypatch.setattr(
        "src.main.GameScreenPicker",
        lambda *_args, **_kwargs: mock_game_screen_picker,
    )
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer",
        lambda *_args, **_kwargs: MagicMock(),
    )

    # Act / Assert
    with pytest.raises((click.ClickException, ValueError), match=error_pattern):
        Main(args=full_args).run()
