"""main.py CLIの単体テスト."""

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import click
import numpy as np
import pytest

from src.main import Main
from src.models.picker_statistics import PickerStatistics
from src.services.game_screen_picker import GameScreenPicker
from tests.conftest import create_image_metrics


@pytest.fixture
def mock_game_screen_picker() -> Generator[MagicMock, None, None]:
    """GameScreenPickerをモック（選択ロジック制御）."""
    picker = MagicMock(spec=GameScreenPicker)
    empty_stats = PickerStatistics(
        total_files=0,
        analyzed_ok=0,
        analyzed_fail=0,
        rejected_by_similarity=0,
        selected_count=0,
    )
    picker.select.return_value = ([], empty_stats)
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
        - モックされた picker がある
    When:
        - CLIが実行される
    Then:
        - 選択された画像が出力ディレクトリにコピーされること
    """
    # Arrange
    test_dir, output_dir = setup_test_dirs

    # テスト用の画像ファイルを作成（コピー対象）
    for i in range(5):
        (test_dir / f"image{i}.jpg").write_bytes(b"fake_image_data")

    # 結果として返すImageMetricsは実際のファイルパスを指すようにする
    results = [
        create_image_metrics(
            path=str(test_dir / f"image{i}.jpg"),
            raw_metrics_dict={"blur_score": 95.0 - i * 3},
            normalized_metrics_dict={"blur_score": (95.0 - i * 3) / 100.0},
            total_score=95.0 - i * 3,
            features=np.random.rand(64),
        )
        for i in range(3)
    ]
    stats = PickerStatistics(
        total_files=5,
        analyzed_ok=5,
        analyzed_fail=0,
        rejected_by_similarity=2,
        selected_count=3,
    )
    mock_game_screen_picker.select.return_value = (results, stats)

    args = ["-n", "3", str(test_dir), str(output_dir)]
    monkeypatch.setattr(
        "src.main.GameScreenPicker",
        lambda *_args, **_kwargs: mock_game_screen_picker,
    )

    # Act
    Main(args=args).run()

    # Assert
    assert (output_dir / "image0.jpg").exists()
    assert (output_dir / "image1.jpg").exists()
    assert (output_dir / "image2.jpg").exists()


@pytest.mark.parametrize(
    "args,input_path_setup",
    [
        ([], "nonexistent"),  # 不存在のディレクトリ
        ([], "file_path"),  # ファイルパス（ディレクトリではない）
        (["-n", "-1"], None),  # 無効な -n 値（負の数値）
        (["-s", "1.5"], None),  # 無効な -s 値（範囲外）
    ],
)
def test_cli_validates_inputs(
    monkeypatch: pytest.MonkeyPatch,
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
    args: list[str],
    input_path_setup: str | None,
) -> None:
    """無効な入力に対して適切なエラーが発生こと."""
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

    # Act & Assert
    with pytest.raises(click.ClickException):
        Main(args=full_args).run()
