"""main.py CLIの単体テスト."""

from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import click
import numpy as np
import pytest

from src.main import Main
from src.models.image_metrics import ImageMetrics
from src.models.normalized_metrics import NormalizedMetrics
from src.models.picker_statistics import PickerStatistics
from src.models.raw_metrics import RawMetrics
from src.services.game_screen_picker import GameScreenPicker


@pytest.fixture
def mock_game_screen_picker() -> MagicMock:
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
    return picker


@pytest.fixture
def sample_image_metrics_factory() -> Callable[[str, float], ImageMetrics]:
    """テスト用ImageMetricsを作成するファクトリー関数."""

    def _create(path: str, score: float) -> ImageMetrics:
        raw = RawMetrics(
            blur_score=score,
            brightness=100.0,
            contrast=50.0,
            edge_density=0.1,
            color_richness=50.0,
            ui_density=10.0,
            action_intensity=30.0,
            visual_balance=80.0,
            dramatic_score=50.0,
        )
        norm = NormalizedMetrics(
            blur_score=score / 100.0,
            contrast=0.5,
            color_richness=0.5,
            edge_density=0.5,
            dramatic_score=0.5,
            visual_balance=0.5,
            action_intensity=0.5,
            ui_density=0.5,
        )
        return ImageMetrics(
            path=path,
            raw_metrics=raw,
            normalized_metrics=norm,
            semantic_score=0.8,
            total_score=score,
            features=np.random.rand(64),
        )

    return _create


def test_cli_selects_and_copies_images(
    monkeypatch: pytest.MonkeyPatch,
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
    sample_image_metrics_factory: Callable[[str, float], ImageMetrics],
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
    test_dir = tmp_path / "test_images"
    test_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # テスト用の画像ファイルを作成（コピー対象）
    for i in range(5):
        img_path = test_dir / f"image{i}.jpg"
        img_path.write_bytes(b"fake_image_data")

    # 結果として返すImageMetricsは実際のファイルパスを指すようにする
    results = [
        sample_image_metrics_factory(str(test_dir / f"image{i}.jpg"), 95.0 - i * 3)
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

    # オプションが前にくる形式: [オプション...] input output
    args = ["-n", "3", str(test_dir), str(output_dir)]

    monkeypatch.setattr(
        "src.main.GameScreenPicker",
        lambda *_a, **_k: mock_game_screen_picker,
    )

    # Act
    Main(args=args).run()

    # Assert
    # 出力ディレクトリにファイルがコピーされたことを確認
    assert (output_dir / "image0.jpg").exists()
    assert (output_dir / "image1.jpg").exists()
    assert (output_dir / "image2.jpg").exists()


@pytest.mark.parametrize(
    "args,input_path_setup",
    [
        # 不存在のディレクトリ
        ([], "nonexistent"),
        # ファイルパス（ディレクトリではない）
        ([], "file_path"),
        # 無効な -n 値
        (["-n", "-1"], None),
        (["-n", "abc"], None),
        # 無効な -s 値
        (["-s", "1.5"], None),
        (["-s", "abc"], None),
    ],
)
def test_cli_validates_inputs(
    monkeypatch: pytest.MonkeyPatch,
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
    args: list[str],
    input_path_setup: str | None,
) -> None:
    """無効な入力に対して適切なエラーが発生すること.

    Given:
        - 無効な入力パス、または無効なコマンドライン引数がある
    When:
        - CLIが実行される
    Then:
        - 適切なエラーが発生すること（clickはClickExceptionを発生させる）
    """
    # Arrange
    if input_path_setup == "nonexistent":
        input_path = "/nonexistent/directory"
    elif input_path_setup == "file_path":
        input_path = str(tmp_path / "file.jpg")
        Path(input_path).touch()
    else:
        # 有効なテストディレクトリを作成
        input_path = str(tmp_path / "valid_dir")
        Path(input_path).mkdir()

    output_path = str(tmp_path / "output")

    # オプションが前にくる形式: [オプション...] input output
    full_args = args + [input_path, output_path]

    monkeypatch.setattr(
        "src.main.GameScreenPicker",
        lambda *_a, **_k: mock_game_screen_picker,
    )

    # Act & Assert
    # Clickのエラーは ClickException
    with pytest.raises(click.ClickException):
        Main(args=full_args).run()
