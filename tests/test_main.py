"""main.py CLIの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. ユーザー視点でのCLI挙動をテスト（argparse、I/O、エラーハンドリング）
2. モック使用を最小化 - 重いMLモデルとファイル操作のみモック化
3. pytestのtmp_pathを使用したリアルなファイルシステムテスト
4. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
5. capsysでstdoutをキャプチャしてユーザー向け出力を検証
"""

from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.analyzers.image_quality_analyzer import ImageQualityAnalyzer
from src.main import Main
from src.models.image_metrics import ImageMetrics
from src.models.normalized_metrics import NormalizedMetrics
from src.models.picker_statistics import PickerStatistics
from src.models.raw_metrics import RawMetrics
from src.services.game_screen_picker import GameScreenPicker


@pytest.fixture
def mock_image_quality_analyzer() -> MagicMock:
    """ImageQualityAnalyzerをモック（CLIPモデル読み込み回避）."""
    analyzer = MagicMock(spec=ImageQualityAnalyzer)
    return analyzer


@pytest.fixture
def mock_game_screen_picker() -> MagicMock:
    """GameScreenPickerをモック（選択ロジック制御）."""
    picker = MagicMock(spec=GameScreenPicker)
    # 戻り値を(結果リスト, 統計情報)のタプルにする
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
    """テスト用ImageMetricsを作成するファクトリ関数."""

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


def test_cli_selects_and_displays_images(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_image_quality_analyzer: MagicMock,
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
    sample_image_metrics_factory: Callable[[str, float], ImageMetrics],
) -> None:
    """画像が選択されて表示されること.

    Given:
        - 有効な入力ディレクトリが存在する
        - モックされた analyzer と picker がある
    When:
        - CLIが実行される
    Then:
        - 選択された画像が表示されること
        - 統計情報が表示されること
    """
    # Arrange
    test_dir = tmp_path / "test_images"
    test_dir.mkdir()
    for i in range(5):
        (test_dir / f"image{i}.jpg").touch()

    results = [
        sample_image_metrics_factory(f"/fake/image{i}.jpg", 95.0 - i * 3)
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

    monkeypatch.setattr("sys.argv", ["main.py", str(test_dir), "-n", "3"])
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer",
        lambda *_a, **_k: mock_image_quality_analyzer,
    )
    monkeypatch.setattr(
        "src.main.GameScreenPicker",
        lambda *_a, **_k: mock_game_screen_picker,
    )

    # Act
    Main().run()

    # Assert
    captured = capsys.readouterr()
    assert "選択された画像一覧" in captured.out
    assert "統計情報" in captured.out
    assert "Score:" in captured.out


@pytest.mark.parametrize(
    "args,input_path_setup,error_type",
    [
        # 不在のディレクトリ
        ([], "nonexistent", FileNotFoundError),
        # ファイルパス（ディレクトリではない）
        ([], "file_path", NotADirectoryError),
        # 無効な -n 値
        (["-n", "-1"], None, SystemExit),
        (["-n", "abc"], None, SystemExit),
        # 無効な -s 値
        (["-s", "1.5"], None, SystemExit),
        (["-s", "abc"], None, SystemExit),
    ],
)
def test_cli_validates_inputs(
    monkeypatch: pytest.MonkeyPatch,
    mock_image_quality_analyzer: MagicMock,
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
    args: list[str],
    input_path_setup: str | None,
    error_type: type[Exception],
) -> None:
    """無効な入力に対して適切なエラーが発生すること.

    Given:
        - 無効な入力パス、または無効なコマンドライン引数がある
    When:
        - CLIが実行される
    Then:
        - 適切なエラーが発生すること
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

    monkeypatch.setattr("sys.argv", ["main.py", input_path] + args)
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer",
        lambda *_a, **_k: mock_image_quality_analyzer,
    )
    monkeypatch.setattr(
        "src.main.GameScreenPicker",
        lambda *_a, **_k: mock_game_screen_picker,
    )

    # Act & Assert
    with pytest.raises(error_type):
        Main().run()
