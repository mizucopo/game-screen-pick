"""main.py CLIの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. ユーザー視点でのCLI挙動をテスト（argparse、I/O、エラーハンドリング）
2. モック使用を最小化 - 重いMLモデルとファイル操作のみモック化
3. pytestのtmp_pathを使用したリアルなファイルシステムテスト
4. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
5. capsysでstdoutをキャプチャしてユーザー向け出力を検証
"""

from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.analyzers.image_quality_analyzer import ImageQualityAnalyzer
from src.main import Main
from src.models.image_metrics import ImageMetrics
from src.models.picker_statistics import PickerStatistics
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
        return ImageMetrics(
            path=path,
            raw_metrics={"blur_score": score},
            normalized_metrics={"blur_score": score / 100.0},
            semantic_score=0.8,
            total_score=score,
            features=np.random.rand(64),
        )

    return _create


@pytest.fixture
def test_image_directory(tmp_path: Path) -> str:
    """テスト用画像ディレクトリを作成."""
    images_dir = tmp_path / "test_images"
    images_dir.mkdir()
    for i in range(5):
        (images_dir / f"image{i}.jpg").touch()
    return str(images_dir)


@pytest.fixture(autouse=True)
def setup_main_mocks(
    monkeypatch: pytest.MonkeyPatch,
    mock_image_quality_analyzer: MagicMock,
    mock_game_screen_picker: MagicMock,
) -> None:
    """すべてのテストで必要なモック設定を自動的に適用する."""
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer",
        lambda *a, **k: mock_image_quality_analyzer,  # noqa: ARG005
    )
    monkeypatch.setattr(
        "src.main.GameScreenPicker",
        lambda *a, **k: mock_game_screen_picker,  # noqa: ARG005
    )


def test_cli_selects_and_displays_images(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_game_screen_picker: MagicMock,
    test_image_directory: str,
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
    results = [
        sample_image_metrics_factory(f"/fake/image{i}.jpg", 95.0 - i * 3)
        for i in range(7)
    ]
    stats = PickerStatistics(
        total_files=10,
        analyzed_ok=10,
        analyzed_fail=0,
        rejected_by_similarity=3,
        selected_count=7,
    )
    mock_game_screen_picker.select.return_value = (results, stats)

    monkeypatch.setattr(
        "sys.argv", ["main.py", test_image_directory, "-n", "7", "--no-cache"]
    )

    # Act
    Main().run()

    # Assert
    captured = capsys.readouterr()
    assert "選択された画像一覧" in captured.out
    assert "統計情報" in captured.out
    assert "Score:" in captured.out


def test_cli_copies_images_to_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
    sample_image_metrics_factory: Callable[[str, float], ImageMetrics],
) -> None:
    """画像が出力ディレクトリにコピーされること.

    Given:
        - 入力ディレクトリにファイルが存在する
        - 出力ディレクトリが指定されている
    When:
        - CLIが `-c` オプションで実行される
    Then:
        - 出力ディレクトリが作成されること
        - 画像が出力ディレクトリにコピーされること
        - 成功メッセージが表示されること
    """
    # Arrange
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    results = []
    for i in range(3):
        file_path = input_dir / f"image{i}.jpg"
        file_path.touch()
        results.append(sample_image_metrics_factory(str(file_path), 95.0 - i * 5))

    stats = PickerStatistics(
        total_files=len(results),
        analyzed_ok=len(results),
        analyzed_fail=0,
        rejected_by_similarity=0,
        selected_count=len(results),
    )
    mock_game_screen_picker.select.return_value = (results, stats)

    args = ["main.py", str(input_dir), "-c", str(output_dir), "--no-cache"]
    monkeypatch.setattr("sys.argv", args)

    # Act
    Main().run()

    # Assert
    captured = capsys.readouterr()
    assert f"{len(results)} 枚を" in captured.out
    assert str(output_dir) in captured.out
    assert "保存しました" in captured.out
    assert output_dir.exists()
    assert output_dir.is_dir()
    output_files = list(output_dir.glob("*"))
    assert len(output_files) == len(results)


def test_cli_handles_empty_input_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
) -> None:
    """空のディレクトリが正しく処理されること.

    Given:
        - 空の入力ディレクトリがある
    When:
        - CLIが実行される
    Then:
        - プログラムがクラッシュせず、0件の結果が適切に表示されること
    """
    # Arrange
    input_dir = str(tmp_path / "empty")
    Path(input_dir).mkdir()
    empty_stats = PickerStatistics(
        total_files=0,
        analyzed_ok=0,
        analyzed_fail=0,
        rejected_by_similarity=0,
        selected_count=0,
    )
    mock_game_screen_picker.select.return_value = ([], empty_stats)

    monkeypatch.setattr("sys.argv", ["main.py", input_dir])

    # Act
    Main().run()

    # Assert
    captured = capsys.readouterr()
    assert "選択された画像一覧" in captured.out
    assert "Score:" not in captured.out  # 0件


@pytest.mark.parametrize(
    "args,input_path_setup,error_type,error_patterns",
    [
        (
            [],
            "nonexistent",
            FileNotFoundError,
            ["入力フォルダが存在しません"],
        ),
        (
            [],
            "file_path",
            NotADirectoryError,
            ["フォルダではありません"],
        ),
        (
            ["-n", "-1"],
            None,
            SystemExit,
            ["正の整数を指定してください"],
        ),
        (
            ["-n", "abc"],
            None,
            SystemExit,
            ["整数ではありません"],
        ),
        (
            ["-s", "1.5"],
            None,
            SystemExit,
            ["0.0~1.0の範囲で指定してください"],
        ),
        (
            ["-s", "abc"],
            None,
            SystemExit,
            ["数値ではありません"],
        ),
    ],
)
def test_cli_validates_inputs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    test_image_directory: str,
    args: list[str],
    input_path_setup: str | None,
    error_type: type[Exception],
    error_patterns: list[str],
) -> None:
    """無効な入力に対して適切なエラーが発生すること.

    Given:
        - 無効な入力パス、または無効なコマンドライン引数がある
    When:
        - CLIが実行される
    Then:
        - 適切なエラーが発生すること
        - エラーメッセージに適切な内容が含まれること
    """
    # Arrange
    if input_path_setup == "nonexistent":
        input_path = "/nonexistent/directory"
    elif input_path_setup == "file_path":
        input_path = str(tmp_path / "file.jpg")
        Path(input_path).touch()
    else:
        input_path = test_image_directory

    monkeypatch.setattr("sys.argv", ["main.py", input_path] + args)

    # Act & Assert
    if error_type.__name__ == "SystemExit":
        with pytest.raises(SystemExit):
            Main().run()
        captured = capsys.readouterr()
        for pattern in error_patterns:
            assert pattern in captured.err
    else:
        with pytest.raises(error_type) as exc_info:
            Main().run()
        for pattern in error_patterns:
            assert pattern in str(exc_info.value)


def test_batch_size_argument_passed_to_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """--batch-size引数がSelectionConfigに正しく渡されること.

    Given:
        - 有効な入力ディレクトリが存在する
        - --batch-size引数を指定
    When:
        - CLIが実行される
    Then:
        - 指定したバッチサイズがSelectionConfigに設定されること
    """
    # Arrange
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    from src.models.picker_statistics import PickerStatistics

    empty_stats = PickerStatistics(
        total_files=0,
        analyzed_ok=0,
        analyzed_fail=0,
        rejected_by_similarity=0,
        selected_count=0,
    )

    # GameScreenPickerのモックを設定
    def create_picker_with_config(
        _analyzer: Any,
        config: Any = None,
        **kwargs: Any,  # noqa: ARG001
    ) -> MagicMock:
        """configが正しく渡されていることを確認する."""
        assert config is not None
        assert config.batch_size == 16  # 指定した値
        picker = MagicMock(spec=GameScreenPicker)
        picker.select.return_value = ([], empty_stats)
        return picker

    monkeypatch.setattr(
        "src.main.GameScreenPicker",
        create_picker_with_config,  # noqa: ARG005
    )

    args = ["main.py", str(input_dir), "--batch-size", "16", "--no-cache"]
    monkeypatch.setattr("sys.argv", args)

    # Act
    Main().run()

    # Assertは関数内で行われる


def test_result_max_workers_argument_passed_to_config(
    monkeypatch: pytest.MonkeyPatch,
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
) -> None:
    """--result-max-workers引数がAnalyzerConfigに正しく渡されること.

    Given:
        - 有効な入力ディレクトリが存在する
        - --result-max-workers引数を指定
    When:
        - CLIが実行される
    Then:
        - 指定したワーカー数がAnalyzerConfigに設定されること
    """
    # Arrange
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    from src.models.picker_statistics import PickerStatistics

    empty_stats = PickerStatistics(
        total_files=0,
        analyzed_ok=0,
        analyzed_fail=0,
        rejected_by_similarity=0,
        selected_count=0,
    )

    mock_game_screen_picker.select.return_value = ([], empty_stats)

    # ImageQualityAnalyzerのモックを設定
    def create_analyzer_with_config(
        _cache: Any = None,  # noqa: ARG001
        config: Any = None,
        **kwargs: Any,  # noqa: ARG001
    ) -> MagicMock:
        """configが正しく渡されていることを確認する."""
        assert config is not None
        assert config.result_max_workers == 4  # 指定した値
        analyzer = MagicMock(spec=ImageQualityAnalyzer)
        return analyzer

    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer",
        create_analyzer_with_config,  # noqa: ARG005
    )

    args = ["main.py", str(input_dir), "--result-max-workers", "4", "--no-cache"]
    monkeypatch.setattr("sys.argv", args)

    # Act
    Main().run()

    # Assertは関数内で行われる


def test_default_cache_enabled_without_copy_to(
    monkeypatch: pytest.MonkeyPatch,
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
) -> None:
    """--copy-toなしでもキャッシュが有効になること.

    Given:
        - 有効な入力ディレクトリが存在する
        - --copy-toを指定しない
        - --no-cacheも指定しない
    When:
        - CLIが実行される
    Then:
        - デフォルトのキャッシュパスが使用されること
        - ~/.cache/game-screen-pick/cache.sqlite3 が使用されること
    """
    # Arrange
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    from src.models.picker_statistics import PickerStatistics

    empty_stats = PickerStatistics(
        total_files=0,
        analyzed_ok=0,
        analyzed_fail=0,
        rejected_by_similarity=0,
        selected_count=0,
    )
    mock_game_screen_picker.select.return_value = ([], empty_stats)

    # FeatureCacheのモックを設定
    cache_mock = MagicMock()

    def create_cache_with_default_path(cache_path: Any) -> MagicMock:
        """デフォルトのキャッシュパスが使用されることを確認."""
        # デフォルトパスが使用されていることを確認
        assert cache_path is not None
        assert "game-screen-pick" in str(cache_path)
        return cache_mock

    monkeypatch.setattr(
        "src.main.FeatureCache",
        create_cache_with_default_path,  # noqa: ARG005
    )

    args = ["main.py", str(input_dir)]
    monkeypatch.setattr("sys.argv", args)

    # Act
    Main().run()

    # Assertは関数内で行われる
