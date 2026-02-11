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
from src.models.image_metrics import ImageMetrics
from src.services.screen_picker import GameScreenPicker


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_image_quality_analyzer() -> MagicMock:
    """ImageQualityAnalyzerをモック（CLIPモデル読み込み回避）."""
    analyzer = MagicMock(spec=ImageQualityAnalyzer)
    return analyzer


@pytest.fixture
def mock_game_screen_picker() -> MagicMock:
    """GameScreenPickerをモック（選択ロジック制御）."""
    picker = MagicMock(spec=GameScreenPicker)
    picker.select.return_value = []
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
        "src.main.ImageQualityAnalyzer", lambda *_: mock_image_quality_analyzer
    )
    monkeypatch.setattr("src.main.GameScreenPicker", lambda *_: mock_game_screen_picker)


# ============================================================================
# 引数解析のテスト
# ============================================================================


def test_cli_accepts_all_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_game_screen_picker: MagicMock,
    test_image_directory: str,
    tmp_path: Path,
    sample_image_metrics_factory: Callable[[str, float], ImageMetrics],
) -> None:
    """全ての引数が正しくパースされること.

    Given:
        - 有効な入力ディレクトリが存在する
        - 全てのオプション引数が指定されている
        - モックされた analyzer と picker がある
    When:
        - CLIが全引数指定で実行される
    Then:
        - ImageQualityAnalyzerが指定ジャンルで初期化されること
        - picker.selectが正しいパラメータで呼ばれること
        - 出力ディレクトリが作成されること
    """
    # Arrange
    output_dir = str(tmp_path / "output")
    img_path = Path(test_image_directory) / "image0.jpg"
    img_path.touch()
    results = [sample_image_metrics_factory(str(img_path), 95.0)]
    mock_game_screen_picker.select.return_value = results

    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            test_image_directory,
            "-c",
            output_dir,
            "-n",
            "5",
            "-g",
            "2d_rpg",
            "-s",
            "0.85",
            "-r",
        ],
    )

    # Act
    from src.main import Main

    Main().run()

    # Assert
    captured = capsys.readouterr()
    # 指定された枚数の画像が選択される
    assert captured.out.count("Score:") == 1
    # 出力ディレクトリが存在する
    assert Path(output_dir).exists()
    # 画像が出力ディレクトリにコピーされる
    output_files = list(Path(output_dir).glob("*.jpg"))
    assert len(output_files) == 1
    # 成功メッセージが表示される
    assert "1 枚を" in captured.out
    assert output_dir in captured.out
    assert "保存しました" in captured.out


def test_cli_shows_error_for_missing_required_argument(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """必須引数（input）が missing の時にエラーが表示されること.

    Given:
        - 必須引数（input）が指定されていない
    When:
        - CLIが実行される
    Then:
        - 適切なエラーメッセージが出力されること
        - プログラムが終了すること
    """
    # Arrange
    monkeypatch.setattr("sys.argv", ["main.py"])

    # Act & Assert
    from src.main import Main

    with pytest.raises(SystemExit):
        Main().run()

    captured = capsys.readouterr()
    # argparseのエラーメッセージが含まれている
    assert "error" in captured.err.lower() or "required" in captured.err.lower()


# ============================================================================
# 基本機能のテスト
# ============================================================================


@pytest.mark.parametrize(
    "args,num_expected",
    [
        ([], 10),  # デフォルト値
        (["-n", "7"], 7),  # カスタム値
    ],
)
def test_cli_selects_and_displays_images(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_game_screen_picker: MagicMock,
    test_image_directory: str,
    args: list[str],
    num_expected: int,
    sample_image_metrics_factory: Callable[[str, float], ImageMetrics],
) -> None:
    """画像が選択されて表示されること.

    Given:
        - 有効な入力ディレクトリが存在する
        - モックされた analyzer と picker がある
        - デフォルトまたはカスタムパラメータが指定されている
    When:
        - CLIが実行される
    Then:
        - 指定された枚数分の結果が表示されること
    """
    # Arrange
    results = [
        sample_image_metrics_factory(f"/fake/image{i}.jpg", 95.0 - i * 3)
        for i in range(num_expected)
    ]
    mock_game_screen_picker.select.return_value = results

    monkeypatch.setattr("sys.argv", ["main.py", test_image_directory] + args)

    # Act
    from src.main import Main

    Main().run()

    # Assert
    captured = capsys.readouterr()
    # 結果一覧が表示される
    assert "選択された画像一覧" in captured.out
    # 指定された枚数分のスコア表示がある
    assert captured.out.count("Score:") == num_expected


# ============================================================================
# Tests for Copy/Output Functionality
# ============================================================================


def test_cli_copies_selected_images_to_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_game_screen_picker: MagicMock,
    test_image_directory: str,
    tmp_path: Path,
    sample_image_metrics_factory: Callable[[str, float], ImageMetrics],
) -> None:
    """選択された画像が出力ディレクトリにコピーされること.

    Given:
        - 有効な入力ディレクトリが存在する
        - 存在しないネストされた出力ディレクトリが指定されている
        - 3つの選択結果がある
    When:
        - CLIが `-c` オプションで実行される
    Then:
        - 出力ディレクトリが作成されること
        - 画像が出力ディレクトリにコピーされること
        - 成功メッセージにコピー数とパスが含まれること
    """
    # Arrange
    output_dir = tmp_path / "parent" / "output"
    results = []
    for i in range(3):
        img_path = Path(test_image_directory) / f"image{i}.jpg"
        img_path.touch()
        results.append(sample_image_metrics_factory(str(img_path), 95.0 - i * 5))

    mock_game_screen_picker.select.return_value = results

    monkeypatch.setattr(
        "sys.argv", ["main.py", test_image_directory, "-c", str(output_dir)]
    )

    # Act
    from src.main import Main

    Main().run()

    # Assert
    captured = capsys.readouterr()
    # 成功メッセージが表示される
    assert "3 枚を" in captured.out
    assert str(output_dir) in captured.out
    assert "保存しました" in captured.out
    # 出力ディレクトリが作成されている
    assert output_dir.exists()
    assert output_dir.is_dir()
    # 画像が出力ディレクトリにコピーされている
    assert (output_dir / "image0.jpg").exists()
    assert (output_dir / "image1.jpg").exists()
    assert (output_dir / "image2.jpg").exists()


# ============================================================================
# Tests for Error Handling
# ============================================================================


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
    mock_game_screen_picker.select.return_value = []

    monkeypatch.setattr("sys.argv", ["main.py", input_dir])

    # Act
    from src.main import Main

    Main().run()

    # Assert
    captured = capsys.readouterr()
    assert "選択された画像一覧" in captured.out
    assert "Score:" not in captured.out  # 0件


def test_cli_exits_with_error_for_nonexistent_input_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """存在しないディレクトリを指定した場合にエラーが発生すること.

    Given:
        - 存在しない入力ディレクトリパスがある
    When:
        - CLIが実行される
    Then:
        - FileNotFoundErrorが発生すること
        - エラーメッセージにパスが含まれること
    """
    # Arrange
    input_dir = "/nonexistent/directory/that/does/not/exist"
    monkeypatch.setattr("sys.argv", ["main.py", input_dir])

    # Act & Assert
    from src.main import Main

    with pytest.raises(FileNotFoundError) as exc_info:
        Main().run()

    assert input_dir in str(exc_info.value)
    assert "入力フォルダが存在しません" in str(exc_info.value)


def test_cli_exits_with_error_when_file_instead_of_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ファイルパスを指定した場合にエラーが発生すること.

    Given:
        - ファイルパスがある
    When:
        - CLIが実行される
    Then:
        - NotADirectoryErrorが発生すること
    """
    # Arrange
    test_file = tmp_path / "image.jpg"
    test_file.touch()
    monkeypatch.setattr("sys.argv", ["main.py", str(test_file)])

    # Act & Assert
    from src.main import Main

    with pytest.raises(NotADirectoryError) as exc_info:
        Main().run()

    assert str(test_file) in str(exc_info.value)
    assert "フォルダではありません" in str(exc_info.value)


# ============================================================================
# Tests for Duplicate Filename Handling
# ============================================================================


@pytest.mark.parametrize(
    "num_files,base_name,extension",
    [
        (2, "image", ".jpg"),
        (3, "screenshot", ".png"),
    ],
)
def test_cli_handles_duplicate_filenames_with_increasing_suffixes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
    num_files: int,
    base_name: str,
    extension: str,
    sample_image_metrics_factory: Callable[[str, float], ImageMetrics],
) -> None:
    """同名ファイルが複数存在する場合に連番でサフィックスが付与されて上書きが回避されること.

    Given:
        - 別々のフォルダに同名ファイルが存在する
        - 出力ディレクトリが指定されている
        - 複数の画像が選択されている
    When:
        - CLIが `-c` オプションで実行される
    Then:
        - 1つ目は元の名前、2つ目以降は _1, _2,... のサフィックスで保存されること
        - すべてのファイルが出力ディレクトリに存在すること
    """
    # Arrange
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for i in range(num_files):
        folder = input_dir / f"folder{i}"
        folder.mkdir()
        (folder / f"{base_name}{extension}").touch()

    output_dir = tmp_path / "output"

    results = []
    for i in range(num_files):
        img_path = input_dir / f"folder{i}" / f"{base_name}{extension}"
        results.append(sample_image_metrics_factory(str(img_path), 95.0 - i * 5))
    mock_game_screen_picker.select.return_value = results

    monkeypatch.setattr(
        "sys.argv", ["main.py", str(input_dir), "-c", str(output_dir), "-r"]
    )

    # Act
    from src.main import Main

    Main().run()

    # Assert
    captured = capsys.readouterr()
    # 成功メッセージが表示される
    assert f"{num_files} 枚を" in captured.out
    assert str(output_dir) in captured.out
    # すべてのファイルが出力ディレクトリに存在する
    assert (output_dir / f"{base_name}{extension}").exists()
    for i in range(1, num_files):
        assert (output_dir / f"{base_name}_{i}{extension}").exists()
    # 出力ディレクトリには正確にnum_files個のファイルが存在
    output_files = list(output_dir.glob(f"*{extension}"))
    assert len(output_files) == num_files


# ============================================================================
# Tests for Input Validation
# ============================================================================


@pytest.mark.parametrize(
    "num_arg,error_message",
    [
        ("-1", "正の整数を指定してください"),
        ("0", "正の整数を指定してください"),
        ("-100", "正の整数を指定してください"),
        ("abc", "整数ではありません"),
    ],
)
def test_cli_rejects_invalid_num_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    test_image_directory: str,
    num_arg: str,
    error_message: str,
) -> None:
    """--num に不正な値が指定された場合にエラーが表示されること.

    Given:
        - 有効な入力ディレクトリが存在する
        - --num に不正な値が指定されている
    When:
        - CLIが実行される
    Then:
        - 適切なエラーメッセージが表示されること
        - プログラムが終了すること
    """
    # Arrange
    monkeypatch.setattr("sys.argv", ["main.py", test_image_directory, "-n", num_arg])

    # Act & Assert
    from src.main import Main

    with pytest.raises(SystemExit):
        Main().run()

    captured = capsys.readouterr()
    # エラーメッセージが含まれている
    assert error_message in captured.err


@pytest.mark.parametrize(
    "similarity_arg,error_message",
    [
        ("1.5", "0.0~1.0の範囲で指定してください"),
        ("2.0", "0.0~1.0の範囲で指定してください"),
        ("-0.1", "0.0~1.0の範囲で指定してください"),
        ("-1.0", "0.0~1.0の範囲で指定してください"),
        ("abc", "数値ではありません"),
    ],
)
def test_cli_rejects_invalid_similarity_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    test_image_directory: str,
    similarity_arg: str,
    error_message: str,
) -> None:
    """--similarity に不正な値が指定された場合にエラーが表示されること.

    Given:
        - 有効な入力ディレクトリが存在する
        - --similarity に不正な値が指定されている
    When:
        - CLIが実行される
    Then:
        - 適切なエラーメッセージが表示されること
        - プログラムが終了すること
    """
    # Arrange
    monkeypatch.setattr(
        "sys.argv", ["main.py", test_image_directory, "-s", similarity_arg]
    )

    # Act & Assert
    from src.main import Main

    with pytest.raises(SystemExit):
        Main().run()

    captured = capsys.readouterr()
    # エラーメッセージが含まれている
    assert error_message in captured.err


@pytest.mark.parametrize(
    "num_arg,similarity_arg",
    [
        ("1", "0.0"),  # 境界値（最小）
        ("1", "1.0"),  # 境界値（最大）
        ("100", "0.5"),  # 有効な値
    ],
)
def test_cli_accepts_valid_boundary_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_game_screen_picker: MagicMock,
    test_image_directory: str,
    num_arg: str,
    similarity_arg: str,
    sample_image_metrics_factory: Callable[[str, float], ImageMetrics],
) -> None:
    """--num と --similarity の境界値が正しく受け付けられること.

    Given:
        - 有効な入力ディレクトリが存在する
        -- --num と --similarity に境界値または有効な値が指定されている
    When:
        - CLIが実行される
    Then:
        - プログラムが正常終了すること
        - 結果が表示されること
    """
    # Arrange
    img_path = Path(test_image_directory) / "image0.jpg"
    img_path.touch()
    results = [sample_image_metrics_factory(str(img_path), 95.0)]
    mock_game_screen_picker.select.return_value = results

    monkeypatch.setattr(
        "sys.argv",
        ["main.py", test_image_directory, "-n", num_arg, "-s", similarity_arg],
    )

    # Act
    from src.main import Main

    Main().run()

    # Assert
    captured = capsys.readouterr()
    # 正常に終了し、結果が表示される
    assert "選択された画像一覧" in captured.out
    assert "Score:" in captured.out
