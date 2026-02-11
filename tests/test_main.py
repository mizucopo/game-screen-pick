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
from src.models.picker_statistics import PickerStatistics
from src.services.screen_picker import GameScreenPicker


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
        "src.main.ImageQualityAnalyzer", lambda *_: mock_image_quality_analyzer
    )
    monkeypatch.setattr("src.main.GameScreenPicker", lambda *_: mock_game_screen_picker)


def test_cli_accepts_all_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_game_screen_picker: MagicMock,
    test_image_directory: str,
    tmp_path: Path,
    sample_image_metrics_factory: Callable[[str, float], ImageMetrics],
) -> None:
    """全ての引数が正しくパースされて処理が完了すること.

    Given:
        - 有効な入力ディレクトリが存在する
        - 全てのオプション引数が指定されている
        - モックされた analyzer と picker がある
    When:
        - CLIが全引数指定で実行される
    Then:
        - プログラムが正常に完了すること
        - 結果が正しく表示されること
    """
    # Arrange
    output_dir = str(tmp_path / "output")
    img_path = Path(test_image_directory) / "image0.jpg"
    img_path.touch()
    results = [sample_image_metrics_factory(str(img_path), 95.0)]
    stats = PickerStatistics(
        total_files=5,
        analyzed_ok=5,
        analyzed_fail=0,
        rejected_by_similarity=4,
        selected_count=1,
    )
    mock_game_screen_picker.select.return_value = (results, stats)

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

    # Assert - 全オプション指定で正常完了すること
    captured = capsys.readouterr()
    assert "選択された画像一覧" in captured.out


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
        - カスタムパラメータが指定されている
    When:
        - CLIが実行される
    Then:
        - 指定された枚数分の結果が表示されること
    """
    # Arrange
    num_expected = 7
    results = [
        sample_image_metrics_factory(f"/fake/image{i}.jpg", 95.0 - i * 3)
        for i in range(num_expected)
    ]
    stats = PickerStatistics(
        total_files=10,
        analyzed_ok=10,
        analyzed_fail=0,
        rejected_by_similarity=3,
        selected_count=7,
    )
    mock_game_screen_picker.select.return_value = (results, stats)

    monkeypatch.setattr("sys.argv", ["main.py", test_image_directory, "-n", "7"])

    # Act
    from src.main import Main

    Main().run()

    # Assert
    captured = capsys.readouterr()
    # 結果一覧が表示される
    assert "選択された画像一覧" in captured.out
    # 統計情報が表示される
    assert "統計情報" in captured.out
    # 指定された枚数分のスコア表示がある
    assert captured.out.count("Score:") == num_expected


@pytest.mark.parametrize(
    "setup_files,expected_files,description",
    [
        # 異なる名前のファイル
        (
            [
                (
                    "setup",
                    [
                        ("image0.jpg", 95.0),
                        ("image1.jpg", 90.0),
                        ("image2.jpg", 85.0),
                    ],
                )
            ],
            ["image0.jpg", "image1.jpg", "image2.jpg"],
            "異なる名前のファイル",
        ),
        # 同名ファイル（サフィックス付与）
        (
            [
                ("folder0", [("image.jpg", 95.0)]),
                ("folder1", [("image.jpg", 90.0)]),
            ],
            ["image.jpg", "image_1.jpg"],
            "同名ファイル（サフィックス付与）",
        ),
    ],
)
def test_cli_copies_images_to_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
    setup_files: list[tuple[str, list[tuple[str, float]]]],
    expected_files: list[str],
    description: str,
    sample_image_metrics_factory: Callable[[str, float], ImageMetrics],
) -> None:
    """画像が出力ディレクトリにコピーされること.

    Given:
        - {description}が入力ディレクトリに存在する
        - 出力ディレクトリが指定されている
    When:
        - CLIが `-c` オプションで実行される
    Then:
        - 出力ディレクトリが作成されること
        - 画像が出力ディレクトリにコピーされること
        - 同名ファイルにはサフィックスが付与されること
        - 成功メッセージが表示されること
    """.format(description=description)
    from src.main import Main

    # Arrange
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    # setup_filesの形式: [(folder_name, [(filename, score), ...]), ...]
    # folder_nameが"setup"の場合はルートディレクトリに直接配置
    files_config = setup_files
    results = []
    all_files = []  # (folder_path, filename, score)

    for folder_name, files in files_config:
        if folder_name == "setup":
            folder_path = input_dir
        else:
            folder_path = input_dir / folder_name
            folder_path.mkdir()

        for filename, score in files:
            file_path = folder_path / filename
            file_path.touch()
            results.append(sample_image_metrics_factory(str(file_path), score))
            all_files.append((folder_path, filename, score))

    stats = PickerStatistics(
        total_files=len(results),
        analyzed_ok=len(results),
        analyzed_fail=0,
        rejected_by_similarity=0,
        selected_count=len(results),
    )
    mock_game_screen_picker.select.return_value = (results, stats)

    args = ["main.py", str(input_dir), "-c", str(output_dir)]
    monkeypatch.setattr("sys.argv", args)

    # Act
    Main().run()

    # Assert
    captured = capsys.readouterr()
    # 成功メッセージが表示される
    assert f"{len(results)} 枚を" in captured.out
    assert str(output_dir) in captured.out
    assert "保存しました" in captured.out
    # 出力ディレクトリが作成されている
    assert output_dir.exists()
    assert output_dir.is_dir()
    # すべての期待されるファイルが出力ディレクトリに存在する
    for expected_file in expected_files:
        assert (output_dir / expected_file).exists()
    # 出力ディレクトリには正確に期待される数のファイルが存在
    output_files = list(output_dir.glob("*"))
    assert len(output_files) == len(expected_files)


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
    from src.main import Main

    Main().run()

    # Assert
    captured = capsys.readouterr()
    assert "選択された画像一覧" in captured.out
    assert "Score:" not in captured.out  # 0件


@pytest.mark.parametrize(
    "input_path_fixture,error_type,error_msg_fragment",
    [
        # 存在しないディレクトリ
        (
            "nonexistent",
            FileNotFoundError,
            "入力フォルダが存在しません",
        ),
        # ファイルパス（ディレクトリではない）
        (
            "file_path",
            NotADirectoryError,
            "フォルダではありません",
        ),
    ],
)
def test_cli_validates_input_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    input_path_fixture: str,
    error_type: type[Exception],
    error_msg_fragment: str,
) -> None:
    """入力ディレクトリのバリデーションが正しく機能こと.

    Given:
        - 無効な入力パス（存在しないディレクトリまたはファイルパス）がある
    When:
        - CLIが実行される
    Then:
        - 適切なエラーが発生すること
        - エラーメッセージにパスが含まれること
    """
    from src.main import Main

    # ファイルパスのテストケースの場合はテストファイルを作成
    if input_path_fixture == "file_path":
        input_path = str(tmp_path / "image.jpg")
        Path(input_path).touch()
    else:
        input_path = "/nonexistent/directory"

    monkeypatch.setattr("sys.argv", ["main.py", input_path])

    with pytest.raises(error_type) as exc_info:
        Main().run()

    assert input_path in str(exc_info.value)
    assert error_msg_fragment in str(exc_info.value)


@pytest.mark.parametrize(
    "args,error_message",
    [
        # --num の無効値
        (["-n", "-1"], "正の整数を指定してください"),
        (["-n", "abc"], "整数ではありません"),
        # --similarity の無効値
        (["-s", "1.5"], "0.0~1.0の範囲で指定してください"),
        (["-s", "abc"], "数値ではありません"),
    ],
)
def test_cli_validates_num_and_similarity_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    test_image_directory: str,
    args: list[str],
    error_message: str,
) -> None:
    """--num と --similarity のバリデーションが正しく機能すること.

    Given:
        - 有効な入力ディレクトリが存在する
        - --num または --similarity に無効値が指定されている
    When:
        - CLIが実行される
    Then:
        - 適切なエラーメッセージが表示されること
        - プログラムが終了すること
    """
    # Arrange & Act & Assert
    monkeypatch.setattr("sys.argv", ["main.py", test_image_directory] + args)

    from src.main import Main

    with pytest.raises(SystemExit):
        Main().run()
    captured = capsys.readouterr()
    assert error_message in captured.err


def test_cli_accepts_seed_argument(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_game_screen_picker: MagicMock,
    test_image_directory: str,
    sample_image_metrics_factory: Callable[[str, float], ImageMetrics],
) -> None:
    """--seed 引数が正しく処理されること.

    Given:
        - 有効な入力ディレクトリが存在する
        - --seed 引数が指定されている
    When:
        - CLIが --seed オプションで実行される
    Then:
        - プログラムが正常に完了すること
        - 結果が正しく表示されること
    """
    # Arrange
    seed_value = 42
    img_path = Path(test_image_directory) / "image0.jpg"
    img_path.touch()
    results = [sample_image_metrics_factory(str(img_path), 95.0)]
    stats = PickerStatistics(
        total_files=5,
        analyzed_ok=5,
        analyzed_fail=0,
        rejected_by_similarity=4,
        selected_count=1,
    )
    mock_game_screen_picker.select.return_value = (results, stats)

    monkeypatch.setattr(
        "sys.argv",
        ["main.py", test_image_directory, "-n", "5", "--seed", str(seed_value)],
    )

    # Act
    from src.main import Main

    Main().run()

    # Assert
    captured = capsys.readouterr()
    assert "選択された画像一覧" in captured.out
    assert captured.out.count("Score:") == 1
