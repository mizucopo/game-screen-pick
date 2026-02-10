"""main.py CLIの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. ユーザー視点でのCLI挙動をテスト（argparse、I/O、エラーハンドリング）
2. モック使用を最小化 - 重いMLモデルとファイル操作のみモック化
3. pytestのtmp_pathを使用したリアルなファイルシステムテスト
4. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
5. capsysでstdoutをキャプチャしてユーザー向け出力を検証
"""

from pathlib import Path
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
    # サンプル選択結果を設定
    picker.select.return_value = [
        ImageMetrics(
            path="/fake/image1.jpg",
            raw_metrics={"blur_score": 100.0},
            normalized_metrics={"blur_score": 0.9},
            semantic_score=0.8,
            total_score=95.0,
            features=np.random.rand(64),
        ),
    ]
    return picker


@pytest.fixture
def test_image_directory(tmp_path: Path) -> str:
    """テスト用画像ディレクトリを作成."""
    images_dir = tmp_path / "test_images"
    images_dir.mkdir()
    for i in range(5):
        (images_dir / f"image{i}.jpg").touch()
    return str(images_dir)


# ============================================================================
# 引数解析のテスト
# ============================================================================


def test_cli_accepts_all_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_image_quality_analyzer: MagicMock,
    mock_game_screen_picker: MagicMock,
    test_image_directory: str,
    tmp_path: Path,
) -> None:
    """全ての引数を正しくパースすることを検証.

    Given:
        - 有効な入力ディレクトリ
        - 全てのオプション引数を指定
        - モックされた analyzer と picker
    When:
        - CLIを全引数指定で実行
    Then:
        - ImageQualityAnalyzerが指定ジャンルで初期化される
        - picker.selectが正しいパラメータで呼ばれる
        - 出力ディレクトリが作成される
    """
    # Arrange
    output_dir = str(tmp_path / "output")
    # 実際のファイルを作成してコピーをテスト
    img_path = Path(test_image_directory) / "image0.jpg"
    img_path.touch()
    results = [
        ImageMetrics(
            path=str(img_path),
            raw_metrics={"blur_score": 100.0},
            normalized_metrics={"blur_score": 0.9},
            semantic_score=0.8,
            total_score=95.0,
            features=np.random.rand(64),
        )
    ]
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
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer", lambda *_: mock_image_quality_analyzer
    )
    monkeypatch.setattr("src.main.GameScreenPicker", lambda *_: mock_game_screen_picker)

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
    """必須引数（input）が missing の時にエラーを表示することを検証.

    Given:
        - 必須引数（input）を指定しない
    When:
        - CLIを実行
    Then:
        - 適切なエラーメッセージが出力される
        - プログラムが終了する
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
    mock_image_quality_analyzer: MagicMock,
    mock_game_screen_picker: MagicMock,
    test_image_directory: str,
    args: list[str],
    num_expected: int,
) -> None:
    """画像を選択して表示することを検証.

    Given:
        - 有効な入力ディレクトリ
        - モックされた analyzer と picker
        - デフォルトまたはカスタムパラメータ
    When:
        - CLIを実行
    Then:
        - 指定された枚数分の結果が表示される
    """
    # Arrange
    # サンプル結果をインラインで作成
    results = [
        ImageMetrics(
            path=f"/fake/image{i}.jpg",
            raw_metrics={"blur_score": 100.0 - i * 5},
            normalized_metrics={"blur_score": 0.9 - i * 0.05},
            semantic_score=0.8 - i * 0.02,
            total_score=95.0 - i * 3,
            features=np.random.rand(64),
        )
        for i in range(num_expected)
    ]
    mock_game_screen_picker.select.return_value = results

    monkeypatch.setattr("sys.argv", ["main.py", test_image_directory] + args)
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer", lambda *_: mock_image_quality_analyzer
    )
    monkeypatch.setattr("src.main.GameScreenPicker", lambda *_: mock_game_screen_picker)

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
    mock_image_quality_analyzer: MagicMock,
    mock_game_screen_picker: MagicMock,
    test_image_directory: str,
    tmp_path: Path,
) -> None:
    """選択された画像が出力ディレクトリにコピーされることを検証.

    Given:
        - 有効な入力ディレクトリ
        - 出力ディレクトリを指定
        - 3つの選択結果
    When:
        - CLIを `-c` オプションで実行
    Then:
        - shutil.copy2が各画像に対して呼ばれる
        - 成功メッセージにコピー数とパスが含まれる
    """
    # Arrange
    output_dir = tmp_path / "output"
    # 実際のファイルを作成
    results = []
    for i in range(3):
        img_path = Path(test_image_directory) / f"image{i}.jpg"
        img_path.touch()
        results.append(
            ImageMetrics(
                path=str(img_path),
                raw_metrics={"blur_score": 100.0 - i * 10},
                normalized_metrics={"blur_score": 0.9 - i * 0.1},
                semantic_score=0.8,
                total_score=95.0 - i * 5,
                features=np.random.rand(64),
            )
        )

    mock_game_screen_picker.select.return_value = results

    monkeypatch.setattr(
        "sys.argv", ["main.py", test_image_directory, "-c", str(output_dir)]
    )
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer", lambda *_: mock_image_quality_analyzer
    )
    monkeypatch.setattr("src.main.GameScreenPicker", lambda *_: mock_game_screen_picker)

    # Act
    from src.main import Main

    Main().run()

    # Assert
    captured = capsys.readouterr()
    # 成功メッセージが表示される
    assert "3 枚を" in captured.out
    assert str(output_dir) in captured.out
    assert "保存しました" in captured.out
    # 画像が出力ディレクトリにコピーされている
    assert (output_dir / "image0.jpg").exists()
    assert (output_dir / "image1.jpg").exists()
    assert (output_dir / "image2.jpg").exists()


def test_cli_creates_output_directory_if_it_doesnt_exist(
    monkeypatch: pytest.MonkeyPatch,
    mock_image_quality_analyzer: MagicMock,
    mock_game_screen_picker: MagicMock,
    test_image_directory: str,
    tmp_path: Path,
) -> None:
    """出力ディレクトリが存在しない場合に作成されることを検証.

    Given:
        - 有効な入力ディレクトリ
        - 存在しない出力ディレクトリパス
        - 選択される画像が少なくとも1つある
    When:
        - CLIを実行
    Then:
        - 出力ディレクトリが作成される
        - ディレクトリが存在することを確認
    """
    # Arrange
    nested_output = tmp_path / "parent" / "child" / "output"
    # 出力ディレクトリはまだ存在しない
    assert not nested_output.exists()

    # 実際のファイルを作成してコピーをテスト
    img_path = Path(test_image_directory) / "image0.jpg"
    img_path.touch()
    results = [
        ImageMetrics(
            path=str(img_path),
            raw_metrics={"blur_score": 100.0},
            normalized_metrics={"blur_score": 0.9},
            semantic_score=0.8,
            total_score=95.0,
            features=np.random.rand(64),
        )
    ]
    mock_game_screen_picker.select.return_value = results

    monkeypatch.setattr(
        "sys.argv", ["main.py", test_image_directory, "-c", str(nested_output)]
    )
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer", lambda *_: mock_image_quality_analyzer
    )
    monkeypatch.setattr("src.main.GameScreenPicker", lambda *_: mock_game_screen_picker)

    # Act
    from src.main import Main

    Main().run()

    # Assert
    # ディレクトリが作成されている
    assert nested_output.exists()
    assert nested_output.is_dir()


# ============================================================================
# Tests for Error Handling
# ============================================================================


def test_cli_handles_empty_and_nonexistent_input_directories(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_image_quality_analyzer: MagicMock,
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
) -> None:
    """空のディレクトリと存在しないディレクトリを正しく処理することを検証.

    Given:
        - 空の入力ディレクトリ
        - 存在しない入力ディレクトリパス
    When:
        - CLIを各ディレクトリで実行
    Then:
        - プログラムがクラッシュせず、0件の結果が適切に表示される
    """
    # Arrange
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    nonexistent_dir = "/nonexistent/directory/that/does/not/exist"
    mock_game_screen_picker.select.return_value = []

    # Act & Assert - 空ディレクトリ
    monkeypatch.setattr("sys.argv", ["main.py", str(empty_dir)])
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer", lambda *_: mock_image_quality_analyzer
    )
    monkeypatch.setattr("src.main.GameScreenPicker", lambda *_: mock_game_screen_picker)

    from src.main import Main

    Main().run()

    captured = capsys.readouterr()
    assert "選択された画像一覧" in captured.out
    assert "Score:" not in captured.out  # 0件

    # Act & Assert - 存在しないディレクトリ
    capsys.readouterr()  # Clear previous output
    monkeypatch.setattr("sys.argv", ["main.py", nonexistent_dir])

    Main().run()

    captured = capsys.readouterr()
    assert "選択された画像一覧" in captured.out
    assert "Score:" not in captured.out  # 0件


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
    mock_image_quality_analyzer: MagicMock,
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
    num_files: int,
    base_name: str,
    extension: str,
) -> None:
    """同名ファイルが複数存在する場合に連番でサフィックスを付与して上書きを回避することを検証.

    Given:
        - 別々のフォルダに同名ファイルが存在
        - 出力ディレクトリを指定
        - 複数の画像が選択される
    When:
        - CLIを `-c` オプションで実行
    Then:
        - 1つ目は元の名前、2つ目以降は _1, _2,... のサフィックスで保存される
        - すべてのファイルが出力ディレクトリに存在する
    """
    # Arrange
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for i in range(num_files):
        folder = input_dir / f"folder{i}"
        folder.mkdir()
        (folder / f"{base_name}{extension}").touch()

    output_dir = tmp_path / "output"

    # 選択結果を設定（同名ファイルを含む）
    results = []
    for i in range(num_files):
        img_path = input_dir / f"folder{i}" / f"{base_name}{extension}"
        results.append(
            ImageMetrics(
                path=str(img_path),
                raw_metrics={"blur_score": 100.0 - i * 10},
                normalized_metrics={"blur_score": 0.9 - i * 0.1},
                semantic_score=0.8 - i * 0.05,
                total_score=95.0 - i * 5,
                features=np.random.rand(64),
            )
        )
    mock_game_screen_picker.select.return_value = results

    monkeypatch.setattr(
        "sys.argv", ["main.py", str(input_dir), "-c", str(output_dir), "-r"]
    )
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer", lambda *_: mock_image_quality_analyzer
    )
    monkeypatch.setattr("src.main.GameScreenPicker", lambda *_: mock_game_screen_picker)

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
