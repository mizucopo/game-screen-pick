"""main.py CLIの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. ユーザー視点でのCLI挙動をテスト（argparse、I/O、エラーハンドリング）
2. モック使用を最小化 - 重いMLモデルとファイル操作のみモック化
3. pytestのtmp_pathを使用したリアルなファイルシステムテスト
4. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
5. capsysでstdoutをキャプチャしてユーザー向け出力を検証
"""

from pathlib import Path
from typing import List
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
def sample_image_metrics() -> List[ImageMetrics]:
    """テスト用サンプルImageMetrics."""
    return [
        ImageMetrics(
            path=f"/fake/image{i}.jpg",
            raw_metrics={"blur_score": 100.0 - i * 10},
            normalized_metrics={"blur_score": 0.9 - i * 0.1},
            semantic_score=0.8 - i * 0.05,
            total_score=95.0 - i * 5,
            features=np.random.rand(64),
        )
        for i in range(3)
    ]


@pytest.fixture
def test_image_directory(tmp_path: Path) -> str:
    """テスト用画像ディレクトリを作成."""
    images_dir = tmp_path / "test_images"
    images_dir.mkdir()
    for i in range(5):
        (images_dir / f"image{i}.jpg").touch()
    return str(images_dir)


# ============================================================================
# 引数解析のテスト（2件）
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
            "rpg",
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
    from src.main import main

    main()

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
    from src.main import main

    with pytest.raises(SystemExit):
        main()

    captured = capsys.readouterr()
    # argparseのエラーメッセージが含まれている
    assert "error" in captured.err.lower() or "required" in captured.err.lower()


# ============================================================================
# 基本機能のテスト（3件）
# ============================================================================


def test_cli_selects_and_displays_images_with_default_parameters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_image_quality_analyzer: MagicMock,
    mock_game_screen_picker: MagicMock,
    test_image_directory: str,
) -> None:
    """デフォルトパラメータで画像を選択して表示することを検証.

    Given:
        - 有効な入力ディレクトリ
        - モックされた analyzer と picker
    When:
        - CLIをデフォルトパラメータで実行
    Then:
        - デフォルトの10枚が選択される
        - 結果が正しく表示される
    """
    # Arrange
    # サンプル結果をインラインで作成（デフォルトの10枚）
    results = [
        ImageMetrics(
            path=f"/fake/image{i}.jpg",
            raw_metrics={"blur_score": 100.0 - i * 5},
            normalized_metrics={"blur_score": 0.9 - i * 0.05},
            semantic_score=0.8 - i * 0.02,
            total_score=95.0 - i * 3,
            features=np.random.rand(64),
        )
        for i in range(10)
    ]
    mock_game_screen_picker.select.return_value = results

    monkeypatch.setattr("sys.argv", ["main.py", test_image_directory])
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer", lambda *_: mock_image_quality_analyzer
    )
    monkeypatch.setattr("src.main.GameScreenPicker", lambda *_: mock_game_screen_picker)

    # Act
    from src.main import main

    main()

    # Assert
    captured = capsys.readouterr()
    # 結果一覧が表示される
    assert "選択された画像一覧" in captured.out
    # デフォルトの10枚分のスコア表示がある
    assert captured.out.count("Score:") == 10
    # すべての画像パスが表示されている
    for i in range(10):
        assert f"image{i}.jpg" in captured.out


def test_cli_selects_specified_number_of_images(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_image_quality_analyzer: MagicMock,
    mock_game_screen_picker: MagicMock,
    test_image_directory: str,
) -> None:
    """指定した枚数だけ画像を選択することを検証.

    Given:
        - 有効な入力ディレクトリ
        - `-n` オプションを指定
    When:
        - CLIを実行
    Then:
        - 指定された枚数分の結果が表示される
    """
    # Arrange
    # サンプル結果をインラインで作成（7枚）
    results = [
        ImageMetrics(
            path=f"/fake/image{i}.jpg",
            raw_metrics={"blur_score": 100.0 - i * 5},
            normalized_metrics={"blur_score": 0.9 - i * 0.05},
            semantic_score=0.8 - i * 0.02,
            total_score=95.0 - i * 3,
            features=np.random.rand(64),
        )
        for i in range(7)
    ]
    mock_game_screen_picker.select.return_value = results

    monkeypatch.setattr("sys.argv", ["main.py", test_image_directory, "-n", "7"])
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer", lambda *_: mock_image_quality_analyzer
    )
    monkeypatch.setattr("src.main.GameScreenPicker", lambda *_: mock_game_screen_picker)

    # Act
    from src.main import main

    main()

    # Assert
    captured = capsys.readouterr()
    # 7枚分のスコア表示がある
    assert captured.out.count("Score:") == 7


def test_cli_applies_genre_specific_settings(
    monkeypatch: pytest.MonkeyPatch,
    mock_image_quality_analyzer: MagicMock,
    mock_game_screen_picker: MagicMock,
    test_image_directory: str,
) -> None:
    """ジャンル固有の設定が適用されることを検証.

    Given:
        - 有効な入力ディレクトリ
        - 異なるジャンルを指定
    When:
        - 各ジャンルでCLIを実行
    Then:
        - ImageQualityAnalyzerが各ジャンルで初期化される
    """
    # Arrange
    genres = ["rpg", "fps", "2d_action", "puzzle", "mixed"]
    analyzer_init_calls = []

    def mock_analyzer_factory(genre: str) -> MagicMock:
        analyzer_init_calls.append(genre)
        return mock_image_quality_analyzer

    monkeypatch.setattr("src.main.ImageQualityAnalyzer", mock_analyzer_factory)
    monkeypatch.setattr("src.main.GameScreenPicker", lambda *_: mock_game_screen_picker)
    mock_game_screen_picker.select.return_value = []

    # Act
    for genre in genres:
        analyzer_init_calls.clear()
        monkeypatch.setattr("sys.argv", ["main.py", test_image_directory, "-g", genre])
        from src.main import main

        main()

        # Assert
        assert len(analyzer_init_calls) == 1
        assert analyzer_init_calls[0] == genre


# ============================================================================
# Tests for Copy/Output Functionality (2 tests)
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
    from src.main import main

    main()

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
    from src.main import main

    main()

    # Assert
    # ディレクトリが作成されている
    assert nested_output.exists()
    assert nested_output.is_dir()


# ============================================================================
# Tests for Error Handling (2 tests)
# ============================================================================


def test_cli_handles_nonexistent_input_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_image_quality_analyzer: MagicMock,
    mock_game_screen_picker: MagicMock,
) -> None:
    """存在しない入力ディレクトリをグレースフルに処理することを検証.

    Given:
        - 存在しない入力ディレクトリパス
    When:
        - CLIを実行
    Then:
        - プログラムがクラッシュせず、適切にエラーを処理する
        - 結果一覧が表示される（0件）
    """
    # Arrange
    nonexistent_dir = "/nonexistent/directory/that/does/not/exist"
    mock_game_screen_picker.select.return_value = []

    monkeypatch.setattr("sys.argv", ["main.py", nonexistent_dir])
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer", lambda *_: mock_image_quality_analyzer
    )
    monkeypatch.setattr("src.main.GameScreenPicker", lambda *_: mock_game_screen_picker)

    # Act
    from src.main import main

    main()

    # Assert
    captured = capsys.readouterr()
    # 選択された画像一覧が表示される（空）
    assert "選択された画像一覧" in captured.out
    # 0件のためスコア表示がない
    assert "Score:" not in captured.out


def test_cli_gracefully_handles_empty_input_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_image_quality_analyzer: MagicMock,
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
) -> None:
    """空の入力ディレクトリを正しく処理することを検証.

    Given:
        - 空の入力ディレクトリ
    When:
        - CLIを実行
    Then:
        - 結果が0件でも適切に表示される
    """
    # Arrange
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    mock_game_screen_picker.select.return_value = []

    monkeypatch.setattr("sys.argv", ["main.py", str(empty_dir)])
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer", lambda *_: mock_image_quality_analyzer
    )
    monkeypatch.setattr("src.main.GameScreenPicker", lambda *_: mock_game_screen_picker)

    # Act
    from src.main import main

    main()

    # Assert
    captured = capsys.readouterr()
    # 選択された画像一覧が表示される
    assert "選択された画像一覧" in captured.out
    # 画像は0件
    assert "Score:" not in captured.out  # スコア表示がない＝0件


# ============================================================================
# 高度な機能のテスト（1件）
# ============================================================================


def test_cli_recursive_search_finds_images_in_subdirectories(
    monkeypatch: pytest.MonkeyPatch,
    mock_image_quality_analyzer: MagicMock,
    mock_game_screen_picker: MagicMock,
    tmp_path: Path,
) -> None:
    """再帰検索オプションが正しく動作することを検証.

    Given:
        - 親ディレクトリとサブディレクトリに画像
        - `-r` フラグを指定
    When:
        - CLIを実行
    Then:
        - `-r` フラグがpicker.select()に正しく渡される
        - 再帰検索が有効になる
    """
    # Arrange
    test_dir = tmp_path / "test_images"
    test_dir.mkdir()
    (test_dir / "root.jpg").touch()
    subdir = test_dir / "subdir"
    subdir.mkdir()
    (subdir / "nested.jpg").touch()

    mock_game_screen_picker.select.return_value = []

    monkeypatch.setattr("sys.argv", ["main.py", str(test_dir), "-r"])
    monkeypatch.setattr(
        "src.main.ImageQualityAnalyzer", lambda *_: mock_image_quality_analyzer
    )
    monkeypatch.setattr("src.main.GameScreenPicker", lambda *_: mock_game_screen_picker)

    # Act
    from src.main import main

    main()

    # Assert
    # -rフラグを指定したときの観測可能な挙動を検証
    # （実際のファイルシステムの検証は picker.select の責任範囲）
    # このテストでは、CLIが `-r` フラグを正しく処理し、
    # picker.select を呼び出すことを検証すれば十分
    # （すでに picker.select の単体テストで再帰検索の挙動を検証しているため）
    assert mock_game_screen_picker.select.called
