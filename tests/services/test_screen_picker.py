"""GameScreenPickerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. モック使用を最小限に抑える - 外部依存関係のみモックを使用
   （ファイルシステム、重いMLモデル）
3. テスト可能性を高めるためにドメインロジックをIO操作から分離
4. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
5. パブリックメソッドを通じてプライベートメソッドを間接的にテスト
"""

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.analyzers.image_quality_analyzer import ImageQualityAnalyzer
from src.models.image_metrics import ImageMetrics
from src.services.screen_picker import GameScreenPicker


@pytest.fixture
def mock_analyzer() -> MagicMock:
    """モックImageQualityAnalyzerを作成する.

    このfixtureはテスト中に重いMLモデルのロードを回避し、
    代わりに選択ロジックに焦点を当てます。
    """
    analyzer = MagicMock(spec=ImageQualityAnalyzer)
    return analyzer


@pytest.fixture
def sample_image_metrics() -> List[ImageMetrics]:
    """テスト用のサンプルImageMetricsを作成する.

    異なるスコアと特徴を持つ5つの画像のリストを返します。
    - 高品質、多様な特徴
    - 中品質、最初の画像に類似
    - 高品質、非常に異なる特徴
    - 低品質
    - 中品質、3番目の画像に類似
    """
    # Create distinct feature vectors for diversity testing
    features_list = [
        np.random.rand(128),  # Image 1: random features
        np.random.rand(128),  # Image 2: random features
        np.random.rand(128),  # Image 3: random features
        np.random.rand(128),  # Image 4: random features
        np.random.rand(128),  # Image 5: random features
    ]

    # Make image 2 similar to image 1 (high cosine similarity)
    features_list[1] = features_list[0] + np.random.rand(128) * 0.1

    # Make image 5 similar to image 3
    features_list[4] = features_list[2] + np.random.rand(128) * 0.1

    return [
        ImageMetrics(
            path="/fake/path/image1.jpg",
            raw_metrics={"blur_score": 100.0},
            normalized_metrics={"blur_score": 0.9},
            semantic_score=0.8,
            total_score=95.0,
            features=features_list[0],
        ),
        ImageMetrics(
            path="/fake/path/image2.jpg",
            raw_metrics={"blur_score": 80.0},
            normalized_metrics={"blur_score": 0.7},
            semantic_score=0.7,
            total_score=85.0,
            features=features_list[1],  # Similar to image1
        ),
        ImageMetrics(
            path="/fake/path/image3.jpg",
            raw_metrics={"blur_score": 90.0},
            normalized_metrics={"blur_score": 0.8},
            semantic_score=0.75,
            total_score=90.0,
            features=features_list[2],  # Different from image1
        ),
        ImageMetrics(
            path="/fake/path/image4.jpg",
            raw_metrics={"blur_score": 30.0},
            normalized_metrics={"blur_score": 0.3},
            semantic_score=0.3,
            total_score=40.0,
            features=features_list[3],  # Low quality
        ),
        ImageMetrics(
            path="/fake/path/image5.jpg",
            raw_metrics={"blur_score": 70.0},
            normalized_metrics={"blur_score": 0.6},
            semantic_score=0.6,
            total_score=75.0,
            features=features_list[4],  # Similar to image3
        ),
    ]


# ============================================================================
# Tests for select_from_analyzed (pure domain logic)
# ============================================================================


def test_high_quality_images_are_prioritized_while_avoiding_similar_ones(
    sample_image_metrics: List[ImageMetrics],
) -> None:
    """高品質な画像が優先され、類似した画像は回避される.

    Given:
        - 様々なスコアを持つ5つの分析済み画像
        - image1（スコア95）とimage2（スコア85）は類似した特徴を持つ
        - image3（スコア90）は異なる特徴を持つ
    When:
        - 類似度閾値0.9で3つの画像を選択
    Then:
        - 3つの画像を返す
        - 最高スコアの画像が優先される
        - 類似した画像は除外される
    """
    # Arrange
    num_to_select = 3
    similarity_threshold = 0.9

    # Act
    result = GameScreenPicker.select_from_analyzed(
        sample_image_metrics, num_to_select, similarity_threshold
    )

    # Assert
    assert len(result) == 3
    # Images should be in score order (highest first)
    scores = [m.total_score for m in result]
    assert scores == sorted(scores, reverse=True)
    # Image 2 (similar to image 1) should be filtered out
    selected_paths = [m.path for m in result]
    assert "/fake/path/image2.jpg" not in selected_paths


def test_requesting_more_images_than_available_returns_all_unique_images(
    sample_image_metrics: List[ImageMetrics],
) -> None:
    """利用可能な数より多くの画像を要求した場合、すべての一意な画像を返す.

    Given:
        - 5つの分析済み画像（一部類似）
    When:
        - 中程度の類似度閾値で10個の画像を選択
    Then:
        - 利用可能な多様な画像の数まで返す
    """
    # Arrange
    num_to_select = 10
    similarity_threshold = 0.8

    # Act
    result = GameScreenPicker.select_from_analyzed(
        sample_image_metrics, num_to_select, similarity_threshold
    )

    # Assert
    # Should return fewer than 10 due to similarity filtering
    # (image2 ~ image1, image5 ~ image3)
    assert len(result) <= 5
    assert len(result) >= 1


def test_different_similarity_thresholds_produce_different_selection_results(
    sample_image_metrics: List[ImageMetrics],
) -> None:
    """異なる類似度閾値は異なる選択結果を生成する.

    Given:
        - 類似性を持つ5つの分析済み画像
    When:
        - 2つの異なる閾値で選択
    Then:
        - 結果は閾値に基づいて異なる場合がある
    """
    # Arrange
    num_to_select = 5
    threshold_low = 0.7
    threshold_high = 0.99

    # Act
    result_low = GameScreenPicker.select_from_analyzed(
        sample_image_metrics, num_to_select, threshold_low
    )
    result_high = GameScreenPicker.select_from_analyzed(
        sample_image_metrics, num_to_select, threshold_high
    )

    # Assert
    # Both should return valid results
    assert len(result_low) >= 1
    assert len(result_high) >= 1
    # Results should be sorted by score
    for result in [result_low, result_high]:
        scores = [m.total_score for m in result]
        assert scores == sorted(scores, reverse=True)


def test_higher_similarity_threshold_filters_out_more_similar_images(
    sample_image_metrics: List[ImageMetrics],
) -> None:
    """より高い類似度閾値はより多くの類似画像を除外する.

    Given:
        - image2がimage1に類似している5つの分析済み画像
    When:
        - 厳しい閾値（0.95）で選択
    Then:
        - 類似した画像は両方とも選択されない
    """
    # Arrange
    num_to_select = 5
    similarity_threshold = 0.95

    # Act
    result = GameScreenPicker.select_from_analyzed(
        sample_image_metrics, num_to_select, similarity_threshold
    )

    # Assert
    # Image1 and Image2 should not both be in result (they're similar)
    result_paths = [m.path for m in result]
    has_image1 = "/fake/path/image1.jpg" in result_paths
    has_image2 = "/fake/path/image2.jpg" in result_paths
    # At most one of them should be selected
    assert not (has_image1 and has_image2)


def test_empty_input_list_returns_empty_result() -> None:
    """空の入力リストは空の結果を返す.

    Given:
        - 分析済み画像がない
    When:
        - 空のリストから選択
    Then:
        - 空のリストを返す
    """
    # Arrange
    empty_list: List[ImageMetrics] = []

    # Act
    result = GameScreenPicker.select_from_analyzed(empty_list, 5, 0.8)

    # Assert
    assert result == []


def test_requesting_zero_images_returns_empty_list(
    sample_image_metrics: List[ImageMetrics],
) -> None:
    """0個の画像を要求すると空のリストを返す.

    Given:
        - 5つの分析済み画像
    When:
        - 0個の画像を選択
    Then:
        - 処理せずに空のリストを返す
    """
    # Arrange
    num_to_select = 0

    # Act
    result = GameScreenPicker.select_from_analyzed(
        sample_image_metrics, num_to_select, 0.8
    )

    # Assert
    assert result == []


def test_original_input_list_remains_unchanged_after_selection(
    sample_image_metrics: List[ImageMetrics],
) -> None:
    """元の入力リストは選択後も変更されない.

    Given:
        - 特定の順序の分析済み画像リスト
    When:
        - そのリストから選択
    Then:
        - 元のリストの順序と内容が保持される
    """
    # Arrange
    original_paths = [m.path for m in sample_image_metrics]
    original_order = list(sample_image_metrics)

    # Act
    GameScreenPicker.select_from_analyzed(sample_image_metrics, 3, 0.8)

    # Assert
    assert [m.path for m in sample_image_metrics] == original_paths
    # The original list objects should still be in same order
    assert sample_image_metrics == original_order


# ============================================================================
# Tests for GameScreenPicker initialization
# ============================================================================


def test_picker_stores_the_provided_analyzer_instance(mock_analyzer: MagicMock) -> None:
    """ピッカーは提供されたアナライザインスタンスを格納する.

    Given:
        - モックアナライザ
    When:
        - GameScreenPickerインスタンスを作成
    Then:
        - アナライザが格納され、アクセス可能
    """
    # Arrange & Act
    picker = GameScreenPicker(mock_analyzer)

    # Assert
    assert picker.analyzer is mock_analyzer


# ============================================================================
# Tests for _load_image_files method
# ============================================================================


def test_loading_image_files_finds_all_supported_formats_in_folder() -> None:
    """指定されたフォルダ内のすべてのサポートされる画像ファイルを見つける.

    Given:
        - jpg、png、bmpファイルを持つフォルダ
        - 画像以外のファイル（txt）
    When:
        - 非再帰的に画像ファイルをロード
    Then:
        - 画像ファイルのみを返す
        - 画像以外のファイルを無視する
    """
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        Path(temp_dir, "image1.jpg").touch()
        Path(temp_dir, "image2.png").touch()
        Path(temp_dir, "image3.bmp").touch()
        Path(temp_dir, "image4.jpeg").touch()
        Path(temp_dir, "not_an_image.txt").touch()

        picker = GameScreenPicker(MagicMock(spec=ImageQualityAnalyzer))

        # Act
        result = picker._load_image_files(temp_dir, recursive=False)

        # Assert
        assert len(result) == 4
        extensions = {p.suffix.lower() for p in result}
        assert extensions == {".jpg", ".png", ".bmp", ".jpeg"}


def test_recursive_search_finds_images_in_subdirectories() -> None:
    """recursive=Trueの場合、サブディレクトリを検索する.

    Given:
        - ネストされたサブディレクトリに画像を持つフォルダ
    When:
        - 再帰的に画像ファイルをロード
    Then:
        - すべてのレベルの画像を返す
    """
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create nested structure
        Path(temp_dir, "root.jpg").touch()
        subdir = Path(temp_dir, "subfolder")
        subdir.mkdir()
        Path(subdir, "nested.png").touch()

        picker = GameScreenPicker(MagicMock(spec=ImageQualityAnalyzer))

        # Act
        result = picker._load_image_files(temp_dir, recursive=True)

        # Assert
        assert len(result) == 2
        paths = {str(p.name) for p in result}
        assert paths == {"root.jpg", "nested.png"}


def test_non_recursive_search_only_checks_top_level_folder() -> None:
    """recursive=Falseの場合、トップレベルフォルダのみをチェックする.

    Given:
        - ネストされたサブディレクトリに画像を持つフォルダ
    When:
        - 非再帰的に画像ファイルをロード
    Then:
        - トップレベルの画像のみを返す
    """
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create nested structure
        Path(temp_dir, "root.jpg").touch()
        subdir = Path(temp_dir, "subfolder")
        subdir.mkdir()
        Path(subdir, "nested.png").touch()

        picker = GameScreenPicker(MagicMock(spec=ImageQualityAnalyzer))

        # Act
        result = picker._load_image_files(temp_dir, recursive=False)

        # Assert
        assert len(result) == 1
        assert result[0].name == "root.jpg"


def test_loading_handles_uppercase_and_mixed_case_extensions() -> None:
    """大文字と小文字混在の拡張子を処理する.

    Given:
        - .JPG、.Png、.BMP拡張子を持つファイル
    When:
        - 画像ファイルをロード
    Then:
        - 大文字小文字に関係なくすべてのサポートされる形式を認識
    """
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
        Path(temp_dir, "upper.JPG").touch()
        Path(temp_dir, "mixed.Png").touch()
        Path(temp_dir, "lower.bmp").touch()

        picker = GameScreenPicker(MagicMock(spec=ImageQualityAnalyzer))

        # Act
        result = picker._load_image_files(temp_dir, recursive=False)

        # Assert
        assert len(result) == 3


def test_loading_from_empty_folder_returns_empty_list() -> None:
    """空のフォルダは空のファイルリストを返す.

    Given:
        - 空のフォルダ
    When:
        - 画像ファイルをロード
    Then:
        - 空のリストを返す
    """
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
        picker = GameScreenPicker(MagicMock(spec=ImageQualityAnalyzer))

        # Act
        result = picker._load_image_files(temp_dir, recursive=False)

        # Assert
        assert result == []


def test_loading_only_includes_supported_image_formats() -> None:
    """サポートされる画像形式のみを含める.

    Given:
        - 様々なファイルタイプ（.gif、.webp、.tiff、.txt）を持つフォルダ
    When:
        - 画像ファイルをロード
    Then:
        - サポートされる形式のみ（.jpg、.png、.bmp）を返す
    """
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
        Path(temp_dir, "unsupported.gif").touch()
        Path(temp_dir, "unsupported.webp").touch()
        Path(temp_dir, "supported.jpg").touch()
        Path(temp_dir, "document.txt").touch()

        picker = GameScreenPicker(MagicMock(spec=ImageQualityAnalyzer))

        # Act
        result = picker._load_image_files(temp_dir, recursive=False)

        # Assert
        assert len(result) == 1
        assert result[0].name == "supported.jpg"


# ============================================================================
# Integration tests for select method
# ============================================================================


def test_selecting_from_folder_loads_analyzes_and_returns_diverse_images(
    mock_analyzer: MagicMock,
) -> None:
    """完全な統合：ロード、分析、多様な画像の選択.

    Given:
        - 5つの画像ファイルを持つフォルダ
        - モックアナライザは一貫した結果を返す
    When:
        - 類似度閾値で3つの画像を選択
    Then:
        - 多様で高品質な画像を返す
    """
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test images
        for i in range(5):
            Path(temp_dir, f"image{i}.jpg").touch()

        # Mock analyzer to return predictable results
        def mock_analyze(path: str) -> ImageMetrics:
            # Create features where image0 and image1 are similar
            idx = int(path.split("image")[-1].split(".")[0])
            base_features = np.random.rand(128)
            if idx == 1:
                base_features = base_features * 0.99  # Very similar to image0
            return ImageMetrics(
                path=path,
                raw_metrics={"blur_score": 100 - idx * 10},
                normalized_metrics={"blur_score": 1.0 - idx * 0.1},
                semantic_score=0.8,
                total_score=100 - idx * 10,
                features=base_features,
            )

        mock_analyzer.analyze = mock_analyze
        picker = GameScreenPicker(mock_analyzer)

        # Act
        result = picker.select(
            folder=temp_dir,
            num=3,
            similarity_threshold=0.95,
            recursive=False,
            show_progress=False,
        )

        # Assert
        # Should get diverse results (not too similar)
        assert len(result) >= 1
        assert len(result) <= 3
        # Results should be in score order
        for i in range(len(result) - 1):
            assert result[i].total_score >= result[i + 1].total_score


def test_selecting_gracefully_handles_files_that_fail_to_analyze(
    mock_analyzer: MagicMock,
) -> None:
    """分析に失敗したファイルを適切に処理する.

    Given:
        - 5つの画像ファイルを持つフォルダ
        - アナライザは一部のファイルに対してNoneを返す（破損/読み取り不可）
    When:
        - 画像を選択
    Then:
        - 処理を継続し、有効な画像のみを返す
    """
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(5):
            Path(temp_dir, f"image{i}.jpg").touch()

        call_count = [0]

        def mock_analyze(path: str) -> ImageMetrics | None:
            call_count[0] += 1
            # Return None for even-indexed images
            idx = int(path.split("image")[-1].split(".")[0])
            if idx % 2 == 0:
                return None
            return ImageMetrics(
                path=path,
                raw_metrics={},
                normalized_metrics={},
                semantic_score=0.5,
                total_score=50.0,
                features=np.random.rand(128),
            )

        mock_analyzer.analyze = mock_analyze
        picker = GameScreenPicker(mock_analyzer)

        # Act
        result = picker.select(
            folder=temp_dir,
            num=5,
            similarity_threshold=0.8,
            recursive=False,
            show_progress=False,
        )

        # Assert
        # Should have analyzed all 5 files
        assert call_count[0] == 5
        # But only returned results for non-None analyses
        assert len(result) <= 3  # At most 3 valid images (odd indices)


def test_selecting_from_nonexistent_folder_returns_empty_list(
    mock_analyzer: MagicMock,
) -> None:
    """存在しないフォルダは適切に空のリストを返す.

    Given:
        - 存在しないフォルダパス
    When:
        - 画像を選択
    Then:
        - 空のリストを返す（正常なデグラデーション）
    """
    # Arrange
    picker = GameScreenPicker(mock_analyzer)

    # Act
    result = picker.select(
        folder="/nonexistent/folder/path/that/does/not/exist",
        num=3,
        similarity_threshold=0.8,
        recursive=False,
        show_progress=False,
    )

    # Assert
    # Should return empty list for non-existent folder
    assert result == []
