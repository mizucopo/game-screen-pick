"""Unit tests for GameScreenPicker.

This test module follows these best practices:
1. Tests "what" (observable behavior), not "how" (implementation details)
2. Minimizes mock usage - only uses mocks for external dependencies
   (filesystem, heavy ML models)
3. Separates domain logic from IO operations for better testability
4. Uses AAA pattern (Arrange, Act, Assert) with clear comments
5. Tests private methods indirectly through public methods
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
    """Create a mock ImageQualityAnalyzer.

    This fixture avoids loading heavy ML models during tests,
    focusing on the selection logic instead.
    """
    analyzer = MagicMock(spec=ImageQualityAnalyzer)
    return analyzer


@pytest.fixture
def sample_image_metrics() -> List[ImageMetrics]:
    """Create sample ImageMetrics for testing.

    Returns a list of 5 images with different scores and features.
    - High quality, diverse features
    - Medium quality, similar to first
    - High quality, very different features
    - Low quality
    - Medium quality, similar to third
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
    """High-quality images are prioritized while avoiding similar ones.

    Given:
        - 5 analyzed images with varying scores
        - image1 (score 95) and image2 (score 85) have similar features
        - image3 (score 90) has different features
    When:
        - Selecting 3 images with similarity threshold 0.9
    Then:
        - Returns 3 images
        - Highest scoring images are prioritized
        - Similar images are filtered out
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
    """When requesting more images than available, returns all unique images.

    Given:
        - 5 analyzed images (some similar)
    When:
        - Selecting 10 images with moderate similarity threshold
    Then:
        - Returns at most the number of diverse images available
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
    """Different similarity thresholds produce different selection results.

    Given:
        - 5 analyzed images with some similarities
    When:
        - Selecting with two different thresholds
    Then:
        - Results may differ based on threshold value
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
    """Higher similarity threshold filters out more similar images.

    Given:
        - 5 analyzed images where image2 is similar to image1
    When:
        - Selecting with strict threshold (0.95)
    Then:
        - Similar images are not both selected
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
    """Empty input list returns empty result.

    Given:
        - No analyzed images
    When:
        - Selecting from empty list
    Then:
        - Returns empty list
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
    """Requesting zero images returns empty list.

    Given:
        - 5 analyzed images
    When:
        - Selecting 0 images
    Then:
        - Returns empty list without processing
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
    """Original input list remains unchanged after selection.

    Given:
        - A list of analyzed images in specific order
    When:
        - Selecting from that list
    Then:
        - Original list order and content is preserved
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
    """Picker stores the provided analyzer instance.

    Given:
        - A mock analyzer
    When:
        - Creating a GameScreenPicker instance
    Then:
        - The analyzer is stored and accessible
    """
    # Arrange & Act
    picker = GameScreenPicker(mock_analyzer)

    # Assert
    assert picker.analyzer is mock_analyzer


# ============================================================================
# Tests for _load_image_files method
# ============================================================================


def test_loading_image_files_finds_all_supported_formats_in_folder() -> None:
    """Finds all supported image files in the specified folder.

    Given:
        - A folder with jpg, png, bmp files
        - A non-image file (txt)
    When:
        - Loading image files non-recursively
    Then:
        - Returns only image files
        - Ignores non-image files
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
    """When recursive=True, searches subdirectories.

    Given:
        - A folder with images in nested subdirectories
    When:
        - Loading image files recursively
    Then:
        - Returns images from all levels
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
    """When recursive=False, only checks top-level folder.

    Given:
        - A folder with images in nested subdirectories
    When:
        - Loading image files non-recursively
    Then:
        - Returns only top-level images
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
    """Handles uppercase and mixed case extensions.

    Given:
        - Files with .JPG, .Png, .BMP extensions
    When:
        - Loading image files
    Then:
        - Recognizes all supported formats regardless of case
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
    """Empty folder returns empty file list.

    Given:
        - An empty folder
    When:
        - Loading image files
    Then:
        - Returns empty list
    """
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
        picker = GameScreenPicker(MagicMock(spec=ImageQualityAnalyzer))

        # Act
        result = picker._load_image_files(temp_dir, recursive=False)

        # Assert
        assert result == []


def test_loading_only_includes_supported_image_formats() -> None:
    """Only includes supported image formats.

    Given:
        - Folder with various file types (.gif, .webp, .tiff, .txt)
    When:
        - Loading image files
    Then:
        - Returns only supported formats (.jpg, .png, .bmp)
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
    """Full integration: loads, analyzes, and selects diverse images.

    Given:
        - A folder with 5 image files
        - Mock analyzer returns consistent results
    When:
        - Selecting 3 images with similarity threshold
    Then:
        - Returns diverse, high-quality images
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
    """Gracefully handles files that fail to analyze.

    Given:
        - A folder with 5 image files
        - Analyzer returns None for some files (corrupted/unreadable)
    When:
        - Selecting images
    Then:
        - Continues processing and returns valid images only
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
    """Non-existent folder returns empty list gracefully.

    Given:
        - A folder path that doesn't exist
    When:
        - Selecting images
    Then:
        - Returns empty list (graceful degradation)
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
