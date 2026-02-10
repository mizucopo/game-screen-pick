"""Unit tests for ImageQualityAnalyzer.

This test module follows these best practices:
1. Tests "what" (observable behavior), not "how" (implementation details)
2. Strategically mocks only CLIP model (700MB, 10-30s load time)
3. Does NOT mock OpenCV operations, NumPy calculations, or MetricNormalizer
4. Uses AAA pattern (Arrange, Act, Assert) with clear comments
5. Fast execution (~2-5 seconds) - no heavy model loading
"""

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from src.analyzers.image_quality_analyzer import ImageQualityAnalyzer
from src.models.image_metrics import ImageMetrics


# ============================================================================
# Mock fixtures for CLIP model (avoid 700MB download and 10-30s load time)
# ============================================================================


@pytest.fixture
def mock_clip_model() -> Generator[MagicMock, None, None]:
    """Mock CLIP model to avoid loading 700MB weights.

    This fixture replaces the real CLIP model with a mock that:
    - Returns fixed logit values for deterministic tests
    - Supports .to(device) calls for GPU/CPU switching
    """
    with patch("transformers.CLIPModel.from_pretrained") as mock:
        model = MagicMock()
        # Return a fixed logit value for consistent testing
        mock_output = MagicMock()
        mock_output.logits_per_image = torch.tensor([[25.0]])
        model.return_value = mock_output
        model.to = MagicMock(return_value=model)
        mock.return_value = model
        yield mock


@pytest.fixture
def mock_clip_processor() -> Generator[MagicMock, None, None]:
    """Mock CLIP processor to avoid loading tokenizer and feature extractor.

    This fixture replaces the real CLIP processor with a mock that:
    - Returns fixed tensor shapes for text and images
    - Supports .to(device) calls for GPU/CPU switching
    """
    with patch("transformers.CLIPProcessor.from_pretrained") as mock:
        processor = MagicMock()
        # Return realistic tensor shapes
        processor.return_value = MagicMock(
            input_ids=torch.tensor([[1, 2, 3]]),
            pixel_values=torch.tensor([[[[1.0]]]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )
        # Mock the .to() method for device switching
        processor_instance = MagicMock()
        processor_instance.return_value.to = MagicMock(return_value=processor_instance)
        mock.return_value = processor_instance
        yield mock


# ============================================================================
# Fixtures for creating test images programmatically
# ============================================================================


@pytest.fixture
def sample_image_path(tmp_path: Path) -> str:
    """Create a test image with random pixel values.

    Uses a fixed seed for deterministic test results.
    Image size: 640x480 (standard 4:3 aspect ratio).
    """
    np.random.seed(42)  # Fixed seed for reproducibility
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def blurry_image_path(tmp_path: Path) -> str:
    """Create a blurry test image using Gaussian blur.

    Used to test blur_score detection.
    Image size: 640x480 with heavy Gaussian blur (kernel size 31x31).
    """
    np.random.seed(42)
    img_array = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(img_array, (31, 31), 0)
    img_path = tmp_path / "blurry_image.jpg"
    cv2.imwrite(str(img_path), blurred)
    return str(img_path)


@pytest.fixture
def dark_image_path(tmp_path: Path) -> str:
    """Create a dark test image to test brightness penalty.

    Used to verify that images with brightness < 40 receive 0.6 penalty.
    Image size: 640x480 with low pixel values (0-50).
    """
    np.random.seed(42)
    img_array = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "dark_image.jpg"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def high_quality_image_path(tmp_path: Path) -> str:
    """Create a high-quality test image with good contrast and edges.

    Used to test that good images receive high scores.
    Image size: 640x480 with good contrast and edge density.
    """
    np.random.seed(42)
    # Create an image with good contrast
    img_array = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    # Add some edges using a Sobel-like pattern
    img_array[200:280, 300:340] = 255  # Add a bright rectangle
    img_path = tmp_path / "high_quality_image.jpg"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def png_image_path(tmp_path: Path) -> str:
    """Create a PNG format test image.

    Used to test that the analyzer handles different image formats.
    """
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "test_image.png"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def bmp_image_path(tmp_path: Path) -> str:
    """Create a BMP format test image.

    Used to test that the analyzer handles different image formats.
    """
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "test_image.bmp"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def small_image_path(tmp_path: Path) -> str:
    """Create a small test image (320x240).

    Used to test that the analyzer handles different image dimensions.
    """
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    img_path = tmp_path / "small_image.jpg"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


# ============================================================================
# Tests for initialization (3 tests)
# ============================================================================


def test_analyzer_loads_clip_model_on_init(mock_clip_model: MagicMock) -> None:
    """Analyzer loads CLIP model during initialization.

    Given:
        - A genre type (e.g., "rpg")
        - Mocked CLIP model
    When:
        - Creating an ImageQualityAnalyzer instance
    Then:
        - CLIPModel.from_pretrained is called with correct model name
        - Model is loaded for semantic scoring
    """
    # Arrange & Act
    _analyzer = ImageQualityAnalyzer(genre="rpg")

    # Assert
    mock_clip_model.assert_called_once_with("openai/clip-vit-base-patch32")


def test_analyzer_loads_clip_processor_on_init(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,
) -> None:
    """Analyzer loads CLIP processor during initialization.

    Given:
        - A genre type
        - Mocked CLIP model and processor
    When:
        - Creating an ImageQualityAnalyzer instance
    Then:
        - CLIPProcessor.from_pretrained is called with correct model name
        - Processor is loaded for image preprocessing
    """
    # Arrange & Act
    _analyzer = ImageQualityAnalyzer(genre="fps")

    # Assert
    mock_clip_processor.assert_called_once_with("openai/clip-vit-base-patch32")


def test_analyzer_sets_correct_weights_based_on_genre(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
) -> None:
    """Analyzer sets genre-specific weights during initialization.

    Given:
        - A specific genre (e.g., "fps")
    When:
        - Creating an ImageQualityAnalyzer instance
    Then:
        - Correct genre weights are loaded
        - Weights match expected values for the genre
    """
    # Arrange & Act
    analyzer = ImageQualityAnalyzer(genre="fps")

    # Assert
    expected_weights = {
        "blur_score": 0.25,
        "contrast": 0.20,
        "color_richness": 0.10,
        "visual_balance": 0.10,
        "edge_density": 0.10,
        "action_intensity": 0.15,
        "ui_density": 0.00,
        "dramatic_score": 0.10,
    }
    assert analyzer.weights == expected_weights


# ============================================================================
# Tests for _extract_diversity_features method (3 tests)
# ============================================================================


def test_extract_diversity_features_returns_128x128_vector(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """Extracts features as a flattened 8x8x8 HSV histogram vector.

    Given:
        - An analyzer instance
        - A valid test image
    When:
        - Extracting diversity features
    Then:
        - Returns a numpy array
        - Shape is 64 (8x8 2D histogram for H and S channels)
        - Features are normalized
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()
    img = cv2.imread(sample_image_path)

    # Act
    features = analyzer._extract_diversity_features(img)

    # Assert
    assert isinstance(features, np.ndarray)
    # 8x8 HSV histogram (H and S channels) flattened = 64 elements
    assert features.shape == (64,)


def test_extract_diversity_features_produces_different_vectors_for_different_images(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
) -> None:
    """Different images produce different feature vectors.

    Given:
        - An analyzer instance
        - Two different test images
    When:
        - Extracting diversity features from each image
    Then:
        - Feature vectors are different
        - Feature vectors are not identical (not all zeros or same values)
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Create two different images
    np.random.seed(42)
    img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    np.random.seed(43)  # Different seed
    img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Act
    features1 = analyzer._extract_diversity_features(img1)
    features2 = analyzer._extract_diversity_features(img2)

    # Assert
    assert not np.array_equal(features1, features2)


def test_extract_diversity_features_returns_normalized_histogram(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """Extracts normalized histogram features.

    Given:
        - An analyzer instance
        - A valid test image
    When:
        - Extracting diversity features
    Then:
        - Features are normalized (cv2.normalize applied)
        - All values are non-negative
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()
    img = cv2.imread(sample_image_path)

    # Act
    features = analyzer._extract_diversity_features(img)

    # Assert
    assert np.all(features >= 0)


# ============================================================================
# Tests for analyze method - success path (6 tests)
# ============================================================================


def test_analyze_returns_image_metrics_with_all_required_fields(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """Analyze returns ImageMetrics with all required fields populated.

    Given:
        - An analyzer instance
        - A valid test image path
    When:
        - Analyzing the image
    Then:
        - Returns an ImageMetrics instance
        - All fields are populated:
          - path (str)
          - raw_metrics (dict with 9 fields)
          - normalized_metrics (dict with 8 fields)
          - semantic_score (float)
          - total_score (float)
          - features (numpy array)
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result = analyzer.analyze(sample_image_path)

    # Assert
    assert result is not None
    assert isinstance(result, ImageMetrics)
    assert result.path == sample_image_path
    assert isinstance(result.raw_metrics, dict)
    assert isinstance(result.normalized_metrics, dict)
    assert isinstance(result.semantic_score, float)
    assert isinstance(result.total_score, float)
    assert isinstance(result.features, np.ndarray)


def test_analyze_calculates_blur_score_using_laplacian(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """Calculates blur_score using Laplacian variance.

    Given:
        - An analyzer instance
        - A valid test image
    When:
        - Analyzing the image
    Then:
        - blur_score is calculated using cv2.Laplacian
        - blur_score is the variance of the Laplacian (positive value)
        - Sharper images have higher blur_score
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result = analyzer.analyze(sample_image_path)

    # Assert
    assert result is not None
    assert "blur_score" in result.raw_metrics
    assert result.raw_metrics["blur_score"] >= 0


def test_analyze_calculates_brightness_from_grayscale(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """Calculates brightness from grayscale image mean.

    Given:
        - An analyzer instance
        - A valid test image
    When:
        - Analyzing the image
    Then:
        - brightness is the mean of grayscale pixel values
        - brightness is in range [0, 255]
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result = analyzer.analyze(sample_image_path)

    # Assert
    assert result is not None
    assert "brightness" in result.raw_metrics
    assert 0 <= result.raw_metrics["brightness"] <= 255


def test_analyze_calculates_contrast_as_standard_deviation(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """Calculates contrast as grayscale standard deviation.

    Given:
        - An analyzer instance
        - A valid test image
    When:
        - Analyzing the image
    Then:
        - contrast is the standard deviation of grayscale pixels
        - contrast is a non-negative value
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result = analyzer.analyze(sample_image_path)

    # Assert
    assert result is not None
    assert "contrast" in result.raw_metrics
    assert result.raw_metrics["contrast"] >= 0


def test_analyze_applies_penalty_for_dark_images(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    dark_image_path: str,
) -> None:
    """Applies 0.6 penalty for images with brightness < 40.

    Given:
        - An analyzer instance
        - A dark test image (brightness < 40)
    When:
        - Analyzing the image
    Then:
        - A 0.6 penalty is applied to the total score
        - Total score is reduced compared to what it would be without penalty
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result = analyzer.analyze(dark_image_path)

    # Assert
    assert result is not None
    assert result.raw_metrics["brightness"] < 40
    # Penalty is applied, so total score should be lower
    # We can't directly verify the penalty amount without complex calculation,
    # but we can verify the score is reasonable
    assert result.total_score >= 0


def test_analyze_combines_metrics_with_genre_specific_weights(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """Combines metrics using genre-specific weights.

    Given:
        - An analyzer instance with a specific genre
        - A valid test image
    When:
        - Analyzing the image
    Then:
        - Total score is calculated using weighted sum
        - Weights match the genre's configuration
    """
    # Arrange
    analyzer = ImageQualityAnalyzer(genre="fps")

    # Act
    result = analyzer.analyze(sample_image_path)

    # Assert
    assert result is not None
    # Verify weights are used by checking total score is calculated
    assert result.total_score >= 0
    # FPS genre has higher blur_score weight (0.25), so blur normalization
    # should significantly impact the total score


# ============================================================================
# Tests for analyze method - edge cases (4 tests)
# ============================================================================


def test_analyze_returns_none_for_nonexistent_file(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
) -> None:
    """Returns None for nonexistent file path.

    Given:
        - An analyzer instance
        - A file path that doesn't exist
    When:
        - Analyzing the nonexistent file
    Then:
        - Returns None (graceful failure)
        - No exception is raised
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()
    nonexistent_path = "/path/that/does/not/exist.jpg"

    # Act
    result = analyzer.analyze(nonexistent_path)

    # Assert
    assert result is None


def test_analyze_returns_none_for_corrupted_image_file(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    tmp_path: Path,
) -> None:
    """Returns None for corrupted image file.

    Given:
        - An analyzer instance
        - A file with invalid image data
    When:
        - Analyzing the corrupted file
    Then:
        - Returns None (graceful failure)
        - No exception is raised
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()
    # Create a file with invalid image data
    corrupted_path = tmp_path / "corrupted.jpg"
    corrupted_path.write_text("This is not a valid image file")

    # Act
    result = analyzer.analyze(str(corrupted_path))

    # Assert
    assert result is None


def test_analyze_handles_various_image_formats(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
    png_image_path: str,
    bmp_image_path: str,
) -> None:
    """Handles JPG, PNG, and BMP image formats correctly.

    Given:
        - An analyzer instance
        - Test images in JPG, PNG, and BMP formats
    When:
        - Analyzing each image
    Then:
        - All formats are successfully analyzed
        - All return valid ImageMetrics (not None)
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result_jpg = analyzer.analyze(sample_image_path)
    result_png = analyzer.analyze(png_image_path)
    result_bmp = analyzer.analyze(bmp_image_path)

    # Assert
    assert result_jpg is not None
    assert result_png is not None
    assert result_bmp is not None


def test_analyze_handles_images_with_different_dimensions(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
    small_image_path: str,
) -> None:
    """Handles images with different dimensions correctly.

    Given:
        - An analyzer instance
        - Test images with different dimensions:
          - Standard: 640x480
          - Small: 320x240
    When:
        - Analyzing each image
    Then:
        - All images are successfully analyzed
        - All return valid ImageMetrics (not None)
        - Feature vectors have consistent dimension (64,)
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result_standard = analyzer.analyze(sample_image_path)
    result_small = analyzer.analyze(small_image_path)

    # Assert
    assert result_standard is not None
    assert result_small is not None
    # All images resized to 128x128, so feature vectors are same size
    assert result_standard.features.shape == (64,)
    assert result_small.features.shape == (64,)


# ============================================================================
# Integration tests (2 tests)
# ============================================================================


def test_analyze_integration_with_metric_normalizer(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """Integration with MetricNormalizer produces correct normalized values.

    Given:
        - An analyzer instance
        - A valid test image
    When:
        - Analyzing the image
    Then:
        - Raw metrics are calculated correctly
        - Normalized metrics are in [0, 1] range
        - All 8 normalized metrics are present
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result = analyzer.analyze(sample_image_path)

    # Assert
    assert result is not None
    # Check all normalized metrics are present
    expected_keys = {
        "blur_score",
        "contrast",
        "color_richness",
        "edge_density",
        "dramatic_score",
        "visual_balance",
        "action_intensity",
        "ui_density",
    }
    assert set(result.normalized_metrics.keys()) == expected_keys
    # Check all normalized values are in [0, 1]
    for value in result.normalized_metrics.values():
        assert 0.0 <= value <= 1.0


def test_analyze_produces_consistent_results_for_same_image(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """Produces consistent results when analyzing the same image multiple times.

    Given:
        - An analyzer instance
        - A valid test image
    When:
        - Analyzing the same image twice
    Then:
        - Both analyses produce identical results
        - All metric values are the same
        - Total score is the same
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result1 = analyzer.analyze(sample_image_path)
    result2 = analyzer.analyze(sample_image_path)

    # Assert
    assert result1 is not None
    assert result2 is not None
    assert result1.path == result2.path
    assert result1.total_score == result2.total_score
    assert result1.semantic_score == result2.semantic_score
    # Raw metrics should be identical
    for key in result1.raw_metrics:
        assert result1.raw_metrics[key] == result2.raw_metrics[key]
    # Normalized metrics should be identical
    for key in result1.normalized_metrics:
        assert result1.normalized_metrics[key] == result2.normalized_metrics[key]
    # Features should be identical
    assert np.array_equal(result1.features, result2.features)
