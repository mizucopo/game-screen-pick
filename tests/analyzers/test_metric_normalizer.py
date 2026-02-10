"""Unit tests for MetricNormalizer.

This test module follows these best practices:
1. Tests "what" (observable behavior), not "how" (implementation details)
2. No mocks - pure function tests with predictable inputs/outputs
3. Uses AAA pattern (Arrange, Act, Assert) with clear comments
4. Fast execution (~0.1 seconds) - no external dependencies
"""

from src.analyzers.metric_normalizer import MetricNormalizer


# ============================================================================
# Tests for sigmoid function (5 tests)
# ============================================================================


def test_sigmoid_returns_0_5_when_x_equals_center() -> None:
    """When x equals center, sigmoid returns exactly 0.5.

    Given:
        - A center value of 500.0
        - x equals the center value
    When:
        - Computing sigmoid(x, center)
    Then:
        - Returns exactly 0.5 (midpoint of sigmoid curve)
    """
    # Arrange
    center = 500.0
    x = center

    # Act
    result = MetricNormalizer.sigmoid(x, center)

    # Assert
    assert result == 0.5


def test_sigmoid_returns_high_values_when_x_above_center() -> None:
    """When x is above center, sigmoid returns values greater than 0.5.

    Given:
        - A center value of 500.0
        - x is moderately above the center (600.0)
    When:
        - Computing sigmoid(x, center) with default steepness
    Then:
        - Returns a value greater than 0.5
        - Value approaches 1.0 for very high x (may reach 1.0 for extreme values)
    """
    # Arrange
    center = 500.0
    x = 600.0  # Moderately above center, not too extreme

    # Act
    result = MetricNormalizer.sigmoid(x, center)

    # Assert
    assert result > 0.5
    assert result <= 1.0  # Can be exactly 1.0 for extreme values


def test_sigmoid_returns_low_values_when_x_below_center() -> None:
    """When x is below center, sigmoid returns values less than 0.5.

    Given:
        - A center value of 500.0
        - x is significantly below the center (100.0)
    When:
        - Computing sigmoid(x, center)
    Then:
        - Returns a value less than 0.5
        - Value approaches 0.0 for very low x
    """
    # Arrange
    center = 500.0
    x = 100.0

    # Act
    result = MetricNormalizer.sigmoid(x, center)

    # Assert
    assert result < 0.5
    assert result > 0.0


def test_sigmoid_with_default_steepness_produces_expected_curve() -> None:
    """Default steepness produces expected sigmoid curve.

    Given:
        - A center value of 50.0 (typical for contrast metric)
        - Default steepness of 0.1
        - Three test points: below, at, and above center
    When:
        - Computing sigmoid for each point
    Then:
        - Produces a smooth curve with expected values
        - Lower values produce lower outputs
        - Higher values produce higher outputs
    """
    # Arrange
    center = 50.0
    steepness = 0.1
    x_below = 0.0
    x_at = 50.0
    x_above = 100.0

    # Act
    result_below = MetricNormalizer.sigmoid(x_below, center, steepness)
    result_at = MetricNormalizer.sigmoid(x_at, center, steepness)
    result_above = MetricNormalizer.sigmoid(x_above, center, steepness)

    # Assert
    assert result_below < 0.5
    assert result_at == 0.5
    assert result_above > 0.5
    # Monotonic increasing property
    assert result_below < result_at < result_above


def test_sigmoid_handles_overflow_without_crashing() -> None:
    """Handles overflow/underflow gracefully without raising exceptions.

    Given:
        - Extreme input values that could cause math.exp overflow
        - Very large positive value (1e10)
        - Very large negative value (-1e10)
    When:
        - Computing sigmoid for extreme values
    Then:
        - Returns 1.0 for extreme positive (no exception)
        - Returns 0.0 for extreme negative (no exception)
        - No OverflowError or underflow exceptions raised
    """
    # Arrange
    center = 500.0
    extreme_positive = 1e10
    extreme_negative = -1e10

    # Act
    result_positive = MetricNormalizer.sigmoid(extreme_positive, center)
    result_negative = MetricNormalizer.sigmoid(extreme_negative, center)

    # Assert
    # Should handle gracefully by returning boundary values
    assert result_positive == 1.0
    assert result_negative == 0.0


# ============================================================================
# Tests for normalize_all method (7 tests)
# ============================================================================


def test_normalize_all_returns_all_expected_metrics() -> None:
    """Returns dictionary with all 8 expected normalized metrics.

    Given:
        - Raw metrics dictionary with all required fields
    When:
        - Calling normalize_all
    Then:
        - Returns all 8 expected metrics:
          - blur_score
          - contrast
          - color_richness
          - edge_density
          - dramatic_score
          - visual_balance
          - action_intensity
          - ui_density
    """
    # Arrange
    raw = {
        "blur_score": 500.0,
        "contrast": 50.0,
        "color_richness": 40.0,
        "edge_density": 0.2,
        "dramatic_score": 50.0,
        "visual_balance": 80.0,
        "action_intensity": 30.0,
        "ui_density": 10.0,
    }

    # Act
    result = MetricNormalizer.normalize_all(raw)

    # Assert
    assert len(result) == 8
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
    assert set(result.keys()) == expected_keys


def test_normalize_all_applies_sigmoid_to_blur_score() -> None:
    """Applies sigmoid normalization to blur_score metric.

    Given:
        - Raw blur_score of 500.0 (center value)
    When:
        - Calling normalize_all
    Then:
        - blur_score is normalized using sigmoid with center=500
        - Result is exactly 0.5 at the center point
    """
    # Arrange
    raw = {
        "blur_score": 500.0,
        "contrast": 50.0,
        "color_richness": 40.0,
        "edge_density": 0.1,
        "dramatic_score": 50.0,
        "visual_balance": 80.0,
        "action_intensity": 30.0,
        "ui_density": 10.0,
    }

    # Act
    result = MetricNormalizer.normalize_all(raw)

    # Assert
    assert result["blur_score"] == 0.5


def test_normalize_all_applies_sigmoid_to_contrast() -> None:
    """Applies sigmoid normalization to contrast metric.

    Given:
        - Raw contrast of 50.0 (center value)
    When:
        - Calling normalize_all
    Then:
        - contrast is normalized using sigmoid with center=50
        - Result is exactly 0.5 at the center point
    """
    # Arrange
    raw = {
        "blur_score": 500.0,
        "contrast": 50.0,
        "color_richness": 40.0,
        "edge_density": 0.1,
        "dramatic_score": 50.0,
        "visual_balance": 80.0,
        "action_intensity": 30.0,
        "ui_density": 10.0,
    }

    # Act
    result = MetricNormalizer.normalize_all(raw)

    # Assert
    assert result["contrast"] == 0.5


def test_normalize_all_clips_edge_density_to_max_1() -> None:
    """Clips edge_density to maximum of 1.0 using min(1.0, raw * 5.0).

    Given:
        - Raw edge_density of 0.3 (which would be 1.5 when multiplied by 5)
    When:
        - Calling normalize_all
    Then:
        - edge_density is clipped to maximum 1.0
        - Formula: min(1.0, raw * 5.0)
    """
    # Arrange
    raw = {
        "blur_score": 500.0,
        "contrast": 50.0,
        "color_richness": 40.0,
        "edge_density": 0.3,  # Will become 1.5, clipped to 1.0
        "dramatic_score": 50.0,
        "visual_balance": 80.0,
        "action_intensity": 30.0,
        "ui_density": 10.0,
    }

    # Act
    result = MetricNormalizer.normalize_all(raw)

    # Assert
    assert result["edge_density"] == 1.0


def test_normalize_all_divides_visual_balance_by_100() -> None:
    """Divides visual_balance by 100 to normalize to [0, 1] range.

    Given:
        - Raw visual_balance of 80.0
    When:
        - Calling normalize_all
    Then:
        - visual_balance is divided by 100
        - Result is 0.8
    """
    # Arrange
    raw = {
        "blur_score": 500.0,
        "contrast": 50.0,
        "color_richness": 40.0,
        "edge_density": 0.1,
        "dramatic_score": 50.0,
        "visual_balance": 80.0,
        "action_intensity": 30.0,
        "ui_density": 10.0,
    }

    # Act
    result = MetricNormalizer.normalize_all(raw)

    # Assert
    assert result["visual_balance"] == 0.8


def test_normalize_all_produces_values_between_0_and_1() -> None:
    """All normalized values are within the valid range [0, 1].

    Given:
        - Raw metrics with various realistic values
    When:
        - Calling normalize_all
    Then:
        - All normalized values are between 0.0 and 1.0 inclusive
        - No negative values or values greater than 1.0
    """
    # Arrange
    raw = {
        "blur_score": 650.0,
        "contrast": 75.0,
        "color_richness": 55.0,
        "edge_density": 0.25,
        "dramatic_score": 80.0,
        "visual_balance": 90.0,
        "action_intensity": 45.0,
        "ui_density": 15.0,
    }

    # Act
    result = MetricNormalizer.normalize_all(raw)

    # Assert
    for value in result.values():
        assert 0.0 <= value <= 1.0


def test_normalize_all_with_different_raw_values_produces_different_results() -> None:
    """Different raw input values produce different normalized outputs.

    Given:
        - Two sets of raw metrics with different values
    When:
        - Calling normalize_all on each set
    Then:
        - Produces different normalized results
        - Results maintain relative ordering (higher raw -> higher normalized)
    """
    # Arrange
    raw_low = {
        "blur_score": 300.0,
        "contrast": 30.0,
        "color_richness": 25.0,
        "edge_density": 0.1,
        "dramatic_score": 30.0,
        "visual_balance": 60.0,
        "action_intensity": 20.0,
        "ui_density": 5.0,
    }

    raw_high = {
        "blur_score": 700.0,
        "contrast": 70.0,
        "color_richness": 55.0,
        "edge_density": 0.2,
        "dramatic_score": 70.0,
        "visual_balance": 95.0,
        "action_intensity": 40.0,
        "ui_density": 15.0,
    }

    # Act
    result_low = MetricNormalizer.normalize_all(raw_low)
    result_high = MetricNormalizer.normalize_all(raw_high)

    # Assert
    # Higher raw values should produce higher normalized values
    assert result_high["blur_score"] > result_low["blur_score"]
    assert result_high["contrast"] > result_low["contrast"]
    assert result_high["color_richness"] > result_low["color_richness"]
    # Edge density uses linear scaling
    assert result_high["edge_density"] > result_low["edge_density"]
    # Visual balance uses linear scaling
    assert result_high["visual_balance"] > result_low["visual_balance"]
