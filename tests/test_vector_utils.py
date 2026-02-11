"""VectorUtilsクラスの単体テスト."""

import numpy as np
import pytest

from src.utils import VectorUtils


def test_safe_l2_normalize_returns_zeros_for_zero_vector() -> None:
    """ゼロベクトルが正規化されること.

    Given:
        - ゼロベクトルがある
    When:
        - VectorUtils.safe_l2_normalizeで正規化される
    Then:
        - ゼロベクトルが返されること（NaNではない）
    """
    # Arrange
    zero_vec = np.array([0.0, 0.0, 0.0])

    # Act
    result = VectorUtils.safe_l2_normalize(zero_vec)

    # Assert
    assert result.shape == zero_vec.shape
    assert np.all(result == 0.0)
    assert not np.any(np.isnan(result))


def test_safe_l2_normalize_normalizes_nonzero_vector() -> None:
    """非ゼロベクトルが正しく正規化されること.

    Given:
        - 非ゼロベクトルがある
    When:
        - VectorUtils.safe_l2_normalizeで正規化される
    Then:
        - L2ノルムが1になること
    """
    # Arrange
    vec = np.array([3.0, 4.0])  # ノルム = 5

    # Act
    result = VectorUtils.safe_l2_normalize(vec)

    # Assert
    expected_norm = 1.0
    actual_norm = np.linalg.norm(result)
    assert actual_norm == pytest.approx(expected_norm)


def test_safe_l2_normalize_handles_small_vectors() -> None:
    """微小なベクトルが正しく処理されること.

    Given:
        - epsよりも小さなノルムを持つベクトルがある
    When:
        - VectorUtils.safe_l2_normalizeで正規化される
    Then:
        - ゼロベクトルが返されること
    """
    # Arrange
    small_vec = np.array([1e-10, 1e-10, 1e-10])  # ノルム < 1e-8

    # Act
    result = VectorUtils.safe_l2_normalize(small_vec)

    # Assert
    assert result.shape == small_vec.shape
    assert np.all(result == 0.0)
