"""VectorUtilsクラスの単体テスト."""

import numpy as np
import pytest

from src.utils.vector_utils import VectorUtils


@pytest.mark.parametrize(
    "vec,expected_is_zero",
    [
        (np.array([0.0, 0.0, 0.0]), True),  # ゼロベクトル
        (np.array([1e-10, 1e-10, 1e-10]), True),  # 微小ベクトル
        (np.array([3.0, 4.0]), False),  # 通常ベクトル
    ],
)
def test_safe_l2_normalize_handles_various_vectors(
    vec: np.ndarray, expected_is_zero: bool
) -> None:
    """様々な種類のベクトルが正しく正規化されること.

    Given:
        - ゼロベクトル、微小ベクトル、または通常ベクトルがある
    When:
        - VectorUtils.safe_l2_normalizeで正規化される
    Then:
        - ゼロ/微小ベクトルはゼロベクトルが返されること
        - 通常ベクトルはL2ノルムが1になること
    """
    # Act
    result = VectorUtils.safe_l2_normalize(vec)

    # Assert
    assert result.shape == vec.shape
    if expected_is_zero:
        assert np.all(result == 0.0)
        assert not np.any(np.isnan(result))
    else:
        assert np.linalg.norm(result) == pytest.approx(1.0)
