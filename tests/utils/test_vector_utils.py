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


def test_select_diverse_indices_empty_features() -> None:
    """空の特徴ベクトルリストを正しく処理すること.

    Given:
        - 空の特徴ベクトルリストがある
    When:
        - select_diverse_indicesを実行する
    Then:
        - 空のセットと0が返されること
    """
    # Arrange
    normalized_features: list[np.ndarray] = []

    # Act
    selected_indices, rejected_by_similarity = VectorUtils.select_diverse_indices(
        normalized_features=normalized_features,
        num=5,
        threshold_steps=[0.9, 0.95],
    )

    # Assert
    assert selected_indices == set()
    assert rejected_by_similarity == 0


def test_select_diverse_indices_all_identical_vectors() -> None:
    """全く同一のベクトルに対して正しく動作すること.

    Given:
        - 10件の全く同一の特徴ベクトルがある
        - 選択数が5件
    When:
        - select_diverse_indicesを実行する
    Then:
        - 最初の1件のみが選択されること
        - 残り9件が類似度で除外されること
    """
    # Arrange: 全く同一のベクトル10件
    vec = np.array([1.0, 0.0, 0.0])
    normalized_features = [vec.copy() for _ in range(10)]

    # Act
    selected_indices, rejected_by_similarity = VectorUtils.select_diverse_indices(
        normalized_features=normalized_features,
        num=5,
        threshold_steps=[0.9, 0.95],
    )

    # Assert: 最初の1件のみ選択、残り9件は類似度で除外
    assert len(selected_indices) == 1
    assert selected_indices == {0}
    assert rejected_by_similarity == 9  # 残り9件は類似度で除外
