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

    Arrange:
        - ゼロベクトル、微小ベクトル、または通常ベクトルがある
    Act:
        - VectorUtils.safe_l2_normalizeで正規化される
    Assert:
        - ゼロ/微小ベクトルはゼロベクトルが返されること
        - 通常ベクトルはL2ノルムが1になること
    """
    # Arrange — パラメタライズド引数から各種ベクトルと期待値を設定
    # Act
    result = VectorUtils.safe_l2_normalize(vec)

    # Assert
    assert result.shape == vec.shape
    if expected_is_zero:
        assert np.all(result == 0.0)
        assert not np.any(np.isnan(result))
    else:
        assert np.linalg.norm(result) == pytest.approx(1.0)


def test_select_diverse_indices_handles_empty_and_identical_vectors() -> None:
    """エッジケース（空リスト・同一ベクトル）で正しく動作すること.

    Arrange:
        - 空または同一の特徴ベクトルリストがある
    Act:
        - select_diverse_indicesを実行する
    Assert:
        - 最初の1件のみ選択され、残りは類似度で除外されること
    """
    # Arrange & Act: 空リスト
    selected_empty, rejected_empty = VectorUtils.select_diverse_indices(
        normalized_features=[],
        num=5,
        threshold_steps=[0.9, 0.95],
    )

    # Arrange & Act: 同一ベクトル10件
    vec = np.array([1.0, 0.0, 0.0])
    identical_features = [vec.copy() for _ in range(10)]
    selected_identical, rejected_identical = VectorUtils.select_diverse_indices(
        normalized_features=identical_features,
        num=5,
        threshold_steps=[0.9, 0.95],
    )

    # Assert
    assert len(selected_empty) == 0 and len(rejected_empty) == 0
    assert selected_identical == [0]
    assert len(rejected_identical) == 9


def test_filter_by_similarity_checks_against_seed_features() -> None:
    """既選択特徴をseedとして類似候補を除外できること.

    Arrange:
        - seed特徴として1件のベクトルが指定されている
        - 候補にseedと同一のベクトルと異なるベクトルが含まれている
    Act:
        - filter_by_similarityが実行される
    Assert:
        - seedと同一の候補は除外され、異なる候補のみが選択されること
    """
    # Arrange
    seed = [np.array([1.0, 0.0, 0.0])]
    candidates = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]

    # Act
    selected, rejected = VectorUtils.filter_by_similarity(
        candidates=candidates,
        num=2,
        similarity_threshold=0.9,
        compute_threshold_steps=lambda base: [base],
        seed_features=seed,
    )

    # Assert
    assert selected == [1]
    assert rejected == {0}
