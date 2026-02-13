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


def test_select_diverse_indices_returns_valid_rejection_count() -> None:
    """選択結果と除外数の整合性が正しいこと.

    Given:
        - 20件のほぼ同一の特徴ベクトルがある
        - 選択数が5件
        - 4段階の閾値ステップがある
    When:
        - select_diverse_indicesを実行する
    Then:
        - 除外数が候補数から選択数を引いた値以下であること
        - 除外数が負でないこと
    """
    # Arrange: 20件のほぼ同一ベクトルを生成（最初の10件は類似、後の10件は少し異なる）
    np.random.seed(42)
    base_vec = np.random.randn(128)
    normalized_features = []
    for i in range(20):
        # 最初の10件はほぼ同一、後の10件は少し変化
        noise = np.random.randn(128) * (0.001 if i < 10 else 0.1)
        vec = base_vec + noise
        vec = vec / np.linalg.norm(vec)  # L2正規化
        normalized_features.append(vec)

    # Act
    threshold_steps = [0.85, 0.88, 0.91, 0.94]
    selected_indices, rejected_by_similarity = VectorUtils.select_diverse_indices(
        normalized_features=normalized_features,
        num=5,
        threshold_steps=threshold_steps,
    )

    # Assert
    assert len(selected_indices) <= 5
    assert (
        0 <= rejected_by_similarity <= len(normalized_features) - len(selected_indices)
    ), (
        f"除外数({rejected_by_similarity})が候補数({len(normalized_features)})"
        f" - 選択数({len(selected_indices)})を超えています"
    )


def test_select_diverse_indices_rejection_count_across_threshold_steps() -> None:
    """閾値ステップ間で除外数が正しく集計されること.

    Given:
        - 5件の特徴ベクトルがある
        - 最初の2件は類似、残り3件は互いに異なる
    When:
        - 緩い閾値から厳しい閾値へ段階的に厳格化して選択する
        - 最初の緩い閾値で必要数選択されるため、2段階目は実行されない
    Then:
        - 最初のステップで選択された候補のみが選択されること
        - 類似度で除外された候補のみが除外数にカウントされること
    """
    # Arrange: 5件の特徴ベクトル（最初の2件は類似、残り3件は互いに異なる）
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([0.99, 0.01, 0.0])  # vec_aと類似（内積≈0.99）
    vec_c = np.array([0.0, 1.0, 0.0])  # vec_aと直交
    vec_d = np.array([0.98, 0.02, 0.0])  # vec_aと類似（内積≈0.98）
    vec_e = np.array([0.0, 0.0, 1.0])  # vec_a, vec_cと直交

    # すべてL2正規化済みとする
    normalized_features = [vec_a, vec_b, vec_c, vec_d, vec_e]

    # Act: 緩い閾値（0.8）→ 厳しい閾値（0.95）
    # 最初の緩い閾値で3件選択されるため、2段階目は実行されない
    threshold_steps = [0.8, 0.95]
    selected_indices, rejected_by_similarity = VectorUtils.select_diverse_indices(
        normalized_features=normalized_features,
        num=3,
        threshold_steps=threshold_steps,
    )

    # Assert
    # vec_a(0), vec_c(2), vec_e(4) が選択される（どれも内積<0.8）
    # vec_b(1), vec_d(3) はそれぞれvec_aとの類似度>0.8で除外される
    assert len(selected_indices) == 3
    assert selected_indices == {0, 2, 4}
    assert rejected_by_similarity == 2  # vec_b, vec_d のみ除外


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
