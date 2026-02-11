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
from typing import Callable, List, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.analyzers.image_quality_analyzer import ImageQualityAnalyzer
from src.models.image_metrics import ImageMetrics
from src.services.screen_picker import GameScreenPicker


def _create_features_with_similarity(
    base_features: np.ndarray,
    target_similarity: float,
) -> np.ndarray:
    """ベース特徴ベクトルに対して目標コサイン類似度を持つ特徴を生成する.

    数学的アプローチを使用して、指定された類似度を持つ特徴ベクトルを生成する。
    類似度 = cos(θ) として、目標のコサイン類似度を達成するベクトルを合成する。

    Args:
        base_features: 基準となる特徴ベクトル
        target_similarity: 目標とするコサイン類似度（0.0-1.0）

    Returns:
        指定された類似度を持つ新しい特徴ベクトル
    """
    # ベースを正規化
    base_normalized = base_features / np.linalg.norm(base_features)

    # 直交成分の大きさを計算: sqrt(1 - cos^2)
    orthogonal_norm = np.sqrt(max(0, 1 - target_similarity**2))

    # ランダムな直交ベクトルを生成
    random_vec = np.random.randn(len(base_features))
    random_vec = random_vec / np.linalg.norm(random_vec) * orthogonal_norm

    # 目標類似度を持つベクトルを合成
    similar_features = target_similarity * base_normalized + random_vec

    return cast(np.ndarray, similar_features)


def _assert_cosine_similarity(
    features1: np.ndarray,
    features2: np.ndarray,
    expected_min: float,
    expected_max: float,
) -> None:
    """2つの特徴ベクトル間のコサイン類似度をアサートする.

    Args:
        features1: 1つ目の特徴ベクトル
        features2: 2つ目の特徴ベクトル
        expected_min: 期待される最小類似度
        expected_max: 期待される最大類似度
    """
    from sklearn.metrics.pairwise import cosine_similarity

    similarity = cosine_similarity(
        features1.reshape(1, -1),
        features2.reshape(1, -1),
    )[0][0]

    assert expected_min <= similarity <= expected_max, (
        f"類似度が期待範囲外: {similarity:.6f} "
        f"(期待: {expected_min:.6f} - {expected_max:.6f})"
    )


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

    類似度が明示的に検証された画像セットを返します：
    - image1: 高品質、ベース特徴
    - image2: 中品質、image1と0.96の類似度（0.9閾値を超過）
    - image3: 高品質、image1と0.85の類似度（0.9閾値以下）
    - image4: 低品質、image1と0.30の類似度（異なる特徴）
    - image5: 中品質、image3と0.96の類似度（0.9閾値を超過）
    """
    np.random.seed(42)  # 再現性のためにシード固定

    # ベース特徴ベクトル（image1）
    base_features = np.random.rand(128)

    # 特徴ベクトルを作成
    features_list = [
        base_features,  # image1: ベース特徴
        _create_features_with_similarity(base_features, 0.96),  # image2: 類似
        _create_features_with_similarity(base_features, 0.85),  # image3: 中程度の類似
        _create_features_with_similarity(base_features, 0.30),  # image4: 低類似
    ]

    # image5をimage3と高類似度で生成
    features_list.append(
        _create_features_with_similarity(features_list[2], 0.96),
    )  # image5: image3に類似

    # 類似度を検証
    _assert_cosine_similarity(
        features_list[0], features_list[1], 0.95, 0.97
    )  # image1 ~ image2
    _assert_cosine_similarity(
        features_list[0], features_list[2], 0.83, 0.87
    )  # image1 ~ image3
    _assert_cosine_similarity(
        features_list[2], features_list[4], 0.95, 0.97
    )  # image3 ~ image5

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
            features=features_list[1],  # image1に類似
        ),
        ImageMetrics(
            path="/fake/path/image3.jpg",
            raw_metrics={"blur_score": 90.0},
            normalized_metrics={"blur_score": 0.8},
            semantic_score=0.75,
            total_score=90.0,
            features=features_list[2],  # image1と中程度の類似
        ),
        ImageMetrics(
            path="/fake/path/image4.jpg",
            raw_metrics={"blur_score": 30.0},
            normalized_metrics={"blur_score": 0.3},
            semantic_score=0.3,
            total_score=40.0,
            features=features_list[3],  # 低品質、異なる特徴
        ),
        ImageMetrics(
            path="/fake/path/image5.jpg",
            raw_metrics={"blur_score": 70.0},
            normalized_metrics={"blur_score": 0.6},
            semantic_score=0.6,
            total_score=75.0,
            features=features_list[4],  # image3に類似
        ),
    ]


def test_high_quality_images_are_prioritized_while_avoiding_similar_ones(
    sample_image_metrics: List[ImageMetrics],
) -> None:
    """高品質な画像が優先され、類似した画像は回避されること.

    Given:
        - 様々なスコアを持つ5つの分析済み画像
        - image1（スコア95）とimage2（スコア85）は類似した特徴を持つ
        - image3（スコア90）は異なる特徴を持つ
    When:
        - 類似度閾値0.9で3つの画像を選択
    Then:
        - 3つの画像が返されること
        - 最高スコアの画像が優先されること
        - 類似した画像が除外されること
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
    # 画像はスコア順になっているはず（高い順）
    scores = [m.total_score for m in result]
    assert scores == sorted(scores, reverse=True)
    # Image 2 (similar to image 1) should be filtered out
    selected_paths = [m.path for m in result]
    assert "/fake/path/image2.jpg" not in selected_paths


def test_requesting_more_images_than_available_returns_all_unique_images(
    sample_image_metrics: List[ImageMetrics],
) -> None:
    """利用可能な数より多くの画像を要求した場合、すべての一意な画像が返されること.

    Given:
        - 5つの分析済み画像（一部類似）
    When:
        - 中程度の類似度閾値で10個の画像を選択
    Then:
        - 利用可能な多様な画像の数まで返されること
    """
    # Arrange
    num_to_select = 10
    similarity_threshold = 0.8

    # Act
    result = GameScreenPicker.select_from_analyzed(
        sample_image_metrics, num_to_select, similarity_threshold
    )

    # Assert
    # 類似性フィルタリングにより10件未満になるはず
    # (image2 ~ image1, image5 ~ image3)
    assert len(result) <= 5
    assert len(result) >= 1


@pytest.mark.parametrize(
    "input_list,num_to_select",
    [
        ([], 5),  # 空のリスト
        (None, 0),  # 0個のリクエスト
    ],
)
def test_edge_cases_return_empty_list(
    input_list: List[ImageMetrics] | None,
    num_to_select: int,
    sample_image_metrics: List[ImageMetrics],
) -> None:
    """エッジケースで空のリストを返すことを検証.

    Given:
        - 空の入力リスト、または0個のリクエスト
    When:
        - 選択を実行
    Then:
        - 空のリストを返す
    """
    # Arrange
    if input_list is None:
        input_list = sample_image_metrics

    # Act
    result = GameScreenPicker.select_from_analyzed(input_list, num_to_select, 0.8)

    # Assert
    assert result == []


def test_original_input_list_remains_unchanged_after_selection(
    sample_image_metrics: List[ImageMetrics],
) -> None:
    """元の入力リストは選択後も変更されないこと.

    Given:
        - 特定の順序の分析済み画像リスト
    When:
        - そのリストから選択
    Then:
        - 元のリストの順序と内容が保持されること
    """
    # Arrange
    original_paths = [m.path for m in sample_image_metrics]
    original_order = list(sample_image_metrics)

    # Act
    GameScreenPicker.select_from_analyzed(sample_image_metrics, 3, 0.8)

    # Assert
    assert [m.path for m in sample_image_metrics] == original_paths
    # 元のリストオブジェクトは同じ順序のままであるはず
    assert sample_image_metrics == original_order


def _create_mock_analyze_for_integration(
    make_similar: bool = False,
    return_none_for_even: bool = False,
    target_similarity: float = 0.96,
) -> Callable[[str], ImageMetrics | None]:
    """統合テスト用のモックanalyze関数を作成するヘルパー.

    Args:
        make_similar: image0とimage1を類似させるかどうか
        return_none_for_even: 偶数インデックスの画像でNoneを返すかどうか
        target_similarity: make_similar=True時の目標類似度

    Returns:
        モックanalyze関数
    """
    base_features = None

    def mock_analyze(path: str) -> ImageMetrics | None:
        nonlocal base_features
        idx = int(path.split("image")[-1].split(".")[0])

        if return_none_for_even and idx % 2 == 0:
            return None

        if idx == 0:
            # 最初の画像はベース特徴を生成
            np.random.seed(42)
            base_features = np.random.rand(128)
            features = base_features
        elif make_similar and idx == 1:
            # 2番目の画像は類似特徴を生成
            # もしbase_featuresが未初期化なら先に生成する
            if base_features is None:
                np.random.seed(42)
                base_features = np.random.rand(128)
            features = _create_features_with_similarity(
                base_features, target_similarity
            )
        else:
            # その他は独立した特徴を生成
            np.random.seed(idx + 100)
            features = np.random.rand(128)

        return ImageMetrics(
            path=path,
            raw_metrics={"blur_score": 100 - idx * 10},
            normalized_metrics={"blur_score": 1.0 - idx * 0.1},
            semantic_score=0.8,
            total_score=100 - idx * 10,
            features=features,
        )

    return mock_analyze


def test_selecting_from_folder_loads_analyzes_and_returns_diverse_images(
    mock_analyzer: MagicMock,
) -> None:
    """完全な統合：ロード、分析、多様な画像の選択が行われること.

    Given:
        - 5つの画像ファイルを持つフォルダ
        - image0とimage1は類似（類似度0.96）
        - モックアナライザは一貫した結果を返す
    When:
        - 類似度閾値0.95で3つの画像を選択
    Then:
        - image0とimage1の両方が選択されないこと（類似除外）
        - 多様で高品質な画像が返されること
    """
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(5):
            Path(temp_dir, f"image{i}.jpg").touch()

        mock_analyzer.analyze = _create_mock_analyze_for_integration(
            make_similar=True, target_similarity=0.96
        )
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
        # 結果の基本検証
        assert len(result) >= 1
        assert len(result) <= 3

        # スコア順の検証
        for i in range(len(result) - 1):
            assert result[i].total_score >= result[i + 1].total_score

        # 類似除外の検証：image0とimage1は両方選択されない
        result_paths = [m.path for m in result]
        has_image0 = any("image0.jpg" in p for p in result_paths)
        has_image1 = any("image1.jpg" in p for p in result_paths)

        # 類似度0.96 > 閾値0.95 なので、両方選択されることはない
        assert not (has_image0 and has_image1), (
            "類似画像（image0とimage1）は両方選択されるべきではない"
        )


def test_selecting_gracefully_handles_files_that_fail_to_analyze(
    mock_analyzer: MagicMock,
) -> None:
    """分析に失敗したファイルが適切に処理されること.

    Given:
        - 5つの画像ファイルを持つフォルダ
        - アナライザは一部のファイルに対してNoneを返す（破損/読み取り不可）
    When:
        - 画像を選択
    Then:
        - 処理が継続され、有効な画像のみが返されること
    """
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(5):
            Path(temp_dir, f"image{i}.jpg").touch()

        call_count = [0]
        original_analyze: Callable[[str], ImageMetrics | None] = (
            _create_mock_analyze_for_integration(return_none_for_even=True)
        )

        def counting_analyze(path: str) -> ImageMetrics | None:
            call_count[0] += 1
            return original_analyze(path)

        mock_analyzer.analyze = counting_analyze
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
        assert call_count[0] == 5
        assert len(result) <= 3  # At most 3 valid images (odd indices)
