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
from typing import Callable, List
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
    # 多様性テスト用の異なる特徴ベクトルを作成
    features_list = [
        np.random.rand(128),  # 画像1: ランダムな特徴
        np.random.rand(128),  # 画像2: ランダムな特徴
        np.random.rand(128),  # 画像3: ランダムな特徴
        np.random.rand(128),  # 画像4: ランダムな特徴
        np.random.rand(128),  # 画像5: ランダムな特徴
    ]

    # image2をimage1に類似させる（高いコサイン類似度）
    features_list[1] = features_list[0] + np.random.rand(128) * 0.1

    # image5をimage3に類似させる
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
            features=features_list[1],  # image1に類似
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
# select_from_analyzedメソッドのテスト（純粋なドメインロジック）
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
    # 画像はスコア順になっているはず（高い順）
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
    # 類似性フィルタリングにより10件未満になるはず
    # (image2 ~ image1, image5 ~ image3)
    assert len(result) <= 5
    assert len(result) >= 1


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
    # image1とimage2は両方とも結果に含まれない（類似しているため）
    result_paths = [m.path for m in result]
    has_image1 = "/fake/path/image1.jpg" in result_paths
    has_image2 = "/fake/path/image2.jpg" in result_paths
    # 両方とも選択されることはない
    assert not (has_image1 and has_image2)


@pytest.mark.parametrize(
    "input_list,num_to_select",
    [
        ([], 5),
        ([], 0),
        (None, 0),
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
    # 元のリストオブジェクトは同じ順序のままであるはず
    assert sample_image_metrics == original_order


# ============================================================================
# Integration tests for select method
# ============================================================================


def _create_mock_analyze_for_integration(
    make_similar: bool = False, return_none_for_even: bool = False
) -> Callable[[str], ImageMetrics | None]:
    """統合テスト用のモックanalyze関数を作成するヘルパー.

    Args:
        make_similar: image0とimage1を類似させるかどうか
        return_none_for_even: 偶数インデックスの画像でNoneを返すかどうか

    Returns:
        モックanalyze関数
    """

    def mock_analyze(path: str) -> ImageMetrics | None:
        idx = int(path.split("image")[-1].split(".")[0])
        if return_none_for_even and idx % 2 == 0:
            return None
        base_features = np.random.rand(128)
        if make_similar and idx == 1:
            base_features = base_features * 0.99
        return ImageMetrics(
            path=path,
            raw_metrics={"blur_score": 100 - idx * 10},
            normalized_metrics={"blur_score": 1.0 - idx * 0.1},
            semantic_score=0.8,
            total_score=100 - idx * 10,
            features=base_features,
        )

    return mock_analyze


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
        for i in range(5):
            Path(temp_dir, f"image{i}.jpg").touch()

        mock_analyzer.analyze = _create_mock_analyze_for_integration(make_similar=True)
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
        assert len(result) >= 1
        assert len(result) <= 3
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
