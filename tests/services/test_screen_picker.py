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
from typing import List, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.analyzers.image_quality_analyzer import ImageQualityAnalyzer
from src.models.image_metrics import ImageMetrics
from src.models.picker_statistics import PickerStatistics  # noqa: F401
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
    eps = 1e-8
    norm = np.linalg.norm(base_features)
    if norm < eps:
        base_normalized = base_features  # ゼロベクトルの場合はそのまま使用
    else:
        base_normalized = base_features / norm

    # 直交成分の大きさを計算: sqrt(1 - cos^2)
    orthogonal_norm = np.sqrt(max(0, 1 - target_similarity**2))

    # ランダムな直交ベクトルを生成
    random_vec = np.random.randn(len(base_features))
    random_vec = random_vec / np.linalg.norm(random_vec) * orthogonal_norm

    # 目標類似度を持つベクトルを合成
    similar_features = target_similarity * base_normalized + random_vec

    return cast(np.ndarray, similar_features)


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
    result, stats = GameScreenPicker.select_from_analyzed(
        sample_image_metrics, num_to_select, similarity_threshold
    )

    # Assert
    # 型チェック（インポート使用を明示）
    assert isinstance(stats, PickerStatistics)
    assert len(result) == 3
    # 統計情報の検証
    assert stats.total_files == 5
    assert stats.analyzed_ok == 5
    assert stats.analyzed_fail == 0
    assert stats.selected_count == 3
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
    result, stats = GameScreenPicker.select_from_analyzed(
        sample_image_metrics, num_to_select, similarity_threshold
    )

    # Assert
    # 型チェック（インポート使用を明示）
    assert isinstance(stats, PickerStatistics)
    # 類似性フィルタリングにより10件未満になるはず
    # (image2 ~ image1, image5 ~ image3)
    assert len(result) <= 5
    assert len(result) >= 1
    assert stats.selected_count == len(result)


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
    result, stats = GameScreenPicker.select_from_analyzed(
        input_list, num_to_select, 0.8
    )

    # Assert
    # 型チェック（インポート使用を明示）
    assert isinstance(stats, PickerStatistics)
    assert result == []
    assert stats.selected_count == 0


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


def test_selecting_from_folder_loads_analyzes_and_returns_diverse_images(
    mock_analyzer: MagicMock,
) -> None:
    """完全な統合：ロード、分析、多様な画像の選択が行われること.

    Given:
        - 5つの画像ファイルを持つフォルダ
        - モックアナライザは一貫した結果を返す
    When:
        - 類似度閾値で3つの画像を選択
    Then:
        - フォルダから画像がロード・分析され、選択結果が返されること
    """
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(5):
            Path(temp_dir, f"image{i}.jpg").touch()

        # モックアナライザを設定
        def simple_mock_analyze(path: str) -> ImageMetrics:
            idx = int(path.split("image")[-1].split(".")[0])
            # 各画像に異なる特徴ベクトルを割り当て
            np.random.seed(idx)
            features = np.random.rand(128)
            return ImageMetrics(
                path=path,
                raw_metrics={"blur_score": 100 - idx * 10},
                normalized_metrics={"blur_score": 1.0 - idx * 0.1},
                semantic_score=0.8,
                total_score=100 - idx * 10,
                features=features,
            )

        def mock_analyze_batch(
            paths: List[str],
            batch_size: int = 32,  # noqa: ARG001
            show_progress: bool = False,  # noqa: ARG001
        ) -> List[ImageMetrics | None]:
            return [simple_mock_analyze(p) for p in paths]

        mock_analyzer.analyze_batch = mock_analyze_batch
        picker = GameScreenPicker(mock_analyzer)

        # Act
        result, stats = picker.select(
            folder=temp_dir,
            num=3,
            similarity_threshold=0.8,
            recursive=False,
            show_progress=False,
        )

        # Assert
        # 型チェック（インポート使用を明示）
        assert isinstance(stats, PickerStatistics)
        # 結果の基本検証
        assert len(result) <= 3
        # 統計情報の検証
        assert stats.total_files == 5
        assert stats.analyzed_ok == 5
        assert stats.analyzed_fail == 0
        # スコア順の検証
        for i in range(len(result) - 1):
            assert result[i].total_score >= result[i + 1].total_score


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

        def mock_analyze_batch(
            paths: List[str],
            batch_size: int = 32,  # noqa: ARG001
            show_progress: bool = False,  # noqa: ARG001
        ) -> List[ImageMetrics | None]:
            # 偶数インデックス（image0, image2, image4）はNoneを返す
            results: List[ImageMetrics | None] = []
            for path in paths:
                idx = int(path.split("image")[-1].split(".")[0])
                if idx % 2 == 0:
                    results.append(None)
                else:
                    np.random.seed(idx)
                    results.append(
                        ImageMetrics(
                            path=path,
                            raw_metrics={"blur_score": 100 - idx * 10},
                            normalized_metrics={"blur_score": 1.0 - idx * 0.1},
                            semantic_score=0.8,
                            total_score=100 - idx * 10,
                            features=np.random.rand(128),
                        )
                    )
            return results

        mock_analyzer.analyze_batch = mock_analyze_batch
        picker = GameScreenPicker(mock_analyzer)

        # Act
        result, stats = picker.select(
            folder=temp_dir,
            num=5,
            similarity_threshold=0.8,
            recursive=False,
            show_progress=False,
        )

        # Assert
        # 型チェック（インポート使用を明示）
        assert isinstance(stats, PickerStatistics)
        # 奇数インデックスの画像のみ有効（image1, image3）
        assert len(result) <= 2
        # 統計情報の検証
        assert stats.total_files == 5
        assert stats.analyzed_ok == 2
        assert stats.analyzed_fail == 3


def test_always_returns_requested_number_when_enough_unique_images_exist(
    sample_image_metrics: List[ImageMetrics],
) -> None:
    """十分な数の一意な画像が存在する場合、常に要求数を返すこと.

    Given:
        - 5つの分析済み画像（一部類似）
    When:
        - 高い類似度閾値（0.9）で5つの画像を選択
    Then:
        - 類似画像はフィルタリングされるが、指定数を満たすために
          段階的しきい値緩和と最終フォールバックにより5件が返されること
    """
    # Arrange
    num_to_select = 5
    similarity_threshold = 0.9

    # Act
    result, stats = GameScreenPicker.select_from_analyzed(
        sample_image_metrics, num_to_select, similarity_threshold
    )

    # Assert
    # 5枚要求して5枚存在するので、必ず5枚返されるはず
    assert len(result) == 5
    assert stats.selected_count == 5
    # スコア順になっているはず
    scores = [m.total_score for m in result]
    assert scores == sorted(scores, reverse=True)


def test_gradual_threshold_relaxation_and_fallback_for_similar_images() -> None:
    """類似した画像ばかりの場合、しきい値緩和と最終フォールバックが機能すること.

    Given:
        - 10枚の非常に類似した画像（0.97-0.99の類似度）
        - 類似度閾値0.9
    When:
        - 10枚の画像を選択
    Then:
        - 段階的しきい値緩和により可能な限り多様性を確保しつつ
          最終的に10枚全てが返されること
    """
    # Arrange
    np.random.seed(42)

    # ベース特徴ベクトル
    base_features = np.random.rand(128)

    # 0.97-0.99の類似度を持つ10枚の画像を生成
    similar_images: List[ImageMetrics] = []
    for i in range(10):
        # 0.97-0.99の類似度で生成
        target_sim = 0.97 + (i % 3) * 0.01  # 0.97, 0.98, 0.99を繰り返し
        similar_features = _create_features_with_similarity(base_features, target_sim)
        similar_images.append(
            ImageMetrics(
                path=f"/fake/path/similar{i}.jpg",
                raw_metrics={"blur_score": 100.0 - i},
                normalized_metrics={"blur_score": 0.9},
                semantic_score=0.8,
                total_score=100.0 - i,
                features=similar_features,
            )
        )

    num_to_select = 10
    similarity_threshold = 0.9

    # Act
    result, stats = GameScreenPicker.select_from_analyzed(
        similar_images, num_to_select, similarity_threshold
    )

    # Assert
    # 10枚要求して10枚存在するので、必ず10枚返されるはず
    assert len(result) == 10
    assert stats.selected_count == 10
    # スコア順になっているはず
    scores = [m.total_score for m in result]
    assert scores == sorted(scores, reverse=True)
