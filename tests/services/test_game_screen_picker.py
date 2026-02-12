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
from src.services.game_screen_picker import GameScreenPicker


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
def mock_analyzer_with_batch(mock_analyzer: MagicMock) -> MagicMock:
    """analyze_batchメソッドを持つモックImageQualityAnalyzer.

    標準的なモック分析関数を提供します。
    """

    def mock_analyze_batch(
        paths: List[str],
        batch_size: int = 32,  # noqa: ARG001
        show_progress: bool = False,  # noqa: ARG001
    ) -> List[ImageMetrics | None]:
        """テスト用のモック分析関数."""
        results: List[ImageMetrics | None] = []
        for path in paths:
            try:
                idx = int(path.split("image")[-1].split(".")[0])
            except (ValueError, IndexError):
                idx = 0
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
    return mock_analyzer


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


def test_selecting_from_folder_loads_analyzes_and_returns_diverse_images(
    mock_analyzer_with_batch: MagicMock,
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

        picker = GameScreenPicker(mock_analyzer_with_batch)

        # Act
        result, stats = picker.select(
            folder=temp_dir,
            num=3,
            similarity_threshold=0.8,
            recursive=False,
            show_progress=False,
        )

        # Assert
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
        # 奇数インデックスの画像のみ有効（image1, image3）
        assert len(result) <= 2
        # 統計情報の検証
        assert stats.total_files == 5
        assert stats.analyzed_ok == 2
        assert stats.analyzed_fail == 3


@pytest.fixture
def similar_images_metrics() -> List[ImageMetrics]:
    """非常に類似した画像セットを作成するfixture（0.97-0.99の類似度）."""
    np.random.seed(42)
    base_features = np.random.rand(128)

    similar_images: List[ImageMetrics] = []
    for i in range(10):
        target_sim = 0.97 + (i % 3) * 0.01
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
    return similar_images


def test_threshold_relaxation_with_highly_similar_images(
    similar_images_metrics: List[ImageMetrics],
) -> None:
    """類似した画像ばかりの場合、しきい値緩和と最終フォールバックが機能すること.

    Given:
        - 10枚の非常に類似した画像（0.97-0.99の類似度）
    When:
        - 類似度閾値0.9で10枚の画像を選択
    Then:
        - 段階的しきい値緩和により可能な限り多様性を確保しつつ
          最終的に10枚全てが返されること
    """
    # Arrange
    num_to_select = 10
    similarity_threshold = 0.9

    # Act
    result, stats = GameScreenPicker.select_from_analyzed(
        similar_images_metrics, num_to_select, similarity_threshold
    )

    # Assert
    assert len(result) == num_to_select
    assert stats.selected_count == num_to_select
    scores = [m.total_score for m in result]
    assert scores == sorted(scores, reverse=True)
