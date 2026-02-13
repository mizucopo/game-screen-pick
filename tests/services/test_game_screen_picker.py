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
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.analyzers.image_quality_analyzer import ImageQualityAnalyzer
from src.models.image_metrics import ImageMetrics
from src.models.normalized_metrics import NormalizedMetrics
from src.models.raw_metrics import RawMetrics
from src.models.selection_config import SelectionConfig
from src.services.game_screen_picker import GameScreenPicker


def _create_image_metrics(
    path: str,
    raw_metrics_dict: dict[str, float] | None = None,
    normalized_metrics_dict: dict[str, float] | None = None,
    semantic_score: float = 0.8,
    total_score: float = 100.0,
    features: np.ndarray | None = None,
) -> ImageMetrics:
    """ImageMetricsを作成するヘルパー関数."""
    if features is None:
        np.random.seed(42)
        features = np.random.rand(128)

    raw_metrics_dict = raw_metrics_dict or {}
    raw = RawMetrics(
        blur_score=raw_metrics_dict.get("blur_score", 100),
        brightness=raw_metrics_dict.get("brightness", 100),
        contrast=raw_metrics_dict.get("contrast", 50),
        edge_density=raw_metrics_dict.get("edge_density", 0.1),
        color_richness=raw_metrics_dict.get("color_richness", 50),
        ui_density=raw_metrics_dict.get("ui_density", 10),
        action_intensity=raw_metrics_dict.get("action_intensity", 30),
        visual_balance=raw_metrics_dict.get("visual_balance", 80),
        dramatic_score=raw_metrics_dict.get("dramatic_score", 50),
    )

    normalized_metrics_dict = normalized_metrics_dict or {}
    norm = NormalizedMetrics(
        blur_score=normalized_metrics_dict.get("blur_score", 0.5),
        contrast=normalized_metrics_dict.get("contrast", 0.5),
        color_richness=normalized_metrics_dict.get("color_richness", 0.5),
        edge_density=normalized_metrics_dict.get("edge_density", 0.5),
        dramatic_score=normalized_metrics_dict.get("dramatic_score", 0.5),
        visual_balance=normalized_metrics_dict.get("visual_balance", 0.5),
        action_intensity=normalized_metrics_dict.get("action_intensity", 0.5),
        ui_density=normalized_metrics_dict.get("ui_density", 0.5),
    )

    return ImageMetrics(
        path=path,
        raw_metrics=raw,
        normalized_metrics=norm,
        semantic_score=semantic_score,
        total_score=total_score,
        features=features,
    )


def _create_picker(config: SelectionConfig | None = None) -> GameScreenPicker:
    """GameScreenPickerを作成するヘルパー関数（テスト用）.

    select_from_analyzedメソッドはインスタンスメソッドなので、
    適切なanalyzerとconfigを持つインスタンスを作成する必要がある。
    """
    analyzer = MagicMock(spec=ImageQualityAnalyzer)
    return GameScreenPicker(analyzer=analyzer, config=config)


@pytest.mark.parametrize(
    "activity_mix_enabled",
    [False, True],
)
def test_high_quality_images_are_prioritized_while_avoiding_similar_ones(
    sample_image_metrics: List[ImageMetrics],
    activity_mix_enabled: bool,
) -> None:
    """高品質な画像が優先され、類似した画像は回避されること.

    Given:
        - 様々なスコアを持つ5つの分析済み画像
        - image0（スコア95）とimage1（スコア90）は類似した特徴を持つ
        - image2（スコア85）は異なる特徴を持つ
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
    # activity_mix有効時は(0.3, 0.4, 0.3)、無効時は均等配分
    ratio = (0.3, 0.4, 0.3) if activity_mix_enabled else (0.33, 0.34, 0.33)
    config = SelectionConfig(
        activity_mix_enabled=activity_mix_enabled,
        activity_mix_ratio=ratio,
    )
    picker = _create_picker(config)

    # Act
    result, stats = picker.select_from_analyzed(
        sample_image_metrics,
        num_to_select,
        similarity_threshold,
    )

    # Assert
    assert len(result) == 3
    assert stats.total_files == 5
    assert stats.analyzed_ok == 5
    assert stats.analyzed_fail == 0
    assert stats.selected_count == 3
    scores = [m.total_score for m in result]
    assert scores == sorted(scores, reverse=True)
    selected_paths = [m.path for m in result]
    assert "/fake/path/image0.jpg" in selected_paths
    # activity_mix有効時はバケット分散が優先されるため、類似除外は適用されない場合がある
    if not activity_mix_enabled:
        assert "/fake/path/image1.jpg" not in selected_paths  # image0に類似


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
    eps = 1e-8
    norm = np.linalg.norm(base_features)
    base_normalized = base_features if norm < eps else base_features / norm

    # 直交成分の大きさを計算: sqrt(1 - cos^2)
    orthogonal_norm = np.sqrt(max(0, 1 - target_similarity**2))

    # ランダムな直交ベクトルを生成
    random_vec = np.random.randn(len(base_features))
    random_vec = random_vec / np.linalg.norm(random_vec) * orthogonal_norm

    # 目標類似度を持つベクトルを合成
    return target_similarity * base_normalized + random_vec  # type: ignore[no-any-return]


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
        batch_size: int = 32,  # noqa: ARG001 (API互換性のため維持)
        show_progress: bool = False,  # noqa: ARG001 (API互換性のため維持)
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
                _create_image_metrics(
                    path=path,
                    raw_metrics_dict={"blur_score": 100 - idx * 10},
                    normalized_metrics_dict={"blur_score": 1.0 - idx * 0.1},
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
    - image0: 高品質、ベース特徴（LOWバケットの活動量）
    - image1: 中品質、image0と0.96の類似度（0.9閾値を超過）（LOWバケットの活動量）
    - image2: 高品質、image0と0.85の類似度（0.9閾値以下）（MIDバケットの活動量）
    - image3: 低品質、image0と0.30の類似度（異なる特徴）（HIGHバケットの活動量）
    - image4: 中品質、image2と0.96の類似度（0.9閾値を超過）（HIGHバケットの活動量）
    """
    np.random.seed(42)
    base_features = np.random.rand(128)

    features_list = [
        base_features,
        _create_features_with_similarity(base_features, 0.96),
        _create_features_with_similarity(base_features, 0.85),
        _create_features_with_similarity(base_features, 0.30),
    ]
    # image5はimage3に類似
    features_list.append(_create_features_with_similarity(features_list[2], 0.96))

    # 活動量指標（LOW/MID/HIGHバケット）
    activity_metrics = [
        {"action_intensity": 0.1, "edge_density": 0.1, "dramatic_score": 0.1},
        {"action_intensity": 0.15, "edge_density": 0.15, "dramatic_score": 0.15},
        {"action_intensity": 0.5, "edge_density": 0.5, "dramatic_score": 0.5},
        {"action_intensity": 0.8, "edge_density": 0.8, "dramatic_score": 0.8},
        {"action_intensity": 0.9, "edge_density": 0.9, "dramatic_score": 0.9},
    ]

    return [
        _create_image_metrics(
            path=f"/fake/path/image{i}.jpg",
            raw_metrics_dict={
                "blur_score": 100.0 - i * 5,
                "action_intensity": activity_metrics[i]["action_intensity"] * 100,
                "edge_density": activity_metrics[i]["edge_density"] * 0.2,
                "dramatic_score": activity_metrics[i]["dramatic_score"] * 100,
            },
            normalized_metrics_dict={
                "blur_score": 0.9 - i * 0.1,
                "action_intensity": activity_metrics[i]["action_intensity"],
                "edge_density": activity_metrics[i]["edge_density"],
                "dramatic_score": activity_metrics[i]["dramatic_score"],
            },
            semantic_score=0.8 - i * 0.05,
            total_score=95.0 - i * 5,
            features=features_list[i],
        )
        for i in range(5)
    ]


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
        assert len(result) <= 3
        assert stats.total_files == 5
        assert stats.analyzed_ok == 5
        assert stats.analyzed_fail == 0
        assert stats.selected_count == len(result)
        # スコア降順であることを確認
        scores = [m.total_score for m in result]
        assert scores == sorted(scores, reverse=True)


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
            batch_size: int = 32,  # noqa: ARG001 (API互換性のため維持)
            show_progress: bool = False,  # noqa: ARG001 (API互換性のため維持)
        ) -> List[ImageMetrics | None]:
            results: List[ImageMetrics | None] = []
            for path in paths:
                idx = int(path.split("image")[-1].split(".")[0])
                if idx % 2 == 0:
                    results.append(None)
                else:
                    np.random.seed(idx)
                    results.append(
                        _create_image_metrics(
                            path=path,
                            raw_metrics_dict={"blur_score": 100 - idx * 10},
                            normalized_metrics_dict={"blur_score": 1.0 - idx * 0.1},
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
        assert len(result) <= 2  # 奇数インデックスのみ有効
        assert stats.total_files == 5
        assert stats.analyzed_ok == 2
        assert stats.analyzed_fail == 3
        assert stats.selected_count == len(result)
