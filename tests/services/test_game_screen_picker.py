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
from src.models.picker_statistics import PickerStatistics
from src.models.selection_config import SelectionConfig
from src.services.game_screen_picker import GameScreenPicker


def _create_image_metrics(
    path: str,
    raw_metrics: RawMetrics,
    normalized_metrics: NormalizedMetrics,
    semantic_score: float,
    total_score: float,
    features: np.ndarray,
) -> ImageMetrics:
    """ImageMetricsを作成するヘルパー関数."""
    return ImageMetrics(
        path=path,
        raw_metrics=raw_metrics,
        normalized_metrics=normalized_metrics,
        semantic_score=semantic_score,
        total_score=total_score,
        features=features,
    )


def _create_image_metrics_from_dict(
    path: str,
    raw_metrics_dict: dict[str, float],
    normalized_metrics_dict: dict[str, float],
    semantic_score: float,
    total_score: float,
    features: np.ndarray,
) -> ImageMetrics:
    """辞書からImageMetricsを作成するヘルパー関数（テスト用）."""
    # RawMetricsのデフォルト値を取得
    blur_raw = raw_metrics_dict.get("blur_score", 0)
    brightness_raw = raw_metrics_dict.get(
        "brightness", raw_metrics_dict.get("blur_score", 100)
    )
    contrast_raw = raw_metrics_dict.get("contrast", 50)
    edge_density_raw = raw_metrics_dict.get("edge_density", 0.1)
    color_richness_raw = raw_metrics_dict.get("color_richness", 50)
    ui_density_raw = raw_metrics_dict.get("ui_density", 10)
    action_intensity_raw = raw_metrics_dict.get("action_intensity", 30)
    visual_balance_raw = raw_metrics_dict.get("visual_balance", 80)
    dramatic_raw = raw_metrics_dict.get(
        "dramatic_score", raw_metrics_dict.get("dramatic_score", 50)
    )

    raw = RawMetrics(
        blur_score=blur_raw,
        brightness=brightness_raw,
        contrast=contrast_raw,
        edge_density=edge_density_raw,
        color_richness=color_richness_raw,
        ui_density=ui_density_raw,
        action_intensity=action_intensity_raw,
        visual_balance=visual_balance_raw,
        dramatic_score=dramatic_raw,
    )

    # NormalizedMetricsのデフォルト値を取得
    blur_norm = normalized_metrics_dict.get("blur_score", 0.5)
    contrast_norm = normalized_metrics_dict.get("contrast", 0.5)
    color_richness_norm = normalized_metrics_dict.get("color_richness", 0.5)
    edge_density_norm = normalized_metrics_dict.get("edge_density", 0.5)
    dramatic_norm = normalized_metrics_dict.get(
        "dramatic_score", normalized_metrics_dict.get("dramatic_score", 0.5)
    )
    visual_balance_norm = normalized_metrics_dict.get("visual_balance", 0.5)
    action_intensity_norm = normalized_metrics_dict.get("action_intensity", 0.5)
    ui_density_norm = normalized_metrics_dict.get("ui_density", 0.5)

    norm = NormalizedMetrics(
        blur_score=blur_norm,
        contrast=contrast_norm,
        color_richness=color_richness_norm,
        edge_density=edge_density_norm,
        dramatic_score=dramatic_norm,
        visual_balance=visual_balance_norm,
        action_intensity=action_intensity_norm,
        ui_density=ui_density_norm,
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


def _assert_stats_valid(
    stats: PickerStatistics,
    expected_total: int,
    expected_ok: int,
    expected_fail: int,
    expected_selected: int,
) -> None:
    """統計情報が期待値と一致することを検証するヘルパー関数."""
    assert stats.total_files == expected_total
    assert stats.analyzed_ok == expected_ok
    assert stats.analyzed_fail == expected_fail
    assert stats.selected_count == expected_selected


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
                _create_image_metrics_from_dict(
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
    - image1: 高品質、ベース特徴（LOWバケットの活動量）
    - image2: 中品質、image1と0.96の類似度（0.9閾値を超過）（LOWバケットの活動量）
    - image3: 高品質、image1と0.85の類似度（0.9閾値以下）（MIDバケットの活動量）
    - image4: 低品質、image1と0.30の類似度（異なる特徴）（HIGHバケットの活動量）
    - image5: 中品質、image3と0.96の類似度（0.9閾値を超過）（HIGHバケットの活動量）

    活動量指標の設計：
    - 画像0-1: LOWバケット（活動量低め）
      action_intensity 0.1-0.15, edge_density 0.1-0.15,
      dramatic_score 0.1-0.15
    - 画像2: MIDバケット（活動量中程度）
      action_intensity 0.5, edge_density 0.5, dramatic_score 0.5
    - 画像3-4: HIGHバケット（活動量高め）
      action_intensity 0.8-0.9, edge_density 0.8-0.9,
      dramatic_score 0.8-0.9
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
        _create_image_metrics_from_dict(
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


@pytest.fixture
def large_sample_image_metrics() -> List[ImageMetrics]:
    """活動量ミックスのテスト用に大きなサンプルImageMetricsを作成する.

    活動量ミックスはnum*3枚の候補を要求するため、除外を発生させるにはnum*3より多くの画像が必要。
    num=5の場合、候補要求数は15枚なので、それ以上の画像（20枚）を用意し、類似グループを含めることで
    除外が発生することを確認する。

    画像セット：
    - image0-4: 類似グループ1（高品質、LOWバケットの活動量）
    - image5-9: 類似グループ2（中品質、MIDバケットの活動量）
    - image10-14: 異なる特徴（HIGHバケットの活動量）
    - image15-19: さらに異なる特徴（HIGHバケットの活動量）

    活動量指標の設計：
    - 画像0-6: LOWバケット（活動量低め）
      action_intensity 0.1-0.2, edge_density 0.1-0.2,
      dramatic_score 0.1-0.2
    - 画像7-13: MIDバケット（活動量中程度）
      action_intensity 0.4-0.6, edge_density 0.4-0.6,
      dramatic_score 0.4-0.6
    - 画像14-19: HIGHバケット（活動量高め）
      action_intensity 0.8-0.9, edge_density 0.8-0.9,
      dramatic_score 0.8-0.9
    """
    np.random.seed(42)
    base_features1 = np.random.rand(128)
    base_features2 = np.random.rand(128)
    base_features3 = np.random.rand(128)
    base_features4 = np.random.rand(128)

    features_list = [
        # image0-4: 類似グループ1（高品質、LOWバケット）
        base_features1,
        _create_features_with_similarity(base_features1, 0.96),
        _create_features_with_similarity(base_features1, 0.96),
        _create_features_with_similarity(base_features1, 0.96),
        _create_features_with_similarity(base_features1, 0.96),
        # image5-9: 類似グループ2（中品質、LOWバケット）
        base_features2,
        _create_features_with_similarity(base_features2, 0.96),
        _create_features_with_similarity(base_features2, 0.96),
        _create_features_with_similarity(base_features2, 0.96),
        _create_features_with_similarity(base_features2, 0.96),
        # image10-14: 異なる特徴（LOW/MIDバケット）
        base_features3,
        _create_features_with_similarity(base_features3, 0.85),
        _create_features_with_similarity(base_features3, 0.85),
        _create_features_with_similarity(base_features3, 0.85),
        _create_features_with_similarity(base_features3, 0.85),
        _create_features_with_similarity(base_features3, 0.85),
        # image15-19: さらに異なる特徴（MID/HIGHバケット）
        base_features4,
        _create_features_with_similarity(base_features4, 0.80),
        _create_features_with_similarity(base_features4, 0.80),
        _create_features_with_similarity(base_features4, 0.80),
        _create_features_with_similarity(base_features4, 0.80),
        _create_features_with_similarity(base_features4, 0.80),
    ]

    # 3つのバケット（LOW/MID/HIGH）に分散する活動量指標を設定
    # LOWバケット（画像0-6）: 活動量低め
    low_metrics = [
        {"action_intensity": 0.1, "edge_density": 0.1, "dramatic_score": 0.1},
        {"action_intensity": 0.12, "edge_density": 0.12, "dramatic_score": 0.12},
        {"action_intensity": 0.14, "edge_density": 0.14, "dramatic_score": 0.14},
        {"action_intensity": 0.16, "edge_density": 0.16, "dramatic_score": 0.16},
        {"action_intensity": 0.18, "edge_density": 0.18, "dramatic_score": 0.18},
        {"action_intensity": 0.2, "edge_density": 0.2, "dramatic_score": 0.2},
        {"action_intensity": 0.22, "edge_density": 0.22, "dramatic_score": 0.22},
    ]

    # MIDバケット（画像7-13）: 活動量中程度
    mid_metrics = [
        {"action_intensity": 0.4, "edge_density": 0.4, "dramatic_score": 0.4},
        {"action_intensity": 0.45, "edge_density": 0.45, "dramatic_score": 0.45},
        {"action_intensity": 0.5, "edge_density": 0.5, "dramatic_score": 0.5},
        {"action_intensity": 0.55, "edge_density": 0.55, "dramatic_score": 0.55},
        {"action_intensity": 0.6, "edge_density": 0.6, "dramatic_score": 0.6},
        {"action_intensity": 0.5, "edge_density": 0.5, "dramatic_score": 0.5},
        {"action_intensity": 0.55, "edge_density": 0.55, "dramatic_score": 0.55},
    ]

    # HIGHバケット（画像14-19）: 活動量高め
    high_metrics = [
        {"action_intensity": 0.8, "edge_density": 0.8, "dramatic_score": 0.8},
        {"action_intensity": 0.82, "edge_density": 0.82, "dramatic_score": 0.82},
        {"action_intensity": 0.85, "edge_density": 0.85, "dramatic_score": 0.85},
        {"action_intensity": 0.88, "edge_density": 0.88, "dramatic_score": 0.88},
        {"action_intensity": 0.9, "edge_density": 0.9, "dramatic_score": 0.9},
        {"action_intensity": 0.85, "edge_density": 0.85, "dramatic_score": 0.85},
    ]

    activity_metrics = low_metrics + mid_metrics + high_metrics

    return [
        _create_image_metrics_from_dict(
            path=f"/fake/path/image{i}.jpg",
            raw_metrics_dict={
                "blur_score": 100.0 - i * 1.5,
                "action_intensity": activity_metrics[i]["action_intensity"] * 100,
                "edge_density": activity_metrics[i]["edge_density"] * 0.2,
                "dramatic_score": activity_metrics[i]["dramatic_score"] * 100,
            },
            normalized_metrics_dict={
                "blur_score": 0.9 - i * 0.02,
                "action_intensity": activity_metrics[i]["action_intensity"],
                "edge_density": activity_metrics[i]["edge_density"],
                "dramatic_score": activity_metrics[i]["dramatic_score"],
            },
            semantic_score=0.8 - i * 0.01,
            total_score=95.0 - i * 1.5,
            features=features_list[i],
        )
        for i in range(20)
    ]


def test_high_quality_images_are_prioritized_while_avoiding_similar_ones(
    sample_image_metrics: List[ImageMetrics],
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
    picker = _create_picker(SelectionConfig(activity_mix_enabled=False))

    # Act
    result, stats = picker.select_from_analyzed(
        sample_image_metrics,
        num_to_select,
        similarity_threshold,
    )

    # Assert
    assert len(result) == 3
    _assert_stats_valid(
        stats,
        expected_total=5,
        expected_ok=5,
        expected_fail=0,
        expected_selected=3,
    )
    scores = [m.total_score for m in result]
    assert scores == sorted(scores, reverse=True)
    selected_paths = [m.path for m in result]
    assert "/fake/path/image0.jpg" in selected_paths
    assert "/fake/path/image1.jpg" not in selected_paths  # image0に類似


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
        _assert_stats_valid(
            stats,
            expected_total=5,
            expected_ok=5,
            expected_fail=0,
            expected_selected=len(result),
        )
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
            batch_size: int = 32,  # noqa: ARG001
            show_progress: bool = False,  # noqa: ARG001
        ) -> List[ImageMetrics | None]:
            results: List[ImageMetrics | None] = []
            for path in paths:
                idx = int(path.split("image")[-1].split(".")[0])
                if idx % 2 == 0:
                    results.append(None)
                else:
                    np.random.seed(idx)
                    results.append(
                        _create_image_metrics_from_dict(
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
        _assert_stats_valid(
            stats,
            expected_total=5,
            expected_ok=2,
            expected_fail=3,
            expected_selected=len(result),
        )


@pytest.fixture
def similar_images_metrics() -> List[ImageMetrics]:
    """非常に類似した画像セットを作成するfixture（0.97-0.99の類似度）.

    10枚の非常に類似した画像を返します。
    類似度が高いため、しきい値緩和のテストに使用されます。
    """
    np.random.seed(42)
    base_features = np.random.rand(128)

    return [
        _create_image_metrics_from_dict(
            path=f"/fake/path/similar{i}.jpg",
            raw_metrics_dict={"blur_score": 100.0 - i},
            normalized_metrics_dict={"blur_score": 0.9},
            semantic_score=0.8,
            total_score=100.0 - i,
            features=_create_features_with_similarity(
                base_features, 0.97 + (i % 3) * 0.01
            ),
        )
        for i in range(10)
    ]


@pytest.fixture
def highly_similar_images_for_rejection_test() -> List[ImageMetrics]:
    """類似度除外が確実に発生する画像セットを作成するfixture.

    0.99の類似度を持つ20枚の画像（1つのベース特徴から生成）を返します。
    最終しきい値（max_threshold=0.98）を超える類似度を持つため、
    候補要求数より多い場合に除外が確実に発生します。

    活動量指標としてMIDバケットの値を設定（活動量ミックスのテストで使用）。
    """
    np.random.seed(42)
    base_features = np.random.rand(128)

    # 0.99の類似度を持つ20枚の画像（最終しきい値0.98を超過）
    return [
        _create_image_metrics_from_dict(
            path=f"/fake/path/highly_similar{i}.jpg",
            raw_metrics_dict={
                "blur_score": 100.0 - i * 0.5,
                "action_intensity": 50.0,
                "edge_density": 0.1,
                "dramatic_score": 50.0,
            },
            normalized_metrics_dict={
                "blur_score": 0.9 - i * 0.01,
                "action_intensity": 0.5,
                "edge_density": 0.5,
                "dramatic_score": 0.5,
            },
            semantic_score=0.8,
            total_score=100.0 - i * 0.5,
            features=_create_features_with_similarity(base_features, 0.99),
        )
        for i in range(20)
    ]


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
    picker = _create_picker(SelectionConfig(activity_mix_enabled=False))

    # Act
    result, stats = picker.select_from_analyzed(
        similar_images_metrics,
        num_to_select,
        similarity_threshold,
    )

    # Assert
    assert len(result) == num_to_select
    _assert_stats_valid(
        stats,
        expected_total=10,
        expected_ok=10,
        expected_fail=0,
        expected_selected=num_to_select,
    )
    scores = [m.total_score for m in result]
    assert scores == sorted(scores, reverse=True)


def test_select_from_analyzed_with_activity_mix_enabled_tracks_similarity_rejections(
    highly_similar_images_for_rejection_test: List[ImageMetrics],
) -> None:
    """活動量ミックス有効時、類似度による除外数が正しく記録されること.

    Given:
        - 20枚の非常に類似した画像（0.99の類似度）
        - 最終しきい値（max_threshold=0.98）を超える類似度
        - 活動量ミックスが有効な設定
    When:
        - select_from_analyzedで画像を選択
    Then:
        - 類似度により一部の候補が除外されること
        - 統計情報のrejected_by_similarity > 0 であること（回帰検知）
        - 期待枚数の画像が選択されること
    """
    # Arrange
    num_to_select = 5  # 5*3=15枚の候補を要求
    similarity_threshold = 0.9
    picker = _create_picker(SelectionConfig(activity_mix_enabled=True))

    # Act
    result, stats = picker.select_from_analyzed(
        highly_similar_images_for_rejection_test,
        num_to_select,
        similarity_threshold,
    )

    # Assert
    assert len(result) == num_to_select
    _assert_stats_valid(
        stats,
        expected_total=20,
        expected_ok=20,
        expected_fail=0,
        expected_selected=num_to_select,
    )
    # 0.99の類似度を持つ20枚の画像から15枚の候補を要求する場合、
    # 最終しきい値0.98で類似判定が行われるため、除外が確実に発生する。
    # これによりrejected_by_similarityの回帰検知が可能になる。
    assert stats.rejected_by_similarity > 0, (
        f"類似度による除外が発生していません。"
        f" rejected_by_similarity={stats.rejected_by_similarity}, expected > 0"
    )


def test_select_from_analyzed_with_activity_mix_returns_diverse_selection(
    sample_image_metrics: List[ImageMetrics],
) -> None:
    """活動量ミックス有効時、異なる活動量バケットから画像が選択されること.

    Given:
        - 5つの分析済み画像（LOW/MID/HIGHバケットの活動量を持つ）
        - 活動量ミックスが有効な設定
    When:
        - 画像を選択
    Then:
        - 選択された画像が返されること
        - 統計情報が正しく記録されること
        - 複数の活動量バケットから選択されていること（活動量ミックスの効果）
    """
    # Arrange
    num_to_select = 3
    similarity_threshold = 0.9
    picker = _create_picker(
        SelectionConfig(activity_mix_enabled=True, activity_mix_ratio=(0.3, 0.4, 0.3))
    )

    # Act
    result, stats = picker.select_from_analyzed(
        sample_image_metrics,
        num_to_select,
        similarity_threshold,
    )

    # Assert
    assert len(result) == num_to_select
    # 活動量ミックスではnum*3枚の候補を要求するが、sample_image_metricsは5枚のみのため、
    # 候補要求数を下回り類似度フィルタリングが発動しない。
    # したがってrejected_by_similarityは0になることが期待される（回帰検知）
    assert stats.rejected_by_similarity == 0, (
        f"候補数不足により除外は発生しないはずですが、"
        f" rejected_by_similarity={stats.rejected_by_similarity}"
    )
    # スコア降順であることを確認
    scores = [m.total_score for m in result]
    assert scores == sorted(scores, reverse=True)

    # 活動量スコアを計算して、複数のバケットから選ばれていることを確認
    # sample_image_metricsの設計:
    # - 画像0-1: LOWバケット (action_intensity 0.1-0.15)
    # - 画像2: MIDバケット (action_intensity 0.5)
    # - 画像3-4: HIGHバケット (action_intensity 0.8-0.9)
    from src.constants.score_weights import ScoreWeights

    activity_weights = ScoreWeights.get_activity_weights()
    activity_scores = [
        (
            activity_weights.get("action_intensity", 0.55)
            * m.normalized_metrics.get("action_intensity", 0)
            + activity_weights.get("edge_density", 0.25)
            * m.normalized_metrics.get("edge_density", 0)
            + activity_weights.get("dramatic_score", 0.20)
            * m.normalized_metrics.get("dramatic_score", 0)
        )
        for m in result
    ]

    # 選択された画像の活動量スコアに十分な分散があることを確認
    # LOWバケット(0.1-0.2)とHIGHバケット(0.8-0.9)の範囲が含まれているか
    assert min(activity_scores) < 0.3, "LOWバケットの画像が選択されている必要があります"
    assert max(activity_scores) > 0.7, (
        "HIGHバケットの画像が選択されている必要があります"
    )
