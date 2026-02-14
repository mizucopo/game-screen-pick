"""GameScreenPickerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. 「What」（観察可能な挙動）をテスト
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

from src.models.image_metrics import ImageMetrics
from src.models.selection_config import SelectionConfig
from src.services.game_screen_picker import GameScreenPicker
from tests.conftest import create_image_metrics


def _create_picker(config: SelectionConfig | None = None) -> GameScreenPicker:
    """GameScreenPickerを作成するヘルパー関数（テスト用）."""
    from src.analyzers.image_quality_analyzer import ImageQualityAnalyzer

    analyzer = MagicMock(spec=ImageQualityAnalyzer)
    return GameScreenPicker(analyzer=analyzer, config=config)


def _create_sample_metrics(count: int) -> List[ImageMetrics]:
    """テスト用のサンプルImageMetricsを作成する."""
    metrics = []
    for i in range(count):
        np.random.seed(i)
        features = np.random.rand(128)
        metrics.append(
            create_image_metrics(
                path=f"/fake/path/image{i}.jpg",
                raw_metrics_dict={"blur_score": 100.0 - i * 10},
                normalized_metrics_dict={"blur_score": 1.0 - i * 0.1},
                semantic_score=0.8,
                total_score=100.0 - i * 10,
                features=features,
            )
        )
    return metrics


@pytest.mark.parametrize(
    "activity_mix_enabled",
    [False, True],
)
def test_select_from_analyzed_returns_diverse_images(
    activity_mix_enabled: bool,
) -> None:
    """分析済み画像から多様な画像が正しく選択されること.

    Given:
        - 様々なスコアを持つ5つの分析済み画像
    When:
        - 3つの画像を選択
    Then:
        - 要求された数の画像が返されること
        - 統計情報が正しく記録されていること
    """
    # Arrange
    sample_metrics = _create_sample_metrics(5)
    num_to_select = 3
    similarity_threshold = 0.9
    ratio = (0.3, 0.4, 0.3) if activity_mix_enabled else (0.33, 0.34, 0.33)
    config = SelectionConfig(
        activity_mix_enabled=activity_mix_enabled,
        activity_mix_ratio=ratio,
    )
    picker = _create_picker(config)

    # Act
    result, stats = picker.select_from_analyzed(
        sample_metrics,
        num_to_select,
        similarity_threshold,
    )

    # Assert
    assert len(result) == 3
    assert stats.total_files == 5
    assert stats.analyzed_ok == 5
    assert stats.selected_count == 3


def test_select_from_folder_processes_images_and_handles_failures() -> None:
    """フォルダから画像をロード・分析し、失敗時も適切に処理すること.

    Given:
        - 5つの画像ファイルを持つフォルダ
        - アナライザは一部のファイルに対してNoneを返す
    When:
        - 画像を選択
    Then:
        - 処理が継続され、有効な画像のみが返されること
        - 統計情報が正しく記録されていること
    """
    # Arrange
    from src.analyzers.image_quality_analyzer import ImageQualityAnalyzer

    mock_analyzer = MagicMock(spec=ImageQualityAnalyzer)

    def mock_analyze_batch(
        paths: List[str],
        batch_size: int = 32,  # noqa: ARG001
        show_progress: bool = False,  # noqa: ARG001
    ) -> List[ImageMetrics | None]:
        results: List[ImageMetrics | None] = []
        for path in paths:
            try:
                idx = int(path.split("image")[-1].split(".")[0])
            except (ValueError, IndexError):
                idx = 0
            # 偶数インデックスは失敗とする
            if idx % 2 == 0:
                results.append(None)
            else:
                np.random.seed(idx)
                results.append(
                    create_image_metrics(
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

    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(5):
            Path(temp_dir, f"image{i}.jpg").touch()

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
