"""BatchPipelineの単体テスト."""

import logging
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from src.analyzers.batch_pipeline import BatchPipeline
from src.analyzers.feature_extractor import FeatureExtractor
from src.analyzers.metric_calculator import MetricCalculator
from src.models.analyzer_config import AnalyzerConfig
from tests.fake_feature_extractor import FakeFeatureExtractor


def test_process_batch_logs_chunk_preparation_before_clip_extraction(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CLIP抽出前のチャンク準備状況が進捗ログに出力されること.

    Arrange:
        - 読み込み可能な画像ファイルがある
        - 進捗表示が有効である
    Act:
        - BatchPipelineで中立解析される
    Assert:
        - チャンク準備、画像読み込み、CLIP特徴抽出開始がログ出力されること
    """
    # Arrange
    image_path = tmp_path / "frame1.jpg"
    Image.new("RGB", (16, 16), color=(120, 120, 120)).save(image_path)
    config = AnalyzerConfig(
        max_dim=16,
        min_chunk_size=1,
        result_max_workers=1,
        io_max_workers=1,
    )
    pipeline = BatchPipeline(
        cast(FeatureExtractor, FakeFeatureExtractor()),
        MetricCalculator(config),
        config,
    )
    caplog.set_level(logging.INFO)

    # Act
    try:
        pipeline.process_batch([str(image_path)], show_progress=True)
    finally:
        pipeline.close()

    # Assert
    assert "中立解析を開始します: 1件" in caplog.text
    assert "解析チャンク: 1個" in caplog.text
    assert "画像読み込みを開始します: chunk 1/1" in caplog.text
    assert "CLIP特徴抽出を開始します: chunk 1/1" in caplog.text
