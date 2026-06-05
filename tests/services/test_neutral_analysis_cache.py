"""NeutralAnalysisCacheの単体テスト."""

from pathlib import Path

import numpy as np

from src.services.neutral_analysis_cache import NeutralAnalysisCache
from tests.conftest import create_analyzed_image


def test_read_returns_none_when_image_path_disappears(tmp_path: Path) -> None:
    """画像pathが消えている場合はcache missとして扱われること.

    Arrange:
        - 存在しない画像pathと中立解析cacheがある
    Act:
        - cacheが読み込まれる
    Assert:
        - 例外ではなくNoneが返されること
    """
    # Arrange
    cache = NeutralAnalysisCache(tmp_path, analyzer_fingerprint="test")

    # Act
    result = cache.read(tmp_path / "missing.jpg")

    # Assert
    assert result is None


def test_write_many_stores_cache_without_pickle_payload(tmp_path: Path) -> None:
    """中立解析cacheがpickleではない形式で保存されること.

    Arrange:
        - 入力画像と中立解析結果がある
    Act:
        - cacheが保存される
    Assert:
        - cache fileはnpz形式で、読み込むと元の解析結果が返されること
    """
    # Arrange
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"\xff\xd8\xff")
    analyzed_image = create_analyzed_image(path=str(image_path))
    cache = NeutralAnalysisCache(tmp_path, analyzer_fingerprint="test")

    # Act
    cache.write_many([analyzed_image])
    restored = cache.read(image_path)

    # Assert
    cache_files = list((tmp_path / ".game-screen-pick").rglob("*"))
    assert any(path.suffix == ".npz" for path in cache_files)
    assert not any(path.suffix == ".pickle" for path in cache_files)
    assert restored is not None
    assert restored.path == analyzed_image.path
    assert restored.raw_metrics == analyzed_image.raw_metrics
    assert restored.normalized_metrics == analyzed_image.normalized_metrics
    assert restored.layout_heuristics == analyzed_image.layout_heuristics
    np.testing.assert_array_equal(restored.clip_features, analyzed_image.clip_features)
    np.testing.assert_array_equal(
        restored.combined_features,
        analyzed_image.combined_features,
    )
    np.testing.assert_array_equal(
        restored.content_features,
        analyzed_image.content_features,
    )
