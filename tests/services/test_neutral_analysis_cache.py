"""NeutralAnalysisCacheの単体テスト."""

from pathlib import Path

import numpy as np
import pytest

from src.models.analyzed_image import AnalyzedImage
from src.services.neutral_analysis_cache import NeutralAnalysisCache
from tests.conftest import create_analyzed_image


def _assert_restored_image_matches(
    restored: AnalyzedImage | None,
    expected: AnalyzedImage,
) -> None:
    """復元された中立解析結果が元の値と一致することを検証する."""
    assert restored is not None
    assert restored.path == expected.path
    assert restored.raw_metrics == expected.raw_metrics
    assert restored.normalized_metrics == expected.normalized_metrics
    assert restored.layout_heuristics == expected.layout_heuristics
    np.testing.assert_array_equal(restored.clip_features, expected.clip_features)
    np.testing.assert_array_equal(
        restored.combined_features,
        expected.combined_features,
    )
    np.testing.assert_array_equal(restored.content_features, expected.content_features)


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
    _assert_restored_image_matches(restored, analyzed_image)


def test_read_reuses_cache_when_same_path_is_spelled_differently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """同じ画像pathの表記が変わってもcacheが再利用されること.

    Arrange:
        - 相対pathで保存された中立解析cacheがある
    Act:
        - 同じ画像が絶対pathで読み込まれる
    Assert:
        - cacheが返され、復元後のpathは現在の指定pathになること
    """
    # Arrange
    monkeypatch.chdir(tmp_path)
    input_dir = Path("input")
    input_dir.mkdir()
    image_path = input_dir / "frame.jpg"
    image_path.write_bytes(b"\xff\xd8\xff")
    analyzed_image = create_analyzed_image(path=str(image_path))
    cache = NeutralAnalysisCache(input_dir, analyzer_fingerprint="test")
    cache.write_many([analyzed_image])

    # Act
    restored = NeutralAnalysisCache(
        input_dir.resolve(),
        analyzer_fingerprint="test",
    ).read(image_path.resolve())

    # Assert
    assert restored is not None
    assert restored.path == str(image_path.resolve())
    assert restored.raw_metrics == analyzed_image.raw_metrics
