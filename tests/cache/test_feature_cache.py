"""FeatureCacheのユニットテスト."""

import tempfile
from pathlib import Path

import numpy as np

from src.cache.feature_cache import FeatureCache


def test_init_creates_database_schema() -> None:
    """データベーススキーマが正しく作成されること.

    Given:
        - インメモリデータベースでFeatureCacheを初期化
    When:
        - データベーススキーマを確認
    Then:
        - テーブルとインデックスが正しく作成されていること
    """
    # Arrange & Act
    with FeatureCache(None) as cache:
        conn = cache._get_connection()
        cursor = conn.cursor()

        # テーブルが存在することを確認
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='feature_cache'"
        )
        table = cursor.fetchone()
        assert table is not None

        # インデックスが存在することを確認
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND name='idx_file_signature'"
        )
        index = cursor.fetchone()
        assert index is not None


def test_put_and_get_roundtrip() -> None:
    """キャッシュへの保存と取得が正しく動作すること.

    Given:
        - テストデータ（特徴量とメトリクス）を作成
    When:
        - キャッシュに保存してから取得
    Then:
        - 保存したデータが正しく取得できること
    """
    # Arrange
    clip_features = np.random.randn(512).astype(np.float32)
    hsv_features = np.random.randn(64).astype(np.float32)
    raw_metrics = {
        "blur_score": 100.0,
        "brightness": 128.0,
        "contrast": 50.0,
        "edge_density": 0.1,
        "color_richness": 60.0,
        "ui_density": 0.05,
        "action_intensity": 40.0,
        "visual_balance": 80.0,
        "dramatic_score": 50.0,
    }
    cache_key: dict[str, str | int] = {
        "absolute_path": "/test/image.jpg",
        "file_size": 1024,
        "mtime_ns": 1234567890,
        "model_name": "openai/clip-vit-base-patch32",
        "target_text": "epic game scenery",
        "max_dim": 1280,
        "metrics_version": "1",
    }

    with FeatureCache(None) as cache:
        # Act: 保存
        cache.put(
            cache_key=cache_key,
            clip_features=clip_features,
            raw_metrics=raw_metrics,
            hsv_features=hsv_features,
        )

        # Act: 取得
        result = cache.get(cache_key)

        # Assert
        assert result is not None
        np.testing.assert_array_almost_equal(result.clip_features, clip_features)
        np.testing.assert_array_almost_equal(result.hsv_features, hsv_features)
        assert result.raw_metrics == raw_metrics


def test_get_returns_none_for_nonexistent_key() -> None:
    """存在しないキーで取得するとNoneが返されること.

    Given:
        - 存在しないキーを作成
    When:
        - キャッシュから取得
    Then:
        - Noneが返されること
    """
    # Arrange
    cache_key: dict[str, str | int] = {
        "absolute_path": "/nonexistent/image.jpg",
        "file_size": 999,
        "mtime_ns": 999,
        "model_name": "test_model",
        "target_text": "test",
        "max_dim": 1280,
        "metrics_version": "1",
    }

    with FeatureCache(None) as cache:
        # Act
        result = cache.get(cache_key)

        # Assert
        assert result is None


def test_get_returns_none_when_file_size_changed() -> None:
    """ファイルサイズが変更されるとキャッシュミスすること.

    Given:
        - キャッシュに保存した後、file_sizeを変更したキーを作成
    When:
        - 変更後のキーでキャッシュから取得
    Then:
        - Noneが返されること（ファイル変更検出）
    """
    # Arrange
    clip_features = np.random.randn(512).astype(np.float32)
    hsv_features = np.random.randn(64).astype(np.float32)
    raw_metrics = {"blur_score": 100.0}
    original_key: dict[str, str | int] = {
        "absolute_path": "/test/image.jpg",
        "file_size": 1024,
        "mtime_ns": 1234567890,
        "model_name": "openai/clip-vit-base-patch32",
        "target_text": "epic game scenery",
        "max_dim": 1280,
        "metrics_version": "1",
    }

    with FeatureCache(None) as cache:
        cache.put(
            cache_key=original_key,
            clip_features=clip_features,
            raw_metrics=raw_metrics,
            hsv_features=hsv_features,
        )

        # Act: file_sizeを変更して取得
        modified_key = original_key.copy()
        modified_key["file_size"] = 2048
        result = cache.get(modified_key)

        # Assert
        assert result is None


def test_put_replaces_existing_entry() -> None:
    """同じキーで保存すると上書きされること.

    Given:
        - キャッシュにエントリを保存
    When:
        - 同じキーで異なるデータを保存
    Then:
        - 上書きされたデータが取得できること
    """
    # Arrange
    original_features = np.ones(512, dtype=np.float32)
    updated_features = np.zeros(512, dtype=np.float32)
    raw_metrics = {"blur_score": 100.0}
    hsv_features = np.ones(64, dtype=np.float32)
    cache_key: dict[str, str | int] = {
        "absolute_path": "/test/image.jpg",
        "file_size": 1024,
        "mtime_ns": 1234567890,
        "model_name": "openai/clip-vit-base-patch32",
        "target_text": "epic game scenery",
        "max_dim": 1280,
        "metrics_version": "1",
    }

    with FeatureCache(None) as cache:
        # 初期データを保存
        cache.put(
            cache_key=cache_key,
            clip_features=original_features,
            raw_metrics=raw_metrics,
            hsv_features=hsv_features,
        )

        # Act: 同じキーで更新
        cache.put(
            cache_key=cache_key,
            clip_features=updated_features,
            raw_metrics={"blur_score": 200.0},
            hsv_features=hsv_features,
        )

        # Assert: 更新後のデータが取得できる
        result = cache.get(cache_key)
        assert result is not None
        np.testing.assert_array_equal(result.clip_features, updated_features)
        assert result.raw_metrics["blur_score"] == 200.0


def test_persistent_storage() -> None:
    """キャッシュデータが永続化されること.

    Given:
        - 一時ファイルでキャッシュを作成し、データを保存
    When:
        - 接続を閉じてから再度開いて取得
    Then:
        - 保存したデータが永続化されていること
    """
    # Arrange
    clip_features = np.random.randn(512).astype(np.float32)
    hsv_features = np.random.randn(64).astype(np.float32)
    raw_metrics = {"blur_score": 100.0}
    cache_key: dict[str, str | int] = {
        "absolute_path": "/test/persistent.jpg",
        "file_size": 2048,
        "mtime_ns": 9876543210,
        "model_name": "openai/clip-vit-base-patch32",
        "target_text": "epic game scenery",
        "max_dim": 1280,
        "metrics_version": "1",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.sqlite3"

        # 最初のセッションで保存
        with FeatureCache(cache_path) as cache1:
            cache1.put(
                cache_key=cache_key,
                clip_features=clip_features,
                raw_metrics=raw_metrics,
                hsv_features=hsv_features,
            )

        # Act: 別セッションで取得
        with FeatureCache(cache_path) as cache2:
            result = cache2.get(cache_key)

        # Assert
        assert result is not None
        np.testing.assert_array_almost_equal(result.clip_features, clip_features)


def test_generate_cache_key() -> None:
    """キャッシュキーが正しく生成されること.

    Given:
        - FeatureCacheインスタンスを作成
    When:
        - generate_cache_keyを呼び出し
    Then:
        - 正しい構造のキーが生成されること
    """
    # Arrange & Act
    with FeatureCache(None) as cache:
        cache_key = cache.generate_cache_key(
            absolute_path="/test/image.jpg",
            file_size=1024,
            mtime_ns=1234567890,
            model_name="test_model",
            target_text="test target",
            max_dim=1920,
        )

    # Assert
    assert cache_key["absolute_path"] == "/test/image.jpg"
    assert cache_key["file_size"] == 1024
    assert cache_key["mtime_ns"] == 1234567890
    assert cache_key["model_name"] == "test_model"
    assert cache_key["target_text"] == "test target"
    assert cache_key["max_dim"] == 1920
    assert cache_key["metrics_version"] == FeatureCache.METRICS_VERSION
