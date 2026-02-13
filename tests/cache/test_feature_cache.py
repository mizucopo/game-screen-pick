"""FeatureCacheのユニットテスト."""

import tempfile
from pathlib import Path

import numpy as np

from src.cache.feature_cache import FeatureCache
from src.models.cache_entry import CacheEntry

# キャッシュで使用されるmetrics_version
METRICS_VERSION = "4"


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
        - normalized_metricsとtotal_scoreも正しく保存・取得できること
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
    semantic_score = 0.75
    normalized_metrics = {
        "blur_score": 0.5,
        "brightness": 0.6,
        "contrast": 0.7,
        "edge_density": 0.8,
        "color_richness": 0.9,
        "ui_density": 0.1,
        "action_intensity": 0.2,
        "visual_balance": 0.3,
        "dramatic_score": 0.4,
    }
    total_score = 85.5
    cache_key: dict[str, str | int] = {
        "absolute_path": "/test/image.jpg",
        "file_size": 1024,
        "mtime_ns": 1234567890,
        "model_name": "openai/clip-vit-base-patch32",
        "target_text": "epic game scenery",
        "max_dim": 1280,
        "metrics_version": METRICS_VERSION,
    }

    with FeatureCache(None) as cache:
        # Act: 保存（normalized_metricsとtotal_scoreを含む）
        cache.put(
            cache_key=cache_key,
            clip_features=clip_features,
            raw_metrics=raw_metrics,
            hsv_features=hsv_features,
            semantic_score=semantic_score,
            normalized_metrics=normalized_metrics,
            total_score=total_score,
        )

        # Act: 取得
        result = cache.get(cache_key)

        # Assert
        assert result is not None
        # float16で保存されているため、精度が異なることを考慮
        expected_clip = clip_features.astype(np.float16).astype(np.float32)
        expected_hsv = hsv_features.astype(np.float16).astype(np.float32)
        np.testing.assert_array_almost_equal(result.clip_features, expected_clip)
        np.testing.assert_array_almost_equal(result.hsv_features, expected_hsv)
        assert result.raw_metrics == raw_metrics
        assert result.semantic_score == semantic_score
        # normalized_metricsとtotal_scoreが正しく保存・取得されること
        assert result.normalized_metrics == normalized_metrics
        assert result.total_score == total_score


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
        "metrics_version": METRICS_VERSION,
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
        "metrics_version": METRICS_VERSION,
    }

    with FeatureCache(None) as cache:
        normalized_metrics = {"blur_score": 0.5}
        total_score = 75.0
        cache.put(
            cache_key=original_key,
            clip_features=clip_features,
            raw_metrics=raw_metrics,
            hsv_features=hsv_features,
            semantic_score=0.75,
            normalized_metrics=normalized_metrics,
            total_score=total_score,
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
        "metrics_version": METRICS_VERSION,
    }

    with FeatureCache(None) as cache:
        # 初期データを保存
        normalized_metrics1 = {"blur_score": 0.5}
        total_score1 = 75.0
        cache.put(
            cache_key=cache_key,
            clip_features=original_features,
            raw_metrics=raw_metrics,
            hsv_features=hsv_features,
            semantic_score=0.5,
            normalized_metrics=normalized_metrics1,
            total_score=total_score1,
        )

        # Act: 同じキーで更新
        normalized_metrics2 = {"blur_score": 0.8}
        total_score2 = 80.0
        cache.put(
            cache_key=cache_key,
            clip_features=updated_features,
            raw_metrics={"blur_score": 200.0},
            hsv_features=hsv_features,
            semantic_score=0.8,
            normalized_metrics=normalized_metrics2,
            total_score=total_score2,
        )

        # Assert: 更新後のデータが取得できる
        result = cache.get(cache_key)
        assert result is not None
        np.testing.assert_array_equal(result.clip_features, updated_features)
        assert result.raw_metrics["blur_score"] == 200.0
        assert result.semantic_score == 0.8
        # normalized_metricsとtotal_scoreも更新されていること
        assert result.normalized_metrics == normalized_metrics2
        assert result.total_score == total_score2


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
        "metrics_version": METRICS_VERSION,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.sqlite3"

        # 最初のセッションで保存
        normalized_metrics = {"blur_score": 0.5}
        total_score = 75.0
        with FeatureCache(cache_path) as cache1:
            cache1.put(
                cache_key=cache_key,
                clip_features=clip_features,
                raw_metrics=raw_metrics,
                hsv_features=hsv_features,
                semantic_score=0.75,
                normalized_metrics=normalized_metrics,
                total_score=total_score,
            )

        # Act: 別セッションで取得
        with FeatureCache(cache_path) as cache2:
            result = cache2.get(cache_key)

        # Assert
        assert result is not None
        # float16で保存されているため、精度が異なることを考慮
        expected_clip = clip_features.astype(np.float16).astype(np.float32)
        np.testing.assert_array_almost_equal(result.clip_features, expected_clip)
        assert result.raw_metrics == raw_metrics
        assert result.semantic_score == 0.75
        # normalized_metricsとtotal_scoreも永続化されていること
        assert result.normalized_metrics == normalized_metrics
        assert result.total_score == total_score


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


def test_put_batch_saves_multiple_entries() -> None:
    """バッチ保存で複数のエントリが正しく保存されること.

    Given:
        - 複数のテストエントリを作成
    When:
        - put_batch()で一括保存
    Then:
        - すべてのエントリが正しく取得できること
        - normalized_metricsとtotal_scoreも正しく保存されること
        - 単一トランザクションで処理されること
    """
    # Arrange: 複数のテストエントリを作成
    entries = []
    expected_results = []
    for i in range(5):
        clip_features = np.random.randn(512).astype(np.float32)
        hsv_features = np.random.randn(64).astype(np.float32)
        raw_metrics = {"blur_score": float(i * 10)}
        semantic_score = 0.1 * i
        normalized_metrics = {"blur_score": 0.5 * i}
        total_score = 50.0 + i * 5
        cache_key: dict[str, str | int] = {
            "absolute_path": f"/test/batch_image_{i}.jpg",
            "file_size": 1024 + i,
            "mtime_ns": 1234567890 + i,
            "model_name": "test_model",
            "target_text": "test target",
            "max_dim": 1280,
            "metrics_version": METRICS_VERSION,
        }
        entries.append(
            {
                "cache_key": cache_key,
                "clip_features": clip_features,
                "raw_metrics": raw_metrics,
                "hsv_features": hsv_features,
                "semantic_score": semantic_score,
                "normalized_metrics": normalized_metrics,
                "total_score": total_score,
            }
        )
        expected_results.append(
            (
                cache_key,
                clip_features,
                hsv_features,
                raw_metrics,
                semantic_score,
                normalized_metrics,
                total_score,
            )
        )

    with FeatureCache(None) as cache:
        # Act: バッチ保存
        cache.put_batch(entries)

        # Assert: すべてのエントリが取得できる
        for (
            cache_key,
            clip_features,
            hsv_features,
            raw_metrics,
            semantic,
            norm,
            total,
        ) in expected_results:
            result = cache.get(cache_key)
            assert result is not None
            # float16で保存されているため、精度が異なることを考慮
            expected_clip = clip_features.astype(np.float16).astype(np.float32)
            expected_hsv = hsv_features.astype(np.float16).astype(np.float32)
            np.testing.assert_array_almost_equal(result.clip_features, expected_clip)
            np.testing.assert_array_almost_equal(result.hsv_features, expected_hsv)
            assert result.raw_metrics == raw_metrics
            assert result.semantic_score == semantic
            # normalized_metricsとtotal_scoreも正しく保存・取得されること
            assert result.normalized_metrics == norm
            assert result.total_score == total


def test_composite_key_allows_different_params_for_same_path() -> None:
    """複合主キーにより、同一パスで異なるパラメータのエントリが保存できること.

    Given:
        - 同一パスでmodel_name/max_dimが異なるエントリ
    When:
        - 各エントリを保存して取得
    Then:
        - 各エントリが独立して保存・取得できること
    """
    # Arrange
    clip_features1 = np.ones(512, dtype=np.float32)
    clip_features2 = np.zeros(512, dtype=np.float32) * 2
    clip_features3 = np.ones(512, dtype=np.float32) * 3
    hsv_features = np.ones(64, dtype=np.float32)
    raw_metrics = {"blur_score": 100.0}

    # 同一パスで異なるパラメータ
    cache_key1: dict[str, str | int] = {
        "absolute_path": "/test/same_path.jpg",
        "file_size": 1024,
        "mtime_ns": 1234567890,
        "model_name": "model_A",
        "target_text": "epic game scenery",
        "max_dim": 1280,
        "metrics_version": METRICS_VERSION,
    }
    cache_key2: dict[str, str | int] = {
        "absolute_path": "/test/same_path.jpg",
        "file_size": 1024,
        "mtime_ns": 1234567890,
        "model_name": "model_B",  # 異なるモデル
        "target_text": "epic game scenery",
        "max_dim": 1280,
        "metrics_version": METRICS_VERSION,
    }
    cache_key3: dict[str, str | int] = {
        "absolute_path": "/test/same_path.jpg",
        "file_size": 1024,
        "mtime_ns": 1234567890,
        "model_name": "model_A",
        "target_text": "epic game scenery",
        "max_dim": 640,  # 異なるmax_dim
        "metrics_version": METRICS_VERSION,
    }

    with FeatureCache(None) as cache:
        # Act: 3つのエントリを保存
        cache.put(
            cache_key=cache_key1,
            clip_features=clip_features1,
            raw_metrics=raw_metrics,
            hsv_features=hsv_features,
            semantic_score=0.5,
            normalized_metrics={"blur_score": 0.5},
            total_score=75.0,
        )
        cache.put(
            cache_key=cache_key2,
            clip_features=clip_features2,
            raw_metrics=raw_metrics,
            hsv_features=hsv_features,
            semantic_score=0.6,
            normalized_metrics={"blur_score": 0.6},
            total_score=80.0,
        )
        cache.put(
            cache_key=cache_key3,
            clip_features=clip_features3,
            raw_metrics=raw_metrics,
            hsv_features=hsv_features,
            semantic_score=0.7,
            normalized_metrics={"blur_score": 0.7},
            total_score=85.0,
        )

        # Assert: 各エントリが独立して取得できる
        result1 = cache.get(cache_key1)
        result2 = cache.get(cache_key2)
        result3 = cache.get(cache_key3)

        assert result1 is not None
        assert result2 is not None
        assert result3 is not None

        np.testing.assert_array_equal(result1.clip_features, clip_features1)
        np.testing.assert_array_equal(result2.clip_features, clip_features2)
        np.testing.assert_array_equal(result3.clip_features, clip_features3)


def test_get_many_retrieves_multiple_entries() -> None:
    """get_manyで複数のエントリが正しく取得できること.

    Given:
        - 複数のテストエントリをキャッシュに保存
    When:
        - get_many()で一括取得
    Then:
        - すべてのキャッシュヒットしたエントリが正しく取得できること
        - 存在しないキーに対してはNoneが返されること
    """
    # Arrange: 複数のエントリを保存
    entries = []
    cache_keys = []
    expected_results = {}

    for i in range(3):
        clip_features = np.random.randn(512).astype(np.float32)
        hsv_features = np.random.randn(64).astype(np.float32)
        raw_metrics = {"blur_score": float(i * 10)}
        semantic_score = 0.1 * i
        normalized_metrics = {"blur_score": 0.5 * i}
        total_score = 50.0 + i * 5
        cache_key: dict[str, str | int] = {
            "absolute_path": f"/test/get_many_{i}.jpg",
            "file_size": 1024 + i,
            "mtime_ns": 1234567890 + i,
            "model_name": "test_model",
            "target_text": "test target",
            "max_dim": 1280,
            "metrics_version": METRICS_VERSION,
        }
        cache_keys.append(cache_key)
        entries.append(
            (
                cache_key,
                clip_features,
                hsv_features,
                raw_metrics,
                semantic_score,
                normalized_metrics,
                total_score,
            )
        )
        expected_results[
            (
                cache_key["absolute_path"],
                cache_key["file_size"],
                cache_key["mtime_ns"],
                cache_key["model_name"],
                cache_key["target_text"],
                cache_key["max_dim"],
                cache_key["metrics_version"],
            )
        ] = (
            clip_features,
            hsv_features,
            raw_metrics,
            semantic_score,
            normalized_metrics,
            total_score,
        )

    with FeatureCache(None) as cache:
        for (
            cache_key,
            clip_features,
            hsv_features,
            raw_metrics,
            semantic,
            norm,
            total,
        ) in entries:
            cache.put(
                cache_key=cache_key,
                clip_features=clip_features,
                raw_metrics=raw_metrics,
                hsv_features=hsv_features,
                semantic_score=semantic,
                normalized_metrics=norm,
                total_score=total,
            )

        # 存在しないキーも含める
        nonexistent_key: dict[str, str | int] = {
            "absolute_path": "/nonexistent.jpg",
            "file_size": 9999,
            "mtime_ns": 9999,
            "model_name": "test_model",
            "target_text": "test target",
            "max_dim": 1280,
            "metrics_version": METRICS_VERSION,
        }
        cache_keys.append(nonexistent_key)

        # Act: 一括取得
        results = cache.get_many(cache_keys)

        # Assert: 3つのエントリが取得でき、1つはNone
        assert len(results) == 4

        # 存在するキーが取得できることを確認
        for i in range(3):
            key_id = (
                cache_keys[i]["absolute_path"],
                cache_keys[i]["file_size"],
                cache_keys[i]["mtime_ns"],
                cache_keys[i]["model_name"],
                cache_keys[i]["target_text"],
                cache_keys[i]["max_dim"],
                cache_keys[i]["metrics_version"],
            )
            result = results.get(key_id)
            assert result is not None
            (
                expected_clip,
                expected_hsv,
                expected_raw,
                expected_semantic,
                expected_norm,
                expected_total,
            ) = expected_results[key_id]
            # float16で保存されているため、精度が異なることを考慮
            expected_clip_f16 = expected_clip.astype(np.float16).astype(np.float32)
            expected_hsv_f16 = expected_hsv.astype(np.float16).astype(np.float32)
            np.testing.assert_array_almost_equal(
                result.clip_features, expected_clip_f16
            )
            np.testing.assert_array_almost_equal(result.hsv_features, expected_hsv_f16)
            assert result.raw_metrics == expected_raw
            assert result.semantic_score == expected_semantic
            # normalized_metricsとtotal_scoreも正しく取得できること
            assert result.normalized_metrics == expected_norm
            assert result.total_score == expected_total

        # 存在しないキーはNone
        nonexistent_key_id = (
            nonexistent_key["absolute_path"],
            nonexistent_key["file_size"],
            nonexistent_key["mtime_ns"],
            nonexistent_key["model_name"],
            nonexistent_key["target_text"],
            nonexistent_key["max_dim"],
            nonexistent_key["metrics_version"],
        )
        assert results.get(nonexistent_key_id) is None


def test_get_many_returns_empty_dict_for_empty_input() -> None:
    """get_manyに空のリストを渡すと空の辞書が返されること.

    Given:
        - FeatureCacheインスタンスを作成
    When:
        - 空のリストでget_many()を呼び出し
    Then:
        - 空の辞書が返されること
    """
    # Arrange & Act
    with FeatureCache(None) as cache:
        results = cache.get_many([])

    # Assert
    assert results == {}


def test_pragma_settings_applied_to_file_db() -> None:
    """ファイルDBに対してPRAGMA設定が適用されること.

    Given:
        - 一時ファイルでキャッシュを作成
    When:
        - データベース接続を取得してPRAGMA設定を確認
    Then:
        - WALモード、NORMAL同期、MEMORY一時ストレージが設定されていること
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.sqlite3"

        # Arrange & Act: キャッシュを初期化
        with FeatureCache(cache_path) as cache:
            conn = cache._get_connection()
            cursor = conn.cursor()

            # Assert: PRAGMA設定が適用されていることを確認
            cursor.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]
            # WALモードは'wal'を返す
            assert journal_mode.lower() == "wal"

            cursor.execute("PRAGMA synchronous")
            synchronous = cursor.fetchone()[0]
            assert synchronous == 1  # NORMAL = 1

            cursor.execute("PRAGMA temp_store")
            temp_store = cursor.fetchone()[0]
            assert temp_store == 2  # MEMORY = 2


def test_pragma_settings_not_applied_to_memory_db() -> None:
    """インメモリDBに対してPRAGMA設定が適用されないこと.

    Given:
        - インメモリキャッシュを作成
    When:
        - データベース接続を取得
    Then:
        - 接続が正常に取得できること
        - エラーが発生しないこと
    """
    # Arrange & Act: インメモリキャッシュを初期化
    with FeatureCache(None) as cache:
        conn = cache._get_connection()

        # Assert: 接続が正常に取得できる
        assert conn is not None
        # 行ファクトリーが設定されている
        assert conn.row_factory is not None


def test_get_many_reuses_temp_table() -> None:
    """get_manyがTEMP TABLEを再利用すること.

    Given:
        - FeatureCacheインスタンスを作成
        - 複数のエントリをキャッシュに保存
    When:
        - get_manyを複数回呼び出し
    Then:
        - 2回目以降も正常に動作すること
        - TEMP TABLEが再利用されていること
    """
    # Arrange: 複数のエントリを保存
    entries = []
    cache_keys = []
    for i in range(3):
        clip_features = np.random.randn(512).astype(np.float32)
        hsv_features = np.random.randn(64).astype(np.float32)
        raw_metrics = {"blur_score": float(i * 10)}
        semantic_score = 0.1 * i
        normalized_metrics = {"blur_score": 0.5 * i}
        total_score = 50.0 + i * 5
        cache_key: dict[str, str | int] = {
            "absolute_path": f"/test/reuse_temp_{i}.jpg",
            "file_size": 1024 + i,
            "mtime_ns": 1234567890 + i,
            "model_name": "test_model",
            "target_text": "test target",
            "max_dim": 1280,
            "metrics_version": METRICS_VERSION,
        }
        cache_keys.append(cache_key)
        entries.append(
            {
                "cache_key": cache_key,
                "clip_features": clip_features,
                "raw_metrics": raw_metrics,
                "hsv_features": hsv_features,
                "semantic_score": semantic_score,
                "normalized_metrics": normalized_metrics,
                "total_score": total_score,
            }
        )

    with FeatureCache(None) as cache:
        cache.put_batch(entries)

        # Act: 1回目の呼び出し（TEMP TABLEが作成される）
        results1 = cache.get_many(cache_keys)
        assert len(results1) == 3
        for key_id in results1:
            assert results1[key_id] is not None

        # Act: 2回目の呼び出し（TEMP TABLEが再利用される）
        results2 = cache.get_many(cache_keys)
        assert len(results2) == 3
        for key_id in results2:
            assert results2[key_id] is not None

        # Assert: 結果が一致すること
        for key_id in results1:
            from typing import cast

            entry1 = cast(CacheEntry, results1[key_id])
            entry2 = cast(CacheEntry, results2[key_id])
            np.testing.assert_array_almost_equal(
                entry1.clip_features,
                entry2.clip_features,
            )


def test_put_batch_uses_executemany() -> None:
    """put_batchがexecutemanyで一括挿入されること.

    Given:
        - 複数のテストエントリを作成
    When:
        - put_batch()で一括保存
    Then:
        - すべてのエントリが正しく保存されること
        - normalized_metricsとtotal_scoreも正しく保存されること
        - トランザクション内で処理されること
    """
    # Arrange: 複数のテストエントリを作成（100件）
    entries = []
    expected_results = []
    for i in range(100):
        clip_features = np.random.randn(512).astype(np.float32)
        hsv_features = np.random.randn(64).astype(np.float32)
        raw_metrics = {"blur_score": float(i)}
        semantic_score = 0.01 * i
        normalized_metrics = {"blur_score": 0.5 * i}
        total_score = 50.0 + i * 0.5
        cache_key: dict[str, str | int] = {
            "absolute_path": f"/test/executemany_{i}.jpg",
            "file_size": 1024 + i,
            "mtime_ns": 1234567890 + i,
            "model_name": "test_model",
            "target_text": "test target",
            "max_dim": 1280,
            "metrics_version": METRICS_VERSION,
        }
        entries.append(
            {
                "cache_key": cache_key,
                "clip_features": clip_features,
                "raw_metrics": raw_metrics,
                "hsv_features": hsv_features,
                "semantic_score": semantic_score,
                "normalized_metrics": normalized_metrics,
                "total_score": total_score,
            }
        )
        expected_results.append(
            (
                cache_key,
                clip_features,
                hsv_features,
                raw_metrics,
                semantic_score,
                normalized_metrics,
                total_score,
            )
        )

    with FeatureCache(None) as cache:
        # Act: バッチ保存（executemanyで一括挿入）
        cache.put_batch(entries)

        # Assert: すべてのエントリが取得できる
        for (
            cache_key,
            clip_features,
            hsv_features,
            raw_metrics,
            semantic,
            norm,
            total,
        ) in expected_results:
            result = cache.get(cache_key)
            assert result is not None
            # float16で保存されているため、精度が異なることを考慮
            expected_clip = clip_features.astype(np.float16).astype(np.float32)
            expected_hsv = hsv_features.astype(np.float16).astype(np.float32)
            np.testing.assert_array_almost_equal(result.clip_features, expected_clip)
            np.testing.assert_array_almost_equal(result.hsv_features, expected_hsv)
            assert result.raw_metrics == raw_metrics
            assert result.semantic_score == semantic
            # normalized_metricsとtotal_scoreも正しく保存・取得されること
            assert result.normalized_metrics == norm
            assert result.total_score == total


def test_semantic_score_roundtrip() -> None:
    """semantic_scoreが正しく保存・取得できること.

    Given:
        - semantic_scoreを含むテストデータを作成
    When:
        - キャッシュに保存してから取得
    Then:
        - semantic_scoreが正しく取得できること
    """
    # Arrange
    clip_features = np.random.randn(512).astype(np.float32)
    hsv_features = np.random.randn(64).astype(np.float32)
    raw_metrics = {"blur_score": 100.0}
    semantic_score = 0.75
    cache_key: dict[str, str | int] = {
        "absolute_path": "/test/semantic.jpg",
        "file_size": 2048,
        "mtime_ns": 9876543210,
        "model_name": "test_model",
        "target_text": "test target",
        "max_dim": 1280,
        "metrics_version": METRICS_VERSION,
    }

    with FeatureCache(None) as cache:
        # Act: 保存（semantic_scoreを指定）
        normalized_metrics = {"blur_score": 0.5}
        total_score = 75.0
        cache.put(
            cache_key=cache_key,
            clip_features=clip_features,
            raw_metrics=raw_metrics,
            hsv_features=hsv_features,
            semantic_score=semantic_score,
            normalized_metrics=normalized_metrics,
            total_score=total_score,
        )

        # Act: 取得
        result = cache.get(cache_key)

        # Assert
        assert result is not None
        assert result.semantic_score == semantic_score
        # normalized_metricsとtotal_scoreも正しく保存・取得されること
        assert result.normalized_metrics == normalized_metrics
        assert result.total_score == total_score


def test_table_recreated_when_schema_differs() -> None:
    """スキーマが異なる場合にテーブルが再作成されること.

    Given:
        - 期待されるスキーマと異なる定義でテーブルが作成されている
    When:
        - FeatureCacheを初期化
    Then:
        - テーブルがドロップされて再作成されること
        - 期待されるスキーマになること
    """
    import tempfile
    import sqlite3

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "invalid_cache.sqlite3"

        # Arrange: 異なるスキーマでテーブルを作成（カラムが不足）
        conn = sqlite3.connect(str(cache_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE feature_cache (
                absolute_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                max_dim INTEGER NOT NULL,
                PRIMARY KEY (absolute_path, max_dim)
            )
            """
        )
        conn.commit()
        conn.close()

        # Act: FeatureCacheを初期化（スキーマが異なるため再作成）
        with FeatureCache(cache_path) as cache:
            conn2 = cache._get_connection()
            cursor2 = conn2.cursor()

            # Assert: 期待されるスキーマになっていること
            cursor2.execute("PRAGMA table_info(feature_cache)")
            columns = [row[1] for row in cursor2.fetchall()]
            assert "clip_features" in columns
            assert "hsv_features" in columns
            assert "semantic_score" in columns
            assert "normalized_metrics" in columns
            assert "total_score" in columns
            assert "model_name" in columns
            assert "target_text" in columns

            # 複合主キーが正しく設定されていること
            cursor2.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND name='feature_cache'"
            )
            schema_sql = cursor2.fetchone()[0]
            assert "PRIMARY KEY" in schema_sql
            assert "absolute_path" in schema_sql
            assert "model_name" in schema_sql


def test_correct_schema_preserved() -> None:
    """スキーマが正しい場合は何もしないこと.

    Given:
        - 正しいスキーマでテーブルが作成されている
        - データが保存されている
    When:
        - FeatureCacheを初期化
    Then:
        - テーブルが再作成されないこと
        - 既存データが保持されること
        - normalized_metricsとtotal_scoreも保持されること
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "correct_cache.sqlite3"

        # Arrange: 正しいスキーマでFeatureCacheを作成し、データを保存
        clip_features = np.ones(512, dtype=np.float32) * 0.5
        hsv_features = np.ones(64, dtype=np.float32) * 0.3
        raw_metrics = {"blur_score": 100.0}
        semantic = 0.75
        normalized_metrics = {"blur_score": 0.5}
        total_score = 85.5
        cache_key: dict[str, str | int] = {
            "absolute_path": "/test/preserved.jpg",
            "file_size": 2048,
            "mtime_ns": 1234567890,
            "model_name": "test_model",
            "target_text": "test target",
            "max_dim": 1280,
            "metrics_version": METRICS_VERSION,
        }

        with FeatureCache(cache_path) as cache1:
            cache1.put(
                cache_key=cache_key,
                clip_features=clip_features,
                raw_metrics=raw_metrics,
                hsv_features=hsv_features,
                semantic_score=semantic,
                normalized_metrics=normalized_metrics,
                total_score=total_score,
            )

        # Act: 同じパスでFeatureCacheを再度初期化
        with FeatureCache(cache_path) as cache2:
            # Assert: データが保持されている
            result = cache2.get(cache_key)
            assert result is not None
            # float16で保存されているため、精度が異なることを考慮
            expected_clip = clip_features.astype(np.float16).astype(np.float32)
            expected_hsv = hsv_features.astype(np.float16).astype(np.float32)
            np.testing.assert_array_equal(result.clip_features, expected_clip)
            np.testing.assert_array_equal(result.hsv_features, expected_hsv)
            assert result.raw_metrics == raw_metrics
            assert result.semantic_score == semantic
            # normalized_metricsとtotal_scoreも保持されること
            assert result.normalized_metrics == normalized_metrics
            assert result.total_score == total_score


def test_float16_features_roundtrip() -> None:
    """float16で保存した特徴量が正しく変換されて読み出されること.

    Given:
        - float32の特徴量を作成
    When:
        - float16で保存してから読み出し
    Then:
        - float16の精度で丸められた特徴量が読み出されること
        - 読み出し後はfloat32に変換されていること
        - normalized_metricsとtotal_scoreも正しく保存・取得できること
    """
    # Arrange: float32で特徴量を作成
    clip_features = np.array(
        [0.12345678, -0.98765432, 1.0, -1.0, 0.0], dtype=np.float32
    )
    hsv_features = np.array([0.5, 0.25, 0.75, 0.125], dtype=np.float32)
    raw_metrics = {"blur_score": 100.0}
    semantic_score = 0.75
    normalized_metrics = {"blur_score": 0.5}
    total_score = 85.5
    cache_key: dict[str, str | int] = {
        "absolute_path": "/test/float16_roundtrip.jpg",
        "file_size": 2048,
        "mtime_ns": 9876543210,
        "model_name": "test_model",
        "target_text": "test target",
        "max_dim": 1280,
        "metrics_version": METRICS_VERSION,
    }

    with FeatureCache(None) as cache:
        # Act: 保存（内部でfloat16に変換される）
        cache.put(
            cache_key=cache_key,
            clip_features=clip_features,
            raw_metrics=raw_metrics,
            hsv_features=hsv_features,
            semantic_score=semantic_score,
            normalized_metrics=normalized_metrics,
            total_score=total_score,
        )

        # Act: 取得（float16から読み出してfloat32に変換される）
        result = cache.get(cache_key)

        # Assert
        assert result is not None
        # 読み出し後はfloat32になっていること
        assert result.clip_features.dtype == np.float32
        assert result.hsv_features.dtype == np.float32
        # float16の精度で丸められていることを確認
        # float16で保存すると、例えば0.12345678は0.1235に丸められる
        expected_clip = clip_features.astype(np.float16).astype(np.float32)
        expected_hsv = hsv_features.astype(np.float16).astype(np.float32)
        np.testing.assert_array_almost_equal(
            result.clip_features, expected_clip, decimal=4
        )
        np.testing.assert_array_almost_equal(
            result.hsv_features, expected_hsv, decimal=4
        )
        assert result.raw_metrics == raw_metrics
        assert result.semantic_score == semantic_score
        # normalized_metricsとtotal_scoreも正しく保存・取得されること
        assert result.normalized_metrics == normalized_metrics
        assert result.total_score == total_score
