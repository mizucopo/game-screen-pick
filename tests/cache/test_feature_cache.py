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


def test_put_batch_saves_multiple_entries() -> None:
    """バッチ保存で複数のエントリが正しく保存されること.

    Given:
        - 複数のテストエントリを作成
    When:
        - put_batch()で一括保存
    Then:
        - すべてのエントリが正しく取得できること
        - 単一トランザクションで処理されること
    """
    # Arrange: 複数のテストエントリを作成
    entries = []
    expected_results = []
    for i in range(5):
        clip_features = np.random.randn(512).astype(np.float32)
        hsv_features = np.random.randn(64).astype(np.float32)
        raw_metrics = {"blur_score": float(i * 10)}
        cache_key: dict[str, str | int] = {
            "absolute_path": f"/test/batch_image_{i}.jpg",
            "file_size": 1024 + i,
            "mtime_ns": 1234567890 + i,
            "model_name": "test_model",
            "target_text": "test target",
            "max_dim": 1280,
            "metrics_version": "1",
        }
        entries.append(
            {
                "cache_key": cache_key,
                "clip_features": clip_features,
                "raw_metrics": raw_metrics,
                "hsv_features": hsv_features,
            }
        )
        expected_results.append((cache_key, clip_features, hsv_features, raw_metrics))

    with FeatureCache(None) as cache:
        # Act: バッチ保存
        cache.put_batch(entries)

        # Assert: すべてのエントリが取得できる
        for cache_key, clip_features, hsv_features, raw_metrics in expected_results:
            result = cache.get(cache_key)
            assert result is not None
            np.testing.assert_array_almost_equal(result.clip_features, clip_features)
            np.testing.assert_array_almost_equal(result.hsv_features, hsv_features)
            assert result.raw_metrics == raw_metrics


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
        "metrics_version": "1",
    }
    cache_key2: dict[str, str | int] = {
        "absolute_path": "/test/same_path.jpg",
        "file_size": 1024,
        "mtime_ns": 1234567890,
        "model_name": "model_B",  # 異なるモデル
        "target_text": "epic game scenery",
        "max_dim": 1280,
        "metrics_version": "1",
    }
    cache_key3: dict[str, str | int] = {
        "absolute_path": "/test/same_path.jpg",
        "file_size": 1024,
        "mtime_ns": 1234567890,
        "model_name": "model_A",
        "target_text": "epic game scenery",
        "max_dim": 640,  # 異なるmax_dim
        "metrics_version": "1",
    }

    with FeatureCache(None) as cache:
        # Act: 3つのエントリを保存
        cache.put(
            cache_key=cache_key1,
            clip_features=clip_features1,
            raw_metrics=raw_metrics,
            hsv_features=hsv_features,
        )
        cache.put(
            cache_key=cache_key2,
            clip_features=clip_features2,
            raw_metrics=raw_metrics,
            hsv_features=hsv_features,
        )
        cache.put(
            cache_key=cache_key3,
            clip_features=clip_features3,
            raw_metrics=raw_metrics,
            hsv_features=hsv_features,
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


def test_migration_from_old_schema_to_composite_key() -> None:
    """古いスキーマ（absolute_pathのみのPRIMARY KEY）から新しいスキーマへ正しく
    マイグレーションされること.

    Given:
        - 古いスキーマで作成されたキャッシュデータベース
    When:
        - FeatureCacheを初期化してマイグレーションを実行
    Then:
        - 新しいスキーマに変換されること
        - 既存データが保持されること
    """
    import tempfile
    import sqlite3

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "old_cache.sqlite3"

        # Arrange: 古いスキーマでテーブルを作成
        conn = sqlite3.connect(str(cache_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE feature_cache (
                absolute_path TEXT PRIMARY KEY,
                file_size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                target_text TEXT NOT NULL,
                max_dim INTEGER NOT NULL,
                metrics_version TEXT NOT NULL,
                clip_features BLOB NOT NULL,
                raw_metrics TEXT NOT NULL,
                hsv_features BLOB NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )

        # テストデータを投入
        import json

        clip_features = np.ones(512, dtype=np.float32) * 0.5
        hsv_features = np.ones(64, dtype=np.float32) * 0.3
        raw_metrics = {"blur_score": 100.0}

        cursor.execute(
            """
            INSERT INTO feature_cache VALUES (
                '/old/image.jpg', 1024, 1234567890, 'old_model',
                'test target', 1280, '1', ?, ?, ?, 1234567890.0
            )
            """,
            (
                clip_features.astype(np.float32).tobytes(),
                json.dumps(raw_metrics),
                hsv_features.astype(np.float32).tobytes(),
            ),
        )
        conn.commit()
        conn.close()

        # Act: 新しいFeatureCacheを初期化（マイグレーションが実行される）
        with FeatureCache(cache_path) as cache:
            # 新しいスキーマでデータが取得できることを確認
            cache_key = cache.generate_cache_key(
                absolute_path="/old/image.jpg",
                file_size=1024,
                mtime_ns=1234567890,
                model_name="old_model",
                target_text="test target",
                max_dim=1280,
            )
            result = cache.get(cache_key)

        # Assert: データが正しくマイグレーションされている
        assert result is not None
        np.testing.assert_array_equal(result.clip_features, clip_features)
        np.testing.assert_array_equal(result.hsv_features, hsv_features)
        assert result.raw_metrics == raw_metrics


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
        cache_key: dict[str, str | int] = {
            "absolute_path": f"/test/get_many_{i}.jpg",
            "file_size": 1024 + i,
            "mtime_ns": 1234567890 + i,
            "model_name": "test_model",
            "target_text": "test target",
            "max_dim": 1280,
            "metrics_version": "1",
        }
        cache_keys.append(cache_key)
        entries.append((cache_key, clip_features, hsv_features, raw_metrics))
        expected_results[
            str(
                (
                    cache_key["absolute_path"],
                    cache_key["file_size"],
                    cache_key["mtime_ns"],
                    cache_key["model_name"],
                    cache_key["target_text"],
                    cache_key["max_dim"],
                    cache_key["metrics_version"],
                )
            )
        ] = (clip_features, hsv_features, raw_metrics)

    with FeatureCache(None) as cache:
        for cache_key, clip_features, hsv_features, raw_metrics in entries:
            cache.put(
                cache_key=cache_key,
                clip_features=clip_features,
                raw_metrics=raw_metrics,
                hsv_features=hsv_features,
            )

        # 存在しないキーも含める
        nonexistent_key: dict[str, str | int] = {
            "absolute_path": "/nonexistent.jpg",
            "file_size": 9999,
            "mtime_ns": 9999,
            "model_name": "test_model",
            "target_text": "test target",
            "max_dim": 1280,
            "metrics_version": "1",
        }
        cache_keys.append(nonexistent_key)

        # Act: 一括取得
        results = cache.get_many(cache_keys)

        # Assert: 3つのエントリが取得でき、1つはNone
        assert len(results) == 4

        # 存在するキーが取得できることを確認
        for i in range(3):
            key_id = str(
                (
                    cache_keys[i]["absolute_path"],
                    cache_keys[i]["file_size"],
                    cache_keys[i]["mtime_ns"],
                    cache_keys[i]["model_name"],
                    cache_keys[i]["target_text"],
                    cache_keys[i]["max_dim"],
                    cache_keys[i]["metrics_version"],
                )
            )
            result = results.get(key_id)
            assert result is not None
            expected_clip, expected_hsv, expected_raw = expected_results[key_id]
            np.testing.assert_array_almost_equal(result.clip_features, expected_clip)
            np.testing.assert_array_almost_equal(result.hsv_features, expected_hsv)
            assert result.raw_metrics == expected_raw

        # 存在しないキーはNone
        nonexistent_key_id = str(
            (
                nonexistent_key["absolute_path"],
                nonexistent_key["file_size"],
                nonexistent_key["mtime_ns"],
                nonexistent_key["model_name"],
                nonexistent_key["target_text"],
                nonexistent_key["max_dim"],
                nonexistent_key["metrics_version"],
            )
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
