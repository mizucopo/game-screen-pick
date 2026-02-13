"""特徴量キャッシュ - 画像解析の中間結果をSQLiteで永続化する."""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..models.cache_entry import CacheEntry


class FeatureCache:
    """特徴量キャッシュを管理するクラス.

    SQLiteを使用して画像解析の中間結果を永続化し、
    2回目以降の実行で再利用する。
    """

    # スキーマ初期化用のクラスロック
    _init_lock = threading.Lock()

    # キャッシュスキーマのバージョン（アルゴリズム変更時に更新）
    METRICS_VERSION: str = "1"

    def __init__(self, cache_path: Optional[str | Path] = None):
        """特徴量キャッシュを初期化する.

        Args:
            cache_path: キャッシュデータベースのパス
                       Noneの場合はインメモリキャッシュ（テスト用）
        """
        self.cache_path = Path(cache_path) if cache_path else None
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> Any:
        """スレッドローカルなデータベース接続を取得する.

        Returns:
            データベース接続
        """
        if not hasattr(self._local, "conn"):
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._local.conn = sqlite3.connect(
                    str(self.cache_path), check_same_thread=False
                )
            else:
                # インメモリデータベース（テスト用）
                self._local.conn = sqlite3.connect(":memory:")
            # 行を辞書形式でアクセス可能にする
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """データベーススキーマを初期化する.

        必要に応じて既存スキーマから複合主キースキーマへマイグレーションを実行する。
        スレッドセーフのため、クラスロックを使用して初期化を保護する。
        """
        with FeatureCache._init_lock:
            # ロック内でもう一度テーブル存在チェック（ダブルチェックロック）
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='feature_cache'
                """
            )
            if cursor.fetchone() is None:
                # テーブルが存在しない場合、新しいスキーマで作成
                self._create_new_schema(cursor)
                conn.commit()
                return

            # 既存テーブルのスキーマを確認
            cursor.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type='table' AND name='feature_cache'
                """
            )
            result = cursor.fetchone()
            existing_sql = result[0] if result else ""

            # 既存スキーマの主キーを確認（PRIMARY KEYがabsolute_pathのみかチェック）
            if "absolute_path TEXT PRIMARY KEY" in existing_sql:
                # マイグレーション必要: 古いスキーマから新しいスキーマへ移行
                self._migrate_to_composite_key(cursor)
                conn.commit()
            # 新しいスキーマの場合は何もしない（既に複合主キー）
        """データベーススキーマを初期化する.

        必要に応じて既存スキーマから複合主キースキーマへマイグレーションを実行する。
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # 既存テーブルのスキーマを確認
        cursor.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type='table' AND name='feature_cache'
            """
        )
        result = cursor.fetchone()

        if result is None:
            # テーブルが存在しない場合、新しいスキーマで作成
            self._create_new_schema(cursor)
        else:
            existing_sql = result[0]
            # 既存スキーマの主キーを確認（PRIMARY KEYがabsolute_pathのみかチェック）
            if "absolute_path TEXT PRIMARY KEY" in existing_sql:
                # マイグレーション必要: 古いスキーマから新しいスキーマへ移行
                self._migrate_to_composite_key(cursor)
            # 新しいスキーマの場合は何もしない（既に複合主キー）

        conn.commit()

    def _create_new_schema(self, cursor: sqlite3.Cursor) -> None:
        """新しい複合主キースキーマでテーブルを作成する.

        Args:
            cursor: データベースカーソル
        """
        cursor.execute(
            """
            CREATE TABLE feature_cache (
                absolute_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                target_text TEXT NOT NULL,
                max_dim INTEGER NOT NULL,
                metrics_version TEXT NOT NULL,
                clip_features BLOB NOT NULL,
                raw_metrics TEXT NOT NULL,
                hsv_features BLOB NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (
                    absolute_path, model_name, target_text, max_dim, metrics_version
                )
            )
            """
        )
        # パスとファイルサイズ、更新時刻で高速検索するためのインデックス
        cursor.execute(
            """
            CREATE INDEX idx_file_signature
            ON feature_cache(absolute_path, file_size, mtime_ns)
            """
        )

    def _migrate_to_composite_key(self, cursor: sqlite3.Cursor) -> None:
        """古いスキーマから新しい複合主キースキーマへマイグレーションする.

        手順:
        1. 既存データをfeature_cache_backupに退避
        2. 古いテーブルを削除
        3. 新しいスキーマでテーブル作成
        4. データを復元（INSERT OR REPLACEで重複処理）
        5. バックアップテーブルを削除

        Args:
            cursor: データベースカーソル
        """
        import logging

        logger = logging.getLogger(__name__)

        # ステップ1: 既存テーブルをリネーム
        cursor.execute("ALTER TABLE feature_cache RENAME TO feature_cache_backup")

        # ステップ2: 新しいスキーマでテーブル作成
        self._create_new_schema(cursor)

        # ステップ3: データを復元
        cursor.execute(
            """
            INSERT OR REPLACE INTO feature_cache (
                absolute_path, file_size, mtime_ns, model_name, target_text,
                max_dim, metrics_version, clip_features, raw_metrics,
                hsv_features, created_at
            )
            SELECT
                absolute_path, file_size, mtime_ns, model_name, target_text,
                max_dim, metrics_version, clip_features, raw_metrics,
                hsv_features, created_at
            FROM feature_cache_backup
            """
        )

        # 移行した行数をログ出力
        migrated_count = cursor.rowcount
        logger.info(f"キャッシュスキーマをマイグレーションしました: {migrated_count}件")

        # ステップ4: バックアップテーブルを削除
        cursor.execute("DROP TABLE feature_cache_backup")

    def generate_cache_key(
        self,
        absolute_path: str,
        file_size: int,
        mtime_ns: int,
        model_name: str,
        target_text: str,
        max_dim: int,
    ) -> dict[str, str | int]:
        """キャッシュキーを生成する.

        Args:
            absolute_path: 画像ファイルの絶対パス
            file_size: ファイルサイズ（バイト）
            mtime_ns: 更新時刻（ナノ秒）
            model_name: CLIPモデル名
            target_text: ターゲットテキスト
            max_dim: 最大解像度

        Returns:
            キャッシュキーの辞書
        """
        return {
            "absolute_path": absolute_path,
            "file_size": file_size,
            "mtime_ns": mtime_ns,
            "model_name": model_name,
            "target_text": target_text,
            "max_dim": max_dim,
            "metrics_version": self.METRICS_VERSION,
        }

    def get(self, cache_key: dict[str, str | int]) -> Optional[CacheEntry]:
        """キャッシュから特徴量を取得する.

        Args:
            cache_key: キャッシュキー

        Returns:
            キャッシュエントリ（存在しない場合はNone）
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT clip_features, raw_metrics, hsv_features
            FROM feature_cache
            WHERE absolute_path = ?
              AND file_size = ?
              AND mtime_ns = ?
              AND model_name = ?
              AND target_text = ?
              AND max_dim = ?
              AND metrics_version = ?
            """,
            (
                cache_key["absolute_path"],
                cache_key["file_size"],
                cache_key["mtime_ns"],
                cache_key["model_name"],
                cache_key["target_text"],
                cache_key["max_dim"],
                cache_key["metrics_version"],
            ),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return CacheEntry(
            clip_features=np.frombuffer(row["clip_features"], dtype=np.float32),
            raw_metrics=json.loads(row["raw_metrics"]),
            hsv_features=np.frombuffer(row["hsv_features"], dtype=np.float32),
        )

    def put(
        self,
        cache_key: dict[str, str | int],
        clip_features: np.ndarray,
        raw_metrics: dict[str, float],
        hsv_features: np.ndarray,
    ) -> None:
        """特徴量をキャッシュに保存する.

        Args:
            cache_key: キャッシュキー
            clip_features: CLIP特徴（512次元）
            raw_metrics: 生メトリクス
            hsv_features: HSV特徴（64次元）
        """
        import time

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO feature_cache (
                absolute_path, file_size, mtime_ns, model_name, target_text,
                max_dim, metrics_version, clip_features, raw_metrics,
                hsv_features, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key["absolute_path"],
                cache_key["file_size"],
                cache_key["mtime_ns"],
                cache_key["model_name"],
                cache_key["target_text"],
                cache_key["max_dim"],
                cache_key["metrics_version"],
                clip_features.astype(np.float32).tobytes(),
                json.dumps(raw_metrics),
                hsv_features.astype(np.float32).tobytes(),
                time.time(),
            ),
        )
        conn.commit()

    def get_many(
        self, cache_keys: list[dict[str, str | int]]
    ) -> dict[str, Optional[CacheEntry]]:
        """複数のキャッシュキーで特徴量を一括取得する.

        パフォーマンス最適化:
        - 単一のINクエリで複数キーを取得
        - SQLiteの複合主キー（7フィールド）を考慮
        - 結果をキャッシュキーの文字列表現でマッピング

        Args:
            cache_keys: キャッシュキーのリスト

        Returns:
            キャッシュキーの文字列表現をキー、CacheEntry（ヒットしない場合はNone）を値とする辞書
        """
        if not cache_keys:
            return {}

        conn = self._get_connection()
        cursor = conn.cursor()

        # キャッシュキーの一意識別子を生成（主キーフィールドのタプル）
        key_identifiers = [
            (
                k["absolute_path"],
                k["file_size"],
                k["mtime_ns"],
                k["model_name"],
                k["target_text"],
                k["max_dim"],
                k["metrics_version"],
            )
            for k in cache_keys
        ]

        # 結果を格納する辞書（キーはタプルの文字列表現）
        results: dict[str, Optional[CacheEntry]] = {
            str(k_id): None for k_id in key_identifiers
        }

        # 一時テーブルを作成してルックアップ用データを挿入
        cursor.execute(
            """
            CREATE TEMP TABLE temp_lookup (
                absolute_path TEXT,
                file_size INTEGER,
                mtime_ns INTEGER,
                model_name TEXT,
                target_text TEXT,
                max_dim INTEGER,
                metrics_version TEXT
            )
            """
        )

        # ルックアップテーブルにデータを一括挿入
        cursor.executemany(
            """
            INSERT INTO temp_lookup VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            key_identifiers,
        )

        # JOINで一括取得
        cursor.execute(
            """
            SELECT
                fc.absolute_path,
                fc.file_size,
                fc.mtime_ns,
                fc.model_name,
                fc.target_text,
                fc.max_dim,
                fc.metrics_version,
                fc.clip_features,
                fc.raw_metrics,
                fc.hsv_features
            FROM feature_cache fc
            INNER JOIN temp_lookup tl ON
                fc.absolute_path = tl.absolute_path AND
                fc.file_size = tl.file_size AND
                fc.mtime_ns = tl.mtime_ns AND
                fc.model_name = tl.model_name AND
                fc.target_text = tl.target_text AND
                fc.max_dim = tl.max_dim AND
                fc.metrics_version = tl.metrics_version
            """
        )

        # 結果をマッピング
        for row in cursor.fetchall():
            key_id = str(
                (
                    row["absolute_path"],
                    row["file_size"],
                    row["mtime_ns"],
                    row["model_name"],
                    row["target_text"],
                    row["max_dim"],
                    row["metrics_version"],
                )
            )
            results[key_id] = CacheEntry(
                clip_features=np.frombuffer(row["clip_features"], dtype=np.float32),
                raw_metrics=json.loads(row["raw_metrics"]),
                hsv_features=np.frombuffer(row["hsv_features"], dtype=np.float32),
            )

        # 一時テーブルを削除
        cursor.execute("DROP TABLE temp_lookup")

        return results

    def put_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> None:
        """複数のエントリを単一トランザクションで一括保存する.

        パフォーマンス最適化:
        - 単一トランザクションで複数件を一括挿入
        - SQLiteのロック競合を回避
        - トランザクションオーバーヘッドを削減

        Args:
            entries: 保存するエントリのリスト。各エントリは以下のキーを持つ辞書:
                - cache_key: キャッシュキー（辞書）
                - clip_features: CLIP特徴（512次元、np.ndarray）
                - raw_metrics: 生メトリクス（辞書）
                - hsv_features: HSV特徴（64次元、np.ndarray）
        """
        import time

        if not entries:
            return

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("BEGIN TRANSACTION")
        try:
            for entry in entries:
                cache_key = entry["cache_key"]
                cursor.execute(
                    """INSERT OR REPLACE INTO feature_cache (
                        absolute_path, file_size, mtime_ns, model_name, target_text,
                        max_dim, metrics_version, clip_features, raw_metrics,
                        hsv_features, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key["absolute_path"],
                        cache_key["file_size"],
                        cache_key["mtime_ns"],
                        cache_key["model_name"],
                        cache_key["target_text"],
                        cache_key["max_dim"],
                        cache_key["metrics_version"],
                        entry["clip_features"].astype(np.float32).tobytes(),
                        json.dumps(entry["raw_metrics"]),
                        entry["hsv_features"].astype(np.float32).tobytes(),
                        time.time(),
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        """データベース接続を閉じる."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            delattr(self._local, "conn")

    def __enter__(self) -> "FeatureCache":
        """コンテキストマネージャーのエントリー."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """コンテキストマネージャーの終了時に接続を閉じる."""
        self.close()
