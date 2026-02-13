"""特徴量キャッシュ - 画像解析の中間結果をSQLiteで永続化する."""

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..models.cache_entry import CacheEntry

logger = logging.getLogger(__name__)


class FeatureCache:
    """特徴量キャッシュを管理するクラス.

    SQLiteを使用して画像解析の中間結果を永続化し、
    2回目以降の実行で再利用する。
    """

    # スキーマ初期化用のクラスロック
    _init_lock = threading.Lock()

    # キャッシュスキーマのバージョン（アルゴリズム変更時に更新）
    METRICS_VERSION: str = "2"  # float16 features

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

        接続時にTEMP TABLEを事前に作成し、get_many呼び出し時に再利用する。
        これにより毎回のCREATE/DROP TABLEオーバーヘッドを削減する。

        Returns:
            データベース接続
        """
        if not hasattr(self._local, "conn"):
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(self.cache_path), check_same_thread=False)
                # ファイルDBの場合はパフォーマンス最適化PRAGMAを設定
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA temp_store=MEMORY")
                self._local.conn = conn
            else:
                # インメモリデータベース（テスト用）
                self._local.conn = sqlite3.connect(":memory:")
            # 行を辞書形式でアクセス可能にする
            self._local.conn.row_factory = sqlite3.Row

            # TEMP TABLEを事前に作成（接続時に1回のみ）
            # get_manyでクリアして再利用する
            cursor = self._local.conn.cursor()
            cursor.execute(
                """
                CREATE TEMP TABLE IF NOT EXISTS temp_lookup (
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
        return self._local.conn

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        """SQL文字列を正規化して比較可能にする.

        空白、改行、大文字小文字を正規化し、スキーマ比較を安定させる。

        Args:
            sql: 正規化するSQL文字列

        Returns:
            正規化されたSQL文字列
        """
        return " ".join(sql.strip().split()).upper()

    def _get_expected_schema_sql(self) -> str:
        """期待されるテーブル作成SQLを取得.

        Returns:
            期待されるCREATE TABLE文
        """
        return """CREATE TABLE feature_cache (
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
            semantic_score REAL,
            created_at REAL NOT NULL,
            PRIMARY KEY (
                absolute_path, model_name, target_text, max_dim, metrics_version
            )
        )"""

    def _init_db(self) -> None:
        """データベーススキーマを初期化する.

        スキーマが期待と異なる場合はテーブルをドロップして再作成する。
        キャッシュデータは破棄されるが、次回実行時に再構築される。
        スレッドセーフのため、クラスロックを使用して初期化を保護する。
        """
        with FeatureCache._init_lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # テーブルが存在する場合、スキーマを確認
            cursor.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND name='feature_cache'"
            )
            result = cursor.fetchone()

            if result is not None:
                # 既存テーブルのスキーマを取得
                existing_sql = result[0]
                # スキーマを正規化して比較（空白、改行を正規化）
                normalized_existing = self._normalize_sql(existing_sql)
                normalized_expected = self._normalize_sql(
                    self._get_expected_schema_sql()
                )

                if normalized_existing != normalized_expected:
                    # スキーマが異なる場合はドロップして再作成
                    logger.info(
                        f"キャッシュスキーマが異なるためテーブルを再作成します: "
                        f"expected={normalized_expected[:50]}..., "
                        f"got={normalized_existing[:50]}..."
                    )
                    cursor.execute("DROP TABLE feature_cache")

            # テーブル作成（存在しないか、ドロップされた場合）
            cursor.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='feature_cache'"
            )
            if cursor.fetchone() is None:
                self._create_new_schema(cursor)
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
                semantic_score REAL,
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
            SELECT clip_features, raw_metrics, hsv_features, semantic_score
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
            clip_features=np.frombuffer(row["clip_features"], dtype=np.float16).astype(
                np.float32
            ),
            raw_metrics=json.loads(row["raw_metrics"]),
            hsv_features=np.frombuffer(row["hsv_features"], dtype=np.float16).astype(
                np.float32
            ),
            semantic_score=row["semantic_score"],
        )

    def put(
        self,
        cache_key: dict[str, str | int],
        clip_features: np.ndarray,
        raw_metrics: dict[str, float],
        hsv_features: np.ndarray,
        semantic_score: float,
    ) -> None:
        """特徴量をキャッシュに保存する.

        Args:
            cache_key: キャッシュキー
            clip_features: CLIP特徴（512次元）
            raw_metrics: 生メトリクス
            hsv_features: HSV特徴（64次元）
            semantic_score: セマンティックスコア
        """
        import time

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO feature_cache (
                absolute_path, file_size, mtime_ns, model_name, target_text,
                max_dim, metrics_version, clip_features, raw_metrics,
                hsv_features, semantic_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key["absolute_path"],
                cache_key["file_size"],
                cache_key["mtime_ns"],
                cache_key["model_name"],
                cache_key["target_text"],
                cache_key["max_dim"],
                cache_key["metrics_version"],
                clip_features.astype(np.float16).tobytes(),
                json.dumps(raw_metrics),
                hsv_features.astype(np.float16).tobytes(),
                semantic_score,
                time.time(),
            ),
        )
        conn.commit()

    def get_many(
        self, cache_keys: list[dict[str, str | int]]
    ) -> dict[tuple[Any, ...], Optional[CacheEntry]]:
        """複数のキャッシュキーで特徴量を一括取得する.

        パフォーマンス最適化:
        - 単一のINクエリで複数キーを取得
        - SQLiteの複合主キー（7フィールド）を考慮
        - 結果をキャッシュキーのタプルでマッピング（文字列化を回避）
        - TEMP TABLEを再利用してCREATE/DROPオーバーヘッドを削減

        Args:
            cache_keys: キャッシュキーのリスト

        Returns:
            キャッシュキーのタプルをキー、CacheEntry（ヒットしない場合はNone）を値とする辞書
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

        # 結果を格納する辞書（キーはタプル、文字列化を回避）
        results: dict[tuple[Any, ...], Optional[CacheEntry]] = {
            k_id: None for k_id in key_identifiers
        }

        # TEMP TABLEをクリアして再利用（接続時に作成済み）
        cursor.execute("DELETE FROM temp_lookup")

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
                fc.hsv_features,
                fc.semantic_score
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

        # 結果をマッピング（タプルを直接使用して文字列化を回避）
        for row in cursor.fetchall():
            key_id = (
                row["absolute_path"],
                row["file_size"],
                row["mtime_ns"],
                row["model_name"],
                row["target_text"],
                row["max_dim"],
                row["metrics_version"],
            )
            # float16で読み出してfloat32に変換
            clip = np.frombuffer(row["clip_features"], dtype=np.float16).astype(
                np.float32
            )
            hsv = np.frombuffer(row["hsv_features"], dtype=np.float16).astype(
                np.float32
            )
            results[key_id] = CacheEntry(
                clip_features=clip,
                raw_metrics=json.loads(row["raw_metrics"]),
                hsv_features=hsv,
                semantic_score=row["semantic_score"],
            )

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
        - executemanyで一括挿入してパフォーマンス向上

        Args:
            entries: 保存するエントリのリスト。各エントリは以下のキーを持つ辞書:
                - cache_key: キャッシュキー（辞書）
                - clip_features: CLIP特徴（512次元、np.ndarray）
                - raw_metrics: 生メトリクス（辞書）
                - hsv_features: HSV特徴（64次元、np.ndarray）
                - semantic_score: セマンティックスコア（オプション）
        """
        import time

        if not entries:
            return

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("BEGIN TRANSACTION")
        try:
            insert_data = []
            for entry in entries:
                cache_key = entry["cache_key"]
                insert_data.append(
                    (
                        cache_key["absolute_path"],
                        cache_key["file_size"],
                        cache_key["mtime_ns"],
                        cache_key["model_name"],
                        cache_key["target_text"],
                        cache_key["max_dim"],
                        cache_key["metrics_version"],
                        entry["clip_features"].astype(np.float16).tobytes(),
                        json.dumps(entry["raw_metrics"]),
                        entry["hsv_features"].astype(np.float16).tobytes(),
                        entry.get("semantic_score"),
                        time.time(),
                    )
                )

            # executemanyで一括挿入（ループ内のexecuteより20-200倍高速）
            cursor.executemany(
                """
                INSERT OR REPLACE INTO feature_cache (
                    absolute_path, file_size, mtime_ns, model_name, target_text,
                    max_dim, metrics_version, clip_features, raw_metrics,
                    hsv_features, semantic_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_data,
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
        """コンテキストマネージャーの終了時に接続を閉じる.

        Args:
            exc_type: 例外タイプ（使用しない）
            exc_val: 例外値（使用しない）
            exc_tb: 例外トレースバック（使用しない）
        """
        _ = (exc_type, exc_val, exc_tb)  # 未使用警告を抑制
        self.close()
