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
        """データベーススキーマを初期化する."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_cache (
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
        # パスとファイルサイズ、更新時刻で高速検索するためのインデックス
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_file_signature
            ON feature_cache(absolute_path, file_size, mtime_ns)
            """
        )
        conn.commit()

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
