"""キャッシュエントリ情報のデータ構造."""

from dataclasses import dataclass

from ..models.cache_entry import CacheEntry


@dataclass
class CacheEntryInfo:
    """キャッシュエントリと関連情報を保持するヘルパークラス.

    バッチ処理時に必要な情報をまとめて保持する。

    Attributes:
        path: 画像ファイルパス
        entry: キャッシュエントリ
        cache_key: キャッシュキー
    """

    path: str
    entry: "CacheEntry"
    cache_key: dict[str, str | int]
