"""パスとファイルメタ情報のデータ構造."""

import os
from dataclasses import dataclass


@dataclass
class PathMetadata:
    """パスとファイルメタ情報を保持するデータクラス.

    Attributes:
        path: 元のパス
        absolute_path: 絶対パス（resolve結果）
        file_stat: ファイルstat結果
        cache_key: キャッシュキー
    """

    path: str
    absolute_path: str | None = None
    file_stat: os.stat_result | None = None
    cache_key: dict[str, str | int] | None = None
