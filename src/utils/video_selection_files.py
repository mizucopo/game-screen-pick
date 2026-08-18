"""単一動画選定の再開可能なファイル操作."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


def file_sha256(path: Path) -> str:
    """ファイル全体のSHA-256を返す."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampled_file_sha256(path: Path) -> str:
    """巨大動画の先頭・中央・末尾を用いた安定指紋を返す."""
    size = path.stat().st_size
    block_size = 1024 * 1024
    offsets = {
        0,
        max(0, size // 2 - block_size // 2),
        max(0, size - block_size),
    }
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as file:
        for offset in sorted(offsets):
            file.seek(offset)
            block = file.read(block_size)
            digest.update(offset.to_bytes(8, "big"))
            digest.update(len(block).to_bytes(8, "big"))
            digest.update(block)
    return digest.hexdigest()


def json_digest(payload: Any) -> str:
    """JSON互換値の決定的なSHA-256を返す."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Any:
    """UTF-8のJSONファイルを読む."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: Path, payload: Any) -> None:
    """JSONを同一ディレクトリ内でatomicに置換する."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def is_valid_image(path: Path) -> bool:
    """画像ファイルが存在し、Pillowで検証できるか返す."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.width > 0 and image.height > 0
    except (OSError, ValueError):
        return False
