"""単一動画選定の再開可能なファイル操作."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from PIL import Image

RUN_LOCK_FILENAME = "run.lock"


def create_exclusive_temporary_file(
    directory: Path,
    *,
    prefix: str,
    suffix: str,
) -> tuple[int, Path]:
    """umaskを反映した予測不能なexclusive fileを作る."""
    directory.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    for _attempt in range(100):
        path = directory / f"{prefix}{secrets.token_hex(16)}{suffix}"
        try:
            return os.open(path, flags, 0o666), path
        except FileExistsError:
            continue
    raise RuntimeError(f"temporary fileを作成できません: {directory}")


def file_sha256(path: Path) -> str:
    """ファイル全体のSHA-256を返す."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def cache_directory_lock(cache_dir: Path) -> Iterator[None]:
    """同一cache rootのpipelineをadvisory lockで一つに制限する."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / RUN_LOCK_FILENAME
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_APPEND
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
        | os.O_NOFOLLOW
    )
    try:
        lock_fd = os.open(lock_path, flags, 0o600)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise RuntimeError(
                f"cache lockにsymlinkは使用できません: {lock_path}"
            ) from error
        raise
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise RuntimeError(f"cache lockがregular fileではありません: {lock_path}")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            raise RuntimeError(
                f"同じInput Video Directoryの処理がすでに実行中です: {cache_dir.parent}"
            ) from error
        locked = True
        yield
    finally:
        if locked:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


@contextmanager
def output_directory_lock(output_dir: Path) -> Iterator[None]:
    """同一Output Folderへのartifact書き込みを一つに制限する."""
    output_dir.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(output_dir, flags)
    locked = False
    try:
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            raise RuntimeError(
                f"同じOutput Folderを使う処理がすでに実行中です: {output_dir}"
            ) from error
        locked = True
        yield
    finally:
        if locked:
            fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)


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


def write_text_atomic(path: Path, content: str) -> None:
    """UTF-8 textを同一directory内でatomicに置換する."""
    temporary_fd, temporary_path = create_exclusive_temporary_file(
        path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
    )
    try:
        with os.fdopen(
            temporary_fd,
            mode="w",
            encoding="utf-8",
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    """JSONを同一ディレクトリ内でatomicに置換する."""
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    write_text_atomic(path, content)


def is_valid_image(path: Path) -> bool:
    """画像ファイルが存在し、Pillowで検証できるか返す."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.width > 0 and image.height > 0
    except (OSError, ValueError, Image.DecompressionBombError):
        return False
