"""acceptance state/record用のatomic JSON操作。"""

import json
import os
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4


def write_atomic_json(path: Path, value: Mapping[str, object]) -> None:
    """同一directoryの一時fileをflushしてJSON objectをatomic replaceする。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def read_json_object(path: Path) -> dict[str, object] | None:
    """存在するvalid JSON objectだけを返す。"""
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, TypeError, ValueError):
        raise ValueError("Acceptance JSON stateが破損しています") from None
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Acceptance JSON stateにはobjectが必要です")
    return value
