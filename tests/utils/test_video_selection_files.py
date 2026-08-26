"""video selection用file操作の単体テスト."""

import json
from pathlib import Path

from src.utils.video_selection_files import write_json_atomic


def test_write_json_atomic_does_not_follow_fixed_temporary_symlink(
    tmp_path: Path,
) -> None:
    """予測可能な旧temporary pathのsymlinkを辿らないこと."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    target = cache_dir / "run-manifest.json"
    legacy_temporary = cache_dir / ".run-manifest.json.partial"
    external = tmp_path / "external.txt"
    external.write_text("user-owned", encoding="utf-8")
    legacy_temporary.symlink_to(external)

    write_json_atomic(target, {"status": "complete"})

    assert external.read_text(encoding="utf-8") == "user-owned"
    assert legacy_temporary.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "complete"}
