"""video selection用file操作の単体テスト."""

import json
import stat
from pathlib import Path

import pytest

from src.utils.video_selection_files import cache_directory_lock, write_json_atomic


def test_cache_directory_lock_rejects_symlink_without_creating_target(
    tmp_path: Path,
) -> None:
    """run.lock symlinkを辿らず外部targetも作成しないこと."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    external = tmp_path / "external.lock"
    (cache_dir / "run.lock").symlink_to(external)

    with (
        pytest.raises(RuntimeError, match="cache lock.*symlink"),
        cache_directory_lock(cache_dir),
    ):
        pass

    assert not external.exists()


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
    mode_probe = tmp_path / "mode-probe"
    mode_probe.touch()
    expected_mode = stat.S_IMODE(mode_probe.stat().st_mode)

    write_json_atomic(target, {"status": "complete"})

    assert external.read_text(encoding="utf-8") == "user-owned"
    assert legacy_temporary.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "complete"}
    assert stat.S_IMODE(target.stat().st_mode) == expected_mode
