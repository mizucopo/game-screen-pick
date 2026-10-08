"""他言語からも使えるInput Video Identity・digestの固定vector."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.services.video_phase_cache import (
    build_video_identity,
    phase_key,
    stable_frame_id,
)
from src.utils.video_selection_files import json_digest

ROOT = Path(__file__).parents[1] / "fixtures" / "rust_migration"
EXPECTED = json.loads((ROOT / "wire_expectations.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("vector", EXPECTED["json_digest_vectors"])
def test_digest_uses_recorded_utf8_bytes(vector: dict[str, Any]) -> None:
    """Unicode・整数/float・exponentのwire表現まで旧cacheとの境界に含める."""
    assert (
        hashlib.sha256(vector["canonical_utf8"].encode("utf-8")).hexdigest()
        == vector["sha256"]
    )
    assert json_digest(vector["payload"]) == vector["sha256"]


def test_identity_and_stable_frame_vectors(tmp_path: Path) -> None:
    """絶対path・mtime・内容ではなく相対名とsizeで同一性を決める."""
    expected = EXPECTED["identity"]
    for name, byte in (("original", b"a"), ("moved", b"b")):
        directory = tmp_path / name
        directory.mkdir()
        video = directory / expected["relative_path"]
        video.write_bytes(byte * expected["size"])
        identity = build_video_identity(directory, video)
        assert identity.key == expected["key"]
        assert identity.relative_path == expected["relative_path"]
        for sample in expected["frame_ids"]:
            assert (
                stable_frame_id(identity.key, sample["sample_index"])
                == sample["frame_id"]
            )


def test_phase_key_vector_and_version_boundary() -> None:
    """versionを変えたphaseは同じsemantic inputでも別keyになる."""
    expected = EXPECTED["phase"]
    assert (
        phase_key(expected["name"], expected["version"], expected["conditions"])
        == expected["key"]
    )
    assert (
        phase_key(expected["name"], expected["version"] + 1, expected["conditions"])
        != expected["key"]
    )
