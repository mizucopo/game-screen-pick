"""PcmAudioCheckpointの中断再開test。"""

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import cast

import pytest

from src.video_selection.models.pcm_audio_chunk import PcmAudioChunk
from src.video_selection.services.pcm_audio_checkpoint import PcmAudioCheckpoint


def test_resume_extracts_only_unfinished_sample_ranges(tmp_path: Path) -> None:
    """途中失敗後に完了rangeが再抽出されず連続実行と同じPCMになること。

    Arrange:
        - 3つのcanonical rangeと2番目で失敗するextractorが用意される
    Act:
        - 初回失敗後に同じcheckpointから処理が再開される
    Assert:
        - 再開extractorが未完了rangeだけを呼ばれること
        - 復元PCM列が失敗なしの連続実行と一致すること
    """
    # Arrange
    source_fingerprint = "a" * 64
    chunks = {
        0: _chunk(0, 4, b"\x01\x00" * 4),
        4: _chunk(4, 4, b"\x02\x00" * 4),
        8: _chunk(8, 2, b"\x03\x00" * 2),
    }
    first_calls: list[int] = []

    def interrupted_extract(
        sample_start: int,
        _maximum_sample_count: int,
    ) -> PcmAudioChunk | None:
        first_calls.append(sample_start)
        if sample_start == 4:
            raise RuntimeError("injected interruption")
        return chunks.get(sample_start)

    checkpoint = _checkpoint(tmp_path / "resumed", source_fingerprint)
    with pytest.raises(RuntimeError, match="injected interruption"):
        tuple(checkpoint.resolve(interrupted_extract))
    retry_calls: list[int] = []

    def retry_extract(
        sample_start: int,
        _maximum_sample_count: int,
    ) -> PcmAudioChunk | None:
        retry_calls.append(sample_start)
        return chunks.get(sample_start)

    continuous_calls: list[int] = []

    def continuous_extract(
        sample_start: int,
        _maximum_sample_count: int,
    ) -> PcmAudioChunk | None:
        continuous_calls.append(sample_start)
        return chunks.get(sample_start)

    # Act
    resumed = tuple(
        _checkpoint(tmp_path / "resumed", source_fingerprint).resolve(retry_extract)
    )
    continuous = tuple(
        _checkpoint(tmp_path / "continuous", source_fingerprint).resolve(
            continuous_extract
        )
    )

    # Assert
    assert first_calls == [0, 4]
    assert retry_calls == [4, 8]
    assert continuous_calls == [0, 4, 8]
    assert resumed == continuous


def test_discontinuous_cached_range_is_recomputed_without_losing_prior_range(
    tmp_path: Path,
) -> None:
    """前rangeと不連続な破損checkpointだけが再計算されること。

    Arrange:
        - 連続する2 rangeの確定checkpointが用意される
        - 2件目のPTSとmanifest hashが整合したまま不連続値へ変更される
    Act:
        - 同じPCM checkpointが再開される
    Assert:
        - 健全な先頭rangeが再抽出されないこと
        - 不連続な2件目だけが再抽出され連続PCM列へ修復されること
    """
    # Arrange
    cache = tmp_path / "cache"
    source_fingerprint = "a" * 64
    chunks = {
        0: _chunk(0, 4, b"\x01\x00" * 4),
        4: _chunk(4, 2, b"\x02\x00" * 2),
    }
    tuple(
        _checkpoint(cache, source_fingerprint).resolve(
            lambda sample_start, _maximum: chunks.get(sample_start)
        )
    )
    manifests = sorted(
        (cache / "work-units" / source_fingerprint / "pcm-audio-chunk").glob(
            "*/manifest.json"
        )
    )
    second_manifest_path = next(
        path
        for path in manifests
        if json.loads(path.read_text(encoding="utf-8"))["semantic_input"][
            "sample_start"
        ]
        == 4
    )
    second_root = second_manifest_path.parent
    artifact_path = second_root / "artifact.json"
    artifact = cast(
        dict[str, object],
        json.loads(artifact_path.read_text(encoding="utf-8")),
    )
    artifact["pts"] = 99
    artifact_bytes = (
        json.dumps(
            artifact,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    artifact_path.write_bytes(artifact_bytes)
    manifest = cast(
        dict[str, object],
        json.loads(second_manifest_path.read_text(encoding="utf-8")),
    )
    records = cast(list[dict[str, object]], manifest["artifacts"])
    artifact_record = next(
        record for record in records if record["path"] == "artifact.json"
    )
    artifact_record["size_bytes"] = len(artifact_bytes)
    artifact_record["sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
    second_manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    calls: list[int] = []

    def extract(
        sample_start: int,
        _maximum_sample_count: int,
    ) -> PcmAudioChunk | None:
        calls.append(sample_start)
        return chunks.get(sample_start)

    # Act
    resumed = tuple(_checkpoint(cache, source_fingerprint).resolve(extract))

    # Assert
    assert calls == [4]
    assert resumed == tuple(chunks.values())


def _checkpoint(cache: Path, source_fingerprint: str) -> PcmAudioCheckpoint:
    return PcmAudioCheckpoint(
        cache,
        source_fingerprint=source_fingerprint,
        stream_index=1,
        sample_rate=4,
        frame_sample_count=4,
        extraction_semantic_input={"media_runtime_identity": "runtime-a"},
    )


def _chunk(
    sample_start: int,
    sample_count: int,
    content: bytes,
) -> PcmAudioChunk:
    return PcmAudioChunk(
        stream_index=1,
        sample_start=sample_start,
        sample_count=sample_count,
        sample_rate=4,
        channel_count=1,
        sample_format="s16le",
        pts=sample_start,
        time_base=Fraction(1, 4),
        pcm_bytes=content,
    )
