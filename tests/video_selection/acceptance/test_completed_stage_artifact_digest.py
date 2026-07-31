"""Completed Stage artifact digestのtest。"""

from pathlib import Path

from src.video_selection.acceptance.completed_stage_artifact_digest import (
    completed_stage_artifact_digest,
)
from src.video_selection.models.completed_stage import CompletedStage
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.stage_fingerprint import StageFingerprint
from src.video_selection.services.completed_stage_writer import CompletedStageWriter


def test_digest_compares_actual_semantic_artifact_content(tmp_path: Path) -> None:
    """性能値を除く実artifact内容の差だけがdigestへ反映されること。

    Arrange:
        - 同じfingerprintで性能値だけが異なる二つのcacheが用意される
        - binary artifactだけが異なる第三のcacheが用意される
    Act:
        - 各cacheのCompleted Stage artifact digestが生成される
    Assert:
        - 性能値だけの差は一致しbinary内容の差は不一致になること
    """
    # Arrange
    same_binary = b"same-proxy"
    fixed_cache, fixed_stage = _write_stage(
        tmp_path / "fixed",
        wall_seconds=12.0,
        binary=same_binary,
    )
    auto_cache, auto_stage = _write_stage(
        tmp_path / "auto",
        wall_seconds=8.0,
        binary=same_binary,
    )
    changed_cache, changed_stage = _write_stage(
        tmp_path / "changed",
        wall_seconds=8.0,
        binary=b"changed-proxy",
    )

    # Act
    fixed_digest = completed_stage_artifact_digest(
        fixed_cache,
        (fixed_stage,),
    )
    auto_digest = completed_stage_artifact_digest(
        auto_cache,
        (auto_stage,),
    )
    changed_digest = completed_stage_artifact_digest(
        changed_cache,
        (changed_stage,),
    )

    # Assert
    assert auto_digest == fixed_digest
    assert changed_digest != fixed_digest


def _write_stage(
    cache_folder: Path,
    *,
    wall_seconds: float,
    binary: bytes,
) -> tuple[Path, CompletedStage]:
    fingerprint = StageFingerprint("a" * 64)

    def produce(stage_root: Path) -> dict[str, object]:
        (stage_root / "proxy.bin").write_bytes(binary)
        return {
            "schema": "test/artifact@1",
            "semantic_value": "same",
            "metrics": {
                "wall_seconds": wall_seconds,
                "cpu_seconds": wall_seconds / 2,
                "duration_seconds": wall_seconds,
                "input_seconds_per_wall_second": 2.0,
                "semantic_count": 1,
            },
        }

    writer = CompletedStageWriter(
        cache_folder,
        subject_namespace="videos",
        subject_fingerprint="b" * 64,
    )
    stage = writer.write_artifacts(
        ProcessingStage.SCAN_VIDEO,
        fingerprint,
        (),
        {"semantic": "same"},
        produce,
    )
    return cache_folder, stage
