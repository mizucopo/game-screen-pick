"""Completed Stage writerのconcurrency test。"""

import json
import threading
from pathlib import Path

import pytest

from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.services.build_stage_fingerprint import (
    build_stage_fingerprint,
)
from src.video_selection.services.completed_stage_writer import CompletedStageWriter


def test_same_fingerprint_writes_are_serialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同じStage Fingerprintの同時書き込みが直列化されること。

    Arrange:
        - first artifact writeを一時停止するfilesystem boundaryが用意される
        - 同じfingerprintへ書く2つのwriter threadが用意される
    Act:
        - first write中にsecond writeが開始される
    Assert:
        - first write解放までsecond writeが完了しないこと
        - 最初に開始したartifactがCompleted Stageとして保持されること
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    stage = ProcessingStage.DISCOVER_VIDEO_SET
    semantic_input = {"videos": ["video.mp4"]}
    fingerprint = build_stage_fingerprint(stage, (), semantic_input)
    first_write_started = threading.Event()
    release_first_write = threading.Event()
    second_write_finished = threading.Event()
    errors: list[BaseException] = []
    original_write_bytes = Path.write_bytes

    def controlled_write_bytes(path: Path, content: bytes) -> int:
        if (
            threading.current_thread().name == "first-stage-writer"
            and path.name == "artifact.json"
        ):
            first_write_started.set()
            if not release_first_write.wait(timeout=5):
                msg = "first artifact write was not released"
                raise TimeoutError(msg)
        return original_write_bytes(path, content)

    monkeypatch.setattr(Path, "write_bytes", controlled_write_bytes)

    def write_artifact(value: str, finished: threading.Event | None = None) -> None:
        try:
            CompletedStageWriter(cache_folder).write(
                stage,
                fingerprint,
                (),
                {"value": value},
            )
        except BaseException as error:
            errors.append(error)
        finally:
            if finished is not None:
                finished.set()

    first_thread = threading.Thread(
        target=write_artifact,
        args=("first",),
        name="first-stage-writer",
    )
    second_thread = threading.Thread(
        target=write_artifact,
        args=("second", second_write_finished),
        name="second-stage-writer",
    )

    # Act
    first_thread.start()
    assert first_write_started.wait(timeout=5)
    second_thread.start()
    try:
        second_completed_early = second_write_finished.wait(timeout=0.2)
    finally:
        release_first_write.set()
        first_thread.join(timeout=5)
        second_thread.join(timeout=5)

    # Assert
    assert not second_completed_early
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []
    artifact_path = (
        cache_folder
        / "walking-skeleton"
        / stage.value
        / fingerprint.value
        / "artifact.json"
    )
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == {"value": "first"}
