"""Processing Stage runnerの単体テスト。"""

import json
from pathlib import Path

import pytest

from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.services.build_stage_fingerprint import (
    build_stage_fingerprint,
)
from src.video_selection.services.processing_stage_runner import (
    ProcessingStageRunner,
)
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver


def test_processing_stage_cannot_complete_out_of_order(tmp_path: Path) -> None:
    """順序外のProcessing Stageが完了されないこと。

    Arrange:
        - まだ一つも完了していないStage runnerが用意される
    Act:
        - 2番目のFrame Candidate抽出Stageが先に完了される
    Assert:
        - 順序違反として拒否され、cache artifactが作成されないこと
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    observer = RecordingRunObserver()
    runner = ProcessingStageRunner(cache_folder, observer)

    # Act / Assert
    with pytest.raises(
        ValueError,
        match="expected=discover-video-set, actual=extract-frame-candidates",
    ):
        runner.complete(
            ProcessingStage.EXTRACT_FRAME_CANDIDATES,
            semantic_input={},
            artifact={},
        )
    assert runner.completed_stages == ()
    assert observer.completed_stages == []
    assert not cache_folder.exists()


def test_partial_stage_artifact_is_replaced_before_completion(tmp_path: Path) -> None:
    """完了manifestのないStage artifactが再利用されないこと。

    Arrange:
        - 次Stageと同じfingerprint位置にpartial artifactだけがある
    Act:
        - Stage runnerがfresh artifactでStageを完了する
    Assert:
        - partial内容が使われず、fresh artifactと完了manifestが確定されること
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    observer = RecordingRunObserver()
    semantic_input = {"videos": ["fresh.mp4"]}
    fingerprint = build_stage_fingerprint(
        ProcessingStage.DISCOVER_VIDEO_SET,
        (),
        semantic_input,
    )
    stage_folder = (
        cache_folder
        / "walking-skeleton"
        / ProcessingStage.DISCOVER_VIDEO_SET.value
        / fingerprint.value
    )
    stage_folder.mkdir(parents=True)
    artifact_path = stage_folder / "artifact.json"
    artifact_path.write_text('{"videos": ["poison.mp4"]}\n', encoding="utf-8")
    runner = ProcessingStageRunner(cache_folder, observer)

    # Act
    runner.complete(
        ProcessingStage.DISCOVER_VIDEO_SET,
        semantic_input=semantic_input,
        artifact={"videos": ["fresh.mp4"]},
    )

    # Assert
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == {
        "videos": ["fresh.mp4"]
    }
    manifest = json.loads((stage_folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["fingerprint"] == fingerprint.value
