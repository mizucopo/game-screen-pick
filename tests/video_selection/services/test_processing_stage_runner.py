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
    runner = ProcessingStageRunner(
        cache_folder,
        observer,
        subject_namespace="video-sets",
        subject_fingerprint="a" * 64,
    )

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
    subject_fingerprint = "b" * 64
    observer = RecordingRunObserver()
    semantic_input = {"videos": ["fresh.mp4"]}
    fingerprint = build_stage_fingerprint(
        ProcessingStage.DISCOVER_VIDEO_SET,
        (),
        semantic_input,
    )
    stage_folder = (
        cache_folder
        / "video-sets"
        / subject_fingerprint
        / ProcessingStage.DISCOVER_VIDEO_SET.value
        / fingerprint.value
    )
    stage_folder.mkdir(parents=True)
    artifact_path = stage_folder / "artifact.json"
    artifact_path.write_text('{"videos": ["poison.mp4"]}\n', encoding="utf-8")
    runner = ProcessingStageRunner(
        cache_folder,
        observer,
        subject_namespace="video-sets",
        subject_fingerprint=subject_fingerprint,
    )

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
    assert manifest["stage_fingerprint"] == fingerprint.value


def test_completed_stage_artifact_is_immutable_for_same_fingerprint(
    tmp_path: Path,
) -> None:
    """同じStage FingerprintのCompleted Stageが上書きされないこと。

    Arrange:
        - first artifactを持つCompleted Stageが確定済みである
    Act:
        - 別内容のartifactで同じStage Fingerprintが再度完了される
    Assert:
        - 最初に確定したartifactとmanifestがそのまま保持されること
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    subject_fingerprint = "c" * 64
    semantic_input = {"videos": [{"path": "video.mp4", "sha256": "digest"}]}
    first_runner = ProcessingStageRunner(
        cache_folder,
        RecordingRunObserver(),
        subject_namespace="video-sets",
        subject_fingerprint=subject_fingerprint,
    )
    completed_stage = first_runner.complete(
        ProcessingStage.DISCOVER_VIDEO_SET,
        semantic_input=semantic_input,
        artifact={"value": "first"},
    )

    # Act
    ProcessingStageRunner(
        cache_folder,
        RecordingRunObserver(),
        subject_namespace="video-sets",
        subject_fingerprint=subject_fingerprint,
    ).complete(
        ProcessingStage.DISCOVER_VIDEO_SET,
        semantic_input=semantic_input,
        artifact={"value": "second"},
    )

    # Assert
    artifact_path = (
        cache_folder
        / "video-sets"
        / subject_fingerprint
        / ProcessingStage.DISCOVER_VIDEO_SET.value
        / completed_stage.fingerprint.value
        / "artifact.json"
    )
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == {"value": "first"}


def test_snapshot_validation_failure_prevents_stage_cache_mutation(
    tmp_path: Path,
) -> None:
    """Stage直前のsnapshot validation失敗でcacheが変更されないこと。

    Arrange:
        - snapshot変更errorを返すbefore-stage callbackが用意される
    Act:
        - 最初のProcessing Stage completionが試行される
    Assert:
        - errorが返されsubject cacheとobserverが未変更であること
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    observer = RecordingRunObserver()

    def reject_changed_snapshot() -> None:
        msg = "Video Set snapshotが変更されました"
        raise ValueError(msg)

    runner = ProcessingStageRunner(
        cache_folder,
        observer,
        subject_namespace="video-sets",
        subject_fingerprint="d" * 64,
        before_stage=reject_changed_snapshot,
    )

    # Act / Assert
    with pytest.raises(ValueError, match="Video Set snapshotが変更されました"):
        runner.complete(
            ProcessingStage.DISCOVER_VIDEO_SET,
            semantic_input={"videos": []},
            artifact={"videos": []},
        )
    assert observer.completed_stages == []
    assert not (cache_folder / "video-sets").exists()
