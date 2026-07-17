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
from src.video_selection.services.run_progress_tracker import RunProgressTracker
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


def test_video_stage_runner_completes_scan_before_candidate_extraction(
    tmp_path: Path,
) -> None:
    """動画単位のStageがscanからcandidate抽出の順で確定されること。

    Arrange:
        - Video Stage専用の順序を持つrunnerが用意される
    Act:
        - scanの複数artifactとcandidate抽出artifactが順番に確定される
    Assert:
        - 2つのCompleted Stageが動画用順序で通知されること
        - scanのproxy artifactが検証済みbundleから参照されること
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    observer = RecordingRunObserver()
    runner = ProcessingStageRunner(
        cache_folder,
        observer,
        subject_namespace="videos",
        subject_fingerprint="2" * 64,
        stage_order=(
            ProcessingStage.SCAN_VIDEO,
            ProcessingStage.EXTRACT_FRAME_CANDIDATES,
        ),
    )

    def produce_scan(stage_folder: Path) -> dict[str, object]:
        heartbeat_folder = stage_folder / "heartbeats"
        heartbeat_folder.mkdir()
        (heartbeat_folder / "000.jpg").write_bytes(b"heartbeat")
        return {"heartbeat_proxy_paths": ["heartbeats/000.jpg"]}

    # Act
    scan_bundle = runner.complete_artifacts(
        ProcessingStage.SCAN_VIDEO,
        {"scan_algorithm": "v1"},
        produce_scan,
    )
    runner.complete(
        ProcessingStage.EXTRACT_FRAME_CANDIDATES,
        {"candidate_algorithm": "v1"},
        {"candidate_ids": []},
    )

    # Assert
    assert [item.stage for item in runner.completed_stages] == [
        ProcessingStage.SCAN_VIDEO,
        ProcessingStage.EXTRACT_FRAME_CANDIDATES,
    ]
    assert observer.completed_stages == list(runner.completed_stages)
    assert scan_bundle.root.joinpath("heartbeats/000.jpg").read_bytes() == b"heartbeat"


def test_stage_can_select_only_semantic_upstream_dependencies(tmp_path: Path) -> None:
    """Stageが順序上流から意味的に依存するStageだけを選べること。

    Arrange:
        - discovery、model resolution、annotationの順序が用意される
        - model resolution fingerprintだけが異なる2 runが用意される
    Act:
        - annotationがdiscoveryだけを上流依存として確定・再利用される
    Assert:
        - model resolution変更後も同じannotationが再利用されること
        - manifestへ選択された上流fingerprintだけが記録されること
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    subject_fingerprint = "3" * 64
    stage_order = (
        ProcessingStage.DISCOVER_VIDEO_SET,
        ProcessingStage.RESOLVE_MODELS,
        ProcessingStage.ANNOTATE_CANDIDATES,
    )
    first = ProcessingStageRunner(
        cache_folder,
        RecordingRunObserver(),
        subject_namespace="video-sets",
        subject_fingerprint=subject_fingerprint,
        stage_order=stage_order,
    )
    discovery = first.complete(
        ProcessingStage.DISCOVER_VIDEO_SET,
        {"video_set": "same"},
        {"video_set": "same"},
    )
    first.complete(
        ProcessingStage.RESOLVE_MODELS,
        {"speech_model": "a"},
        {"speech_model": "a"},
    )
    annotation_input = {"candidate_model": "same"}
    annotation = first.complete(
        ProcessingStage.ANNOTATE_CANDIDATES,
        annotation_input,
        {"summary": "cached"},
        upstream_stages=(ProcessingStage.DISCOVER_VIDEO_SET,),
    )
    second = ProcessingStageRunner(
        cache_folder,
        RecordingRunObserver(),
        subject_namespace="video-sets",
        subject_fingerprint=subject_fingerprint,
        stage_order=stage_order,
    )
    second.complete(
        ProcessingStage.DISCOVER_VIDEO_SET,
        {"video_set": "same"},
        {"video_set": "same"},
    )
    second.complete(
        ProcessingStage.RESOLVE_MODELS,
        {"speech_model": "b"},
        {"speech_model": "b"},
    )

    # Act
    restored = second.reuse(
        ProcessingStage.ANNOTATE_CANDIDATES,
        annotation_input,
        lambda artifact: artifact["summary"],
        upstream_stages=(ProcessingStage.DISCOVER_VIDEO_SET,),
    )

    # Assert
    assert restored == "cached"
    manifest_path = (
        cache_folder
        / "video-sets"
        / subject_fingerprint
        / ProcessingStage.ANNOTATE_CANDIDATES.value
        / annotation.fingerprint.value
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["upstream_stage_fingerprints"] == [discovery.fingerprint.value]


def test_processing_stage_runner_emits_cache_and_completion_events(
    tmp_path: Path,
) -> None:
    """cache missからrecompute完了までtyped eventが順番に通知されること。

    Arrange:
        - run開始済みProgress Trackerとcold Stage cacheが用意される
    Act:
        - cache lookup後に一つのProcessing Stageが確定される
    Assert:
        - miss、recompute、Stage完了が同じStage番号で通知されること
    """
    # Arrange
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer, clock=lambda: 10.0)
    progress.start_run()
    runner = ProcessingStageRunner(
        tmp_path / "cache",
        observer,
        subject_namespace="video-sets",
        subject_fingerprint="4" * 64,
        stage_order=(ProcessingStage.DISCOVER_VIDEO_SET,),
        progress=progress,
        total_stage_count=1,
    )
    semantic_input = {"video_set": "cold"}

    # Act
    restored = runner.reuse(
        ProcessingStage.DISCOVER_VIDEO_SET,
        semantic_input,
        lambda artifact: artifact,
    )
    runner.complete(
        ProcessingStage.DISCOVER_VIDEO_SET,
        semantic_input,
        {"video_set": "completed"},
    )

    # Assert
    assert restored is None
    assert tuple(
        (
            event.kind,
            event.stage_index,
            event.cache_hit_count,
            event.cache_miss_count,
            event.reuse_count,
            event.recompute_count,
        )
        for event in observer.progress_events
    ) == (
        ("run_started", None, 0, 0, 0, 0),
        ("stage_started", 1, 0, 0, 0, 0),
        ("cache", 1, 0, 1, 0, 0),
        ("cache", 1, 0, 1, 0, 1),
        ("stage_completed", 1, 0, 0, 0, 0),
    )


def test_processing_stage_runner_records_recompute_duration_for_eta(
    tmp_path: Path,
) -> None:
    """Stage runnerのrecompute完了時間がETA sampleへ記録されること。

    Arrange:
        - 同じComparable Work Seriesを5回完了するcold Stage runnerが用意される
    Act:
        - 6件目のStageで残り1件のETAが通知される
    Assert:
        - 実完了時間を使ったETAがavailableとして通知されること
    """
    # Arrange
    observer = RecordingRunObserver()
    current_time = [0.0]
    progress = RunProgressTracker(observer, clock=lambda: current_time[0])
    progress.start_run()

    def produce_artifact(_stage_folder: Path) -> dict[str, object]:
        current_time[0] += 10.0
        return {"status": "completed"}

    for run_index in range(5):
        runner = ProcessingStageRunner(
            tmp_path / f"cache-{run_index}",
            observer,
            subject_namespace="video-sets",
            subject_fingerprint=f"{run_index:064x}",
            stage_order=(ProcessingStage.DISCOVER_VIDEO_SET,),
            progress=progress,
            work_unit_kind="video_set",
        )
        runner.complete_artifacts(
            ProcessingStage.DISCOVER_VIDEO_SET,
            {"run_index": run_index},
            produce_artifact,
        )
    progress.start_stage(
        ProcessingStage.DISCOVER_VIDEO_SET,
        work_unit_kind="video_set",
    )
    current_time[0] += 30.0

    # Act
    progress.progress(
        remaining_reuse_count=0,
        remaining_recompute_count=1,
    )

    # Assert
    event = observer.progress_events[-1]
    assert (event.estimation_state, event.eta_seconds) == ("available", 10.0)
