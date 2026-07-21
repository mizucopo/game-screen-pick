"""Video Stage processorの統合style test。"""

import threading
from dataclasses import replace
from pathlib import Path

import pytest

from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.media_runtime_identity import MediaRuntimeIdentity
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.services.discover_video_set import discover_video_set
from src.video_selection.services.run_progress_tracker import RunProgressTracker
from src.video_selection.services.video_stage_processor import VideoStageProcessor
from tests.video_selection.fakes.fake_speech_runtime import FakeSpeechRuntime
from tests.video_selection.fakes.fake_video_stage_media_runtime import (
    FakeVideoStageMediaRuntime,
)
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver


def _configuration(input_folder: Path, output_folder: Path) -> EffectiveConfiguration:
    return EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=output_folder,
    )


def test_context_collection_is_the_third_source_local_video_stage(
    tmp_path: Path,
) -> None:
    """Context Collectionがcandidate抽出後の3番目のVideo Stageになること。

    Arrange:
        - context streamを持たない一つのVideo Sourceが用意される
    Act:
        - Video Stage processorが実行される
    Assert:
        - scan、candidate抽出、context収集の順でCompleted Stageになること
        - context stream不在がsource-localな正常結果として保持されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")

    # Act
    result = VideoStageProcessor(
        FakeVideoStageMediaRuntime(),
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        discover_video_set(input_folder),
        _configuration(input_folder, tmp_path / "output"),
    )[0]

    # Assert
    assert [stage.stage for stage in result.completed_stages] == [
        ProcessingStage.SCAN_VIDEO,
        ProcessingStage.EXTRACT_FRAME_CANDIDATES,
        ProcessingStage.COLLECT_CONTEXT,
    ]
    assert [(item.status, item.reason_code) for item in result.context.outcomes] == [
        ("absent", "no_subtitle_stream"),
        ("absent", "no_audio_stream"),
    ]


def test_video_stage_progress_follows_video_order_with_monotonic_stage_index(
    tmp_path: Path,
) -> None:
    """複数Videoの3 StageがVideo Orderと単調なStage番号で通知されること。

    Arrange:
        - 自然順の2動画とrun開始済みProgress Trackerが用意される
    Act:
        - Video Stage processorで両動画が直列処理される
    Assert:
        - scan、extract、contextが各Video Orderに結び付いて通知されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "01-first.mp4").write_bytes(b"first-video")
    (input_folder / "02-second.mp4").write_bytes(b"second-video")
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer, clock=lambda: 10.0)
    progress.start_run()
    processor = VideoStageProcessor(
        FakeVideoStageMediaRuntime(),
        FakeSpeechRuntime(),
        observer,
        progress=progress,
    )

    # Act
    processor.process(
        discover_video_set(input_folder),
        _configuration(input_folder, tmp_path / "output"),
    )

    # Assert
    started = tuple(
        (
            event.stage,
            event.stage_index,
            event.stage_count,
            event.video_order,
            event.video_count,
            event.video_relative_path,
        )
        for event in observer.progress_events
        if event.kind == "stage_started"
    )
    assert started == (
        (ProcessingStage.SCAN_VIDEO, 1, None, 1, 2, "01-first.mp4"),
        (
            ProcessingStage.EXTRACT_FRAME_CANDIDATES,
            2,
            None,
            1,
            2,
            "01-first.mp4",
        ),
        (ProcessingStage.COLLECT_CONTEXT, 3, None, 1, 2, "01-first.mp4"),
        (ProcessingStage.SCAN_VIDEO, 4, None, 2, 2, "02-second.mp4"),
        (
            ProcessingStage.EXTRACT_FRAME_CANDIDATES,
            5,
            None,
            2,
            2,
            "02-second.mp4",
        ),
        (ProcessingStage.COLLECT_CONTEXT, 6, None, 2, 2, "02-second.mp4"),
    )


def test_video_scans_are_prepared_before_ordered_downstream_stages(
    tmp_path: Path,
) -> None:
    """scanが先行確定されdownstreamがVideo Order順に処理されること。

    Arrange:
        - 異なる内容を持つ2動画のVideo SetとVideo Stage processorが用意される
    Act:
        - 初回処理後に動画をrenameして順序とdownstream設定を変えて再実行される
        - 続いてCandidate Moment Densityだけを変えて再実行される
    Assert:
        - 全probeとscanの後にrefinementがVideo Order順で処理されること
        - rename、順序、downstream変更では両Stageが再利用されること
        - density変更ではscanだけが再利用されcandidate抽出だけ再計算されること
        - scene一時画像がCompleted Stageへ永続化されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    first_path = input_folder / "01-first.mp4"
    second_path = input_folder / "02-second.mp4"
    first_path.write_bytes(b"first-video")
    second_path.write_bytes(b"second-video")
    configuration = _configuration(input_folder, tmp_path / "output")
    first_video_set = discover_video_set(input_folder)
    first_runtime = FakeVideoStageMediaRuntime()

    # Act
    first_results = VideoStageProcessor(
        first_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(first_video_set, configuration)

    # Assert
    assert first_runtime.call_order[:2] == [
        ("probe", "01-first.mp4"),
        ("probe", "02-second.mp4"),
    ]
    assert sorted(first_runtime.call_order[2:4]) == [
        ("scan", "01-first.mp4"),
        ("scan", "02-second.mp4"),
    ]
    assert first_runtime.call_order[4:] == [
        ("refine", "01-first.mp4"),
        ("refine", "02-second.mp4"),
    ]
    assert all(
        [item.stage for item in result.completed_stages]
        == [
            ProcessingStage.SCAN_VIDEO,
            ProcessingStage.EXTRACT_FRAME_CANDIDATES,
            ProcessingStage.COLLECT_CONTEXT,
        ]
        for result in first_results
    )
    assert not tuple(configuration.processing_cache_folder.rglob(".scene-proxies"))

    # Arrange
    first_path.rename(input_folder / "99-renamed.mp4")
    second_path.rename(input_folder / "01-renamed.mp4")
    reordered_video_set = discover_video_set(input_folder)
    cached_runtime = FakeVideoStageMediaRuntime()
    downstream_changed = replace(
        configuration,
        image_count=7,
        scene_hint="downstream only",
        spoiler_sensitivity="high",
        similarity_threshold=0.9,
    )

    # Act
    cached_results = VideoStageProcessor(
        cached_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(reordered_video_set, downstream_changed)

    # Assert
    assert cached_runtime.scan_calls == []
    assert cached_runtime.range_calls == []
    assert {
        result.source.fingerprint: tuple(
            stage.fingerprint.value for stage in result.completed_stages
        )
        for result in cached_results
    } == {
        result.source.fingerprint: tuple(
            stage.fingerprint.value for stage in result.completed_stages
        )
        for result in first_results
    }

    # Arrange
    density_runtime = FakeVideoStageMediaRuntime()
    density_changed = replace(configuration, candidate_density_per_minute=4.0)

    # Act
    VideoStageProcessor(
        density_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(reordered_video_set, density_changed)

    # Assert
    assert density_runtime.scan_calls == []
    assert [path.name for path in density_runtime.range_calls] == [
        "01-renamed.mp4",
        "99-renamed.mp4",
    ]


def test_three_video_scans_run_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """24 logical CPUでは独立した3動画のscanが同時実行されること。

    Arrange:
        - 3 workerを許可する24 logical CPUと3動画が用意される
        - 3 scanの開始を同期するbarrierが用意される
    Act:
        - Video Stage processorが実行される
    Assert:
        - 3件のscanが同時にactiveになること
        - 各scanのCPU時間が他の並列scanを重複計上しないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    for index in range(1, 4):
        (input_folder / f"0{index}-video.mp4").write_bytes(f"video-{index}".encode())
    barrier = threading.Barrier(3)
    active_count = 0
    peak_count = 0
    count_lock = threading.Lock()

    def synchronize_scans(_path: Path) -> None:
        nonlocal active_count, peak_count
        with count_lock:
            active_count += 1
            peak_count = max(peak_count, active_count)
        barrier.wait(timeout=2)
        with count_lock:
            active_count -= 1

    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor.os.cpu_count",
        lambda: 24,
    )

    # Act
    results = VideoStageProcessor(
        FakeVideoStageMediaRuntime(
            on_scan_video=synchronize_scans,
            reported_scan_cpu_seconds=7.0,
        ),
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        discover_video_set(input_folder),
        _configuration(input_folder, tmp_path / "output"),
    )

    # Assert
    assert peak_count == 3
    assert all(7.0 <= result.scan.metrics.cpu_seconds < 8.0 for result in results)


def test_interrupt_cancels_active_video_scans(tmp_path: Path) -> None:
    """scan中のKeyboardInterruptでactive subprocess cancellationが要求されること。

    Arrange:
        - scan開始時にKeyboardInterruptとなるMedia Runtimeが用意される
    Act:
        - Video Stage processorが実行される
    Assert:
        - interruptが維持され、active scan cancellationが一度要求されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")

    def interrupt_scan(_path: Path) -> None:
        raise KeyboardInterrupt

    runtime = FakeVideoStageMediaRuntime(on_scan_video=interrupt_scan)

    # Act / Assert
    with pytest.raises(KeyboardInterrupt):
        VideoStageProcessor(
            runtime,
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(
            discover_video_set(input_folder),
            _configuration(input_folder, tmp_path / "output"),
        )
    assert runtime.cancel_video_scans_call_count == 1


@pytest.mark.parametrize("failure_position", [0, 1, 2])
def test_completed_parallel_scans_survive_first_middle_last_video_failure(
    tmp_path: Path,
    failure_position: int,
) -> None:
    """scan失敗後も並行して正常完了したVideo Scanが再利用されること。

    Arrange:
        - 自然順の3動画と指定Videoのscanだけ失敗するMedia Runtimeが用意される
    Act:
        - 失敗runの後に同じVideo Setとcacheで再実行される
    Assert:
        - 正常完了したscanは再利用され、未確定scanだけが再計算されること
        - extractionはretry時にVideo Order順で処理されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_names = ("01-first.mp4", "02-middle.mp4", "03-last.mp4")
    for index, name in enumerate(video_names, start=1):
        (input_folder / name).write_bytes(f"video-{index}".encode())
    configuration = _configuration(input_folder, tmp_path / "output")
    video_set = discover_video_set(input_folder)
    failed_name = video_names[failure_position]

    def fail_selected_video(path: Path) -> None:
        if path.name == failed_name:
            raise OSError("injected video scan failure")

    failing_runtime = FakeVideoStageMediaRuntime(on_scan_video=fail_selected_video)

    # Act
    with pytest.raises(OSError, match="injected video scan failure"):
        VideoStageProcessor(
            failing_runtime,
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(video_set, configuration)
    retry_runtime = FakeVideoStageMediaRuntime()
    results = VideoStageProcessor(
        retry_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)

    # Assert
    attempted = {path.name for path in failing_runtime.scan_calls}
    completed = attempted - {failed_name}
    expected_scan_recompute = [name for name in video_names if name not in completed]
    assert failed_name in attempted
    assert failing_runtime.range_calls == []
    assert [path.name for path in retry_runtime.scan_calls] == expected_scan_recompute
    assert [path.name for path in retry_runtime.range_calls] == list(video_names)
    assert len(results) == 3


def test_corrupt_candidate_proxy_recomputes_only_candidate_stage(
    tmp_path: Path,
) -> None:
    """破損したcandidate proxyが検知され上流scanを残して再計算されること。

    Arrange:
        - scanとcandidate抽出がCompleted Stageとして確定済みである
        - candidate proxyの一つがmanifest確定後に破損される
    Act:
        - 同じVideo Identityと設定でVideo Stageが再実行される
    Assert:
        - scanは再利用されcandidate抽出だけが再実行されること
        - 破損bytesが新しいMJPEG proxyへ置き換えられること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    configuration = _configuration(input_folder, tmp_path / "output")
    video_set = discover_video_set(input_folder)
    initial_result = VideoStageProcessor(
        FakeVideoStageMediaRuntime(),
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)[0]
    proxy_path = initial_result.extraction.candidates[0].proxy_path
    assert proxy_path is not None
    proxy_path.write_bytes(b"corrupt-proxy")
    repair_runtime = FakeVideoStageMediaRuntime()

    # Act
    repaired_result = VideoStageProcessor(
        repair_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)[0]

    # Assert
    assert repair_runtime.scan_calls == []
    assert [path.name for path in repair_runtime.range_calls] == ["video.mp4"]
    repaired_proxy_path = repaired_result.extraction.candidates[0].proxy_path
    assert repaired_proxy_path is not None
    assert repaired_proxy_path.read_bytes() != b"corrupt-proxy"


def test_metadata_change_is_checked_before_video_stage(
    tmp_path: Path,
) -> None:
    """Video Setのmetadata変更が最初のsourceのprobe前に検知されること。

    Arrange:
        - 異なる内容を持つ2動画の発見済みVideo Setが用意される
        - initial validation後に2本目の内容とmetadataが変更される
    Act:
        - Video Stage処理がVideo Order順に実行される
    Assert:
        - 最初のsourceのprobe前に変更が拒否されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    first_path = input_folder / "01-first.mp4"
    second_path = input_folder / "02-second.mp4"
    first_path.write_bytes(b"first-video")
    second_path.write_bytes(b"second-video")
    configuration = _configuration(input_folder, tmp_path / "output")
    video_set = discover_video_set(input_folder)

    def rewrite_second_source() -> None:
        second_path.write_bytes(b"changed-second-video")

    runtime = FakeVideoStageMediaRuntime(on_preflight=rewrite_second_source)

    # Act / Assert
    with pytest.raises(ValueError, match="Video Set snapshotが変更されました"):
        VideoStageProcessor(
            runtime,
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(
            video_set,
            configuration,
        )
    assert runtime.call_order == []


def test_refinement_is_streamed_between_distant_moment_groups(
    tmp_path: Path,
) -> None:
    """離れたMoment groupの全RGB frameが同時に保持されないこと。

    Arrange:
        - 離れた2つのCandidate Momentとstreaming検査付きruntimeが用意される
    Act:
        - 一つのrange scanからFrame Candidateが抽出される
    Assert:
        - 後側groupのdecode継続前に前側groupのproxyが書かれること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    configuration = _configuration(input_folder, tmp_path / "output")
    runtime = FakeVideoStageMediaRuntime(
        distant_moments=True,
        require_streaming_refinement=True,
    )

    # Act
    result = VideoStageProcessor(
        runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        discover_video_set(input_folder),
        configuration,
    )[0]

    # Assert
    assert len(result.extraction.moments) == 2
    assert all(moment.frame_candidate_ids for moment in result.extraction.moments)


def test_runtime_build_identity_change_recomputes_scan_stage(tmp_path: Path) -> None:
    """同じversionでもbuild identity変更時にscan cacheが再計算されること。

    Arrange:
        - 同じVideo Identityとversionで異なるbuild digestを返すruntimeが用意される
    Act:
        - 最初のruntimeでcache確定後、別buildのruntimeで再実行される
    Assert:
        - scan-videoと下流candidate抽出が再計算されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_path = input_folder / "video.mp4"
    video_path.write_bytes(b"video-content")
    configuration = _configuration(input_folder, tmp_path / "output")
    video_set = discover_video_set(input_folder)
    first_runtime = FakeVideoStageMediaRuntime(
        runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "a" * 64,
        )
    )
    VideoStageProcessor(
        first_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        video_set,
        configuration,
    )
    changed_runtime = FakeVideoStageMediaRuntime(
        runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "b" * 64,
        )
    )

    # Act
    VideoStageProcessor(
        changed_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        video_set,
        configuration,
    )

    # Assert
    assert changed_runtime.scan_calls == [video_path]
    assert changed_runtime.range_calls == [video_path]


def test_stage_metrics_include_current_process_and_full_stage_time(
    tmp_path: Path,
) -> None:
    """Video Stage metricにcurrent processを含む全Stage costが記録されること。

    Arrange:
        - native metricを0で返しscanとrefinement中にCPUを消費するruntimeが用意される
    Act:
        - Video Stageが初回計算される
    Assert:
        - scanとcandidate抽出のwall時間とCPU時間が正であること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    configuration = _configuration(input_folder, tmp_path / "output")
    runtime = FakeVideoStageMediaRuntime(
        cpu_burn_seconds=0.02,
        reported_scan_wall_seconds=0.0,
        reported_scan_cpu_seconds=0.0,
    )

    # Act
    result = VideoStageProcessor(
        runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        discover_video_set(input_folder),
        configuration,
    )[0]

    # Assert
    assert result.scan.metrics.wall_seconds >= 0.01
    assert result.scan.metrics.cpu_seconds >= 0.01
    assert result.extraction_metrics.wall_seconds >= 0.01
    assert result.extraction_metrics.cpu_seconds >= 0.01
