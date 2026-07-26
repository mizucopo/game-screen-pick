"""Video Stage processorの統合style test。"""

import json
import os
import signal
import threading
import time
from concurrent.futures import CancelledError
from dataclasses import replace
from pathlib import Path

import pytest

from src.video_selection.acceptance.completed_stage_artifact_digest import (
    canonicalize_completed_stage_artifact_value,
)
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.media_runtime_identity import MediaRuntimeIdentity
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.video_scan_resource_sample import (
    VideoScanResourceSample,
)
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


def _semantic_stage_artifacts(root: Path) -> dict[Path, bytes]:
    """run時間metricとmanifestを除いたCompleted Stage成果物を返す。"""
    artifacts: dict[Path, bytes] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative_path = path.relative_to(root)
        if path.suffix != ".json":
            artifacts[relative_path] = path.read_bytes()
            continue
        value: object = json.loads(path.read_text(encoding="utf-8"))
        artifacts[relative_path] = json.dumps(
            canonicalize_completed_stage_artifact_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    return artifacts


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


def test_video_stage_pipeline_preserves_order_and_source_local_cache(
    tmp_path: Path,
) -> None:
    """pipelining後もVideo Orderとsource-local cacheが維持されること。

    Arrange:
        - 異なる内容を持つ2動画のVideo SetとVideo Stage processorが用意される
    Act:
        - 初回処理後に動画をrenameして順序とdownstream設定を変えて再実行される
        - 続いてCandidate Moment Densityだけを変えて再実行される
    Assert:
        - probeとscan後のrefinementがVideo Order順で処理されること
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
    configuration = replace(
        _configuration(input_folder, tmp_path / "output"),
        video_scan_workers=1,
        video_scan_auto_max_workers=1,
    )
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
    assert sorted(path.name for path in first_runtime.scan_calls) == [
        "01-first.mp4",
        "02-second.mp4",
    ]
    assert [path.name for path in first_runtime.range_calls] == [
        "01-first.mp4",
        "02-second.mp4",
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
        video_scan_workers="auto",
        video_scan_auto_max_workers=6,
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
        replace(
            _configuration(input_folder, tmp_path / "output"),
            video_scan_workers=3,
        ),
    )

    # Assert
    assert peak_count == 3
    assert all(7.0 <= result.scan.metrics.cpu_seconds < 8.0 for result in results)


def test_nvdec_auto_grows_to_six_workers_when_gpu_has_rolling_headroom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU余力が継続するNVDEC環境で6 workerまで増加されること。

    Arrange:
        - NVDEC、24論理CPU、auto上限6、GPU余力sampleと12動画が用意される
    Act:
        - Video Stage processorが実行される
    Assert:
        - 保守的な3 workerからrolling判断で6 workerまで増加されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    for index in range(1, 13):
        (input_folder / f"{index:02d}-video.mp4").write_bytes(f"video-{index}".encode())

    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor.os.cpu_count",
        lambda: 24,
    )
    sample = VideoScanResourceSample(
        cpu_percent=45.0,
        memory_percent=50.0,
        decoder_percent=40.0,
        gpu_percent=20.0,
        vram_percent=22.0,
        disk_busy_percent=40.0,
        disk_read_mib_per_second=300.0,
    )

    configuration = replace(
        _configuration(input_folder, tmp_path / "output"),
        decode_backend="nvdec",
        video_scan_workers="auto",
        video_scan_auto_max_workers=6,
    )
    runtime = FakeVideoStageMediaRuntime(
        on_scan_video=lambda _path: time.sleep(0.02),
    )
    processor = VideoStageProcessor(
        runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
        resource_sampler=lambda: sample,
    )

    # Act
    processor.process(discover_video_set(input_folder), configuration)

    # Assert
    assert len(runtime.scan_calls) == 12
    assert processor.parallelism_diagnostics["initial_workers"] == 3
    assert processor.parallelism_diagnostics["peak_workers"] == 6


def test_fixed_video_scan_workers_skip_resource_sampling(tmp_path: Path) -> None:
    """固定worker指定では動的resource samplingが実行されないこと。

    Arrange:
        - 固定1 worker、一つの動画、呼出回数を記録するsamplerが用意される
    Act:
        - Video Stage processorが実行される
    Assert:
        - samplerが呼ばれずscanが完了されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    sample_calls = 0

    def sample_resources() -> None:
        nonlocal sample_calls
        sample_calls += 1
        return None

    processor = VideoStageProcessor(
        FakeVideoStageMediaRuntime(),
        FakeSpeechRuntime(),
        RecordingRunObserver(),
        resource_sampler=sample_resources,
    )
    configuration = replace(
        _configuration(input_folder, tmp_path / "output"),
        video_scan_workers=1,
    )

    # Act
    processor.process(discover_video_set(input_folder), configuration)

    # Assert
    assert sample_calls == 0


def test_fixed_three_and_auto_produce_identical_completed_stage_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """固定3とautoで同じCompleted Stage成果物が生成されること。

    Arrange:
        - 同じ4動画を持つ独立cacheの固定3用とauto用Video Setが用意される
        - 24論理CPUとNVDEC余力を示すresource sampleが用意される
    Act:
        - 固定3とautoで各Video Setが別々に処理される
    Assert:
        - 全Completed Stage fingerprintとsemantic artifact bytesが一致すること
    """
    # Arrange
    fixed_input = tmp_path / "fixed-videos"
    auto_input = tmp_path / "auto-videos"
    fixed_input.mkdir()
    auto_input.mkdir()
    for index in range(1, 5):
        name = f"{index:02d}-video.mp4"
        payload = f"video-{index}".encode()
        (fixed_input / name).write_bytes(payload)
        (auto_input / name).write_bytes(payload)
    healthy = VideoScanResourceSample(
        cpu_percent=45.0,
        memory_percent=50.0,
        decoder_percent=40.0,
        gpu_percent=20.0,
        vram_percent=22.0,
        disk_busy_percent=40.0,
        disk_read_mib_per_second=300.0,
    )
    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor.os.cpu_count",
        lambda: 24,
    )
    fixed_configuration = replace(
        _configuration(fixed_input, tmp_path / "fixed-output"),
        decode_backend="nvdec",
        video_scan_workers=3,
    )
    auto_configuration = replace(
        _configuration(auto_input, tmp_path / "auto-output"),
        decode_backend="nvdec",
        video_scan_workers="auto",
        video_scan_auto_max_workers=6,
    )

    # Act
    fixed_results = VideoStageProcessor(
        FakeVideoStageMediaRuntime(),
        FakeSpeechRuntime(),
        RecordingRunObserver(),
        resource_sampler=lambda: healthy,
    ).process(discover_video_set(fixed_input), fixed_configuration)
    auto_results = VideoStageProcessor(
        FakeVideoStageMediaRuntime(),
        FakeSpeechRuntime(),
        RecordingRunObserver(),
        resource_sampler=lambda: healthy,
    ).process(discover_video_set(auto_input), auto_configuration)

    # Assert
    assert [
        tuple(stage.fingerprint for stage in result.completed_stages)
        for result in auto_results
    ] == [
        tuple(stage.fingerprint for stage in result.completed_stages)
        for result in fixed_results
    ]
    fixed_stage_root = fixed_configuration.processing_cache_folder / "videos"
    auto_stage_root = auto_configuration.processing_cache_folder / "videos"
    fixed_artifacts = _semantic_stage_artifacts(fixed_stage_root)
    auto_artifacts = _semantic_stage_artifacts(auto_stage_root)
    assert auto_artifacts == fixed_artifacts


def test_pressure_changes_admission_without_interrupting_active_scans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pressure時にactive scanを止めず次taskの投入だけが抑制されること。

    Arrange:
        - 3から6 workerへ増加できる15動画と、先頭6本だけ即時完了するscanが用意される
        - 増加後の3本へ継続するCPU pressureを返すsample列が用意される
    Act:
        - rolling余力による増加後、pressure対象3本が完了される
    Assert:
        - 14本目が開始されずactive scan cancellationも要求されないこと
        - 残りを解放するとVideo Orderどおり全結果が返されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_names = tuple(f"{index:02d}-video.mp4" for index in range(1, 16))
    for index, name in enumerate(video_names, start=1):
        (input_folder / name).write_bytes(f"video-{index}".encode())
    release_pressure_scans = threading.Event()
    release_rest = threading.Event()
    started_names: list[str] = []
    started_lock = threading.Lock()

    def block_scans(path: Path) -> None:
        with started_lock:
            started_names.append(path.name)
        if path.name in video_names[6:9]:
            assert release_pressure_scans.wait(timeout=10)
        elif path.name not in video_names[:6]:
            assert release_rest.wait(timeout=10)
        else:
            time.sleep(0.2)

    healthy = VideoScanResourceSample(
        cpu_percent=45.0,
        memory_percent=50.0,
        decoder_percent=40.0,
        gpu_percent=20.0,
        vram_percent=22.0,
        disk_busy_percent=40.0,
        disk_read_mib_per_second=300.0,
    )
    pressure = replace(healthy, cpu_percent=94.0)
    sample_lock = threading.Lock()
    sample_count = 0

    def sample_resources() -> VideoScanResourceSample:
        nonlocal sample_count
        with sample_lock:
            sample_count += 1
            return healthy if sample_count <= 8 else pressure

    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor.os.cpu_count",
        lambda: 24,
    )
    runtime = FakeVideoStageMediaRuntime(on_scan_video=block_scans)
    processor = VideoStageProcessor(
        runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
        resource_sampler=sample_resources,
    )
    configuration = replace(
        _configuration(input_folder, tmp_path / "output"),
        decode_backend="nvdec",
    )
    results: list[tuple[str, ...]] = []
    failures: list[BaseException] = []

    def run_processor() -> None:
        try:
            processed = processor.process(
                discover_video_set(input_folder),
                configuration,
            )
            results.append(tuple(item.source.relative_path for item in processed))
        except BaseException as error:
            failures.append(error)

    processing_thread = threading.Thread(target=run_processor)
    processing_thread.start()
    growth_deadline = time.monotonic() + 5
    while (
        processor.parallelism_diagnostics.get("peak_workers") != 6
        and time.monotonic() < growth_deadline
    ):
        time.sleep(0.01)
    assert processor.parallelism_diagnostics["peak_workers"] == 6

    # Act
    release_pressure_scans.set()
    deadline = time.monotonic() + 5
    while (
        processor.parallelism_diagnostics.get("final_workers") not in {1, 2, 3, 4, 5}
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)

    # Assert
    assert processor.parallelism_diagnostics["final_workers"] in {1, 2, 3, 4, 5}
    with started_lock:
        assert video_names[14] not in started_names
    assert runtime.cancel_video_scans_call_count == 0

    release_rest.set()
    processing_thread.join(timeout=10)
    assert not processing_thread.is_alive()
    assert failures == []
    assert results == [video_names]


def test_downstream_starts_while_later_video_scans_are_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """先頭Videoのdownstreamが後続scanの完了前に開始されること。

    Arrange:
        - 3 workerで同時開始し、後続2動画だけ待機するscanが用意される
        - 先頭Videoのrefinementが後続scanを解放する同期境界が用意される
    Act:
        - Video Stage processorが実行される
    Assert:
        - 後続scanがactiveな間に先頭Videoのrefinementが開始されること
        - 全Video Stage resultがVideo Order順に返されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_names = ("01-first.mp4", "02-second.mp4", "03-third.mp4")
    for index, name in enumerate(video_names, start=1):
        (input_folder / name).write_bytes(f"video-{index}".encode())
    scans_started = threading.Barrier(3)
    release_later_scans = threading.Event()
    first_refinement_started = threading.Event()

    def synchronize_scans(path: Path) -> None:
        scans_started.wait(timeout=2)
        if path.name != video_names[0]:
            assert release_later_scans.wait(timeout=2)

    def observe_refinement(path: Path) -> None:
        if path.name == video_names[0]:
            first_refinement_started.set()
            release_later_scans.set()

    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor.os.cpu_count",
        lambda: 24,
    )
    runtime = FakeVideoStageMediaRuntime(
        on_scan_video=synchronize_scans,
        on_scan_video_frame_ranges=observe_refinement,
    )

    # Act
    results = VideoStageProcessor(
        runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        discover_video_set(input_folder),
        replace(
            _configuration(input_folder, tmp_path / "output"),
            video_scan_workers=3,
        ),
    )

    # Assert
    assert first_refinement_started.is_set()
    assert [result.source.relative_path for result in results] == list(video_names)


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
    processor = VideoStageProcessor(
        runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    )

    # Act / Assert
    with pytest.raises(KeyboardInterrupt):
        processor.process(
            discover_video_set(input_folder),
            _configuration(input_folder, tmp_path / "output"),
        )
    assert runtime.cancel_video_scans_call_count == 1
    scan_wall_seconds = processor.parallelism_diagnostics["scan_wall_seconds"]
    assert isinstance(scan_wall_seconds, int | float)
    assert scan_wall_seconds > 0


def test_interrupt_does_not_start_queued_video_scans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """割り込み後に待機中のVideo Scanが開始されないこと。

    Arrange:
        - 3 workerに対して4動画と、先頭3 scanを停止するMedia Runtimeが用意される
        - 先頭3 scanの開始後にprocessへSIGINTが送られる
    Act:
        - Video Stage processorが実行される
    Assert:
        - interruptが維持され、待機中の4本目が開始されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_names = (
        "01-first.mp4",
        "02-second.mp4",
        "03-third.mp4",
        "04-queued.mp4",
    )
    for index, name in enumerate(video_names, start=1):
        (input_folder / name).write_bytes(f"video-{index}".encode())
    active_scans_started = threading.Event()
    release_active_scans = threading.Event()
    started_scan_names: list[str] = []
    started_scan_names_lock = threading.Lock()

    def block_active_scans(path: Path) -> None:
        with started_scan_names_lock:
            started_scan_names.append(path.name)
            if len(started_scan_names) == 3:
                active_scans_started.set()
        if path.name != video_names[-1]:
            assert release_active_scans.wait(timeout=5)

    interrupt_wait_timed_out = threading.Event()

    def send_sigint_after_active_scans_start() -> None:
        if not active_scans_started.wait(timeout=5):
            interrupt_wait_timed_out.set()
        os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor.os.cpu_count",
        lambda: 24,
    )
    runtime = FakeVideoStageMediaRuntime(
        on_scan_video=block_active_scans,
        on_cancel_video_scans=release_active_scans.set,
    )
    interrupt_thread = threading.Thread(
        target=send_sigint_after_active_scans_start,
        daemon=True,
    )
    interrupt_thread.start()

    # Act / Assert
    with pytest.raises(KeyboardInterrupt):
        VideoStageProcessor(
            runtime,
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(
            discover_video_set(input_folder),
            replace(
                _configuration(input_folder, tmp_path / "output"),
                video_scan_workers=3,
            ),
        )
    interrupt_thread.join(timeout=1)
    assert not interrupt_thread.is_alive()
    assert not interrupt_wait_timed_out.is_set()
    assert len(started_scan_names) == 3
    assert set(started_scan_names) == set(video_names[:3])


def test_scan_failure_cancels_queued_sibling_scans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """通常のscan失敗後に待機中の兄弟scanが開始されないこと。

    Arrange:
        - 1 workerに対して3動画と先頭で失敗するMedia Runtimeが用意される
    Act:
        - Video Stage processorが実行される
    Assert:
        - 失敗が維持され、待機中scanがcancelされること
        - active scan cancellationが一度要求されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    for index in range(1, 4):
        (input_folder / f"0{index}-video.mp4").write_bytes(f"video-{index}".encode())

    def fail_first_scan(_path: Path) -> None:
        raise OSError("injected scan failure")

    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor.os.cpu_count",
        lambda: 8,
    )
    runtime = FakeVideoStageMediaRuntime(on_scan_video=fail_first_scan)

    # Act / Assert
    with pytest.raises(OSError, match="injected scan failure"):
        VideoStageProcessor(
            runtime,
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(
            discover_video_set(input_folder),
            _configuration(input_folder, tmp_path / "output"),
        )
    assert [path.name for path in runtime.scan_calls] == ["01-video.mp4"]
    assert runtime.cancel_video_scans_call_count == 1


def test_changed_source_content_does_not_commit_prepared_scan(
    tmp_path: Path,
) -> None:
    """scan中に変更されたsourceのprepared cacheが確定されないこと。

    Arrange:
        - 発見後のscan中だけ同じsourceを書き換えて元へ戻すruntimeが用意される
    Act:
        - 失敗run後に現在のVideo Setから再実行される
    Assert:
        - snapshot変更として失敗し、次回scanがcache再利用されず再計算されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_path = input_folder / "video.mp4"
    original = b"video-content"
    video_path.write_bytes(original)
    configuration = _configuration(input_folder, tmp_path / "output")
    discovered = discover_video_set(input_folder)

    def temporarily_change_source(path: Path) -> None:
        path.write_bytes(b"other-content")
        path.write_bytes(original)

    # Act
    with pytest.raises(ValueError, match="Video Set snapshotが変更されました"):
        VideoStageProcessor(
            FakeVideoStageMediaRuntime(
                on_scan_video=temporarily_change_source,
            ),
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(discovered, configuration)
    retry_runtime = FakeVideoStageMediaRuntime()
    VideoStageProcessor(
        retry_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(discover_video_set(input_folder), configuration)

    # Assert
    assert retry_runtime.scan_calls == [video_path]


def test_background_scan_emits_heartbeat_while_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """background scan待機中にactive Stage heartbeatが通知されること。

    Arrange:
        - heartbeat間隔より長く実行されるcold scanが用意される
        - run開始済みProgress Trackerが用意される
    Act:
        - Video Stage processorが実行される
    Assert:
        - scan完了前にscan-videoへ結び付いたheartbeatが通知されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer)
    progress.start_run()
    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor."
        "_SCAN_PROGRESS_HEARTBEAT_SECONDS",
        0.01,
    )

    def slow_scan(_path: Path) -> None:
        threading.Event().wait(timeout=0.04)

    # Act
    VideoStageProcessor(
        FakeVideoStageMediaRuntime(on_scan_video=slow_scan),
        FakeSpeechRuntime(),
        observer,
        progress=progress,
    ).process(
        discover_video_set(input_folder),
        _configuration(input_folder, tmp_path / "output"),
    )

    # Assert
    heartbeats = [
        event for event in observer.progress_events if event.kind == "heartbeat"
    ]
    assert heartbeats
    assert all(event.stage is ProcessingStage.SCAN_VIDEO for event in heartbeats)


@pytest.mark.parametrize("failure_position", [0, 1, 2])
def test_completed_parallel_scans_survive_first_middle_last_video_failure(
    tmp_path: Path,
    failure_position: int,
) -> None:
    """scan失敗後も先に完了したVideo Stageが再利用されること。

    Arrange:
        - 自然順の3動画と指定Videoのscanだけ失敗するMedia Runtimeが用意される
    Act:
        - 失敗runの後に同じVideo Setとcacheで再実行される
    Assert:
        - 正常完了したscanは再利用され、未確定scanだけが再計算されること
        - 失敗検知前に完了した先頭Videoのdownstreamはretryで再計算されないこと
        - 残るextractionはretry時にVideo Order順で処理されること
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
    completed_downstream = [path.name for path in failing_runtime.range_calls]
    assert completed_downstream == list(video_names[: len(completed_downstream)])
    assert len(completed_downstream) <= failure_position
    assert sorted(path.name for path in retry_runtime.scan_calls) == (
        expected_scan_recompute
    )
    assert [path.name for path in retry_runtime.range_calls] == list(
        video_names[len(completed_downstream) :]
    )
    assert len(results) == 3


def test_primary_scan_failure_is_not_masked_by_sibling_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一次scan障害が兄弟scanの取消例外に覆われず返されること。

    Arrange:
        - 2番目のscanが実行中に待機し、3番目のscanが一次障害になる
        - cancellation要求を受けた2番目のscanが取消例外を返す
    Act:
        - 3動画のVideo Stage処理が実行される
    Assert:
        - Video Order上で先に待たれる取消例外ではなく一次障害が返されること
    """
    # Arrange
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    for index, name in enumerate(
        ("01-first.mp4", "02-cancelled.mp4", "03-failed.mp4"),
        start=1,
    ):
        (input_folder / name).write_bytes(f"video-{index}".encode())
    cancelled_scan_started = threading.Event()
    release_cancelled_scan = threading.Event()

    def coordinate_scan_failure(path: Path) -> None:
        if path.name == "02-cancelled.mp4":
            cancelled_scan_started.set()
            release_cancelled_scan.wait(timeout=1)
            raise CancelledError
        if path.name == "03-failed.mp4":
            assert cancelled_scan_started.wait(timeout=1)
            raise OSError("primary video scan failure")

    runtime = FakeVideoStageMediaRuntime(
        on_scan_video=coordinate_scan_failure,
        on_cancel_video_scans=release_cancelled_scan.set,
    )

    # Act
    with pytest.raises(OSError, match="primary video scan failure"):
        VideoStageProcessor(
            runtime,
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(
            discover_video_set(input_folder),
            replace(
                _configuration(input_folder, tmp_path / "output"),
                video_scan_workers=3,
            ),
        )

    # Assert
    assert runtime.cancel_video_scans_call_count == 1


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
    """Video Stage metricに所有threadとchild processのcostが記録されること。

    Arrange:
        - native metricを0で返すruntimeが用意される
        - Stage threadとdecoder childがCPUを消費する
    Act:
        - Video Stageが初回計算される
    Assert:
        - scanとcandidate抽出に所有するwall時間とCPU時間が記録されること
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
        reported_refinement_child_cpu_seconds=1.0,
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
    assert result.extraction_metrics.cpu_seconds >= 1.01


def test_extraction_cpu_metric_excludes_background_scan_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """後続Videoのbackground scan CPUがcandidate抽出へ計上されないこと。

    Arrange:
        - 2 workerで先頭Videoの抽出中に後続scan threadだけがCPUを消費する
    Act:
        - Video Stage processorが両Videoをpipeliningする
    Assert:
        - 先頭candidate抽出のCPU時間へ後続scanの消費が含まれないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "01-first.mp4").write_bytes(b"first-video")
    (input_folder / "02-background.mp4").write_bytes(b"background-video")
    extraction_started = threading.Event()
    background_scan_finished = threading.Event()

    def coordinate_background_scan(path: Path) -> None:
        if path.name != "02-background.mp4":
            return
        assert extraction_started.wait(timeout=5)
        started_at = time.thread_time()
        while time.thread_time() - started_at < 0.4:
            pass
        background_scan_finished.set()

    def wait_for_background_scan(path: Path) -> None:
        if path.name != "01-first.mp4":
            return
        extraction_started.set()
        assert background_scan_finished.wait(timeout=5)

    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor.os.cpu_count",
        lambda: 16,
    )
    runtime = FakeVideoStageMediaRuntime(
        on_scan_video=coordinate_background_scan,
        on_scan_video_frame_ranges=wait_for_background_scan,
    )

    # Act
    first = VideoStageProcessor(
        runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        discover_video_set(input_folder),
        _configuration(input_folder, tmp_path / "output"),
    )[0]

    # Assert
    assert first.extraction_metrics.cpu_seconds < 0.2
