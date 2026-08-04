"""Video Stage processorの統合style test。"""

import hashlib
import json
import os
import shutil
import signal
import threading
import time
from concurrent.futures import CancelledError
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from src.video_selection.acceptance.completed_stage_artifact_digest import (
    canonicalize_completed_stage_artifact_value,
)
from src.video_selection.models.checkpoint_operation import CheckpointOperation
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.media_probe import MediaProbe
from src.video_selection.models.media_runtime_identity import MediaRuntimeIdentity
from src.video_selection.models.media_stream import MediaStream
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.video_scan_resource_sample import (
    VideoScanResourceSample,
)
from src.video_selection.models.video_stage_result import VideoStageResult
from src.video_selection.services.checkpoint_version import checkpoint_version
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


def _rewrite_hash_consistent_artifact(
    checkpoint_folder: Path,
    artifact: dict[str, object],
) -> None:
    """artifactとmanifest recordを同じ破損内容へ揃えて書き換える。"""
    artifact_path = checkpoint_folder / "artifact.json"
    manifest_path = checkpoint_folder / "manifest.json"
    artifact_bytes = (
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    artifact_path.write_bytes(artifact_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_record = next(
        item for item in manifest["artifacts"] if item["path"] == "artifact.json"
    )
    artifact_record["size_bytes"] = len(artifact_bytes)
    artifact_record["sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    assert result.completed_stages[1].semantic_input["refinement_group_contract"] == {
        "version": checkpoint_version(CheckpointOperation.FRAME_REFINEMENT_GROUP)
    }
    assert result.completed_stages[2].semantic_input["checkpoint_contracts"] == {}


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


def test_legacy_scan_without_frame_timing_hint_reuses_all_stage_cache(
    tmp_path: Path,
) -> None:
    """旧Scan artifactでも有効な全Stage cacheが再利用されること。

    Arrange:
        - 全Video Stageが確定され、親Scan artifactからresource hintが除かれる
    Act:
        - 同じVideo Sourceと設定で再実行される
    Assert:
        - Video ScanとRefinement Groupが再decodeされないこと
        - 旧artifactのhint欠落だけが逐次fallbackとして復元されること
        - Frame CandidateとStage Fingerprintが維持されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    configuration = _configuration(input_folder, tmp_path / "output")
    initial = VideoStageProcessor(
        FakeVideoStageMediaRuntime(),
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)[0]
    scan_root = (
        configuration.processing_cache_folder
        / "videos"
        / video_set.sources[0].fingerprint
        / ProcessingStage.SCAN_VIDEO.value
    )
    scan_folder = next(
        path
        for path in scan_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    artifact = json.loads((scan_folder / "artifact.json").read_text(encoding="utf-8"))
    artifact.pop("minimum_frame_delta_ts")
    artifact.pop("maximum_frame_count_per_pts")
    _rewrite_hash_consistent_artifact(scan_folder, artifact)
    retry_runtime = FakeVideoStageMediaRuntime(
        minimum_frame_delta_ts=1,
        maximum_frame_count_per_pts=2,
    )

    # Act
    reused = VideoStageProcessor(
        retry_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)[0]

    # Assert
    assert retry_runtime.scan_partition_calls == []
    assert retry_runtime.range_calls == []
    assert reused.scan.minimum_frame_delta_ts is None
    assert reused.scan.maximum_frame_count_per_pts is None
    assert tuple(stage.fingerprint for stage in reused.completed_stages) == tuple(
        stage.fingerprint for stage in initial.completed_stages
    )
    assert tuple(
        (candidate.identifier, candidate.image_bytes)
        for candidate in reused.extraction.candidates
    ) == tuple(
        (candidate.identifier, candidate.image_bytes)
        for candidate in initial.extraction.candidates
    )


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
        on_scan_video=lambda _path: time.sleep(0.2),
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

    # Act
    # Assert
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

    # Act
    # Assert
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

    # Act
    # Assert
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


def test_completed_scan_partition_survives_later_partition_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """後続scan partition失敗後も完了済みpartitionが再利用されること。

    Arrange:
        - 2秒動画を1秒partitionへ分割し、2件目だけ初回に失敗させる
    Act:
        - 初回失敗後に同じVideo SourceのVideo Stageが再実行される
    Assert:
        - retryでは未完了の末尾partitionだけがdecodeされること
        - 再開結果と中断なし結果のtimeline、proxy、candidateが一致すること
    """
    # Arrange
    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor._SCAN_PARTITION_SECONDS",
        1.0,
    )
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    configuration = _configuration(input_folder, tmp_path / "output")
    scan_call_count = 0

    def fail_second_partition(_path: Path) -> None:
        nonlocal scan_call_count
        scan_call_count += 1
        if scan_call_count == 2:
            raise OSError("injected second scan partition failure")

    failing_runtime = FakeVideoStageMediaRuntime(on_scan_video=fail_second_partition)
    with pytest.raises(
        OSError,
        match="injected second scan partition failure",
    ):
        VideoStageProcessor(
            failing_runtime,
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(video_set, configuration)
    retry_runtime = FakeVideoStageMediaRuntime()

    # Act
    resumed = VideoStageProcessor(
        retry_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)[0]
    uninterrupted_input = tmp_path / "uninterrupted-videos"
    uninterrupted_input.mkdir()
    (uninterrupted_input / "video.mp4").write_bytes(b"video-content")
    uninterrupted = VideoStageProcessor(
        FakeVideoStageMediaRuntime(),
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        discover_video_set(uninterrupted_input),
        _configuration(uninterrupted_input, tmp_path / "uninterrupted-output"),
    )[0]

    # Assert
    assert [
        (start, end) for _path, start, end in failing_runtime.scan_partition_calls
    ] == [(0, 10), (10, None)]
    assert [
        (start, end) for _path, start, end in retry_runtime.scan_partition_calls
    ] == [(10, None)]
    assert resumed.scan.timeline == uninterrupted.scan.timeline
    assert tuple(
        (heartbeat.source_pts, heartbeat.proxy_path.read_bytes())
        for heartbeat in resumed.scan.heartbeats
    ) == tuple(
        (heartbeat.source_pts, heartbeat.proxy_path.read_bytes())
        for heartbeat in uninterrupted.scan.heartbeats
    )
    assert resumed.scan.scene_signals == uninterrupted.scan.scene_signals
    assert tuple(
        (candidate.identifier, candidate.image_bytes)
        for candidate in resumed.extraction.candidates
    ) == tuple(
        (candidate.identifier, candidate.image_bytes)
        for candidate in uninterrupted.extraction.candidates
    )
    assert resumed.scan.metrics.decode_pass_count == 2


def test_hash_consistent_wrong_scan_partition_recomputes_only_that_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """要求範囲と違うScan partitionだけが破棄され再decodeされること。

    Arrange:
        - 2 partitionと親Scan Stageが正常に確定される
        - 後半partitionのoriginだけがhash整合を保って前半範囲へ改変される
        - 親Scan Stageが失われ子partitionからの再集約が必要になる
    Act:
        - 同じVideo SourceのVideo Stageが再実行される
    Assert:
        - 健全な前半partitionが保持され後半partitionだけが再計算されること
        - 修復結果のtimelineとcandidateが元の結果と一致すること
    """
    # Arrange
    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor._SCAN_PARTITION_SECONDS",
        1.0,
    )
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    configuration = _configuration(input_folder, tmp_path / "output")
    initial = VideoStageProcessor(
        FakeVideoStageMediaRuntime(),
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)[0]
    partition_root = (
        configuration.processing_cache_folder
        / "work-units"
        / video_set.sources[0].fingerprint
        / "video-scan-partition"
    )
    partition_folders = tuple(
        path
        for path in partition_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    assert len(partition_folders) == 2
    second_folder = next(
        path
        for path in partition_folders
        if json.loads((path / "artifact.json").read_text(encoding="utf-8"))[
            "origin_pts"
        ]
        == 10
    )
    second_artifact = json.loads(
        (second_folder / "artifact.json").read_text(encoding="utf-8")
    )
    second_artifact["origin_pts"] = 0
    _rewrite_hash_consistent_artifact(second_folder, second_artifact)
    shutil.rmtree(
        configuration.processing_cache_folder
        / "videos"
        / video_set.sources[0].fingerprint
        / ProcessingStage.SCAN_VIDEO.value
    )
    retry_runtime = FakeVideoStageMediaRuntime()

    # Act
    repaired = VideoStageProcessor(
        retry_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)[0]

    # Assert
    assert [
        (start, end) for _path, start, end in retry_runtime.scan_partition_calls
    ] == [(10, None)]
    assert repaired.scan.timeline == initial.scan.timeline
    assert tuple(
        (candidate.identifier, candidate.image_bytes)
        for candidate in repaired.extraction.candidates
    ) == tuple(
        (candidate.identifier, candidate.image_bytes)
        for candidate in initial.extraction.candidates
    )


def test_hash_consistent_wrong_parent_scan_reuses_scan_partitions(
    tmp_path: Path,
) -> None:
    """別streamを指す親Scanだけが破棄されpartitionから修復されること。

    Arrange:
        - 親Scanと子partitionが正常に確定される
        - 親Scanのstream indexがhash整合を保って別streamへ改変される
    Act:
        - 同じVideo SourceのVideo Stageが再実行される
    Assert:
        - 親Scanだけが再構築されpartition decodeは再実行されないこと
        - 修復後の意味的なStage成果物が初回結果と一致すること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    configuration = _configuration(input_folder, tmp_path / "output")
    VideoStageProcessor(
        FakeVideoStageMediaRuntime(),
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)
    scan_root = (
        configuration.processing_cache_folder
        / "videos"
        / source.fingerprint
        / ProcessingStage.SCAN_VIDEO.value
    )
    scan_folder = next(
        path
        for path in scan_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    expected_artifacts = _semantic_stage_artifacts(scan_folder)
    artifact = json.loads((scan_folder / "artifact.json").read_text(encoding="utf-8"))
    artifact["primary_stream"]["index"] = 99
    _rewrite_hash_consistent_artifact(scan_folder, artifact)
    retry_runtime = FakeVideoStageMediaRuntime()

    # Act
    VideoStageProcessor(
        retry_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)

    # Assert
    assert retry_runtime.scan_partition_calls == []
    assert _semantic_stage_artifacts(scan_folder) == expected_artifacts


def test_container_duration_only_schedules_fixed_scan_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stream duration欠落時もcontainer durationで固定partitionが作られること。

    Arrange:
        - stream durationがなく2秒のcontainer durationを持つ動画が用意される
        - Video Scan partitionが1秒へ設定される
    Act:
        - Video Stage processorが実行される
    Assert:
        - container durationがstream tickへ変換され2件のpartitionが処理されること
        - 最後のpartitionがEOFまで開かれること
        - partition境界を含む最小PTS差がresource hintへ集約されること
    """
    # Arrange
    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor._SCAN_PARTITION_SECONDS",
        1.0,
    )
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    media_probe = MediaProbe(
        format_names=("matroska",),
        streams=(
            MediaStream(
                index=0,
                kind="video",
                codec_name="ffv1",
                time_base=Fraction(1, 10),
                start_pts=0,
                duration_ts=None,
                width=64,
                height=48,
                sample_rate=None,
                channels=None,
                language=None,
                is_default=True,
                is_forced=False,
                is_attached_picture=False,
            ),
        ),
        duration=Fraction(2),
    )
    runtime = FakeVideoStageMediaRuntime(media_probe=media_probe)

    # Act
    result = VideoStageProcessor(
        runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        discover_video_set(input_folder),
        _configuration(input_folder, tmp_path / "output"),
    )[0]

    # Assert
    assert [(start, end) for _path, start, end in runtime.scan_partition_calls] == [
        (0, 10),
        (10, None),
    ]
    assert result.scan.metrics.decode_pass_count == 2
    assert result.scan.minimum_frame_delta_ts == 1
    assert result.scan.maximum_frame_count_per_pts == 1


def test_duration_hint_tail_does_not_require_an_empty_scan_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最終frameを越えるduration hintで空の末尾partitionが要求されないこと。

    Arrange:
        - 実frameより0.1秒長い2.1秒のcontainer duration hintが用意される
        - 1秒partitionと、3回目のdecodeを拒否するMedia Runtimeが用意される
    Act:
        - Video Stage processorが実行される
    Assert:
        - 完全区間1件と、その次のopen-ended区間だけが処理されること
    """
    # Arrange
    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor._SCAN_PARTITION_SECONDS",
        1.0,
    )
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    media_probe = MediaProbe(
        format_names=("matroska",),
        streams=(
            MediaStream(
                index=0,
                kind="video",
                codec_name="ffv1",
                time_base=Fraction(1, 10),
                start_pts=0,
                duration_ts=None,
                width=64,
                height=48,
                sample_rate=None,
                channels=None,
                language=None,
                is_default=True,
                is_forced=False,
                is_attached_picture=False,
            ),
        ),
        duration=Fraction(21, 10),
    )
    scan_attempt_count = 0

    def reject_empty_tail(_path: Path) -> None:
        nonlocal scan_attempt_count
        scan_attempt_count += 1
        if scan_attempt_count == 3:
            raise AssertionError("空の末尾partitionをdecodeしてはいけません")

    runtime = FakeVideoStageMediaRuntime(
        media_probe=media_probe,
        on_scan_video=reject_empty_tail,
    )

    # Act
    result = VideoStageProcessor(
        runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        discover_video_set(input_folder),
        _configuration(input_folder, tmp_path / "output"),
    )[0]

    # Assert
    assert [(start, end) for _path, start, end in runtime.scan_partition_calls] == [
        (0, 10),
        (10, None),
    ]
    assert result.scan.metrics.decode_pass_count == 2


def test_overstated_container_duration_stops_after_confirmed_video_eof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """長いcontainer tailが映像EOF後の必須partitionを増やさないこと。

    Arrange:
        - 実frameが約1秒で終わり4秒のcontainer durationを持つ動画が用意される
        - Video Scan partitionが1秒へ設定される
    Act:
        - Video Stage processorが実行される
    Assert:
        - 最初の空区間からEOFまでが確認され、後続境界が処理されないこと
        - 映像frameを持つpartitionのtimelineが失われないこと
    """
    # Arrange
    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor._SCAN_PARTITION_SECONDS",
        1.0,
    )
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    media_probe = MediaProbe(
        format_names=("matroska",),
        streams=(
            MediaStream(
                index=0,
                kind="video",
                codec_name="ffv1",
                time_base=Fraction(1, 10),
                start_pts=0,
                duration_ts=None,
                width=64,
                height=48,
                sample_rate=None,
                channels=None,
                language=None,
                is_default=True,
                is_forced=False,
                is_attached_picture=False,
            ),
        ),
        duration=Fraction(4),
    )
    video_set = discover_video_set(input_folder)
    configuration = _configuration(input_folder, tmp_path / "output")
    runtime = FakeVideoStageMediaRuntime(media_probe=media_probe)

    # Act
    result = VideoStageProcessor(
        runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)[0]
    shutil.rmtree(
        configuration.processing_cache_folder
        / "videos"
        / video_set.sources[0].fingerprint
        / ProcessingStage.SCAN_VIDEO.value
    )
    retry_runtime = FakeVideoStageMediaRuntime(media_probe=media_probe)
    repaired = VideoStageProcessor(
        retry_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)[0]

    # Assert
    assert [(start, end) for _path, start, end in runtime.scan_partition_calls] == [
        (0, 10),
        (10, 20),
        (20, 30),
        (20, None),
    ]
    assert result.scan.timeline.origin_pts == 0
    assert result.scan.timeline.duration.seconds == 2
    assert retry_runtime.scan_partition_calls == []
    assert repaired.scan.timeline == result.scan.timeline


def test_empty_partition_preserves_frames_after_a_long_timestamp_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """空partition後のframeがEOF確認scanで保持されること。

    Arrange:
        - 0秒と40秒にframeを持ち50秒のcontainer durationを持つ動画が用意される
        - Video Scan partitionが10秒へ設定される
    Act:
        - Video Stage processorが実行される
    Assert:
        - 10秒からの空区間が検出され、同じ開始点からEOFまでscanされること
        - 40秒の後半frameがscan結果に保持されること
    """
    # Arrange
    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor._SCAN_PARTITION_SECONDS",
        10.0,
    )
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    media_probe = MediaProbe(
        format_names=("matroska",),
        streams=(
            MediaStream(
                index=0,
                kind="video",
                codec_name="ffv1",
                time_base=Fraction(1, 10),
                start_pts=0,
                duration_ts=None,
                width=64,
                height=48,
                sample_rate=None,
                channels=None,
                language=None,
                is_default=True,
                is_forced=False,
                is_attached_picture=False,
            ),
        ),
        duration=Fraction(50),
    )
    runtime = FakeVideoStageMediaRuntime(
        media_probe=media_probe,
        distant_moments=True,
        scan_frame_pts=(0, 400),
    )

    # Act
    result = VideoStageProcessor(
        runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        discover_video_set(input_folder),
        _configuration(input_folder, tmp_path / "output"),
    )[0]

    # Assert
    assert [(start, end) for _path, start, end in runtime.scan_partition_calls] == [
        (0, 100),
        (100, 200),
        (100, None),
    ]
    assert [frame.source_pts for frame in result.scan.heartbeats] == [0, 400]
    assert result.scan.timeline.duration.seconds == 41


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
        - scanと健全なRefinement Work Unitは再利用されること
        - candidate抽出StageだけがWork Unitから再構築されること
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
    assert repair_runtime.range_calls == []
    repaired_proxy_path = repaired_result.extraction.candidates[0].proxy_path
    assert repaired_proxy_path is not None
    assert repaired_proxy_path.read_bytes() != b"corrupt-proxy"


def test_candidate_proxy_permission_failure_preserves_completed_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """candidate proxyのaccess障害でCompleted Stageが削除されないこと。

    Arrange:
        - scanとcandidate抽出がCompleted Stageとして確定済みである
        - manifest整合確認後のproxy type検査だけがPermissionErrorになる
    Act:
        - 同じVideo Identityと設定でVideo Stageが再実行される
    Assert:
        - access障害が返されcandidate再抽出が開始されないこと
        - 確定済みproxy bytesが変更されないこと
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
    original_bytes = proxy_path.read_bytes()
    original_lstat = Path.lstat
    proxy_lstat_count = 0

    def deny_domain_type_check(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal proxy_lstat_count
        if path == proxy_path:
            proxy_lstat_count += 1
            if proxy_lstat_count == 2:
                raise PermissionError("injected proxy permission failure")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", deny_domain_type_check)
    retry_runtime = FakeVideoStageMediaRuntime()

    # Act
    # Assert
    with pytest.raises(PermissionError, match="injected proxy permission failure"):
        VideoStageProcessor(
            retry_runtime,
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(video_set, configuration)
    assert retry_runtime.range_calls == []
    assert proxy_path.read_bytes() == original_bytes


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

    # Act
    # Assert
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


def test_refinement_keeps_distant_moment_groups_in_separate_work_units(
    tmp_path: Path,
) -> None:
    """離れたMoment groupが独立したrangeとcheckpointへ分離されること。

    Arrange:
        - 離れた2つのCandidate Momentを持つruntimeが用意される
    Act:
        - Frame Candidateが抽出される
    Assert:
        - 各groupが別の単一range requestとして処理されること
        - 各groupのDurable Work Unitが個別に確定されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    configuration = _configuration(input_folder, tmp_path / "output")
    runtime = FakeVideoStageMediaRuntime(distant_moments=True)

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
    assert len(runtime.range_pts_calls) == 2
    assert all(len(ranges) == 1 for ranges in runtime.range_pts_calls)
    checkpoint_root = configuration.processing_cache_folder / "work-units"
    assert (
        len(tuple(checkpoint_root.glob("*/frame-refinement-group/*/manifest.json")))
        == 2
    )


def test_refinement_window_groups_run_concurrently_within_safe_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """独立したRefinement Window GroupがCPU上限内で並列処理されること。

    Arrange:
        - 離れた2 groupと十分なlogical CPUを持つruntimeが用意される
    Act:
        - 一つのVideo SourceからFrame Candidateが抽出される
    Assert:
        - 先頭groupの完了を待たずに次のgroupが開始されること
        - 同時実行数が2件を超えないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    active_count = 0
    peak_count = 0
    overlap_started = threading.Event()
    active_lock = threading.Lock()

    def wait_for_sibling_group(_path: Path) -> None:
        nonlocal active_count, peak_count
        with active_lock:
            active_count += 1
            peak_count = max(peak_count, active_count)
            if active_count == 2:
                overlap_started.set()
        try:
            assert overlap_started.wait(timeout=1)
        finally:
            with active_lock:
                active_count -= 1

    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor.os.cpu_count",
        lambda: 8,
    )
    runtime = FakeVideoStageMediaRuntime(
        distant_moments=True,
        on_scan_video_frame_ranges=wait_for_sibling_group,
    )

    # Act
    result = VideoStageProcessor(
        runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
        available_memory_reader=lambda: 64 * 1024**3,
    ).process(
        discover_video_set(input_folder),
        _configuration(input_folder, tmp_path / "output"),
    )[0]

    # Assert
    assert len(result.extraction.moments) == 2
    assert peak_count == 2
    assert len(runtime.range_pts_calls) == 2


def test_refinement_reserves_cpu_for_background_video_scans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """background Video Scan分を除いたCPU容量でGroupが並列化されること。

    Arrange:
        - 16 logical CPUで3 scanが開始され、先頭scanだけが完了される
        - 残る2 scanが全CPUを予約した後、一方だけが解放される
    Act:
        - Video Stage processorがpipeliningされる
    Assert:
        - Refinement Groupが1件ずつ実行されること
        - background scan解放後に全Video Stageが完了されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    for index, name in enumerate(
        ("01-first.mp4", "02-background.mp4", "03-background.mp4"),
        start=1,
    ):
        (input_folder / name).write_bytes(f"video-{index}".encode())
    scans_started = threading.Barrier(3)
    first_scan_completed = threading.Event()
    release_background_scans = {
        name: threading.Event() for name in ("02-background.mp4", "03-background.mp4")
    }
    refinement_started = threading.Event()
    release_refinement = threading.Event()
    active_refinement_count = 0
    peak_refinement_count = 0
    refinement_lock = threading.Lock()

    def coordinate_scans(path: Path) -> None:
        scans_started.wait(timeout=2)
        if path.name == "01-first.mp4":
            first_scan_completed.set()
            return
        assert release_background_scans[path.name].wait(timeout=2)

    def coordinate_refinement(path: Path) -> None:
        nonlocal active_refinement_count, peak_refinement_count
        if path.name != "01-first.mp4":
            return
        with refinement_lock:
            active_refinement_count += 1
            peak_refinement_count = max(
                peak_refinement_count,
                active_refinement_count,
            )
            refinement_started.set()
        try:
            assert release_refinement.wait(timeout=2)
        finally:
            with refinement_lock:
                active_refinement_count -= 1

    def release_work() -> None:
        assert first_scan_completed.wait(timeout=2)
        time.sleep(0.05)
        release_background_scans["02-background.mp4"].set()
        assert refinement_started.wait(timeout=2)
        release_background_scans["03-background.mp4"].set()
        release_refinement.set()

    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor.os.cpu_count",
        lambda: 16,
    )
    runtime = FakeVideoStageMediaRuntime(
        distant_moments=True,
        on_scan_video=coordinate_scans,
        on_scan_video_frame_ranges=coordinate_refinement,
    )
    releaser = threading.Thread(target=release_work)
    releaser.start()

    # Act
    results = VideoStageProcessor(
        runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
        available_memory_reader=lambda: 64 * 1024**3,
    ).process(
        discover_video_set(input_folder),
        replace(
            _configuration(input_folder, tmp_path / "output"),
            video_scan_workers=3,
        ),
    )

    # Assert
    releaser.join(timeout=1)
    assert not releaser.is_alive()
    assert peak_refinement_count == 1
    assert len(results) == 3


def test_refinement_caps_parallel_groups_by_available_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """available memoryが少ない場合にGroupが1件ずつ処理されること。

    Arrange:
        - CPUには2 Groupを並列化できる余裕がある
        - 高解像度sourceに対してavailable memoryが5 GiBと報告される
        - 一Groupはparallel予算へ収まるが二Groupは収まらない
    Act:
        - 離れた2 GroupからFrame Candidateが抽出される
    Assert:
        - Refinement Groupが1件ずつ実行されること
        - 両Groupの処理が完了されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    media_probe = MediaProbe(
        format_names=("matroska",),
        streams=(
            MediaStream(
                index=0,
                kind="video",
                codec_name="ffv1",
                time_base=Fraction(1, 10),
                start_pts=0,
                duration_ts=500,
                width=1920,
                height=1080,
                sample_rate=None,
                channels=None,
                language=None,
                is_default=True,
                is_forced=False,
                is_attached_picture=False,
            ),
        ),
    )
    refinement_started = threading.Event()
    release_refinement = threading.Event()
    active_count = 0
    peak_count = 0
    active_lock = threading.Lock()

    def coordinate_refinement(_path: Path) -> None:
        nonlocal active_count, peak_count
        with active_lock:
            active_count += 1
            peak_count = max(peak_count, active_count)
            refinement_started.set()
        try:
            assert release_refinement.wait(timeout=2)
        finally:
            with active_lock:
                active_count -= 1

    def release_work() -> None:
        assert refinement_started.wait(timeout=2)
        time.sleep(0.05)
        release_refinement.set()

    monkeypatch.setattr(
        "src.video_selection.services.video_stage_processor.os.cpu_count",
        lambda: 16,
    )
    runtime = FakeVideoStageMediaRuntime(
        distant_moments=True,
        media_probe=media_probe,
        on_scan_video_frame_ranges=coordinate_refinement,
    )
    releaser = threading.Thread(target=release_work)
    releaser.start()

    # Act
    result = VideoStageProcessor(
        runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
        available_memory_reader=lambda: 5 * 1024**3,
    ).process(
        discover_video_set(input_folder),
        _configuration(input_folder, tmp_path / "output"),
    )[0]

    # Assert
    releaser.join(timeout=1)
    assert not releaser.is_alive()
    assert peak_count == 1
    assert len(result.extraction.moments) == 2
    assert len(runtime.range_pts_calls) == 2


def test_refinement_worker_count_does_not_change_semantic_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resource条件がFrame Candidateの意味結果を変えないこと。

    Arrange:
        - 離れた4 groupを持つ同内容のVideo Sourceが2組用意される
    Act:
        - 異なるframe timing hintの1 workerと完了順を反転した4 workerで抽出される
    Assert:
        - 4 worker側のGroup完了順が入力順と異なること
        - Moment、Candidate、親Stage artifact、Fingerprintが一致すること
    """
    # Arrange
    media_probe = MediaProbe(
        format_names=("matroska",),
        streams=(
            MediaStream(
                index=0,
                kind="video",
                codec_name="ffv1",
                time_base=Fraction(1, 10),
                start_pts=0,
                duration_ts=310,
                width=64,
                height=48,
                sample_rate=None,
                channels=None,
                language=None,
                is_default=True,
                is_forced=False,
                is_attached_picture=False,
            ),
        ),
    )

    def run_with_cpu_count(
        name: str,
        cpu_count: int,
        *,
        minimum_frame_delta_ts: int,
        maximum_frame_count_per_pts: int,
        reverse_completion: bool = False,
    ) -> tuple[VideoStageResult, dict[Path, bytes], tuple[int, ...]]:
        input_folder = tmp_path / name
        input_folder.mkdir()
        (input_folder / "video.mp4").write_bytes(b"video-content")
        completion_order: list[int] = []
        call_index = 0
        call_lock = threading.Lock()
        later_group_started = threading.Event()

        def complete_later(_path: Path) -> None:
            nonlocal call_index
            with call_lock:
                current_index = call_index
                call_index += 1
            if current_index == 0:
                assert later_group_started.wait(timeout=1)
                time.sleep(0.05)
            elif current_index == 1:
                later_group_started.set()
            with call_lock:
                completion_order.append(current_index)

        monkeypatch.setattr(
            "src.video_selection.services.video_stage_processor.os.cpu_count",
            lambda: cpu_count,
        )
        runtime = FakeVideoStageMediaRuntime(
            media_probe=media_probe,
            scan_frame_pts=(0, 100, 200, 300),
            on_scan_video_frame_ranges=(complete_later if reverse_completion else None),
            minimum_frame_delta_ts=minimum_frame_delta_ts,
            maximum_frame_count_per_pts=maximum_frame_count_per_pts,
        )
        configuration = replace(
            _configuration(input_folder, tmp_path / f"{name}-output"),
            candidate_density_per_minute=60.0,
        )
        video_set = discover_video_set(input_folder)
        result = VideoStageProcessor(
            runtime,
            FakeSpeechRuntime(),
            RecordingRunObserver(),
            available_memory_reader=lambda: 64 * 1024**3,
        ).process(video_set, configuration)[0]
        stage_root = (
            configuration.processing_cache_folder
            / "videos"
            / video_set.sources[0].fingerprint
            / ProcessingStage.EXTRACT_FRAME_CANDIDATES.value
            / result.completed_stages[1].fingerprint.value
        )
        return result, _semantic_stage_artifacts(stage_root), tuple(completion_order)

    # Act
    serial, serial_artifacts, _serial_completion_order = run_with_cpu_count(
        "serial-videos",
        4,
        minimum_frame_delta_ts=5,
        maximum_frame_count_per_pts=1,
    )
    parallel, parallel_artifacts, parallel_completion_order = run_with_cpu_count(
        "parallel-videos",
        16,
        minimum_frame_delta_ts=1,
        maximum_frame_count_per_pts=2,
        reverse_completion=True,
    )

    # Assert
    assert len(serial.extraction.moments) == 4
    assert parallel_completion_order[0] != 0
    assert (
        serial.completed_stages[1].fingerprint
        == parallel.completed_stages[1].fingerprint
    )
    assert serial.extraction.moments == parallel.extraction.moments
    assert tuple(
        replace(candidate, proxy_path=None)
        for candidate in serial.extraction.candidates
    ) == tuple(
        replace(candidate, proxy_path=None)
        for candidate in parallel.extraction.candidates
    )
    assert serial_artifacts == parallel_artifacts


def test_completed_refinement_group_survives_later_group_failure(
    tmp_path: Path,
) -> None:
    """後続group失敗後も完了済みRefinement Window Groupが再利用されること。

    Arrange:
        - 離れた2 groupの2件目だけ初回に失敗するruntimeが用意される
    Act:
        - 初回失敗後に同じVideo Sourceのcandidate抽出が再実行される
    Assert:
        - retryでは未完了の2件目だけがdecodeされること
        - 再開結果が中断なしの結果と同じcandidate IDと画像bytesになること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    configuration = _configuration(input_folder, tmp_path / "output")
    video_set = discover_video_set(input_folder)
    range_call_count = 0

    def fail_second_group(_path: Path) -> None:
        nonlocal range_call_count
        range_call_count += 1
        if range_call_count == 2:
            raise OSError("injected second refinement group failure")

    failing_runtime = FakeVideoStageMediaRuntime(
        distant_moments=True,
        on_scan_video_frame_ranges=fail_second_group,
    )
    with pytest.raises(
        OSError,
        match="injected second refinement group failure",
    ):
        VideoStageProcessor(
            failing_runtime,
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(video_set, configuration)
    retry_runtime = FakeVideoStageMediaRuntime(distant_moments=True)

    # Act
    resumed = VideoStageProcessor(
        retry_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)[0]
    uninterrupted_input = tmp_path / "uninterrupted-videos"
    uninterrupted_input.mkdir()
    (uninterrupted_input / "video.mp4").write_bytes(b"video-content")
    uninterrupted_configuration = _configuration(
        uninterrupted_input,
        tmp_path / "uninterrupted-output",
    )
    uninterrupted = VideoStageProcessor(
        FakeVideoStageMediaRuntime(distant_moments=True),
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        discover_video_set(uninterrupted_input),
        uninterrupted_configuration,
    )[0]

    # Assert
    assert len(failing_runtime.range_pts_calls) == 2
    assert retry_runtime.range_pts_calls == [failing_runtime.range_pts_calls[1]]
    assert tuple(
        (candidate.identifier, candidate.image_bytes)
        for candidate in resumed.extraction.candidates
    ) == tuple(
        (candidate.identifier, candidate.image_bytes)
        for candidate in uninterrupted.extraction.candidates
    )


def test_hash_consistent_wrong_refinement_group_recomputes_only_that_group(
    tmp_path: Path,
) -> None:
    """要求Momentと違うRefinement Groupだけが再decodeされること。

    Arrange:
        - 離れた2 groupと親Extraction Stageが正常に確定される
        - 先頭groupのMoment IDがhash整合を保って別IDへ改変される
        - 親Extraction Stageが失われ子groupからの再集約が必要になる
    Act:
        - 同じVideo SourceのVideo Stageが再実行される
    Assert:
        - 改変groupだけが再計算され健全な兄弟groupが保持されること
        - 修復後のCandidate IDと画像bytesが元の結果と一致すること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    configuration = _configuration(input_folder, tmp_path / "output")
    initial_runtime = FakeVideoStageMediaRuntime(distant_moments=True)
    initial = VideoStageProcessor(
        initial_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)[0]
    group_root = (
        configuration.processing_cache_folder
        / "work-units"
        / video_set.sources[0].fingerprint
        / "frame-refinement-group"
    )
    group_folders = tuple(
        path
        for path in group_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    assert len(group_folders) == 2
    first_folder = min(
        group_folders,
        key=lambda path: json.loads(
            (path / "artifact.json").read_text(encoding="utf-8")
        )["moments"][0]["source_pts"],
    )
    first_manifest = json.loads(
        (first_folder / "manifest.json").read_text(encoding="utf-8")
    )
    affected_range = tuple(first_manifest["semantic_input"]["pts_range"])
    first_artifact = json.loads(
        (first_folder / "artifact.json").read_text(encoding="utf-8")
    )
    first_artifact["moments"][0]["id"] = "mom_" + "f" * 64
    _rewrite_hash_consistent_artifact(first_folder, first_artifact)
    shutil.rmtree(
        configuration.processing_cache_folder
        / "videos"
        / video_set.sources[0].fingerprint
        / ProcessingStage.EXTRACT_FRAME_CANDIDATES.value
    )
    retry_runtime = FakeVideoStageMediaRuntime(distant_moments=True)

    # Act
    repaired = VideoStageProcessor(
        retry_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)[0]

    # Assert
    assert retry_runtime.scan_calls == []
    assert retry_runtime.range_pts_calls == [(affected_range,)]
    assert tuple(
        (candidate.identifier, candidate.image_bytes)
        for candidate in repaired.extraction.candidates
    ) == tuple(
        (candidate.identifier, candidate.image_bytes)
        for candidate in initial.extraction.candidates
    )


def test_hash_consistent_wrong_parent_extraction_reuses_refinement_groups(
    tmp_path: Path,
) -> None:
    """別動画を指す親Extractionだけが破棄され子groupから修復されること。

    Arrange:
        - 親Extractionと子refinement groupが正常に確定される
        - 親candidateの動画fingerprintがhash整合を保って改変される
    Act:
        - 同じVideo SourceのVideo Stageが再実行される
    Assert:
        - 親Extractionだけが再構築されscanとrefinementは再実行されないこと
        - 修復後の意味的なStage成果物が初回結果と一致すること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    configuration = _configuration(input_folder, tmp_path / "output")
    VideoStageProcessor(
        FakeVideoStageMediaRuntime(),
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)
    extraction_root = (
        configuration.processing_cache_folder
        / "videos"
        / source.fingerprint
        / ProcessingStage.EXTRACT_FRAME_CANDIDATES.value
    )
    extraction_folder = next(
        path
        for path in extraction_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    expected_artifacts = _semantic_stage_artifacts(extraction_folder)
    artifact = json.loads(
        (extraction_folder / "artifact.json").read_text(encoding="utf-8")
    )
    assert artifact["candidates"]
    artifact["candidates"][0]["video_fingerprint"] = "b" * 64
    _rewrite_hash_consistent_artifact(extraction_folder, artifact)
    retry_runtime = FakeVideoStageMediaRuntime()

    # Act
    VideoStageProcessor(
        retry_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)

    # Assert
    assert retry_runtime.scan_partition_calls == []
    assert retry_runtime.range_calls == []
    assert _semantic_stage_artifacts(extraction_folder) == expected_artifacts


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


def test_serial_refinement_cpu_time_is_counted_once(tmp_path: Path) -> None:
    """直列Refinementの所有thread CPU時間が一度だけ計上されること。

    Arrange:
        - 一つのRefinement Groupで0.2秒のCPUを消費するruntimeが用意される
    Act:
        - Video Stageが1 workerで初回計算される
    Assert:
        - candidate抽出のCPU時間が同じthreadの実測値と重複加算されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    runtime = FakeVideoStageMediaRuntime(cpu_burn_seconds=0.2)

    # Act
    result = VideoStageProcessor(
        runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        discover_video_set(input_folder),
        _configuration(input_folder, tmp_path / "output"),
    )[0]

    # Assert
    assert 0.2 <= result.extraction_metrics.cpu_seconds < 0.35


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
