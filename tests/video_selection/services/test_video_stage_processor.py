"""Video Stage processorの統合style test。"""

from dataclasses import replace
from pathlib import Path

from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.services.discover_video_set import discover_video_set
from src.video_selection.services.video_stage_processor import VideoStageProcessor
from tests.video_selection.fakes.fake_video_stage_media_runtime import (
    FakeVideoStageMediaRuntime,
)
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver


def _configuration(input_folder: Path, output_folder: Path) -> EffectiveConfiguration:
    return EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=output_folder,
    )


def test_video_sources_are_serial_and_video_stage_cache_is_source_local(
    tmp_path: Path,
) -> None:
    """動画が直列処理されpath、順序、downstream設定から独立して再利用されること。

    Arrange:
        - 異なる内容を持つ2動画のVideo SetとVideo Stage processorが用意される
    Act:
        - 初回処理後に動画をrenameして順序とdownstream設定を変えて再実行される
        - 続いてCandidate Moment Densityだけを変えて再実行される
    Assert:
        - 各動画がscanからrefinementまでVideo Order順に直列処理されること
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
        RecordingRunObserver(),
    ).process(first_video_set, configuration)

    # Assert
    assert first_runtime.call_order == [
        ("probe", "01-first.mp4"),
        ("scan", "01-first.mp4"),
        ("refine", "01-first.mp4"),
        ("probe", "02-second.mp4"),
        ("scan", "02-second.mp4"),
        ("refine", "02-second.mp4"),
    ]
    assert all(
        [item.stage for item in result.completed_stages]
        == [ProcessingStage.SCAN_VIDEO, ProcessingStage.EXTRACT_FRAME_CANDIDATES]
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
        RecordingRunObserver(),
    ).process(reordered_video_set, density_changed)

    # Assert
    assert density_runtime.scan_calls == []
    assert [path.name for path in density_runtime.range_calls] == [
        "01-renamed.mp4",
        "99-renamed.mp4",
    ]


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
        RecordingRunObserver(),
    ).process(video_set, configuration)[0]
    proxy_path = initial_result.extraction.candidates[0].proxy_path
    assert proxy_path is not None
    proxy_path.write_bytes(b"corrupt-proxy")
    repair_runtime = FakeVideoStageMediaRuntime()

    # Act
    repaired_result = VideoStageProcessor(
        repair_runtime,
        RecordingRunObserver(),
    ).process(video_set, configuration)[0]

    # Assert
    assert repair_runtime.scan_calls == []
    assert [path.name for path in repair_runtime.range_calls] == ["video.mp4"]
    repaired_proxy_path = repaired_result.extraction.candidates[0].proxy_path
    assert repaired_proxy_path is not None
    assert repaired_proxy_path.read_bytes() != b"corrupt-proxy"
