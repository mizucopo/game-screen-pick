"""実Processorを接続するVideo Selection Applicationのtest。"""

import json
from dataclasses import replace
from pathlib import Path

from src.video_selection.application.video_selection_application import (
    VideoSelectionApplication,
)
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.run_status import RunStatus
from src.video_selection.services.run_progress_tracker import RunProgressTracker
from tests.video_selection.fakes.echo_structured_vision_runtime import (
    EchoStructuredVisionRuntime,
)
from tests.video_selection.fakes.fake_model_runtime import FakeModelRuntime
from tests.video_selection.fakes.fake_speech_runtime import FakeSpeechRuntime
from tests.video_selection.fakes.fake_video_stage_media_runtime import (
    FakeVideoStageMediaRuntime,
)
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver


def test_real_processors_publish_canonical_output_and_reuse_warm_cache(
    tmp_path: Path,
) -> None:
    """実Processor列がcanonical outputを公開しwarm時に推論cacheが再利用されること。

    Arrange:
        - 一つのVideo Sourceと決定的なMedia、Model、Speech、Vision fakeが用意される
    Act:
        - cold runと別Output Folderへのexact warm runが実行される
    Assert:
        - 選択画像とcanonical reportがatomicに公開されること
        - warm runでscan、refinement、Vision推論が再実行されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mkv").write_bytes(b"video-content")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "cold-output",
        image_count=1,
    )
    cold_media = FakeVideoStageMediaRuntime()
    cold_vision = EchoStructuredVisionRuntime()
    cold_observer = RecordingRunObserver()
    cold_progress = RunProgressTracker(cold_observer)
    cold_progress.start_run()
    cold_application = _application(
        cold_media,
        cold_vision,
        observer=cold_observer,
        progress=cold_progress,
    )

    # Act
    cold = cold_application.run(configuration)
    warm_media = FakeVideoStageMediaRuntime()
    warm_vision = EchoStructuredVisionRuntime()
    warm_observer = RecordingRunObserver()
    warm_progress = RunProgressTracker(warm_observer)
    warm_progress.start_run()
    warm = _application(
        warm_media,
        warm_vision,
        observer=warm_observer,
        progress=warm_progress,
    ).run(replace(configuration, output_folder=tmp_path / "warm-output"))

    # Assert
    cold_report = json.loads(
        (cold.output_folder / "report.json").read_text(encoding="utf-8")
    )
    warm_report = json.loads(
        (warm.output_folder / "report.json").read_text(encoding="utf-8")
    )
    assert cold.status is RunStatus.COMPLETED
    assert cold.selected_count == 1
    assert len(tuple((cold.output_folder / "images").glob("*.webp"))) == 1
    assert cold_report["run"]["selected_image_count"] == 1
    assert warm_report["run"]["selected_image_count"] == 1
    assert cold_media.scan_calls
    assert cold_media.range_calls
    assert cold_vision.scene_catalog_calls
    assert cold_vision.candidate_annotation_calls
    assert warm_media.scan_calls == []
    assert warm_media.range_calls == []
    assert warm_vision.scene_catalog_calls == []
    assert warm_vision.candidate_annotation_calls == []
    cold_selection_cache = tuple(
        event
        for event in cold_observer.progress_events
        if event.kind == "cache" and event.stage is ProcessingStage.SELECT_IMAGES
    )
    warm_selection_cache = tuple(
        event
        for event in warm_observer.progress_events
        if event.kind == "cache" and event.stage is ProcessingStage.SELECT_IMAGES
    )
    assert len(cold_selection_cache) == 1
    assert cold_selection_cache[0].recompute_count == 1
    assert len(warm_selection_cache) == 1
    assert warm_selection_cache[0].reuse_count == 1
    assert warm_selection_cache[0].recompute_count == 0


def test_no_valid_candidate_skips_vision_and_publishes_zero_shortfall(
    tmp_path: Path,
) -> None:
    """有効Candidateが0件ならVisionを呼ばず0枚warning outputが公開されること。

    Arrange:
        - refinement frameがすべてblackoutになるVideo Sourceが用意される
    Act:
        - Video Selection Applicationが実行される
    Assert:
        - Scene CatalogとCandidate Annotationが呼ばれないこと
        - requested N、selected 0、exhaustedを持つwarning reportが公開されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "black.mkv").write_bytes(b"video-content")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        image_count=3,
    )
    vision = EchoStructuredVisionRuntime()
    application = _application(
        FakeVideoStageMediaRuntime(zero_valid_frames=True),
        vision,
    )

    # Act
    outcome = application.run(configuration)

    # Assert
    report = json.loads(
        (outcome.output_folder / "report.json").read_text(encoding="utf-8")
    )
    assert outcome.status is RunStatus.COMPLETED_WITH_WARNINGS
    assert outcome.selected_count == 0
    assert vision.scene_catalog_calls == []
    assert vision.candidate_annotation_calls == []
    assert report["selection_summary"]["shortfall"] == {
        "requested": 3,
        "selected": 0,
        "all_candidate_moments_exhausted": True,
    }
    assert report["selected"] == []
    assert report["run"]["warnings"][0]["code"] == "selection_shortfall"


def test_shortfall_expands_annotation_without_fixed_maximum(tmp_path: Path) -> None:
    """initial batch不足時に要求枚数単位で全有効Momentまで注釈されること。

    Arrange:
        - 26件の有効Momentと要求2枚、近似画像を返す13 Videoが用意される
    Act:
        - Video Selection Applicationが実行される
    Assert:
        - initial 24件の後に2件が追加され全26件が注釈されること
        - Scene Catalog代表が要求枚数に依存せず24枚に制限されること
        - 全Moment消費後のshortfallが記録されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    for index in range(13):
        (input_folder / f"{index:02d}.mkv").write_bytes(f"video-{index}".encode())
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        image_count=2,
        candidate_density_per_minute=3.0,
    )
    vision = EchoStructuredVisionRuntime()
    application = _application(
        FakeVideoStageMediaRuntime(distant_moments=True),
        vision,
    )

    # Act
    outcome = application.run(configuration)

    # Assert
    report = json.loads(
        (outcome.output_folder / "report.json").read_text(encoding="utf-8")
    )
    assert len(vision.candidate_annotation_calls) == 26
    assert len(vision.scene_catalog_calls) == 1
    assert len(vision.scene_catalog_calls[0].representatives) == 24
    assert report["selection_summary"]["shortlist_expansion_count"] == 1
    assert (
        report["selection_summary"]["shortfall"]["all_candidate_moments_exhausted"]
        is True
    )


def _application(
    media_runtime: FakeVideoStageMediaRuntime,
    vision_runtime: EchoStructuredVisionRuntime,
    *,
    observer: RecordingRunObserver | None = None,
    progress: RunProgressTracker | None = None,
) -> VideoSelectionApplication:
    """実Processorをtest fake境界へ接続する。"""
    actual_observer = observer or RecordingRunObserver()
    return VideoSelectionApplication(
        media_runtime=media_runtime,
        model_runtime=FakeModelRuntime("application-test"),
        speech_runtime_factory=lambda model, _configuration: FakeSpeechRuntime(
            resolved_model_identity=model.execution_identity.identifier
        ),
        vision_runtime=vision_runtime,
        observer=actual_observer,
        progress=progress,
    )
