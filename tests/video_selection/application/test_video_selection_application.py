"""実Processorを接続するVideo Selection Applicationのtest。"""

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
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
    cold_stages = {
        stage["name"]: stage for stage in cold_report["provenance"]["stages"]
    }
    warm_stages = {
        stage["name"]: stage for stage in warm_report["provenance"]["stages"]
    }
    cold_scan = cold_stages["scan_video_001"]
    cold_extraction = cold_stages["extract_frame_candidates_001"]
    cold_context = cold_stages["collect_context_001"]
    cold_catalog = cold_stages["build_scene_catalog_001"]
    cold_annotation = cold_stages["annotate_candidate_001"]
    cold_selection = cold_stages["select_images_001"]
    warm_scan = warm_stages["scan_video_001"]
    assert cold_scan["cache_misses"] == 1
    assert cold_scan["recomputed_items"] == 1
    assert cold_scan["attempt_count"] == 1
    assert cold_extraction["upstream_fingerprints"] == [cold_scan["fingerprint"]]
    assert warm_scan["cache_hits"] == 1
    assert warm_scan["cache_misses"] == 0
    assert warm_scan["recomputed_items"] == 0
    assert warm_scan["attempt_count"] == 1
    provenance = cold_report["provenance"]
    assert provenance["runtime"]["speech_runtime_identity"] == (
        "fake-speech-runtime-v1"
    )
    assert cold_scan["effective_settings"]["decode_backend"] == "cpu"
    assert cold_scan["tool_refs"] == ["ffmpeg"]
    assert cold_scan["contract_refs"] == ["video_scan"]
    assert cold_extraction["tool_refs"] == ["ffmpeg"]
    assert cold_extraction["contract_refs"] == ["frame_candidate_extraction"]
    assert cold_context["tool_refs"] == ["ffmpeg"]
    assert cold_context["model_refs"] == []
    assert cold_context["contract_refs"] == ["context_collection"]
    assert cold_catalog["tool_refs"] == ["ollama"]
    assert cold_catalog["model_refs"] == ["scene_catalog"]
    assert cold_catalog["effective_settings"]["stage_contract_version"]
    assert cold_catalog["prompt_eval_tokens"] == 10
    assert cold_catalog["eval_tokens"] == 5
    assert cold_annotation["tool_refs"] == ["ollama"]
    assert cold_annotation["model_refs"] == ["candidate_annotation"]
    assert cold_annotation["contract_refs"] == [
        "candidate_annotation",
        "nearby_context_policy",
    ]
    assert cold_selection["tool_refs"] == ["video_selection"]
    assert cold_selection["contract_refs"] == ["video_set_selection_policy"]


def test_report_timestamps_cover_the_pipeline_lifecycle(tmp_path: Path) -> None:
    """reportの開始・完了時刻がpipeline全体の境界から記録されること。

    Arrange:
        - run開始時刻とpublication時刻を返すUTC clockが用意される
    Act:
        - Video Selection Applicationが実行される
    Assert:
        - reportへ異なる開始・完了時刻がそのまま記録されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mkv").write_bytes(b"video-content")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        image_count=1,
    )
    media = FakeVideoStageMediaRuntime()
    clock_call_count = 0

    def lifecycle_clock() -> datetime:
        nonlocal clock_call_count
        clock_call_count += 1
        if clock_call_count == 1:
            return datetime(2026, 7, 21, 1, 2, 3, tzinfo=timezone.utc)
        assert media.extracted_original_frame_calls
        return datetime(2026, 7, 21, 1, 3, 4, tzinfo=timezone.utc)

    application = _application(
        media,
        EchoStructuredVisionRuntime(),
        clock=lifecycle_clock,
    )

    # Act
    outcome = application.run(configuration)

    # Assert
    report = json.loads(
        (outcome.output_folder / "report.json").read_text(encoding="utf-8")
    )
    assert report["run"]["started_at"] == "2026-07-21T01:02:03Z"
    assert report["run"]["completed_at"] == "2026-07-21T01:03:04Z"
    assert clock_call_count == 2


def test_speech_runtime_is_closed_before_vision_inference(tmp_path: Path) -> None:
    """STT model資源がVision推論開始前に解放されること。

    Arrange:
        - close状態を記録するSpeech RuntimeとVision callbackが用意される
    Act:
        - Video Selection Applicationが実行される
    Assert:
        - 最初のScene Catalog推論時にSpeech Runtimeがclose済みであること
        - close前に取得したSpeech Runtime Identityがreportへ保持されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mkv").write_bytes(b"video-content")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        image_count=1,
    )
    speech_runtime = FakeSpeechRuntime(runtime_identity="speech-runtime-before-close")
    close_states_at_vision: list[bool] = []
    vision = EchoStructuredVisionRuntime(
        on_create_scene_catalog=lambda: close_states_at_vision.append(
            speech_runtime.closed
        )
    )
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer)
    progress.start_run()
    application = VideoSelectionApplication(
        media_runtime=FakeVideoStageMediaRuntime(),
        model_runtime=FakeModelRuntime("application-test"),
        speech_runtime_factory=lambda _model, _configuration: speech_runtime,
        vision_runtime=vision,
        observer=observer,
        progress=progress,
    )

    # Act
    outcome = application.run(configuration)

    # Assert
    report = json.loads(
        (outcome.output_folder / "report.json").read_text(encoding="utf-8")
    )
    assert close_states_at_vision == [True]
    assert speech_runtime.close_call_count == 1
    assert (
        report["provenance"]["runtime"]["speech_runtime_identity"]
        == "speech-runtime-before-close"
    )


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
    clock: Callable[[], datetime] | None = None,
) -> VideoSelectionApplication:
    """実Processorをtest fake境界へ接続する。"""
    actual_observer = observer or RecordingRunObserver()
    actual_progress = progress or RunProgressTracker(actual_observer)
    if progress is None:
        actual_progress.start_run()
    return VideoSelectionApplication(
        media_runtime=media_runtime,
        model_runtime=FakeModelRuntime("application-test"),
        speech_runtime_factory=lambda model, _configuration: FakeSpeechRuntime(
            resolved_model_identity=model.execution_identity.identifier
        ),
        vision_runtime=vision_runtime,
        observer=actual_observer,
        progress=actual_progress,
        clock=clock,
    )
