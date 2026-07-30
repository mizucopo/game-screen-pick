"""Video Set選定applicationのintegration test。"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.video_selection.application.internal_run_controller import (
    InternalRunController,
)
from src.video_selection.application.walking_skeleton_application import (
    WalkingSkeletonApplication,
)
from src.video_selection.models.candidate_annotation import CandidateAnnotation
from src.video_selection.models.context_cue import ContextCue
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.frame_candidate import FrameCandidate
from src.video_selection.models.legacy_cache_cleanup_diagnostic import (
    LegacyCacheCleanupDiagnostic,
)
from src.video_selection.models.model_role import ModelRole
from src.video_selection.models.model_runtime_error import ModelRuntimeError
from src.video_selection.models.model_runtime_failure_reason import (
    ModelRuntimeFailureReason,
)
from src.video_selection.models.processing_stage import (
    VIDEO_SET_STAGE_ORDER,
    ProcessingStage,
)
from src.video_selection.models.resolved_models import ResolvedModels
from src.video_selection.models.run_failure import RunFailure
from src.video_selection.models.run_outcome import RunOutcome
from src.video_selection.models.run_status import RunStatus
from src.video_selection.models.selected_image import SelectedImage
from src.video_selection.models.video_set import VideoSet
from src.video_selection.services.atomic_output_publisher import AtomicOutputPublisher
from src.video_selection.services.input_folder_lock import InputFolderLock
from src.video_selection.services.prepared_output import PreparedOutput
from src.video_selection.services.run_progress_tracker import RunProgressTracker
from tests.video_selection.fakes.failing_vision_runtime import FailingVisionRuntime
from tests.video_selection.fakes.fake_context_collector import FakeContextCollector
from tests.video_selection.fakes.fake_media_runtime import FakeMediaRuntime
from tests.video_selection.fakes.fake_model_runtime import FakeModelRuntime
from tests.video_selection.fakes.fake_vision_runtime import FakeVisionRuntime
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver


def _video_set_stage_manifests(
    input_folder: Path,
    stage: ProcessingStage,
) -> tuple[Path, ...]:
    return tuple(
        (input_folder / ".game-screen-pick" / "cache" / "video-sets").glob(
            f"*/{stage.value}/*/manifest.json"
        )
    )


def _successful_application(
    observer: RecordingRunObserver,
) -> WalkingSkeletonApplication:
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    return WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="summary"),)
        ),
        observer=observer,
    )


def test_run_publishes_normalized_fake_result_atomically(tmp_path: Path) -> None:
    """fake pipelineの選定結果が一つのOutput Folderへ公開されること。

    Arrange:
        - 2本のdummy videoと5つのfake runtimeが用意される
        - fake mediaから1件のFrame Candidateが返される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - selected image、canonical JSON、Markdownが同時に公開されること
        - 全Processing StageのCompleted Stage manifestが順番に確定されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    (input_folder / "chapter-02.mp4").write_bytes(b"video-02")
    candidate = FrameCandidate(
        identifier="frame-001",
        image_bytes=b"fake-webp-image",
    )
    annotation = CandidateAnnotation(
        candidate=candidate,
        summary="主人公が草原を進んでいる",
    )
    observer = RecordingRunObserver()
    model_runtime = FakeModelRuntime("model-sha-001")
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector((ContextCue(identifier="cue-001"),)),
        model_runtime=model_runtime,
        vision_runtime=FakeVisionRuntime((annotation,)),
        observer=observer,
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=output_folder,
        image_count=1,
    )
    expected_models = model_runtime.resolve_models(configuration).provenance()
    vision_digest = hashlib.sha256(b"model-sha-001").hexdigest()
    speech_commit = hashlib.sha256(b"speech-model").hexdigest()[:40]

    # Act
    outcome = application.run(configuration)

    # Assert
    assert outcome.selected_count == 1
    assert outcome.requested_count == 1
    assert outcome.status == "completed"
    assert outcome.output_folder == output_folder
    assert (output_folder / "images" / "0001_frame-001.webp").read_bytes() == (
        b"fake-webp-image"
    )
    report = json.loads((output_folder / "report.json").read_text(encoding="utf-8"))
    assert report == {
        "schema": "game-screen-pick/walking-skeleton@0",
        "status": "completed",
        "requested_count": 1,
        "selected_count": 1,
        "warnings": [],
        "models": expected_models,
        "video_set": {
            "videos": ["chapter-01.mp4", "chapter-02.mp4"],
        },
        "selected": [
            {
                "id": "frame-001",
                "path": "images/0001_frame-001.webp",
                "summary": "主人公が草原を進んでいる",
                "reason_codes": ["walking_skeleton_selected"],
            }
        ],
    }
    assert (output_folder / "report.md").read_text(encoding="utf-8") == (
        "# Video Selection Report\n\n"
        "Status: completed\n"
        "Requested images: 1\n"
        "Selected images: 1\n\n"
        "## Models\n\n"
        "- candidate_annotation: qwen3-vl:8b-instruct @ "
        f"ollama:sha256:{vision_digest[:12]}… (not_requested)\n"
        "- scene_catalog: qwen3-vl:8b-instruct @ "
        f"ollama:sha256:{vision_digest[:12]}… (not_requested)\n"
        "- speech_to_text: dropbox-dash/faster-whisper-large-v3-turbo @ "
        f"hf:{speech_commit[:12]}… (not_requested)\n\n"
        "## Selected images\n\n"
        "1. [frame-001](images/0001_frame-001.webp) — "
        "主人公が草原を進んでいる\n"
    )
    assert (
        tuple(stage.stage for stage in observer.completed_stages)
        == VIDEO_SET_STAGE_ORDER
    )
    manifests = tuple(
        (input_folder / ".game-screen-pick" / "cache" / "video-sets").glob(
            "*/*/*/manifest.json"
        )
    )
    assert len(manifests) == len(VIDEO_SET_STAGE_ORDER)


def test_internal_controller_emits_application_stage_events(tmp_path: Path) -> None:
    """internal application runが全Stageをtyped eventとして通知すること。

    Arrange:
        - 共有observer、Progress Tracker、fake applicationが用意される
    Act:
        - Internal Run Controllerから成功runが実行される
    Assert:
        - 全Stageの開始・完了とrun terminalが表示文字列なしで通知されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer, clock=lambda: 10.0)
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="summary"),)
        ),
        observer=observer,
        progress=progress,
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        image_count=1,
    )

    # Act
    exit_code, result = InternalRunController(progress).execute(
        lambda: application.run(configuration)
    )

    # Assert
    started = tuple(
        event.stage
        for event in observer.progress_events
        if event.kind == "stage_started"
    )
    completed = tuple(
        event.stage
        for event in observer.progress_events
        if event.kind == "stage_completed"
    )
    assert isinstance(result, RunOutcome)
    assert (
        exit_code,
        result.status,
        observer.progress_events[0].kind,
        observer.progress_events[-1].kind,
        started,
        completed,
    ) == (
        0,
        RunStatus.COMPLETED,
        "run_started",
        "run_completed",
        VIDEO_SET_STAGE_ORDER,
        VIDEO_SET_STAGE_ORDER,
    )


def test_application_omits_unknown_total_when_nested_stages_expand_run(
    tmp_path: Path,
) -> None:
    """内部runtimeがStageを追加するrunで根拠のない総Stage数が省略されること。

    Arrange:
        - candidate抽出中にatomic Video Stageを通知するruntimeが用意される
    Act:
        - 同じProgress Trackerでapplication全体が実行される
    Assert:
        - Stage番号は単調増加し、未確定の総Stage数なしでrunが完了すること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer, clock=lambda: 10.0)

    def emit_nested_video_stage() -> None:
        progress.start_stage(
            ProcessingStage.SCAN_VIDEO,
            work_unit_kind="video",
        )
        progress.complete_stage()

    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime(
            (candidate,),
            on_extract_candidates=emit_nested_video_stage,
        ),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="summary"),)
        ),
        observer=observer,
        progress=progress,
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        image_count=1,
    )

    # Act
    exit_code, result = InternalRunController(progress).execute(
        lambda: application.run(configuration)
    )

    # Assert
    started = tuple(
        (event.stage_index, event.stage_count)
        for event in observer.progress_events
        if event.kind == "stage_started"
    )
    assert isinstance(result, RunOutcome)
    assert (exit_code, started, observer.progress_events[-1].kind) == (
        0,
        tuple((index, None) for index in range(1, 8)),
        "run_completed",
    )


@pytest.mark.parametrize(
    ("invalid_input", "expected_reason_code"),
    (
        ("missing_input", "video_input_folder_not_found"),
        ("empty_input", "video_input_folder_empty"),
        ("duplicate_input", "duplicate_video"),
        ("invalid_output", "output_folder_invalid"),
    ),
)
def test_internal_controller_maps_preflight_path_failures_to_usage_result(
    tmp_path: Path,
    invalid_input: str,
    expected_reason_code: str,
) -> None:
    """実行前に修正できる入力・出力不備がexit 2へ分類されること。

    Arrange:
        - 不存在・動画なし・重複入力、または非空Output Folderが用意される
    Act:
        - Internal Run Controllerからapplicationが実行される
    Assert:
        - run未開始相当の安全なusage failureが返されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    if invalid_input != "missing_input":
        input_folder.mkdir()
    if invalid_input == "duplicate_input":
        (input_folder / "chapter-01.mp4").write_bytes(b"duplicate")
        (input_folder / "chapter-02.mp4").write_bytes(b"duplicate")
    if invalid_input == "invalid_output":
        (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
        output_folder.mkdir()
        (output_folder / "keep.txt").write_text("keep", encoding="utf-8")
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer)
    application = _successful_application(observer)
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=output_folder,
        image_count=1,
    )

    # Act
    exit_code, result = InternalRunController(progress).execute(
        lambda: application.run(configuration)
    )

    # Assert
    assert isinstance(result, RunFailure)
    assert (
        exit_code,
        result.reason_code,
        result.remediation_code,
        result.resume_guidance,
        tuple(event.kind for event in observer.progress_events),
    ) == (
        2,
        expected_reason_code,
        "fix_configuration",
        "run_not_started",
        ("run_started", "run_failed"),
    )


def test_speech_model_is_frozen_before_context_collection(tmp_path: Path) -> None:
    """freeze済みSTT modelがContext Collectionへ渡されること。

    Arrange:
        - 一つのvideoとmodel resolutionを記録するContext Collectorが用意される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - Context Collectionがfreeze済みspeech modelを一度受け取ること
        - Model Resolution StageがContext Collectionより先に確定されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    collector = FakeContextCollector(())
    observer = RecordingRunObserver()
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=collector,
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="summary"),)
        ),
        observer=observer,
    )

    # Act
    application.run(
        EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=output_folder,
            image_count=1,
        )
    )

    # Assert
    assert len(collector.models) == 1
    assert collector.models[0].role is ModelRole.SPEECH_TO_TEXT
    completed = tuple(item.stage for item in observer.completed_stages)
    assert completed.index(ProcessingStage.RESOLVE_MODELS) < completed.index(
        ProcessingStage.COLLECT_CONTEXT
    )


def test_final_snapshot_failure_discards_staged_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最終snapshot検査失敗時にstaging outputが破棄されること。

    Arrange:
        - Output artifact準備直後に入力videoが変更される境界が用意される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - snapshot変更errorが返されること
        - Output Folderとhidden staging directoryが残らないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    video_path = input_folder / "chapter-01.mp4"
    video_path.write_bytes(b"video-01")
    original_prepare = AtomicOutputPublisher.prepare

    def prepare_then_change_input(
        publisher: AtomicOutputPublisher,
        prepared_output_folder: Path,
        video_set: VideoSet,
        selected_images: tuple[SelectedImage, ...],
        requested_count: int,
        run_status: RunStatus,
        resolved_models: ResolvedModels,
    ) -> PreparedOutput:
        prepared_output = original_prepare(
            publisher,
            prepared_output_folder,
            video_set,
            selected_images,
            requested_count,
            run_status,
            resolved_models,
        )
        video_path.write_bytes(b"changed-video")
        return prepared_output

    monkeypatch.setattr(AtomicOutputPublisher, "prepare", prepare_then_change_input)

    # Act
    # Assert
    with pytest.raises(ValueError, match="Video Set snapshotが変更されました"):
        _successful_application(RecordingRunObserver()).run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=output_folder,
                image_count=1,
            )
        )
    assert not output_folder.exists()
    assert tuple(tmp_path.glob(".output.*.staging")) == ()


def test_run_removes_recognized_legacy_cache_and_reports_diagnostic(
    tmp_path: Path,
) -> None:
    """認識済みLegacy Cacheだけが削除され診断が通知されること。

    Arrange:
        - valid Video Setと二種類の認識済みLegacy Cacheが用意される
        - 新cache namespaceに保護対象のmarkerが用意される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - Legacy Cacheだけが削除されること
        - 削除entry数とbyte数がobserverへ通知されること
        - 新cache namespaceが保持されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    cache_folder = input_folder / ".game-screen-pick" / "cache"
    neutral_analysis = cache_folder / "neutral-analysis"
    protected_marker = cache_folder / "videos" / "keep.txt"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    neutral_analysis.mkdir(parents=True)
    legacy_analysis_bytes = b"legacy-analysis"
    legacy_scene_bytes = b'{"legacy": true}'
    (neutral_analysis / "result.json").write_bytes(legacy_analysis_bytes)
    (cache_folder / "ollama-scenes.json").write_bytes(legacy_scene_bytes)
    protected_marker.parent.mkdir()
    protected_marker.write_text("keep", encoding="utf-8")
    observer = RecordingRunObserver()

    # Act
    _successful_application(observer).run(
        EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=output_folder,
            image_count=1,
        )
    )

    # Assert
    assert not neutral_analysis.exists()
    assert not (cache_folder / "ollama-scenes.json").exists()
    assert protected_marker.read_text(encoding="utf-8") == "keep"
    assert observer.legacy_cache_diagnostics == [
        LegacyCacheCleanupDiagnostic(
            removed_entry_count=2,
            removed_bytes=len(legacy_analysis_bytes) + len(legacy_scene_bytes),
        )
    ]


def test_input_lock_failure_preserves_cache_and_output(
    tmp_path: Path,
) -> None:
    """Input Lock取得失敗時にcacheとOutput Folderが変更されないこと。

    Arrange:
        - valid Video Setと認識済みLegacy Cacheが用意される
        - 同じVideo Input FolderのInput Lockが既に保持される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - 非待機の実行中errorが返されること
        - Legacy CacheとOutput Folderが変更されないこと
        - cleanup diagnosticが通知されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    legacy_folder = input_folder / ".game-screen-pick" / "cache" / "neutral-analysis"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    legacy_folder.mkdir(parents=True)
    legacy_marker = legacy_folder / "result.json"
    legacy_marker.write_text("legacy", encoding="utf-8")
    observer = RecordingRunObserver()

    # Act
    with (
        InputFolderLock(input_folder),
        pytest.raises(RuntimeError, match="既に実行中"),
    ):
        _successful_application(observer).run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=output_folder,
                image_count=1,
            )
        )

    # Assert
    assert legacy_marker.read_text(encoding="utf-8") == "legacy"
    assert not output_folder.exists()
    assert observer.legacy_cache_diagnostics == []


def test_reset_cache_removes_entire_processing_cache_before_run(
    tmp_path: Path,
) -> None:
    """reset_cacheでprocessing cache全体が削除後に再構築されること。

    Arrange:
        - valid Video Setと未知のprocessing cache entryが用意される
    Act:
        - reset_cacheを有効にしてapplicationが実行される
    Assert:
        - 既存entryが削除されること
        - 新しいVideo Set Stage cacheとOutput Folderが作成されること
        - Legacy Cache削除件数はゼロとして通知されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    cache_folder = input_folder / ".game-screen-pick" / "cache"
    stale_marker = cache_folder / "unknown" / "keep-unless-reset.txt"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    stale_marker.parent.mkdir(parents=True)
    stale_marker.write_text("stale", encoding="utf-8")
    observer = RecordingRunObserver()

    # Act
    _successful_application(observer).run(
        EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=output_folder,
            image_count=1,
            reset_cache=True,
        )
    )

    # Assert
    assert not stale_marker.exists()
    assert (
        len(
            _video_set_stage_manifests(
                input_folder,
                ProcessingStage.DISCOVER_VIDEO_SET,
            )
        )
        == 1
    )
    assert (output_folder / "report.json").is_file()
    assert observer.legacy_cache_diagnostics == [
        LegacyCacheCleanupDiagnostic(removed_entry_count=0, removed_bytes=0)
    ]


def test_model_preflight_failure_preserves_cache_requested_for_reset(
    tmp_path: Path,
) -> None:
    """model preflight失敗時にreset対象のprocessing cacheが保持されること。

    Arrange:
        - 既存processing cacheとreset_cache有効の設定が用意される
        - model解決で失敗するruntimeが用意される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - model errorが返され既存cacheが変更されないこと
        - cache cleanup通知とOutput Folder公開が行われないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    cache_folder = input_folder / ".game-screen-pick" / "cache"
    stale_marker = cache_folder / "existing" / "preserved.txt"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    stale_marker.parent.mkdir(parents=True)
    stale_marker.write_text("preserve", encoding="utf-8")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    observer = RecordingRunObserver()
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime(
            "model-sha-001",
            resolution_error=ModelRuntimeError(
                ModelRuntimeFailureReason.MODEL_STORE_UNAVAILABLE,
                ModelRole.SPEECH_TO_TEXT,
            ),
        ),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="summary"),)
        ),
        observer=observer,
    )

    # Act
    # Assert
    with pytest.raises(ModelRuntimeError):
        application.run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=output_folder,
                image_count=1,
                reset_cache=True,
            )
        )
    assert stale_marker.read_text(encoding="utf-8") == "preserve"
    assert observer.legacy_cache_diagnostics == []
    assert not output_folder.exists()


def test_input_change_during_model_preflight_preserves_cache_requested_for_reset(
    tmp_path: Path,
) -> None:
    """model preflight中の入力変更がreset前に再検出されること。

    Arrange:
        - 既存processing cacheとreset_cache有効の設定が用意される
        - model解決中に入力videoを変更するruntimeが用意される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - snapshot errorが返され既存cacheが変更されないこと
        - cache cleanup通知とOutput Folder公開が行われないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    cache_folder = input_folder / ".game-screen-pick" / "cache"
    stale_marker = cache_folder / "existing" / "preserved.txt"
    input_folder.mkdir()
    video_path = input_folder / "chapter-01.mp4"
    video_path.write_bytes(b"video-01")
    stale_marker.parent.mkdir(parents=True)
    stale_marker.write_text("preserve", encoding="utf-8")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    observer = RecordingRunObserver()

    def mutate_input() -> None:
        video_path.write_bytes(b"changed-video")

    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime(
            "model-sha-001",
            resolution_action=mutate_input,
        ),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="summary"),)
        ),
        observer=observer,
    )

    # Act
    # Assert
    with pytest.raises(ValueError, match="Video Set snapshotが変更されました"):
        application.run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=output_folder,
                image_count=1,
                reset_cache=True,
            )
        )
    assert stale_marker.read_text(encoding="utf-8") == "preserve"
    assert observer.legacy_cache_diagnostics == []
    assert not output_folder.exists()


def test_stage_failure_leaves_output_unpublished(tmp_path: Path) -> None:
    """Processing Stage失敗時にOutput Folderが公開されないこと。

    Arrange:
        - Candidate Annotationで失敗するfake VisionRuntimeが用意される
        - それ以前のStageは正常に完了される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - runtime failureが返され、Output Folderが存在しないこと
        - 完了済みStageだけがmanifestへ確定されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(
        identifier="frame-001",
        image_bytes=b"fake-webp-image",
    )
    observer = RecordingRunObserver()
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FailingVisionRuntime(),
        observer=observer,
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=output_folder,
        image_count=1,
    )

    # Act
    # Assert
    with pytest.raises(RuntimeError, match="fake vision failure"):
        application.run(configuration)
    assert not output_folder.exists()
    assert tuple(stage.stage for stage in observer.completed_stages) == (
        ProcessingStage.DISCOVER_VIDEO_SET,
        ProcessingStage.EXTRACT_FRAME_CANDIDATES,
        ProcessingStage.RESOLVE_MODELS,
        ProcessingStage.COLLECT_CONTEXT,
    )


def test_existing_empty_output_folder_is_removed_before_stage_failure(
    tmp_path: Path,
) -> None:
    """入力preflight成功後のStage失敗時に空Output Folderが残らないこと。

    Arrange:
        - valid Video Setと既存の空Output Folderが用意される
        - Candidate Annotationで失敗するfake VisionRuntimeが用意される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - runtime failureが返されること
        - Processing Stage cacheは確定され、Output Folderは存在しないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    output_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FailingVisionRuntime(),
        observer=RecordingRunObserver(),
    )

    # Act
    with pytest.raises(RuntimeError, match="fake vision failure"):
        application.run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=output_folder,
                image_count=1,
            )
        )

    # Assert
    assert (input_folder / ".game-screen-pick" / "cache").is_dir()
    assert not output_folder.exists()


def test_invalid_output_folder_is_rejected_before_cache_side_effects(
    tmp_path: Path,
) -> None:
    """利用できないOutput FolderがStage開始前に拒否されること。

    Arrange:
        - 既存fileを持つOutput Folderが指定される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - Output Folder validation errorが返されること
        - processing cacheが作成されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    output_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    existing_output = output_folder / "keep.txt"
    existing_output.write_text("keep", encoding="utf-8")
    candidate = FrameCandidate(
        identifier="frame-001",
        image_bytes=b"fake-webp-image",
    )
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="summary"),)
        ),
        observer=RecordingRunObserver(),
    )

    # Act
    # Assert
    with pytest.raises(ValueError, match="Output Folderは存在しないか空"):
        application.run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=output_folder,
                image_count=1,
            )
        )
    assert existing_output.read_text(encoding="utf-8") == "keep"
    assert not (input_folder / ".game-screen-pick").exists()


def test_resolved_model_identity_changes_model_stage_fingerprint(
    tmp_path: Path,
) -> None:
    """Resolved Model Identity変更時にmodel Stageが共存されること。

    Arrange:
        - 同じVideo Setに対して異なるmodel identityを返す2実行がある
    Act:
        - それぞれ別のOutput Folderへ選定結果が公開される
    Assert:
        - Resolved Model Stageに異なるfingerprint artifactが保存されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(
        identifier="frame-001",
        image_bytes=b"fake-webp-image",
    )
    annotation = CandidateAnnotation(candidate=candidate, summary="summary")

    # Act
    for run_index, model_identity in enumerate(
        ("model-sha-001", "model-sha-002"),
        start=1,
    ):
        application = WalkingSkeletonApplication(
            media_runtime=FakeMediaRuntime((candidate,)),
            speech_runtime=FakeContextCollector(()),
            model_runtime=FakeModelRuntime(model_identity),
            vision_runtime=FakeVisionRuntime((annotation,)),
            observer=RecordingRunObserver(),
        )
        application.run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=tmp_path / f"output-{run_index}",
                image_count=1,
            )
        )

    # Assert
    assert (
        len(_video_set_stage_manifests(input_folder, ProcessingStage.RESOLVE_MODELS))
        == 2
    )
    assert (
        len(_video_set_stage_manifests(input_folder, ProcessingStage.COLLECT_CONTEXT))
        == 1
    )
    assert (
        len(
            _video_set_stage_manifests(
                input_folder,
                ProcessingStage.ANNOTATE_CANDIDATES,
            )
        )
        == 2
    )


def test_candidate_model_context_setting_changes_annotation_fingerprint(
    tmp_path: Path,
) -> None:
    """Candidate Annotationのcontext設定変更でそのStageが再計算されること。

    Arrange:
        - 同じVideo Setとmodel identityに異なるnum_ctxを持つ2 runが用意される
    Act:
        - 各runが別の空Output Folderへ実行される
    Assert:
        - Candidate Annotationに異なるfingerprintが保存されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    annotation = CandidateAnnotation(candidate=candidate, summary="summary")

    # Act
    for run_index, num_ctx in enumerate((32768, 65536), start=1):
        WalkingSkeletonApplication(
            media_runtime=FakeMediaRuntime((candidate,)),
            speech_runtime=FakeContextCollector(()),
            model_runtime=FakeModelRuntime("same-model"),
            vision_runtime=FakeVisionRuntime((annotation,)),
            observer=RecordingRunObserver(),
        ).run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=tmp_path / f"output-{run_index}",
                image_count=1,
                candidate_annotation_num_ctx=num_ctx,
            )
        )

    # Assert
    assert (
        len(
            _video_set_stage_manifests(
                input_folder,
                ProcessingStage.ANNOTATE_CANDIDATES,
            )
        )
        == 2
    )


def test_video_content_change_changes_discovery_stage_fingerprint(
    tmp_path: Path,
) -> None:
    """同じpathのvideo内容変更時にVideo Set snapshotが変更されること。

    Arrange:
        - 同じrelative pathへ異なる内容が順番に保存される
    Act:
        - 各内容で別のOutput Folderへ選定結果が公開される
    Assert:
        - discovery Stageに異なるfingerprint artifactが保存されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_path = input_folder / "chapter-01.mp4"
    candidate = FrameCandidate(
        identifier="frame-001",
        image_bytes=b"fake-webp-image",
    )
    annotation = CandidateAnnotation(candidate=candidate, summary="summary")

    # Act
    for run_index, video_content in enumerate((b"video-01", b"video-02"), start=1):
        video_path.write_bytes(video_content)
        WalkingSkeletonApplication(
            media_runtime=FakeMediaRuntime((candidate,)),
            speech_runtime=FakeContextCollector(()),
            model_runtime=FakeModelRuntime("model-sha-001"),
            vision_runtime=FakeVisionRuntime((annotation,)),
            observer=RecordingRunObserver(),
        ).run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=tmp_path / f"output-{run_index}",
                image_count=1,
            )
        )

    # Assert
    assert (
        len(
            _video_set_stage_manifests(
                input_folder,
                ProcessingStage.DISCOVER_VIDEO_SET,
            )
        )
        == 2
    )


def test_warm_run_reuses_cached_candidate_annotations(tmp_path: Path) -> None:
    """warm runでCompleted Candidate Annotationが再利用されること。

    Arrange:
        - Candidate Annotationが確定済みの初回runがある
        - 同じ入力で呼ばれると失敗するVisionRuntimeが用意される
    Act:
        - 別の空Output Folderへwarm runが実行される
    Assert:
        - VisionRuntimeを再実行せずcached annotationからreportが公開されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(
        identifier="frame-001",
        image_bytes=b"fake-webp-image",
    )
    model_runtime = FakeModelRuntime("model-sha-001")
    WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=model_runtime,
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="cached summary"),)
        ),
        observer=RecordingRunObserver(),
    ).run(
        EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=tmp_path / "output-1",
            image_count=1,
        )
    )
    warm_output_folder = tmp_path / "output-2"
    warm_application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=model_runtime,
        vision_runtime=FailingVisionRuntime(),
        observer=RecordingRunObserver(),
    )

    # Act
    warm_application.run(
        EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=warm_output_folder,
            image_count=1,
        )
    )

    # Assert
    report = json.loads(
        (warm_output_folder / "report.json").read_text(encoding="utf-8")
    )
    assert report["selected"][0]["summary"] == "cached summary"


def test_speech_model_change_does_not_invalidate_candidate_annotation_cache(
    tmp_path: Path,
) -> None:
    """STT model identity変更でCandidate Annotation cacheが無効化されないこと。

    Arrange:
        - candidate modelが同じでSTT modelだけが異なる2 runが用意される
        - 初回runでCandidate Annotationが確定される
        - 再実行されると失敗するVisionRuntimeが2回目に用意される
    Act:
        - 2回目のrunが別の空Output Folderへ実行される
    Assert:
        - model resolutionは別fingerprintとなること
        - Candidate Annotationは既存cacheから再利用されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    annotation = CandidateAnnotation(candidate=candidate, summary="cached summary")
    WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(
            (),
            speech_runtime_identity="speech-runtime-v1",
        ),
        model_runtime=FakeModelRuntime(
            "candidate-model",
            speech_identity_seed="speech-model-a",
        ),
        vision_runtime=FakeVisionRuntime((annotation,)),
        observer=RecordingRunObserver(),
    ).run(
        EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=tmp_path / "output-1",
            image_count=1,
        )
    )

    # Act
    WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(
            (),
            speech_runtime_identity="speech-runtime-v1",
        ),
        model_runtime=FakeModelRuntime(
            "candidate-model",
            speech_identity_seed="speech-model-b",
        ),
        vision_runtime=FailingVisionRuntime(),
        observer=RecordingRunObserver(),
    ).run(
        EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=tmp_path / "output-2",
            image_count=1,
        )
    )

    # Assert
    assert (
        len(_video_set_stage_manifests(input_folder, ProcessingStage.RESOLVE_MODELS))
        == 2
    )
    assert (
        len(_video_set_stage_manifests(input_folder, ProcessingStage.COLLECT_CONTEXT))
        == 2
    )
    assert (
        len(
            _video_set_stage_manifests(
                input_folder,
                ProcessingStage.ANNOTATE_CANDIDATES,
            )
        )
        == 1
    )
    assert (
        len(_video_set_stage_manifests(input_folder, ProcessingStage.SELECT_IMAGES))
        == 1
    )


def test_context_fingerprint_includes_speech_inputs_only_when_stt_runs(
    tmp_path: Path,
) -> None:
    """STT関連identityとprofileがSTT実行時だけfingerprintへ入ること。

    Arrange:
        - 同じCueを返すsubtitle-only runとSTT runが用意される
    Act:
        - 両runが別のOutput Folderへ実行される
    Assert:
        - Context cacheが別fingerprintとして共存すること
        - subtitle-only入力にSTT fieldがなくSTT入力に全profileがあること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    cue = ContextCue(identifier="cue-001")
    annotation = CandidateAnnotation(candidate=candidate, summary="summary")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "unused",
        image_count=1,
        speech_to_text_beam_size=7,
        speech_vad_filter=False,
        speech_chunk_seconds=120.0,
        speech_overlap_seconds=3.0,
    )

    # Act
    for run_name, collector in (
        ("subtitle", FakeContextCollector((cue,))),
        (
            "stt",
            FakeContextCollector(
                (cue,),
                speech_runtime_identity="speech-runtime-v2",
            ),
        ),
    ):
        WalkingSkeletonApplication(
            media_runtime=FakeMediaRuntime((candidate,)),
            speech_runtime=collector,
            model_runtime=FakeModelRuntime("model-sha-001"),
            vision_runtime=FakeVisionRuntime((annotation,)),
            observer=RecordingRunObserver(),
        ).run(replace(configuration, output_folder=tmp_path / f"output-{run_name}"))

    # Assert
    manifests = _video_set_stage_manifests(
        input_folder,
        ProcessingStage.COLLECT_CONTEXT,
    )
    assert len(manifests) == 2
    semantic_inputs = [
        json.loads(path.read_text(encoding="utf-8"))["semantic_input"]
        for path in manifests
    ]
    subtitle_input = next(
        item for item in semantic_inputs if "speech_to_text" not in item
    )
    speech_input = next(item for item in semantic_inputs if "speech_to_text" in item)
    assert subtitle_input == {"context_cue_ids": ["cue-001"]}
    assert speech_input["speech_to_text"] == {
        "beam_size": 7,
        "chunk_seconds": 120.0,
        "compute_type": "float16",
        "device": "cuda",
        "language": "ja",
        "model": FakeModelRuntime("model-sha-001")
        .resolve_models(configuration)
        .for_role(ModelRole.SPEECH_TO_TEXT)
        .semantic_input(),
        "overlap_seconds": 3.0,
        "runtime_identity": "speech-runtime-v2",
        "vad_filter": False,
    }


def test_annotation_order_is_normalized_to_frame_candidate_order(
    tmp_path: Path,
) -> None:
    """runtimeとcacheのAnnotation順序がFrame Candidate順へ正規化されること。

    Arrange:
        - Frame Candidateと逆順のCandidate Annotationが返されるcold runがある
        - 同じ入力でVisionRuntimeを再実行できないwarm runがある
    Act:
        - cold runとwarm runが別のOutput Folderへ実行される
    Assert:
        - 両方のselected imageがFrame Candidate順に公開されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    first_candidate = FrameCandidate(identifier="frame-001", image_bytes=b"first")
    second_candidate = FrameCandidate(identifier="frame-002", image_bytes=b"second")
    candidates = (first_candidate, second_candidate)
    model_runtime = FakeModelRuntime("model-sha-001")
    cold_output_folder = tmp_path / "output-cold"
    warm_output_folder = tmp_path / "output-warm"
    cold_application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime(candidates),
        speech_runtime=FakeContextCollector(()),
        model_runtime=model_runtime,
        vision_runtime=FakeVisionRuntime(
            (
                CandidateAnnotation(candidate=second_candidate, summary="second"),
                CandidateAnnotation(candidate=first_candidate, summary="first"),
            )
        ),
        observer=RecordingRunObserver(),
    )
    warm_application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime(candidates),
        speech_runtime=FakeContextCollector(()),
        model_runtime=model_runtime,
        vision_runtime=FailingVisionRuntime(),
        observer=RecordingRunObserver(),
    )

    # Act
    cold_application.run(
        EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=cold_output_folder,
            image_count=2,
        )
    )
    warm_application.run(
        EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=warm_output_folder,
            image_count=2,
        )
    )

    # Assert
    for output_folder in (cold_output_folder, warm_output_folder):
        report = json.loads((output_folder / "report.json").read_text(encoding="utf-8"))
        assert [item["id"] for item in report["selected"]] == [
            "frame-001",
            "frame-002",
        ]


def test_candidate_identity_changes_extraction_stage_fingerprint(
    tmp_path: Path,
) -> None:
    """Frame Candidate identity変更時に抽出Stageが共存されること。

    Arrange:
        - 同じVideo Setから異なるCandidate IDを返す2実行がある
        - どちらの実行もCandidate件数は同じである
    Act:
        - それぞれ別のOutput Folderへ選定結果が公開される
    Assert:
        - Frame Candidate抽出Stageに異なるfingerprintが保存されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")

    # Act
    for run_index, candidate_id in enumerate(("frame-001", "frame-002"), start=1):
        candidate = FrameCandidate(
            identifier=candidate_id,
            image_bytes=f"image-{run_index}".encode(),
        )
        WalkingSkeletonApplication(
            media_runtime=FakeMediaRuntime((candidate,)),
            speech_runtime=FakeContextCollector(()),
            model_runtime=FakeModelRuntime("model-sha-001"),
            vision_runtime=FakeVisionRuntime(
                (CandidateAnnotation(candidate=candidate, summary="summary"),)
            ),
            observer=RecordingRunObserver(),
        ).run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=tmp_path / f"output-{run_index}",
                image_count=1,
            )
        )

    # Assert
    assert (
        len(
            _video_set_stage_manifests(
                input_folder,
                ProcessingStage.EXTRACT_FRAME_CANDIDATES,
            )
        )
        == 2
    )


def test_context_identity_changes_context_stage_fingerprint(tmp_path: Path) -> None:
    """Context Cue identity変更時にcontext Stageが共存されること。

    Arrange:
        - 同じVideo Setから異なるContext Cue IDを返す2実行がある
        - どちらの実行もContext Cue件数は同じである
    Act:
        - それぞれ別のOutput Folderへ選定結果が公開される
    Assert:
        - Context収集Stageに異なるfingerprintが保存されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(
        identifier="frame-001",
        image_bytes=b"fake-webp-image",
    )
    annotation = CandidateAnnotation(candidate=candidate, summary="summary")

    # Act
    for run_index, context_id in enumerate(("cue-001", "cue-002"), start=1):
        WalkingSkeletonApplication(
            media_runtime=FakeMediaRuntime((candidate,)),
            speech_runtime=FakeContextCollector((ContextCue(identifier=context_id),)),
            model_runtime=FakeModelRuntime("model-sha-001"),
            vision_runtime=FakeVisionRuntime((annotation,)),
            observer=RecordingRunObserver(),
        ).run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=tmp_path / f"output-{run_index}",
                image_count=1,
            )
        )

    # Assert
    assert (
        len(_video_set_stage_manifests(input_folder, ProcessingStage.COLLECT_CONTEXT))
        == 2
    )


def test_duplicate_video_content_is_rejected_before_cache_side_effects(
    tmp_path: Path,
) -> None:
    """同一内容の重複videoがcache作成前に拒否されること。

    Arrange:
        - 異なるrelative pathに同一内容のvideoが用意される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - duplicate video errorが返されること
        - processing cacheとOutput Folderが作成されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"same-video")
    (input_folder / "chapter-02.mp4").write_bytes(b"same-video")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="summary"),)
        ),
        observer=RecordingRunObserver(),
    )

    # Act
    with pytest.raises(ValueError, match="Duplicate Video"):
        application.run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=output_folder,
                image_count=1,
            )
        )

    # Assert
    assert not (input_folder / ".game-screen-pick").exists()
    assert not output_folder.exists()


def test_existing_empty_output_folder_is_preserved_when_input_preflight_fails(
    tmp_path: Path,
) -> None:
    """入力preflight失敗時に既存の空Output Folderが保持されること。

    Arrange:
        - 既存の空Output Folderと同一内容の重複videoが用意される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - duplicate video errorが返されること
        - Output Folderが空directoryのまま保持されること
        - processing cacheが作成されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    output_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"same-video")
    (input_folder / "chapter-02.mp4").write_bytes(b"same-video")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="summary"),)
        ),
        observer=RecordingRunObserver(),
    )

    # Act
    with pytest.raises(ValueError, match="Duplicate Video"):
        application.run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=output_folder,
                image_count=1,
            )
        )

    # Assert
    assert output_folder.is_dir()
    assert tuple(output_folder.iterdir()) == ()
    assert not (input_folder / ".game-screen-pick").exists()


def test_output_nested_in_input_is_rejected_before_cache_side_effects(
    tmp_path: Path,
) -> None:
    """input配下のOutput Folderがcache作成前に拒否されること。

    Arrange:
        - Video Input Folder配下の未作成pathがOutput Folderに指定される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - input/output relationship errorが返されること
        - processing cacheとOutput Folderが作成されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = input_folder / "output"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="summary"),)
        ),
        observer=RecordingRunObserver(),
    )

    # Act
    with pytest.raises(ValueError, match="相互の親子pathにできません"):
        application.run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=output_folder,
                image_count=1,
            )
        )

    # Assert
    assert not (input_folder / ".game-screen-pick").exists()
    assert not output_folder.exists()


def test_existing_empty_output_folder_is_accepted(tmp_path: Path) -> None:
    """既存の空Output Folderへatomic outputが公開されること。

    Arrange:
        - 既存の空Output Folderと一つのdummy videoが用意される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - 空folderが安全に引き渡され、選定reportが公開されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    output_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="summary"),)
        ),
        observer=RecordingRunObserver(),
    )

    # Act
    application.run(
        EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=output_folder,
            image_count=1,
        )
    )

    # Assert
    assert (
        json.loads((output_folder / "report.json").read_text(encoding="utf-8"))[
            "status"
        ]
        == "completed"
    )


def test_changed_candidate_content_changes_extraction_stage_fingerprint(
    tmp_path: Path,
) -> None:
    """同じCandidate IDの画像内容変更時に抽出Stageが共存されること。

    Arrange:
        - 同じVideo SetとCandidate IDに対して異なるimage bytesを返す2実行がある
    Act:
        - それぞれ別のOutput Folderへ選定結果が公開される
    Assert:
        - Frame Candidate抽出Stageに異なるfingerprintが保存されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")

    # Act
    for run_index, image_bytes in enumerate((b"image-1", b"image-2"), start=1):
        candidate = FrameCandidate(
            identifier="frame-001",
            image_bytes=image_bytes,
        )
        WalkingSkeletonApplication(
            media_runtime=FakeMediaRuntime((candidate,)),
            speech_runtime=FakeContextCollector(()),
            model_runtime=FakeModelRuntime("model-sha-001"),
            vision_runtime=FakeVisionRuntime(
                (CandidateAnnotation(candidate=candidate, summary="summary"),)
            ),
            observer=RecordingRunObserver(),
        ).run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=tmp_path / f"output-{run_index}",
                image_count=1,
            )
        )

    # Assert
    assert (
        len(
            _video_set_stage_manifests(
                input_folder,
                ProcessingStage.EXTRACT_FRAME_CANDIDATES,
            )
        )
        == 2
    )


def test_duplicate_candidate_ids_are_rejected_before_candidate_cache(
    tmp_path: Path,
) -> None:
    """重複Frame Candidate IDが抽出Stage確定前に拒否されること。

    Arrange:
        - 同じidentifierを持つ2つのFrame Candidateが用意される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - duplicate Candidate ID errorが返されること
        - 抽出Stage cacheとOutput Folderが作成されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    first_candidate = FrameCandidate(identifier="frame-001", image_bytes=b"first")
    second_candidate = FrameCandidate(identifier="frame-001", image_bytes=b"second")
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((first_candidate, second_candidate)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FakeVisionRuntime(
            (
                CandidateAnnotation(candidate=first_candidate, summary="first"),
                CandidateAnnotation(candidate=second_candidate, summary="second"),
            )
        ),
        observer=RecordingRunObserver(),
    )

    # Act
    with pytest.raises(ValueError, match="Frame Candidate IDが重複"):
        application.run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=output_folder,
                image_count=2,
            )
        )

    # Assert
    assert not _video_set_stage_manifests(
        input_folder,
        ProcessingStage.EXTRACT_FRAME_CANDIDATES,
    )
    assert not output_folder.exists()


def test_unsafe_candidate_id_is_rejected_before_candidate_cache(
    tmp_path: Path,
) -> None:
    """path separatorを含むFrame Candidate IDがcache確定前に拒否されること。

    Arrange:
        - path separatorを含むidentifierのFrame Candidateが用意される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - unsafe Candidate ID errorが返されること
        - 抽出Stage cacheとOutput Folderが作成されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(identifier="scene/frame", image_bytes=b"image")
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="summary"),)
        ),
        observer=RecordingRunObserver(),
    )

    # Act
    with pytest.raises(ValueError, match="Frame Candidate IDが安全ではありません"):
        application.run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=output_folder,
                image_count=1,
            )
        )

    # Assert
    assert not _video_set_stage_manifests(
        input_folder,
        ProcessingStage.EXTRACT_FRAME_CANDIDATES,
    )
    assert not output_folder.exists()


def test_foreign_candidate_annotation_is_rejected_before_caching(
    tmp_path: Path,
) -> None:
    """抽出結果に属さないCandidate Annotationがcache前に拒否されること。

    Arrange:
        - MediaRuntimeの抽出結果と異なるcandidateを返すVisionRuntimeが用意される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - foreign Candidate Annotation errorが返されること
        - annotation Stage cacheとOutput Folderが作成されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    extracted_candidate = FrameCandidate(
        identifier="frame-001",
        image_bytes=b"extracted",
    )
    foreign_candidate = FrameCandidate(
        identifier="frame-foreign",
        image_bytes=b"foreign",
    )
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((extracted_candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=foreign_candidate, summary="foreign"),)
        ),
        observer=RecordingRunObserver(),
    )

    # Act
    with pytest.raises(ValueError, match="未知のFrame Candidate"):
        application.run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=output_folder,
                image_count=1,
            )
        )

    # Assert
    assert not _video_set_stage_manifests(
        input_folder,
        ProcessingStage.ANNOTATE_CANDIDATES,
    )
    assert not output_folder.exists()


def test_selection_shortfall_is_published_with_warning(tmp_path: Path) -> None:
    """要求枚数未満の選定結果がwarning付き成功として公開されること。

    Arrange:
        - 要求2枚に対して1件のCandidate Annotationが用意される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - outcomeとreportがcompleted_with_warningsになること
        - requested countとselected countを持つshortfall warningが公開されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="summary"),)
        ),
        observer=RecordingRunObserver(),
    )

    # Act
    outcome = application.run(
        EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=output_folder,
            image_count=2,
        )
    )

    # Assert
    assert outcome.status == "completed_with_warnings"
    assert outcome.requested_count == 2
    assert outcome.selected_count == 1
    report = json.loads((output_folder / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_with_warnings"
    assert report["requested_count"] == 2
    assert report["selected_count"] == 1
    assert report["warnings"] == [
        {
            "code": "selection_shortfall",
            "requested_count": 2,
            "selected_count": 1,
        }
    ]
    assert "Selection Shortfall: requested=2, selected=1" in (
        output_folder / "report.md"
    ).read_text(encoding="utf-8")


def test_unavailable_model_update_is_published_with_warning(tmp_path: Path) -> None:
    """offline local model利用がwarning付き成功として公開されること。

    Arrange:
        - 全枚数を選定できSTT model更新だけがunavailableのrunが用意される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - outcomeとreportがcompleted_with_warningsになること
        - warningと完全model provenanceがJSONへ公開されること
        - Markdownには短縮identityだけが公開されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime(
            "model-sha-001",
            unavailable_roles=frozenset({ModelRole.SPEECH_TO_TEXT}),
        ),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="summary"),)
        ),
        observer=RecordingRunObserver(),
    )
    full_sha = hashlib.sha256(b"speech-model").hexdigest()[:40]

    # Act
    outcome = application.run(
        EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=output_folder,
            image_count=1,
        )
    )

    # Assert
    assert outcome.status is RunStatus.COMPLETED_WITH_WARNINGS
    report = json.loads((output_folder / "report.json").read_text(encoding="utf-8"))
    assert report["warnings"] == [
        {
            "code": "model_update_unavailable",
            "roles": ["speech_to_text"],
        }
    ]
    assert report["models"]["speech_to_text"]["execution_identity"] == (
        f"hf:{full_sha}"
    )
    markdown = (output_folder / "report.md").read_text(encoding="utf-8")
    assert f"hf:{full_sha[:12]}…" in markdown
    assert f"hf:{full_sha}" not in markdown


def test_run_diagnostics_do_not_change_or_stale_model_resolution_cache(
    tmp_path: Path,
) -> None:
    """同じ実行identityのrun診断がcache fingerprintと分離されること。

    Arrange:
        - 同じmodel identityでnot_requestedとunavailableになる2 runが用意される
    Act:
        - 各runが別のOutput Folderへ実行される
    Assert:
        - Model Resolution cacheは一つだけ共存すること
        - cache artifactへrun固有update statusが保存されないこと
        - 2回目reportへ現在runのunavailableが記録されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    annotation = CandidateAnnotation(candidate=candidate, summary="summary")

    # Act
    for run_index, unavailable_roles in enumerate(
        (frozenset(), frozenset({ModelRole.SPEECH_TO_TEXT})),
        start=1,
    ):
        WalkingSkeletonApplication(
            media_runtime=FakeMediaRuntime((candidate,)),
            speech_runtime=FakeContextCollector(()),
            model_runtime=FakeModelRuntime(
                "model-sha-001",
                unavailable_roles=unavailable_roles,
            ),
            vision_runtime=FakeVisionRuntime((annotation,)),
            observer=RecordingRunObserver(),
        ).run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=tmp_path / f"output-{run_index}",
                image_count=1,
            )
        )

    # Assert
    manifests = _video_set_stage_manifests(
        input_folder,
        ProcessingStage.RESOLVE_MODELS,
    )
    assert len(manifests) == 1
    cache_artifact = json.loads(
        (manifests[0].parent / "artifact.json").read_text(encoding="utf-8")
    )
    assert "update_status" not in json.dumps(cache_artifact)
    report = json.loads(
        (tmp_path / "output-2" / "report.json").read_text(encoding="utf-8")
    )
    assert report["models"]["speech_to_text"]["update_status"] == "unavailable"


def test_invalid_output_parent_is_rejected_before_cache_side_effects(
    tmp_path: Path,
) -> None:
    """directoryでないOutput親pathがcache作成前に拒否されること。

    Arrange:
        - Output Folderの親componentにregular fileが用意される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - invalid output parent errorが返されること
        - processing cacheとOutput Folderが作成されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    blocked_parent = tmp_path / "blocked-parent"
    output_folder = blocked_parent / "output"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    blocked_parent.write_text("blocked", encoding="utf-8")
    candidate = FrameCandidate(identifier="frame-001", image_bytes=b"image")
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=candidate, summary="summary"),)
        ),
        observer=RecordingRunObserver(),
    )

    # Act
    with pytest.raises(ValueError, match="Output Folderの親path"):
        application.run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=output_folder,
                image_count=1,
            )
        )

    # Assert
    assert blocked_parent.read_text(encoding="utf-8") == "blocked"
    assert not (input_folder / ".game-screen-pick").exists()
    assert not output_folder.exists()


def test_incomplete_candidate_annotations_are_rejected_before_caching(
    tmp_path: Path,
) -> None:
    """抽出候補を欠落したCandidate Annotation集合がcache前に拒否されること。

    Arrange:
        - 2つのFrame Candidateに対して1件だけannotationが返される
    Act:
        - Video Set選定applicationが実行される
    Assert:
        - incomplete Candidate Annotation errorが返されること
        - annotation Stage cacheとOutput Folderが作成されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"video-01")
    first_candidate = FrameCandidate(identifier="frame-001", image_bytes=b"first")
    second_candidate = FrameCandidate(identifier="frame-002", image_bytes=b"second")
    application = WalkingSkeletonApplication(
        media_runtime=FakeMediaRuntime((first_candidate, second_candidate)),
        speech_runtime=FakeContextCollector(()),
        model_runtime=FakeModelRuntime("model-sha-001"),
        vision_runtime=FakeVisionRuntime(
            (CandidateAnnotation(candidate=first_candidate, summary="first"),)
        ),
        observer=RecordingRunObserver(),
    )

    # Act
    with pytest.raises(ValueError, match="Candidate Annotationが不足"):
        application.run(
            EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=output_folder,
                image_count=2,
            )
        )

    # Assert
    assert not _video_set_stage_manifests(
        input_folder,
        ProcessingStage.ANNOTATE_CANDIDATES,
    )
    assert not output_folder.exists()
