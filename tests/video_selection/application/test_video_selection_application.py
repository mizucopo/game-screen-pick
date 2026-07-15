"""Video Set選定applicationのintegration test。"""

import json
from pathlib import Path

import pytest

from src.video_selection.application.video_selection_application import (
    VideoSelectionApplication,
)
from src.video_selection.models.candidate_annotation import CandidateAnnotation
from src.video_selection.models.context_cue import ContextCue
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.frame_candidate import FrameCandidate
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.resolved_model_identity import ResolvedModelIdentity
from tests.video_selection.fakes.failing_vision_runtime import FailingVisionRuntime
from tests.video_selection.fakes.fake_media_runtime import FakeMediaRuntime
from tests.video_selection.fakes.fake_model_runtime import FakeModelRuntime
from tests.video_selection.fakes.fake_speech_runtime import FakeSpeechRuntime
from tests.video_selection.fakes.fake_vision_runtime import FakeVisionRuntime
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver


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
    application = VideoSelectionApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeSpeechRuntime((ContextCue(identifier="cue-001"),)),
        model_runtime=FakeModelRuntime(
            ResolvedModelIdentity(identifier="model-sha-001")
        ),
        vision_runtime=FakeVisionRuntime((annotation,)),
        observer=observer,
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=output_folder,
        image_count=1,
    )

    # Act
    outcome = application.run(configuration)

    # Assert
    assert outcome.selected_count == 1
    assert outcome.output_folder == output_folder
    assert (output_folder / "images" / "0001_frame-001.webp").read_bytes() == (
        b"fake-webp-image"
    )
    report = json.loads((output_folder / "report.json").read_text(encoding="utf-8"))
    assert report == {
        "schema": "game-screen-pick/walking-skeleton@0",
        "status": "completed",
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
        "Status: completed\n\n"
        "## Selected images\n\n"
        "1. [frame-001](images/0001_frame-001.webp) — "
        "主人公が草原を進んでいる\n"
    )
    assert tuple(stage.stage for stage in observer.completed_stages) == tuple(
        ProcessingStage
    )
    manifests = tuple(
        (input_folder / ".game-screen-pick" / "cache" / "walking-skeleton").glob(
            "*/*/manifest.json"
        )
    )
    assert len(manifests) == len(ProcessingStage)


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
    application = VideoSelectionApplication(
        media_runtime=FakeMediaRuntime((candidate,)),
        speech_runtime=FakeSpeechRuntime(()),
        model_runtime=FakeModelRuntime(
            ResolvedModelIdentity(identifier="model-sha-001")
        ),
        vision_runtime=FailingVisionRuntime(),
        observer=observer,
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=output_folder,
        image_count=1,
    )

    # Act / Assert
    with pytest.raises(RuntimeError, match="fake vision failure"):
        application.run(configuration)
    assert not output_folder.exists()
    assert tuple(stage.stage for stage in observer.completed_stages) == (
        ProcessingStage.DISCOVER_VIDEO_SET,
        ProcessingStage.EXTRACT_FRAME_CANDIDATES,
        ProcessingStage.COLLECT_CONTEXT,
        ProcessingStage.RESOLVE_MODELS,
    )
