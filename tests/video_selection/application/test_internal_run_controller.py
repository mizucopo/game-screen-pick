from pathlib import Path

import pytest

from src.video_selection.application.internal_run_controller import (
    InternalRunController,
)
from src.video_selection.configuration.configuration_error import ConfigurationError
from src.video_selection.models.context_stage_error import ContextStageError
from src.video_selection.models.context_stage_failure_reason import (
    ContextStageFailureReason,
)
from src.video_selection.models.media_runtime_error import MediaRuntimeError
from src.video_selection.models.media_runtime_failure_reason import (
    MediaRuntimeFailureReason,
)
from src.video_selection.models.model_role import ModelRole
from src.video_selection.models.model_runtime_error import ModelRuntimeError
from src.video_selection.models.model_runtime_failure_reason import (
    ModelRuntimeFailureReason,
)
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.run_failure import RunFailure
from src.video_selection.models.vision_runtime_error import VisionRuntimeError
from src.video_selection.models.vision_runtime_failure_reason import (
    VisionRuntimeFailureReason,
)
from src.video_selection.services.processing_stage_runner import (
    ProcessingStageRunner,
)
from src.video_selection.services.run_progress_tracker import RunProgressTracker
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver


def test_internal_run_controller_maps_typed_operation_failure() -> None:
    """reason-coded operation errorが安全なexit 1結果へ変換されること。

    Arrange:
        - raw absolute pathをmessageに含むMedia Runtime errorが用意される
    Act:
        - internal run controllerから失敗するoperationが実行される
    Assert:
        - stable reasonだけのRun Failureとrun_failed eventが返されること
    """
    # Arrange
    observer = RecordingRunObserver()
    tracker = RunProgressTracker(observer)
    controller = InternalRunController(tracker)
    error = MediaRuntimeError(
        MediaRuntimeFailureReason.DECODER_FAILURE,
        "/private/video.mkv: ffmpeg raw output",
    )

    def fail_operation() -> object:
        raise error

    # Act
    exit_code, result = controller.execute(fail_operation)

    # Assert
    assert isinstance(result, RunFailure)
    assert (
        exit_code,
        result.reason_code,
        result.remediation_code,
        result.resume_guidance,
        result.cause,
        tuple(event.kind for event in observer.progress_events),
        tuple(event.reason_code for event in observer.progress_events),
    ) == (
        1,
        "decoder_failure",
        "check_media_runtime",
        "completed_stages_reusable",
        error,
        ("run_started", "run_failed"),
        ("run_started", "decoder_failure"),
    )
    assert "/private" not in repr(result)


def test_internal_run_controller_maps_keyboard_interrupt() -> None:
    """active Stage中のCtrl+Cがrun_interruptedとexit 130へ変換されること。

    Arrange:
        - Processing Stage開始後にKeyboardInterruptとなるoperationが用意される
    Act:
        - internal run controllerからoperationが実行される
    Assert:
        - active Stage付きuser_interrupt eventと安全なRun Failureが返されること
    """
    # Arrange
    observer = RecordingRunObserver()
    tracker = RunProgressTracker(observer, clock=lambda: 10.0)
    controller = InternalRunController(tracker)

    def interrupt_operation() -> object:
        tracker.start_stage(
            ProcessingStage.SCAN_VIDEO,
            video_order=2,
            video_count=3,
            video_relative_path="chapter-02.mkv",
            work_unit_kind="video",
        )
        raise KeyboardInterrupt

    # Act
    exit_code, result = controller.execute(interrupt_operation)

    # Assert
    assert isinstance(result, RunFailure)
    terminal = observer.progress_events[-1]
    assert (
        exit_code,
        result.reason_code,
        result.resume_guidance,
        terminal.kind,
        terminal.reason_code,
        terminal.stage,
        terminal.stage_index,
        terminal.video_relative_path,
    ) == (
        130,
        "user_interrupt",
        "completed_stages_reusable",
        "run_interrupted",
        "user_interrupt",
        ProcessingStage.SCAN_VIDEO,
        1,
        "chapter-02.mkv",
    )


def test_internal_run_controller_maps_configuration_error() -> None:
    """設定errorがraw messageを使わずexit 2へ変換されること。

    Arrange:
        - secret相当文字列をmessageに含むConfiguration Errorが用意される
    Act:
        - internal run controllerから失敗するoperationが実行される
    Assert:
        - stable reasonとrun_not_started guidanceだけが返されること
    """
    # Arrange
    observer = RecordingRunObserver()
    tracker = RunProgressTracker(observer)
    controller = InternalRunController(tracker)
    error = ConfigurationError(
        "invalid_configuration",
        "token=/private/secret",
    )

    def fail_configuration() -> object:
        raise error

    # Act
    exit_code, result = controller.execute(fail_configuration)

    # Assert
    assert isinstance(result, RunFailure)
    assert (
        exit_code,
        result.reason_code,
        result.remediation_code,
        result.resume_guidance,
        observer.progress_events[-1].kind,
    ) == (
        2,
        "invalid_configuration",
        "fix_configuration",
        "run_not_started",
        "run_failed",
    )
    assert "secret" not in repr(result)


def test_internal_run_controller_normalizes_configuration_reason_code() -> None:
    """設定層の大文字reason codeがstable codeへ正規化されること。

    Arrange:
        - 実設定層と同じ大文字reason codeのConfiguration Errorが用意される
    Act:
        - internal run controllerから失敗するoperationが実行される
    Assert:
        - 小文字reason codeを持つexit 2のRun Failureが返されること
    """
    # Arrange
    observer = RecordingRunObserver()
    tracker = RunProgressTracker(observer)
    controller = InternalRunController(tracker)
    error = ConfigurationError(
        "CONFIG_INVALID_TYPE",
        "video_input_folderはpathである必要があります",
    )

    def fail_configuration() -> object:
        raise error

    # Act
    exit_code, result = controller.execute(fail_configuration)

    # Assert
    assert isinstance(result, RunFailure)
    assert (
        exit_code,
        result.reason_code,
        result.remediation_code,
        result.resume_guidance,
        observer.progress_events[-1].reason_code,
    ) == (
        2,
        "config_invalid_type",
        "fix_configuration",
        "run_not_started",
        "config_invalid_type",
    )


def test_internal_run_controller_keeps_runtime_value_error_operational() -> None:
    """実行中のValueErrorがusage errorへ誤分類されないこと。

    Arrange:
        - snapshot変更を表すplain ValueErrorが用意される
    Act:
        - internal run controllerから失敗するoperationが実行される
    Assert:
        - internal operation failureとしてexit 1が返されること
    """
    # Arrange
    observer = RecordingRunObserver()
    tracker = RunProgressTracker(observer)
    controller = InternalRunController(tracker)
    error = ValueError("Video Set snapshotがfingerprint計算中に変更されました")

    def fail_during_snapshot() -> object:
        raise error

    # Act
    exit_code, result = controller.execute(fail_during_snapshot)

    # Assert
    assert isinstance(result, RunFailure)
    assert (
        exit_code,
        result.reason_code,
        result.remediation_code,
        result.resume_guidance,
    ) == (
        1,
        "internal_error",
        "report_internal_error",
        "completed_stages_reusable",
    )


def test_internal_run_controller_hides_unexpected_error_detail() -> None:
    """未知例外がraw detailを出さずinternal_errorへ変換されること。

    Arrange:
        - absolute pathとraw textを含む未知のRuntimeErrorが用意される
    Act:
        - internal run controllerから失敗するoperationが実行される
    Assert:
        - internal_errorと安全なterminal eventだけが返されること
    """
    # Arrange
    observer = RecordingRunObserver()
    tracker = RunProgressTracker(observer)
    controller = InternalRunController(tracker)
    error = RuntimeError("/private/cache: raw transcript")

    def fail_unexpectedly() -> object:
        raise error

    # Act
    exit_code, result = controller.execute(fail_unexpectedly)

    # Assert
    assert isinstance(result, RunFailure)
    assert (
        exit_code,
        result.reason_code,
        result.remediation_code,
        result.cause,
        observer.progress_events[-1].reason_code,
    ) == (
        1,
        "internal_error",
        "report_internal_error",
        error,
        "internal_error",
    )
    assert "private" not in repr(result)


def test_internal_run_controller_rejects_unsafe_typed_error_metadata() -> None:
    """typed errorに混入したraw metadataがinternal errorへ安全化されること。

    Arrange:
        - absolute pathとraw responseをvalidation codeに持つVision errorが用意される
    Act:
        - internal run controllerから失敗するoperationが実行される
    Assert:
        - raw値を捨てたinternal_errorだけがeventとRun Failureへ返されること
    """
    # Arrange
    observer = RecordingRunObserver()
    tracker = RunProgressTracker(observer)
    controller = InternalRunController(tracker)
    error = VisionRuntimeError(
        VisionRuntimeFailureReason.SCHEMA_INVALID,
        validation_code="/private/model: raw response",
    )

    def fail_with_unsafe_metadata() -> object:
        raise error

    # Act
    exit_code, result = controller.execute(fail_with_unsafe_metadata)

    # Assert
    assert isinstance(result, RunFailure)
    assert (
        exit_code,
        result.reason_code,
        result.observed_values,
        result.cause,
        observer.progress_events[-1].reason_code,
    ) == (1, "internal_error", (), error, "internal_error")
    assert "private" not in repr(result)


def test_internal_run_controller_normalizes_unclosed_stage_on_success() -> None:
    """active Stageを残した正常returnがinternal errorへ変換されること。

    Arrange:
        - Stage開始後に完了通知せず値を返すoperationが用意される
    Act:
        - internal run controllerがrun完了を試行する
    Assert:
        - lifecycle例外が漏れずactive Stage付きrun_failedへ変換されること
    """
    # Arrange
    observer = RecordingRunObserver()
    tracker = RunProgressTracker(observer)
    controller = InternalRunController(tracker)

    def leave_stage_active() -> str:
        tracker.start_stage(
            ProcessingStage.SELECT_IMAGES,
            work_unit_kind="candidate",
        )
        return "incomplete"

    # Act
    exit_code, result = controller.execute(leave_stage_active)

    # Assert
    assert isinstance(result, RunFailure)
    terminal = observer.progress_events[-1]
    assert (
        exit_code,
        result.reason_code,
        terminal.kind,
        terminal.reason_code,
        terminal.stage,
    ) == (
        1,
        "internal_error",
        "run_failed",
        "internal_error",
        ProcessingStage.SELECT_IMAGES,
    )


@pytest.mark.parametrize(
    ("error", "reason_code", "remediation_code", "observed_values"),
    [
        (
            ModelRuntimeError(
                ModelRuntimeFailureReason.MODEL_NOT_AVAILABLE,
                ModelRole.CANDIDATE_ANNOTATION,
            ),
            "model_not_available",
            "check_model_runtime",
            (("model_role", "candidate_annotation"),),
        ),
        (
            VisionRuntimeError(
                VisionRuntimeFailureReason.SCHEMA_INVALID,
                validation_code="required_field_missing",
                attempt_count=2,
            ),
            "schema_invalid",
            "check_vision_runtime",
            (("attempt_count", 2), ("validation_code", "required_field_missing")),
        ),
        (
            ContextStageError(ContextStageFailureReason.TIMESTAMP_DRIFT),
            "timestamp_drift",
            "check_context_source",
            (),
        ),
    ],
)
def test_internal_run_controller_preserves_typed_runtime_reason(
    error: Exception,
    reason_code: str,
    remediation_code: str,
    observed_values: tuple[tuple[str, str | int], ...],
) -> None:
    """各typed runtime errorのstable reasonと安全な観測値が保持されること。

    Arrange:
        - Model、Vision、Contextいずれかのtyped errorが用意される
    Act:
        - internal run controllerから失敗するoperationが実行される
    Assert:
        - runtime境界固有のreason、remediation、安全な観測値が返されること
    """
    # Arrange
    observer = RecordingRunObserver()
    tracker = RunProgressTracker(observer)
    controller = InternalRunController(tracker)

    def fail_runtime() -> object:
        raise error

    # Act
    exit_code, result = controller.execute(fail_runtime)

    # Assert
    assert isinstance(result, RunFailure)
    assert (
        exit_code,
        result.reason_code,
        result.remediation_code,
        result.observed_values,
        result.cause,
    ) == (
        1,
        reason_code,
        remediation_code,
        observed_values,
        error,
    )


@pytest.mark.parametrize(
    ("failure", "expected_exit_code"),
    [
        (RuntimeError("/private/raw operation failure"), 1),
        (KeyboardInterrupt(), 130),
    ],
)
def test_rerun_reuses_only_completed_stages_after_terminal_failure(
    tmp_path: Path,
    failure: BaseException,
    expected_exit_code: int,
) -> None:
    """運用errorまたはCtrl+C後にCompleted Stageだけが再利用されること。

    Arrange:
        - 最初のStage完了後、次Stage途中で終了するrunが用意される
    Act:
        - 同じcacheとsemantic inputで新しいrunが実行される
    Assert:
        - 完了済みStageだけが再利用され、未完了Stageが再計算されること
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    subject_fingerprint = "a" * 64
    stage_order = (
        ProcessingStage.DISCOVER_VIDEO_SET,
        ProcessingStage.RESOLVE_MODELS,
    )
    recomputed_stages: list[ProcessingStage] = []
    first_observer = RecordingRunObserver()
    first_progress = RunProgressTracker(first_observer)
    first_controller = InternalRunController(first_progress)

    def fail_during_second_stage() -> str:
        runner = ProcessingStageRunner(
            cache_folder,
            first_observer,
            subject_namespace="video-sets",
            subject_fingerprint=subject_fingerprint,
            stage_order=stage_order,
            progress=first_progress,
        )
        assert (
            runner.reuse(
                ProcessingStage.DISCOVER_VIDEO_SET,
                {"input": "same"},
                lambda artifact: artifact["value"],
            )
            is None
        )
        recomputed_stages.append(ProcessingStage.DISCOVER_VIDEO_SET)
        runner.complete(
            ProcessingStage.DISCOVER_VIDEO_SET,
            {"input": "same"},
            {"value": "completed"},
        )
        assert (
            runner.reuse(
                ProcessingStage.RESOLVE_MODELS,
                {"model": "same"},
                lambda artifact: artifact["value"],
            )
            is None
        )
        raise failure

    # Act
    first_exit_code, first_result = first_controller.execute(fail_during_second_stage)
    second_observer = RecordingRunObserver()
    second_progress = RunProgressTracker(second_observer)
    second_controller = InternalRunController(second_progress)

    def resume() -> str:
        runner = ProcessingStageRunner(
            cache_folder,
            second_observer,
            subject_namespace="video-sets",
            subject_fingerprint=subject_fingerprint,
            stage_order=stage_order,
            progress=second_progress,
        )
        restored = runner.reuse(
            ProcessingStage.DISCOVER_VIDEO_SET,
            {"input": "same"},
            lambda artifact: artifact["value"],
        )
        if restored is None:
            recomputed_stages.append(ProcessingStage.DISCOVER_VIDEO_SET)
        assert (
            runner.reuse(
                ProcessingStage.RESOLVE_MODELS,
                {"model": "same"},
                lambda artifact: artifact["value"],
            )
            is None
        )
        recomputed_stages.append(ProcessingStage.RESOLVE_MODELS)
        runner.complete(
            ProcessingStage.RESOLVE_MODELS,
            {"model": "same"},
            {"value": "resolved"},
        )
        assert restored == "completed"
        return "resumed"

    second_exit_code, second_result = second_controller.execute(resume)

    # Assert
    assert isinstance(first_result, RunFailure)
    assert (
        first_exit_code,
        second_exit_code,
        second_result,
        recomputed_stages,
        tuple(
            event.reason_code
            for event in second_observer.progress_events
            if event.kind == "cache"
        ),
        second_observer.progress_events[-1].kind,
    ) == (
        expected_exit_code,
        0,
        "resumed",
        [
            ProcessingStage.DISCOVER_VIDEO_SET,
            ProcessingStage.RESOLVE_MODELS,
        ],
        ("cache_reused", "cache_miss", "stage_recomputed"),
        "run_completed",
    )
