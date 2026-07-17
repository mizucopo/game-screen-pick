"""内部application runのterminal result境界。"""

from collections.abc import Callable
from typing import Literal, TypeVar

from ..configuration.configuration_error import ConfigurationError
from ..models.context_stage_error import ContextStageError
from ..models.media_runtime_error import MediaRuntimeError
from ..models.model_runtime_error import ModelRuntimeError
from ..models.run_failure import (
    ResumeGuidance,
    RunFailure,
    RunFailureExitCode,
    SafeObservedValue,
)
from ..models.vision_runtime_error import VisionRuntimeError
from ..services.run_progress_tracker import RunProgressTracker

RunExitCode = Literal[0, 1, 2, 130]
RunValue = TypeVar("RunValue")


class InternalRunController:
    """operation errorを安全なRun Failureとexit codeへ変換する。"""

    def __init__(self, progress: RunProgressTracker) -> None:
        self._progress = progress

    def execute(
        self,
        operation: Callable[[], RunValue],
    ) -> tuple[RunExitCode, RunValue | RunFailure]:
        """内部operationを一つのrun lifecycleとして実行する。"""
        self._progress.start_run()
        try:
            outcome = operation()
            self._progress.complete_run()
        except KeyboardInterrupt as error:
            failure = _safe_run_failure(
                reason_code="user_interrupt",
                exit_code=130,
                remediation_code="rerun_command",
                resume_guidance="completed_stages_reusable",
                cause=error,
            )
            self._progress.interrupt_run()
            return (failure.exit_code, failure)
        except ConfigurationError as error:
            failure = _safe_run_failure(
                reason_code=error.reason_code.lower(),
                exit_code=2,
                remediation_code="fix_configuration",
                resume_guidance="run_not_started",
                cause=error,
            )
            self._progress.fail_run(failure.reason_code)
            return (failure.exit_code, failure)
        except MediaRuntimeError as error:
            failure = _safe_run_failure(
                reason_code=error.reason.value,
                exit_code=1,
                remediation_code="check_media_runtime",
                resume_guidance="completed_stages_reusable",
                cause=error,
            )
            self._progress.fail_run(failure.reason_code)
            return (failure.exit_code, failure)
        except ModelRuntimeError as error:
            failure = _safe_run_failure(
                reason_code=error.reason.value,
                exit_code=1,
                remediation_code="check_model_runtime",
                resume_guidance="completed_stages_reusable",
                observed_values=(("model_role", error.role.value),),
                cause=error,
            )
            self._progress.fail_run(failure.reason_code)
            return (failure.exit_code, failure)
        except VisionRuntimeError as error:
            observed_values: tuple[tuple[str, str | int], ...] = (
                (("attempt_count", error.attempt_count),)
                if error.validation_code is None
                else (
                    ("attempt_count", error.attempt_count),
                    ("validation_code", error.validation_code),
                )
            )
            failure = _safe_run_failure(
                reason_code=error.reason.value,
                exit_code=1,
                remediation_code="check_vision_runtime",
                resume_guidance="completed_stages_reusable",
                observed_values=observed_values,
                cause=error,
            )
            self._progress.fail_run(failure.reason_code)
            return (failure.exit_code, failure)
        except ContextStageError as error:
            failure = _safe_run_failure(
                reason_code=error.reason.value,
                exit_code=1,
                remediation_code="check_context_source",
                resume_guidance="completed_stages_reusable",
                cause=error,
            )
            self._progress.fail_run(failure.reason_code)
            return (failure.exit_code, failure)
        except Exception as error:
            failure = _safe_run_failure(
                reason_code="internal_error",
                exit_code=1,
                remediation_code="report_internal_error",
                resume_guidance="completed_stages_reusable",
                cause=error,
            )
            self._progress.fail_run(failure.reason_code)
            return (failure.exit_code, failure)
        return (0, outcome)


def _safe_run_failure(
    *,
    reason_code: str,
    exit_code: RunFailureExitCode,
    remediation_code: str,
    resume_guidance: ResumeGuidance,
    cause: BaseException,
    observed_values: tuple[tuple[str, SafeObservedValue], ...] = (),
) -> RunFailure:
    """unsafeなtyped metadataも最外周から漏らさずRun Failureへ変換する。"""
    try:
        return RunFailure(
            reason_code=reason_code,
            exit_code=exit_code,
            remediation_code=remediation_code,
            resume_guidance=resume_guidance,
            observed_values=observed_values,
            cause=cause,
        )
    except (TypeError, ValueError):
        return RunFailure(
            reason_code="internal_error",
            exit_code=1,
            remediation_code="report_internal_error",
            resume_guidance="completed_stages_reusable",
            cause=cause,
        )
