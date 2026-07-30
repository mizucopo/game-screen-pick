"""runとProcessing StageのProgress Event lifecycle。"""

import time
from collections.abc import Callable

from ..models.processing_stage import ProcessingStage
from ..models.progress_event import (
    EstimationState,
    ProgressEvent,
    ProgressEventKind,
    ProgressSeverity,
)
from ..models.stage_fingerprint import StageFingerprint
from ..protocols.run_observer import RunObserver
from .stage_eta_estimator import StageEtaEstimator, WorkDisposition

MonotonicClock = Callable[[], float]


class RunProgressTracker:
    """一つのrunで直列なProcessing Stage eventを発行する。"""

    def __init__(
        self,
        observer: RunObserver,
        *,
        clock: MonotonicClock = time.monotonic,
    ) -> None:
        self._observer = observer
        self._clock = clock
        self._state = "idle"
        self._stage_index = 0
        self._active_stage: ProcessingStage | None = None
        self._stage_started_at: float | None = None
        self._stage_count: int | None = None
        self._video_order: int | None = None
        self._video_count: int | None = None
        self._video_relative_path: str | None = None
        self._work_unit_kind: str | None = None
        self._stage_cache_hit_count = 0
        self._stage_cache_miss_count = 0
        self._stage_reuse_count = 0
        self._stage_recompute_count = 0
        self._completed_stage_events: list[ProgressEvent] = []
        self._eta_estimator = StageEtaEstimator()

    @property
    def completed_stage_events(self) -> tuple[ProgressEvent, ...]:
        """fingerprint付きで確定したStageのrun別観測値を返す。"""
        return tuple(self._completed_stage_events)

    def start_run(self) -> None:
        """runを開始して最初のeventを発行する。"""
        if self._state != "idle":
            msg = "runは一度だけ開始できます"
            raise RuntimeError(msg)
        self._state = "running"
        self._observer.observe(
            ProgressEvent(
                kind="run_started",
                severity="info",
                reason_code="run_started",
            )
        )

    def start_stage(
        self,
        stage: ProcessingStage,
        *,
        stage_count: int | None = None,
        video_order: int | None = None,
        video_count: int | None = None,
        video_relative_path: str | None = None,
        work_unit_kind: str | None = None,
    ) -> None:
        """次のatomicなProcessing Stageを開始する。"""
        self._require_running_without_stage()
        self._stage_index += 1
        self._active_stage = stage
        self._stage_started_at = self._clock()
        self._stage_count = stage_count
        self._video_order = video_order
        self._video_count = video_count
        self._video_relative_path = video_relative_path
        self._work_unit_kind = work_unit_kind
        self._observer.observe(
            self._stage_event(
                kind="stage_started",
                reason_code="stage_started",
                elapsed_seconds=0.0,
            )
        )

    def progress(
        self,
        *,
        processed_count: int | None = None,
        total_count: int | None = None,
        remaining_reuse_count: int | None = None,
        remaining_recompute_count: int | None = None,
    ) -> None:
        """active Stageの観測可能な件数を通知する。"""
        elapsed_seconds = self._stage_elapsed()
        estimation_state: EstimationState = "unavailable"
        eta_seconds: float | None = None
        if self._work_unit_kind is not None:
            stage, work_unit_kind = self._active_work_series()
            estimation_state, eta_seconds = self._eta_estimator.estimate(
                stage,
                work_unit_kind,
                remaining_reuse_count=remaining_reuse_count,
                remaining_recompute_count=remaining_recompute_count,
                stage_elapsed_seconds=elapsed_seconds,
            )
        self._observer.observe(
            self._stage_event(
                kind="progress",
                reason_code="stage_progress",
                elapsed_seconds=elapsed_seconds,
                processed_count=processed_count,
                total_count=total_count,
                estimation_state=estimation_state,
                eta_seconds=eta_seconds,
            )
        )

    def record_work_sample(
        self,
        disposition: WorkDisposition,
        duration_seconds: float | None = None,
    ) -> None:
        """active Comparable Work Seriesへ完了sampleを記録する。"""
        stage, work_unit_kind = self._active_work_series()
        observed_duration = (
            self._stage_elapsed() if duration_seconds is None else duration_seconds
        )
        if observed_duration == 0:
            return
        self._eta_estimator.record_sample(
            stage,
            work_unit_kind,
            disposition,
            observed_duration,
        )

    def external_work_started(self, reason_code: str) -> None:
        """active Stage内のblocking external work開始を通知する。"""
        self._observer.observe(
            self._stage_event(
                kind="external_work_started",
                reason_code=reason_code,
                elapsed_seconds=self._stage_elapsed(),
            )
        )

    def heartbeat(self) -> None:
        """active Stageが継続中であることを経過時間だけで通知する。"""
        self._observer.observe(
            self._stage_event(
                kind="heartbeat",
                reason_code="external_work_heartbeat",
                elapsed_seconds=self._stage_elapsed(),
            )
        )

    def cache_observed(
        self,
        *,
        cache_hit_count: int,
        cache_miss_count: int,
        reuse_count: int,
        recompute_count: int,
        reason_code: str,
        stage_fingerprint: StageFingerprint | None = None,
    ) -> None:
        """active Stageのcache lookupと実処理結果を通知する。"""
        event = self._stage_event(
            kind="cache",
            reason_code=reason_code,
            elapsed_seconds=self._stage_elapsed(),
            cache_hit_count=cache_hit_count,
            cache_miss_count=cache_miss_count,
            reuse_count=reuse_count,
            recompute_count=recompute_count,
            stage_fingerprint=(
                None if stage_fingerprint is None else stage_fingerprint.value
            ),
        )
        self._stage_cache_hit_count += cache_hit_count
        self._stage_cache_miss_count += cache_miss_count
        self._stage_reuse_count += reuse_count
        self._stage_recompute_count += recompute_count
        self._observer.observe(event)

    def complete_stage(
        self,
        duration_seconds: float | None = None,
        *,
        stage_fingerprint: StageFingerprint | None = None,
    ) -> None:
        """active Stageを完了して次のStage開始を許可する。"""
        event = self._stage_event(
            kind="stage_completed",
            reason_code="stage_completed",
            elapsed_seconds=(
                self._stage_elapsed() if duration_seconds is None else duration_seconds
            ),
            cache_hit_count=self._stage_cache_hit_count,
            cache_miss_count=self._stage_cache_miss_count,
            reuse_count=self._stage_reuse_count,
            recompute_count=self._stage_recompute_count,
            stage_fingerprint=(
                None if stage_fingerprint is None else stage_fingerprint.value
            ),
        )
        if stage_fingerprint is not None:
            self._completed_stage_events.append(event)
        self._observer.observe(event)
        self._clear_active_stage()

    def complete_run(self) -> None:
        """active Stageのないrunを正常完了する。"""
        self._require_running_without_stage()
        self._state = "terminal"
        self._observer.observe(
            ProgressEvent(
                kind="run_completed",
                severity="info",
                reason_code="run_completed",
            )
        )

    def fail_run(self, reason_code: str) -> None:
        """runをoperation failureとして終了する。"""
        self._terminate_run("run_failed", "error", reason_code)

    def interrupt_run(self) -> None:
        """runを利用者による中断として終了する。"""
        self._terminate_run("run_interrupted", "warning", "user_interrupt")

    def _stage_event(
        self,
        *,
        kind: ProgressEventKind,
        severity: ProgressSeverity = "info",
        reason_code: str,
        elapsed_seconds: float,
        processed_count: int | None = None,
        total_count: int | None = None,
        cache_hit_count: int = 0,
        cache_miss_count: int = 0,
        reuse_count: int = 0,
        recompute_count: int = 0,
        stage_fingerprint: str | None = None,
        estimation_state: EstimationState = "unavailable",
        eta_seconds: float | None = None,
    ) -> ProgressEvent:
        if self._state != "running" or self._active_stage is None:
            msg = "activeなProcessing Stageがありません"
            raise RuntimeError(msg)
        return ProgressEvent(
            kind=kind,
            severity=severity,
            stage=self._active_stage,
            stage_fingerprint=stage_fingerprint,
            stage_index=self._stage_index,
            stage_count=self._stage_count,
            video_order=self._video_order,
            video_count=self._video_count,
            video_relative_path=self._video_relative_path,
            processed_count=processed_count,
            total_count=total_count,
            cache_hit_count=cache_hit_count,
            cache_miss_count=cache_miss_count,
            reuse_count=reuse_count,
            recompute_count=recompute_count,
            elapsed_seconds=elapsed_seconds,
            eta_seconds=eta_seconds,
            estimation_state=estimation_state,
            work_unit_kind=self._work_unit_kind,
            reason_code=reason_code,
        )

    def _terminate_run(
        self,
        kind: ProgressEventKind,
        severity: ProgressSeverity,
        reason_code: str,
    ) -> None:
        if self._state != "running":
            msg = "runが開始されていません"
            raise RuntimeError(msg)
        if self._active_stage is None:
            event = ProgressEvent(
                kind=kind,
                severity=severity,
                reason_code=reason_code,
            )
        else:
            event = self._stage_event(
                kind=kind,
                severity=severity,
                reason_code=reason_code,
                elapsed_seconds=self._stage_elapsed(),
            )
            self._clear_active_stage()
        self._state = "terminal"
        self._observer.observe(event)

    def _stage_elapsed(self) -> float:
        if self._stage_started_at is None:
            msg = "activeなProcessing Stageがありません"
            raise RuntimeError(msg)
        return self._clock() - self._stage_started_at

    def _active_work_series(self) -> tuple[ProcessingStage, str]:
        if self._active_stage is None or self._work_unit_kind is None:
            msg = "activeなComparable Work Seriesがありません"
            raise RuntimeError(msg)
        return self._active_stage, self._work_unit_kind

    def _require_running_without_stage(self) -> None:
        if self._state != "running":
            msg = "runが開始されていません"
            raise RuntimeError(msg)
        if self._active_stage is not None:
            msg = "Processing Stageは同時に一つだけ開始できます"
            raise RuntimeError(msg)

    def _clear_active_stage(self) -> None:
        self._active_stage = None
        self._stage_started_at = None
        self._stage_count = None
        self._video_order = None
        self._video_count = None
        self._video_relative_path = None
        self._work_unit_kind = None
        self._stage_cache_hit_count = 0
        self._stage_cache_miss_count = 0
        self._stage_reuse_count = 0
        self._stage_recompute_count = 0
