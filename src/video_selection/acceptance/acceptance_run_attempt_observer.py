"""一つのAcceptance Run AttemptのProgress Event収集observer。"""

from collections.abc import Callable
from threading import Lock

from ..models.completed_stage import CompletedStage
from ..models.legacy_cache_cleanup_diagnostic import LegacyCacheCleanupDiagnostic
from ..models.processing_stage import ProcessingStage
from ..models.progress_event import ProgressEvent
from ..protocols.run_observer import RunObserver

AttemptSnapshotWriter = Callable[
    [dict[str, object], dict[str, str]],
    None,
]


class AcceptanceRunAttemptObserver:
    """rendererへforwardしつつrecord用のsafe event/countを収集する。"""

    def __init__(
        self,
        downstream: RunObserver | None = None,
        *,
        snapshot_writer: AttemptSnapshotWriter | None = None,
    ) -> None:
        self._downstream = downstream
        self._snapshot_writer = snapshot_writer
        self._lock = Lock()
        self._current_stage: ProcessingStage | None = None
        self._cache_hit_count = 0
        self._cache_miss_count = 0
        self._reuse_count = 0
        self._recompute_count = 0
        self._stage_durations: dict[str, float] = {}
        self._completed_stage_counts: dict[str, int] = {}
        self._work_unit_resolutions: dict[str, str] = {}
        self.progress_events: list[ProgressEvent] = []
        self.completed_stages: list[CompletedStage] = []
        self.legacy_cache_diagnostics: list[LegacyCacheCleanupDiagnostic] = []

    @property
    def current_stage(self) -> ProcessingStage | None:
        """resource samplerが参照するactive Stageを返す。"""
        with self._lock:
            return self._current_stage

    def observe(self, event: ProgressEvent) -> None:
        """eventを記録しactive Stageを更新してrendererへ渡す。"""
        snapshot: tuple[dict[str, object], dict[str, str]] | None = None
        with self._lock:
            self.progress_events.append(event)
            if event.kind == "cache":
                self._cache_hit_count += event.cache_hit_count
                self._cache_miss_count += event.cache_miss_count
                self._reuse_count += event.reuse_count
                self._recompute_count += event.recompute_count
                checkpoint_fingerprint = (
                    event.work_unit_fingerprint or event.stage_fingerprint
                )
                if checkpoint_fingerprint is not None:
                    status = (
                        "recomputed"
                        if event.recompute_count > 0
                        else "reused"
                        if event.reuse_count > 0
                        else "miss_started"
                        if event.cache_miss_count > 0
                        else None
                    )
                    if status is not None:
                        self._work_unit_resolutions[checkpoint_fingerprint] = (
                            _stronger_resolution(
                                self._work_unit_resolutions.get(checkpoint_fingerprint),
                                status,
                            )
                        )
            if event.kind == "stage_started":
                self._current_stage = event.stage
            elif event.kind in {
                "stage_completed",
                "run_completed",
                "run_failed",
                "run_interrupted",
            }:
                self._current_stage = None
            if (
                event.kind
                in {
                    "stage_completed",
                    "run_failed",
                    "run_interrupted",
                }
                and event.stage is not None
            ):
                if event.elapsed_seconds is None:
                    raise ValueError("Completed Stage eventにelapsedがありません")
                name = event.stage.value
                self._stage_durations[name] = (
                    self._stage_durations.get(name, 0.0) + event.elapsed_seconds
                )
            if event.kind in {
                "cache",
                "stage_completed",
                "run_failed",
                "run_interrupted",
            }:
                snapshot = self._snapshot_locked()
        if snapshot is not None:
            self._write_snapshot(*snapshot)
        if self._downstream is not None:
            self._downstream.observe(event)

    def stage_completed(self, completed_stage: CompletedStage) -> None:
        """Completed Stageをrecord用に保持してdownstreamへ渡す。"""
        with self._lock:
            self.completed_stages.append(completed_stage)
            name = completed_stage.stage.value
            self._completed_stage_counts[name] = (
                self._completed_stage_counts.get(name, 0) + 1
            )
            snapshot = self._snapshot_locked()
        self._write_snapshot(*snapshot)
        if self._downstream is not None:
            self._downstream.stage_completed(completed_stage)

    def legacy_cache_cleaned(
        self,
        diagnostic: LegacyCacheCleanupDiagnostic,
    ) -> None:
        """legacy cache診断を保持してdownstreamへ渡す。"""
        self.legacy_cache_diagnostics.append(diagnostic)
        if self._downstream is not None:
            self._downstream.legacy_cache_cleaned(diagnostic)

    def attempt_metrics(self) -> dict[str, object]:
        """pathやraw内容を含まない一試行のStage/cache aggregateを返す。"""
        with self._lock:
            metrics, _resolutions = self._snapshot_locked()
        return metrics

    def _snapshot_locked(
        self,
    ) -> tuple[dict[str, object], dict[str, str]]:
        """lock内のincremental aggregateをimmutable copyとして返す。"""
        return (
            {
                "cache_hit_count": self._cache_hit_count,
                "cache_miss_count": self._cache_miss_count,
                "reuse_count": self._reuse_count,
                "unexpected_recompute_count": self._recompute_count,
                "stage_durations_seconds": dict(self._stage_durations),
                "completed_stage_counts": dict(self._completed_stage_counts),
            },
            dict(self._work_unit_resolutions),
        )

    def _write_snapshot(
        self,
        metrics: dict[str, object],
        resolutions: dict[str, str],
    ) -> None:
        """設定されたdurable writerへobserver lock外でsnapshotを渡す。"""
        if self._snapshot_writer is not None:
            self._snapshot_writer(metrics, resolutions)


def _stronger_resolution(previous: str | None, current: str) -> str:
    """並行eventの順序にかかわらずcheckpoint状態を後退させない。"""
    priority = {"miss_started": 0, "reused": 1, "recomputed": 2}
    if previous is None or priority[current] >= priority[previous]:
        return current
    return previous
