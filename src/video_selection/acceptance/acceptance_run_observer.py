"""一つのacceptance phaseのProgress Event収集observer。"""

from threading import Lock

from ..models.completed_stage import CompletedStage
from ..models.legacy_cache_cleanup_diagnostic import LegacyCacheCleanupDiagnostic
from ..models.processing_stage import ProcessingStage
from ..models.progress_event import ProgressEvent
from ..protocols.run_observer import RunObserver


class AcceptanceRunObserver:
    """rendererへforwardしつつrecord用のsafe event/countを収集する。"""

    def __init__(self, downstream: RunObserver | None = None) -> None:
        self._downstream = downstream
        self._lock = Lock()
        self._current_stage: ProcessingStage | None = None
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
        with self._lock:
            self.progress_events.append(event)
            if event.kind == "stage_started":
                self._current_stage = event.stage
            elif event.kind in {
                "stage_completed",
                "run_completed",
                "run_failed",
                "run_interrupted",
            }:
                self._current_stage = None
        if self._downstream is not None:
            self._downstream.observe(event)

    def stage_completed(self, completed_stage: CompletedStage) -> None:
        """Completed Stageをrecord用に保持してdownstreamへ渡す。"""
        self.completed_stages.append(completed_stage)
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

    def phase_metrics(self) -> dict[str, object]:
        """pathやraw内容を含まないStage/cache aggregateを返す。"""
        cache_events = tuple(
            event for event in self.progress_events if event.kind == "cache"
        )
        cache_hits = sum(event.cache_hit_count for event in cache_events)
        cache_misses = sum(event.cache_miss_count for event in cache_events)
        reuse_count = sum(event.reuse_count for event in cache_events)
        recompute_count = sum(event.recompute_count for event in cache_events)
        durations: dict[str, float] = {}
        for event in self.progress_events:
            if (
                event.kind
                not in {
                    "stage_completed",
                    "run_failed",
                    "run_interrupted",
                }
                or event.stage is None
            ):
                continue
            name = event.stage.value
            if event.elapsed_seconds is None:
                raise ValueError("Completed Stage eventにelapsedがありません")
            durations[name] = durations.get(name, 0.0) + event.elapsed_seconds
        stage_counts: dict[str, int] = {}
        for stage in self.completed_stages:
            name = stage.stage.value
            stage_counts[name] = stage_counts.get(name, 0) + 1
        return {
            "cache_hit_count": cache_hits,
            "cache_miss_count": cache_misses,
            "reuse_count": reuse_count,
            "unexpected_recompute_count": recompute_count,
            "stage_durations_seconds": durations,
            "completed_stage_counts": stage_counts,
        }
