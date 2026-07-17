from src.video_selection.models.completed_stage import CompletedStage
from src.video_selection.models.legacy_cache_cleanup_diagnostic import (
    LegacyCacheCleanupDiagnostic,
)
from src.video_selection.models.progress_event import ProgressEvent


class RecordingRunObserver:
    """完了したProcessing Stageを記録するobserver。"""

    def __init__(self) -> None:
        self.progress_events: list[ProgressEvent] = []
        self.completed_stages: list[CompletedStage] = []
        self.legacy_cache_diagnostics: list[LegacyCacheCleanupDiagnostic] = []

    def observe(self, event: ProgressEvent) -> None:
        """Progress Eventを順番に記録する。"""
        self.progress_events.append(event)

    def stage_completed(self, completed_stage: CompletedStage) -> None:
        """完了したStageを順番に記録する。"""
        self.completed_stages.append(completed_stage)

    def legacy_cache_cleaned(
        self,
        diagnostic: LegacyCacheCleanupDiagnostic,
    ) -> None:
        """Legacy Cache cleanup diagnosticを記録する。"""
        self.legacy_cache_diagnostics.append(diagnostic)
