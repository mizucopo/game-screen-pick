from src.video_selection.models.completed_stage import CompletedStage
from src.video_selection.models.legacy_cache_cleanup_diagnostic import (
    LegacyCacheCleanupDiagnostic,
)


class RecordingRunObserver:
    """完了したProcessing Stageを記録するobserver。"""

    def __init__(self) -> None:
        self.completed_stages: list[CompletedStage] = []
        self.legacy_cache_diagnostics: list[LegacyCacheCleanupDiagnostic] = []

    def stage_completed(self, completed_stage: CompletedStage) -> None:
        """完了したStageを順番に記録する。"""
        self.completed_stages.append(completed_stage)

    def legacy_cache_cleaned(
        self,
        diagnostic: LegacyCacheCleanupDiagnostic,
    ) -> None:
        """Legacy Cache cleanup diagnosticを記録する。"""
        self.legacy_cache_diagnostics.append(diagnostic)
