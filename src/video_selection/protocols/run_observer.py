"""RunObserverのsemantic port。"""

from typing import Protocol

from ..models.completed_stage import CompletedStage
from ..models.legacy_cache_cleanup_diagnostic import LegacyCacheCleanupDiagnostic


class RunObserver(Protocol):
    """Processing Stageの完了を外部へ通知する境界。"""

    def stage_completed(self, completed_stage: CompletedStage) -> None:
        """atomicに完了したStageを通知する。"""

    def legacy_cache_cleaned(
        self,
        diagnostic: LegacyCacheCleanupDiagnostic,
    ) -> None:
        """認識済みLegacy Cache cleanupの結果を通知する。"""
