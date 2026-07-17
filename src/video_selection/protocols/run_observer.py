"""RunObserverのsemantic port。"""

from typing import Protocol

from ..models.completed_stage import CompletedStage
from ..models.legacy_cache_cleanup_diagnostic import LegacyCacheCleanupDiagnostic
from ..models.progress_event import ProgressEvent


class RunObserver(Protocol):
    """runのstructured observationを外部へ通知する境界。"""

    def observe(self, event: ProgressEvent) -> None:
        """renderer非依存のProgress Eventを通知する。"""

    def stage_completed(self, completed_stage: CompletedStage) -> None:
        """atomicに完了したStageを通知する。"""

    def legacy_cache_cleaned(
        self,
        diagnostic: LegacyCacheCleanupDiagnostic,
    ) -> None:
        """認識済みLegacy Cache cleanupの結果を通知する。"""
