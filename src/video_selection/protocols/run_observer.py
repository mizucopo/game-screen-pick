"""RunObserverのsemantic port。"""

from typing import Protocol

from ..models.completed_stage import CompletedStage


class RunObserver(Protocol):
    """Processing Stageの完了を外部へ通知する境界。"""

    def stage_completed(self, completed_stage: CompletedStage) -> None:
        """atomicに完了したStageを通知する。"""
