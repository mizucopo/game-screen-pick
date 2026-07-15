from src.video_selection.models.completed_stage import CompletedStage


class RecordingRunObserver:
    """完了したProcessing Stageを記録するobserver。"""

    def __init__(self) -> None:
        self.completed_stages: list[CompletedStage] = []

    def stage_completed(self, completed_stage: CompletedStage) -> None:
        """完了したStageを順番に記録する。"""
        self.completed_stages.append(completed_stage)
