from pathlib import Path

from src.video_selection.models.completed_stage import CompletedStage
from src.video_selection.models.processing_stage import ProcessingStage


class BlockingPublicationObserver:
    """Publish Output Stageのcache pathをfileで塞ぐobserver。"""

    def __init__(self, cache_folder: Path) -> None:
        self._cache_folder = cache_folder

    def stage_completed(self, completed_stage: CompletedStage) -> None:
        """Select Images完了後にPublish Output用folderの作成を妨げる。"""
        if completed_stage.stage is not ProcessingStage.SELECT_IMAGES:
            return
        stage_root = self._cache_folder / "walking-skeleton"
        stage_root.mkdir(parents=True, exist_ok=True)
        (stage_root / ProcessingStage.PUBLISH_OUTPUT.value).write_text(
            "blocked",
            encoding="utf-8",
        )
