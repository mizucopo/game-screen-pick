"""walking skeleton用Context Cue collectorのport。"""

from typing import Protocol

from ..models.collected_context import CollectedContext
from ..models.resolved_model import ResolvedModel
from ..models.video_set import VideoSet


class ContextCollector(Protocol):
    """Video Set単位の旧walking-skeleton収集境界。"""

    def collect_context(
        self,
        video_set: VideoSet,
        model: ResolvedModel,
    ) -> CollectedContext:
        """CueとSTT実行時runtime identityを返す。"""
