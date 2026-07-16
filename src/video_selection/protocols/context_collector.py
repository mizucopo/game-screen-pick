"""walking skeleton用Context Cue collectorのport。"""

from typing import Protocol

from ..models.context_cue import ContextCue
from ..models.video_set import VideoSet


class ContextCollector(Protocol):
    """Video Set単位の旧walking-skeleton収集境界。"""

    def collect_context(self, video_set: VideoSet) -> tuple[ContextCue, ...]:
        """Video Setに対応するContext Cueを返す。"""
