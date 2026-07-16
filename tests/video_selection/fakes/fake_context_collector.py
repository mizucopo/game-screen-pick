from src.video_selection.models.context_cue import ContextCue
from src.video_selection.models.video_set import VideoSet


class FakeContextCollector:
    """固定されたContext Cueを返すwalking-skeleton用fake。"""

    def __init__(self, context_cues: tuple[ContextCue, ...]) -> None:
        self._context_cues = context_cues

    def collect_context(
        self,
        video_set: VideoSet,
    ) -> tuple[ContextCue, ...]:
        """Video Setに対応するContext Cueを返す。"""
        del video_set
        return self._context_cues
