"""SpeechRuntimeのsemantic port。"""

from typing import Protocol

from ..models.context_cue import ContextCue
from ..models.video_set import VideoSet


class SpeechRuntime(Protocol):
    """Video SetからContext Cueを取得する境界。"""

    def collect_context(
        self,
        video_set: VideoSet,
    ) -> tuple[ContextCue, ...]:
        """Video SetのContext Cueを返す。"""
