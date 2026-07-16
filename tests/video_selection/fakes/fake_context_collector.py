from src.video_selection.models.collected_context import CollectedContext
from src.video_selection.models.context_cue import ContextCue
from src.video_selection.models.resolved_model import ResolvedModel
from src.video_selection.models.video_set import VideoSet


class FakeContextCollector:
    """固定されたContext Cueを返すwalking-skeleton用fake。"""

    def __init__(
        self,
        context_cues: tuple[ContextCue, ...],
        *,
        speech_runtime_identity: str | None = None,
    ) -> None:
        self._context_cues = context_cues
        self._speech_runtime_identity = speech_runtime_identity
        self.models: list[ResolvedModel] = []

    def collect_context(
        self,
        video_set: VideoSet,
        model: ResolvedModel,
    ) -> CollectedContext:
        """Video Setに対応するContext Cueを返す。"""
        del video_set
        self.models.append(model)
        return CollectedContext(
            cues=self._context_cues,
            speech_runtime_identity=self._speech_runtime_identity,
        )
