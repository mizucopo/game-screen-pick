"""Progress EventのTTY更新renderer。"""

from ..models.progress_event import ProgressEvent
from .line_progress_renderer import LineProgressRenderer

_TERMINAL_KINDS = {"run_completed", "run_failed", "run_interrupted"}


class TtyProgressRenderer:
    """Progress Eventを同じTTY行へ描画する。"""

    def __init__(self) -> None:
        self._line_renderer = LineProgressRenderer()

    def render(self, event: ProgressEvent) -> str:
        """行頭へ戻して描画し、run terminal eventだけ改行する。"""
        suffix = "\x1b[K\n" if event.kind in _TERMINAL_KINDS else "\x1b[K"
        return f"\r{self._line_renderer.render(event)}{suffix}"
