"""Progress Eventをstderr互換streamへ描画するobserver。"""

import sys
from threading import Lock
from typing import TextIO

from ..models.completed_stage import CompletedStage
from ..models.legacy_cache_cleanup_diagnostic import LegacyCacheCleanupDiagnostic
from ..models.progress_event import ProgressEvent
from .line_progress_renderer import LineProgressRenderer
from .tty_progress_renderer import TtyProgressRenderer


class ProgressStreamObserver:
    """streamのTTY状態に対応するrendererを自動選択する。"""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._is_tty = self._stream.isatty()
        self._renderer = (
            TtyProgressRenderer() if self._is_tty else LineProgressRenderer()
        )
        self._lock = Lock()

    def observe(self, event: ProgressEvent) -> None:
        """一つのProgress Eventをatomicにstreamへ書く。"""
        rendered = self._renderer.render(event)
        if not self._is_tty:
            rendered = f"{rendered}\n"
        with self._lock:
            self._stream.write(rendered)
            self._stream.flush()

    def stage_completed(self, _completed_stage: CompletedStage) -> None:
        """legacy Stage callbackはProgress Eventへ統合するため描画しない。"""

    def legacy_cache_cleaned(
        self,
        _diagnostic: LegacyCacheCleanupDiagnostic,
    ) -> None:
        """legacy cleanup callbackはProgress Eventへ統合するため描画しない。"""
