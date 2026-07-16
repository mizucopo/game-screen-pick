"""faster-whisper external modelのtest fake。"""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


class FakeFasterWhisperModel:
    """固定segmentを返しwaveformとoptionを記録するboundary fake。"""

    def __init__(self, segments: tuple[object, ...], info: object) -> None:
        self._segments = segments
        self._info = info
        self.audio: NDArray[np.float32] | None = None
        self.options: dict[str, object] | None = None

    def transcribe(
        self,
        audio: NDArray[np.float32],
        **options: object,
    ) -> tuple[Iterable[object], object]:
        """waveformとoptionを記録して固定結果を返す。"""
        self.audio = audio.copy()
        self.options = options
        return iter(self._segments), self._info
