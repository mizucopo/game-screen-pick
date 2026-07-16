"""faster-whisper model objectの最小external port。"""

from collections.abc import Iterable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class FasterWhisperModel(Protocol):
    """waveformをfaster-whisper resultへ変換する外部境界。"""

    def transcribe(
        self,
        audio: NDArray[np.float32],
        **options: object,
    ) -> tuple[Iterable[object], object]:
        """lazy segment列とtranscription infoを返す。"""
