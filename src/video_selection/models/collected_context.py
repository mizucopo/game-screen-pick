"""Video Set単位で収集されたContext CueとSTT実行診断。"""

import re
from dataclasses import dataclass

from .context_cue import ContextCue

_SPEECH_RUNTIME_IDENTITY_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z._:+-]{0,255}")


@dataclass(frozen=True)
class CollectedContext:
    """Context Cueと、STT実行時だけ存在するruntime identityの組。"""

    cues: tuple[ContextCue, ...]
    speech_runtime_identity: str | None = None

    def __post_init__(self) -> None:
        """privacy-safeなSpeech Runtime Identityだけを受け入れる。"""
        identity = self.speech_runtime_identity
        if (
            identity is not None
            and _SPEECH_RUNTIME_IDENTITY_PATTERN.fullmatch(identity) is None
        ):
            msg = "安全なSpeech Runtime Identityが必要です"
            raise ValueError(msg)
