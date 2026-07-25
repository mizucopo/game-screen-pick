"""word timestamp付きSpeechRuntimeのsemantic port。"""

from typing import Protocol

from ..models.pcm_audio_chunk import PcmAudioChunk
from ..models.speech_recognition_result import SpeechRecognitionResult


class SpeechRuntime(Protocol):
    """解決済みmodelでPCM chunkを認識する境界。"""

    @property
    def runtime_identity(self) -> str:
        """STT adapterとbackend runtimeのidentityを返す。"""

    @property
    def resolved_model_identity(self) -> str:
        """run内でfreezeされた完全model identityを返す。"""

    def transcribe(
        self,
        chunk: PcmAudioChunk,
        *,
        language: str,
        vad_filter: bool,
        beam_size: int,
    ) -> SpeechRecognitionResult:
        """chunk内のinteger sample位置付き認識結果を返す。"""

    def close(self) -> None:
        """model資源を解放し、以降の認識を禁止する。"""
