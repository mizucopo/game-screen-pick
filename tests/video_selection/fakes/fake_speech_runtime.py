"""Context Collection Stage用SpeechRuntime fake。"""

from src.video_selection.models.pcm_audio_chunk import PcmAudioChunk
from src.video_selection.models.speech_recognition_result import (
    SpeechRecognitionResult,
)


class FakeSpeechRuntime:
    """固定されたword timestamp結果を返すexternal-boundary fake。"""

    def __init__(
        self,
        results: tuple[SpeechRecognitionResult, ...] = (),
        *,
        runtime_identity: str = "fake-speech-runtime-v1",
        resolved_model_identity: str = "hf:" + "0" * 40,
        error_on_call: int | None = None,
        error_message: str = "speech recognition failed",
    ) -> None:
        self._results = results
        self._runtime_identity = runtime_identity
        self._resolved_model_identity = resolved_model_identity
        self._error_on_call = error_on_call
        self._error_message = error_message
        self.transcribe_calls: list[PcmAudioChunk] = []
        self.transcribe_options: list[tuple[str, bool, int]] = []

    @property
    def runtime_identity(self) -> str:
        """固定runtime identityを返す。"""
        return self._runtime_identity

    @property
    def resolved_model_identity(self) -> str:
        """固定model identityを返す。"""
        return self._resolved_model_identity

    def transcribe(
        self,
        chunk: PcmAudioChunk,
        *,
        language: str,
        vad_filter: bool,
        beam_size: int,
    ) -> SpeechRecognitionResult:
        """呼び出しを記録して対応する固定結果を返す。"""
        call_index = len(self.transcribe_calls)
        self.transcribe_calls.append(chunk)
        self.transcribe_options.append((language, vad_filter, beam_size))
        if call_index == self._error_on_call:
            raise RuntimeError(self._error_message)
        if call_index < len(self._results):
            return self._results[call_index]
        return SpeechRecognitionResult(vad_speech_detected=False, segments=())
