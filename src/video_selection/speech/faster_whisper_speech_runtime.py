"""faster-whisperをSpeechRuntimeへ変換するadapter。"""

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from decimal import ROUND_HALF_EVEN, Decimal
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel

from ..models.pcm_audio_chunk import PcmAudioChunk
from ..models.speech_recognition_result import SpeechRecognitionResult
from ..models.speech_segment import SpeechSegment
from ..models.speech_word import SpeechWord
from ..protocols.faster_whisper_model import FasterWhisperModel
from ..services.gpu_work_coordinator import GpuWorkCoordinator

FasterWhisperModelLoader = Callable[
    [Path, str, str, bool],
    FasterWhisperModel,
]

_ADAPTER_VERSION = "faster-whisper-speech-runtime-v1"
_GPU_RUNTIME_DISTRIBUTIONS = (
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
)


class FasterWhisperSpeechRuntime:
    """解決済みlocal modelでword timestamp付きSTTを実行する。"""

    def __init__(
        self,
        model: FasterWhisperModel,
        *,
        runtime_identity: str,
        resolved_model_identity: str,
        gpu_coordinator: GpuWorkCoordinator | None = None,
    ) -> None:
        self._model = model
        self._runtime_identity = runtime_identity
        self._resolved_model_identity = resolved_model_identity
        self._gpu_coordinator = gpu_coordinator

    @classmethod
    def load_local(
        cls,
        model_artifact: Path,
        *,
        resolved_model_identity: str,
        device: str,
        compute_type: str,
        model_loader: FasterWhisperModelLoader | None = None,
        gpu_coordinator: GpuWorkCoordinator | None = None,
    ) -> "FasterWhisperSpeechRuntime":
        """ModelRuntimeが解決したlocal artifactだけをloadして構築する。"""
        if not model_artifact.is_dir():
            msg = "解決済みSTT model artifactがdirectoryではありません"
            raise ValueError(msg)
        loader = model_loader or _load_local_faster_whisper_model
        model = loader(model_artifact, device, compute_type, True)
        return cls(
            model,
            runtime_identity=_build_runtime_identity(device),
            resolved_model_identity=resolved_model_identity,
            gpu_coordinator=gpu_coordinator,
        )

    @property
    def runtime_identity(self) -> str:
        """adapter構築時に解決されたSpeech Runtime Identityを返す。"""
        return self._runtime_identity

    @property
    def resolved_model_identity(self) -> str:
        """run内でfreeze済みのResolved Model Identityを返す。"""
        return self._resolved_model_identity

    def transcribe(
        self,
        chunk: PcmAudioChunk,
        *,
        language: str,
        vad_filter: bool,
        beam_size: int,
    ) -> SpeechRecognitionResult:
        """mono s16le PCMをbackend非依存のsample timestampへ変換する。"""
        operation = partial(
            self._transcribe,
            chunk,
            language=language,
            vad_filter=vad_filter,
            beam_size=beam_size,
        )

        if self._gpu_coordinator is None:
            return operation()
        return self._gpu_coordinator.run("speech_to_text", operation)

    def _transcribe(
        self,
        chunk: PcmAudioChunk,
        *,
        language: str,
        vad_filter: bool,
        beam_size: int,
    ) -> SpeechRecognitionResult:
        """model呼び出しとlazy segment変換を一つのGPU workとして実行する。"""
        waveform = np.frombuffer(chunk.pcm_bytes, dtype="<i2").astype(np.float32)
        waveform /= 32768.0
        segments, info = self._model.transcribe(
            waveform,
            language=language,
            vad_filter=vad_filter,
            beam_size=beam_size,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        return SpeechRecognitionResult(
            vad_speech_detected=(_number_attribute(info, "duration_after_vad") > 0.0),
            segments=tuple(
                _convert_segment(segment, chunk.sample_rate, chunk.sample_count)
                for segment in segments
            ),
            detected_language=_optional_string_attribute(info, "language"),
        )


def _convert_segment(
    segment: object,
    sample_rate: int,
    sample_count: int,
) -> SpeechSegment:
    words_value = getattr(segment, "words", None)
    if not isinstance(words_value, Iterable) or isinstance(words_value, str | bytes):
        msg = "faster-whisper segmentにword timestampがありません"
        raise ValueError(msg)
    words = tuple(
        _convert_word(word, sample_rate, sample_count) for word in words_value
    )
    return SpeechSegment(
        words=words,
        average_log_probability=_number_attribute(segment, "avg_logprob"),
        no_speech_probability=_optional_number_attribute(
            segment,
            "no_speech_prob",
        ),
    )


def _convert_word(
    word: object,
    sample_rate: int,
    sample_count: int,
) -> SpeechWord:
    text = _string_attribute(word, "word")
    start_sample = _timestamp_sample(word, "start", sample_rate)
    end_sample = _timestamp_sample(word, "end", sample_rate)
    if start_sample < 0 or end_sample < start_sample or end_sample > sample_count:
        msg = "faster-whisper word timestampがPCM chunk範囲外です"
        raise ValueError(msg)
    return SpeechWord(
        text=text,
        start_sample=start_sample,
        end_sample=end_sample,
        probability=_optional_number_attribute(word, "probability"),
    )


def _timestamp_sample(value: object, attribute: str, sample_rate: int) -> int:
    seconds = _number_attribute(value, attribute)
    samples = Decimal(str(seconds)) * sample_rate
    return int(samples.to_integral_value(rounding=ROUND_HALF_EVEN))


def _number_attribute(value: object, attribute: str) -> float:
    result = getattr(value, attribute, None)
    if not isinstance(result, int | float) or isinstance(result, bool):
        msg = f"faster-whisper {attribute}がnumberではありません"
        raise ValueError(msg)
    normalized = float(result)
    if not math.isfinite(normalized):
        msg = f"faster-whisper {attribute}がfiniteではありません"
        raise ValueError(msg)
    return normalized


def _optional_number_attribute(value: object, attribute: str) -> float | None:
    return (
        None
        if getattr(value, attribute, None) is None
        else _number_attribute(value, attribute)
    )


def _string_attribute(value: object, attribute: str) -> str:
    result = getattr(value, attribute, None)
    if not isinstance(result, str):
        msg = f"faster-whisper {attribute}がstringではありません"
        raise ValueError(msg)
    return result


def _optional_string_attribute(value: object, attribute: str) -> str | None:
    return (
        None
        if getattr(value, attribute, None) is None
        else _string_attribute(value, attribute)
    )


def _load_local_faster_whisper_model(
    model_artifact: Path,
    device: str,
    compute_type: str,
    local_files_only: bool,
) -> FasterWhisperModel:
    model = WhisperModel(
        str(model_artifact),
        device=device,
        compute_type=compute_type,
        local_files_only=local_files_only,
    )
    return cast(FasterWhisperModel, model)


def _build_runtime_identity(device: str) -> str:
    payload = {
        "adapter": _ADAPTER_VERSION,
        "ctranslate2": version("ctranslate2"),
        "faster_whisper": version("faster-whisper"),
        "backend_capabilities": _backend_capabilities(device),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "speech_" + hashlib.sha256(canonical).hexdigest()


def _backend_capabilities(device: str) -> dict[str, object]:
    capabilities: dict[str, object] = {
        "cpu_compute_types": sorted(ctranslate2.get_supported_compute_types("cpu")),
    }
    if device == "cuda" or device == "auto" or device.startswith("cuda:"):
        cuda_device_count = ctranslate2.get_cuda_device_count()
        capabilities["cuda"] = {
            "device_count": cuda_device_count,
            "compute_types": (
                sorted(ctranslate2.get_supported_compute_types("cuda", 0))
                if cuda_device_count > 0
                else []
            ),
            "runtime_distributions": {
                distribution: _optional_distribution_version(distribution)
                for distribution in _GPU_RUNTIME_DISTRIBUTIONS
            },
        }
    return capabilities


def _optional_distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None
