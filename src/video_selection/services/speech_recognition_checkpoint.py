"""Speech RecognitionをPCM chunk単位で確定する。"""

import hashlib
import math
from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import cast

from ..models.checkpoint_operation import CheckpointOperation
from ..models.pcm_audio_chunk import PcmAudioChunk
from ..models.speech_recognition_result import SpeechRecognitionResult
from ..models.speech_segment import SpeechSegment
from ..models.speech_word import SpeechWord
from ..protocols.run_observer import RunObserver
from .durable_work_unit_cache import DurableWorkUnitCache

_SCHEMA = "game-screen-pick/speech-recognition-chunk@1.0.0"


class SpeechRecognitionCheckpoint:
    """同じ意味入力のSpeech Recognition Resultをchunkごとに再利用する。"""

    def __init__(
        self,
        cache_folder: Path,
        *,
        source_fingerprint: str,
        recognition_semantic_input: Mapping[str, object],
        validate_source: Callable[[], None] | None = None,
        observer: RunObserver | None = None,
    ) -> None:
        self._recognition_semantic_input = dict(recognition_semantic_input)
        self._validate_source = validate_source or _skip_validation
        self._cache = DurableWorkUnitCache(
            cache_folder,
            subject_fingerprint=source_fingerprint,
            operation=CheckpointOperation.SPEECH_RECOGNITION_CHUNK,
            observer=observer,
        )

    def resolve(
        self,
        chunk: PcmAudioChunk,
        recognize: Callable[[], SpeechRecognitionResult],
    ) -> SpeechRecognitionResult:
        """chunk checkpointを復元し、miss時だけSpeech Runtimeを呼ぶ。"""
        semantic_input = {
            "speech_recognition_semantic_input": self._recognition_semantic_input,
            "chunk": {
                "stream_index": chunk.stream_index,
                "sample_start": chunk.sample_start,
                "sample_count": chunk.sample_count,
                "sample_rate": chunk.sample_rate,
                "pts": chunk.pts,
                "time_base": _fraction_value(chunk.time_base),
                "pcm_sha256": hashlib.sha256(chunk.pcm_bytes).hexdigest(),
            },
        }
        bundle, _reused = self._cache.resolve(
            f"samples-{chunk.sample_start}-{chunk.sample_count}",
            semantic_input,
            lambda _folder: self._produce_recognition(
                recognize,
                chunk.sample_count,
            ),
            validate_bundle=lambda value: _restore_recognition(
                value.artifact,
                chunk.sample_count,
            ),
        )
        self._validate_source()
        return _restore_recognition(bundle.artifact, chunk.sample_count)

    def _produce_recognition(
        self,
        recognize: Callable[[], SpeechRecognitionResult],
        maximum_sample_count: int,
    ) -> dict[str, object]:
        """認識後のsource検証に成功した結果だけを確定候補にする。"""
        recognition = recognize()
        _validate_recognition(recognition, maximum_sample_count)
        self._validate_source()
        return _serialize_recognition(recognition)


def _serialize_recognition(
    recognition: SpeechRecognitionResult,
) -> dict[str, object]:
    return {
        "schema": _SCHEMA,
        "vad_speech_detected": recognition.vad_speech_detected,
        "detected_language": recognition.detected_language,
        "segments": [
            {
                "average_log_probability": segment.average_log_probability,
                "no_speech_probability": segment.no_speech_probability,
                "words": [
                    {
                        "text": word.text,
                        "start_sample": word.start_sample,
                        "end_sample": word.end_sample,
                        "probability": word.probability,
                    }
                    for word in segment.words
                ],
            }
            for segment in recognition.segments
        ],
    }


def _restore_recognition(
    artifact: Mapping[str, object],
    maximum_sample_count: int,
) -> SpeechRecognitionResult:
    if artifact.get("schema") != _SCHEMA:
        msg = "Speech Recognition chunk artifact schemaが不正です"
        raise ValueError(msg)
    detected_language = artifact.get("detected_language")
    recognition = SpeechRecognitionResult(
        vad_speech_detected=_boolean(artifact.get("vad_speech_detected")),
        detected_language=(
            None if detected_language is None else _string(detected_language)
        ),
        segments=tuple(
            SpeechSegment(
                words=tuple(
                    SpeechWord(
                        text=_string(word.get("text")),
                        start_sample=_integer(word.get("start_sample")),
                        end_sample=_integer(word.get("end_sample")),
                        probability=_optional_number(word.get("probability")),
                    )
                    for word in _mapping_list(segment.get("words"))
                ),
                average_log_probability=_number(segment.get("average_log_probability")),
                no_speech_probability=_optional_number(
                    segment.get("no_speech_probability")
                ),
            )
            for segment in _mapping_list(artifact.get("segments"))
        ),
    )
    _validate_recognition(recognition, maximum_sample_count)
    return recognition


def _validate_recognition(
    recognition: SpeechRecognitionResult,
    maximum_sample_count: int,
) -> None:
    """chunk外時刻、非有限score、順序不正をcheckpointへ入れない。"""
    if maximum_sample_count < 1 or (
        recognition.detected_language is not None
        and not recognition.detected_language.strip()
    ):
        raise ValueError("Speech Recognition chunk resultが不正です")
    previous_segment_start = -1
    for segment in recognition.segments:
        if not math.isfinite(segment.average_log_probability):
            raise ValueError("Speech Recognition chunk scoreが不正です")
        _validate_probability(segment.no_speech_probability)
        previous_start = -1
        previous_end = -1
        for word in segment.words:
            if (
                word.start_sample < 0
                or word.end_sample < word.start_sample
                or word.end_sample > maximum_sample_count
                or word.start_sample < previous_start
                or word.end_sample < previous_end
            ):
                raise ValueError("timestamp_drift")
            _validate_probability(word.probability)
            previous_start = word.start_sample
            previous_end = word.end_sample
        if segment.words:
            segment_start = segment.words[0].start_sample
            if segment_start < previous_segment_start:
                raise ValueError("timestamp_drift")
            previous_segment_start = segment_start


def _validate_probability(value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Speech Recognition chunk probabilityが不正です")


def _fraction_value(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _mapping_list(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, dict) and all(isinstance(key, str) for key in item)
        for item in value
    ):
        msg = "Speech Recognition chunk artifact listが不正です"
        raise ValueError(msg)
    return cast(tuple[Mapping[str, object], ...], tuple(value))


def _string(value: object) -> str:
    if not isinstance(value, str):
        msg = "Speech Recognition chunk artifact stringが不正です"
        raise ValueError(msg)
    return value


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = "Speech Recognition chunk artifact integerが不正です"
        raise ValueError(msg)
    return value


def _number(value: object) -> float:
    if type(value) not in {int, float}:
        msg = "Speech Recognition chunk artifact numberが不正です"
        raise ValueError(msg)
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        msg = "Speech Recognition chunk artifact numberが不正です"
        raise ValueError(msg)
    return result


def _optional_number(value: object) -> float | None:
    return None if value is None else _number(value)


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        msg = "Speech Recognition chunk artifact booleanが不正です"
        raise ValueError(msg)
    return value


def _skip_validation() -> None:
    return
