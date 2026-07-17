"""audio sample gridからSpeech Context Cueを収集する。"""

import unicodedata
from collections.abc import Iterable, Iterator
from fractions import Fraction
from functools import partial

from ..models.context_cue import ContextCue
from ..models.context_cue_diagnostics import ContextCueDiagnostics
from ..models.context_cue_provenance import ContextCueProvenance
from ..models.context_source_outcome import ContextSourceOutcome
from ..models.context_stage_error import ContextStageError
from ..models.context_stage_failure_reason import ContextStageFailureReason
from ..models.context_stage_result import ContextStageResult
from ..models.effective_configuration import EffectiveConfiguration
from ..models.media_stream import MediaStream
from ..models.pcm_audio_chunk import PcmAudioChunk
from ..models.rejected_speech_diagnostic import RejectedSpeechDiagnostic
from ..models.speech_word import SpeechWord
from ..models.video_scan_result import VideoScanResult
from ..models.video_source import VideoSource
from ..protocols.speech_runtime import SpeechRuntime
from ..protocols.video_stage_media_runtime import VideoStageMediaRuntime
from .build_context_cue_id import build_context_cue_id
from .external_work_monitor import ExternalWorkMonitor
from .iter_overlapping_pcm_chunks import iter_overlapping_pcm_chunks
from .normalize_context_language import normalize_context_language

_AUDIO_SAMPLE_RATE = 16000
_WORD_GAP_SECONDS = Fraction(3, 2)
_MINIMUM_AVERAGE_LOG_PROBABILITY = -0.8
_MINIMUM_SEMANTIC_CHARACTER_COUNT = 3


def collect_speech_context(
    *,
    media_runtime: VideoStageMediaRuntime,
    speech_runtime: SpeechRuntime,
    source: VideoSource,
    scan: VideoScanResult,
    stream: MediaStream,
    configuration: EffectiveConfiguration,
    external_work_monitor: ExternalWorkMonitor | None = None,
) -> ContextStageResult:
    """選択audioをSTTしCue、低reliability診断、outcomeを返す。"""
    chunk_samples = int(configuration.speech_chunk_seconds * _AUDIO_SAMPLE_RATE)
    overlap_samples = int(configuration.speech_overlap_seconds * _AUDIO_SAMPLE_RATE)
    frame_sample_count = chunk_samples - overlap_samples
    cues: list[ContextCue] = []
    rejected_diagnostics: list[RejectedSpeechDiagnostic] = []
    detected_speech = False
    processed_chunk_count = 0
    speech_language = normalize_context_language(configuration.language)
    source_chunks = _iter_validated_pcm_chunks(
        media_runtime.scan_pcm_audio(
            source.path,
            stream.index,
            _AUDIO_SAMPLE_RATE,
            frame_sample_count,
        ),
        stream,
    )
    overlapping_chunks = iter_overlapping_pcm_chunks(
        source_chunks,
        overlap_samples,
    )
    for chunk, ownership_start, ownership_end in _iter_chunk_ownership(
        overlapping_chunks
    ):
        try:
            transcribe = partial(
                speech_runtime.transcribe,
                chunk,
                language=speech_language,
                vad_filter=configuration.speech_vad_filter,
                beam_size=configuration.speech_to_text_beam_size,
            )
            recognition = (
                transcribe()
                if external_work_monitor is None
                else external_work_monitor.run(
                    transcribe,
                    reason_code="speech_recognition_started",
                )
            )
        except Exception:
            raise ContextStageError(
                ContextStageFailureReason.CHUNK_FAILED
                if processed_chunk_count > 0
                else ContextStageFailureReason.STT_ANALYSIS_FAILED
            ) from None
        processed_chunk_count += 1
        detected_speech = detected_speech or recognition.vad_speech_detected
        for segment in recognition.segments:
            for words in _group_words(segment.words, chunk.sample_rate):
                text = "".join(word.text for word in words).strip()
                provenance = _speech_provenance(
                    stream,
                    chunk,
                    speech_runtime,
                    configuration,
                    detected_language=recognition.detected_language,
                )
                diagnostics = ContextCueDiagnostics(
                    average_log_probability=segment.average_log_probability,
                    no_speech_probability=segment.no_speech_probability,
                    word_probabilities=tuple(word.probability for word in words),
                )
                cue_midpoint = chunk.sample_start + Fraction(
                    words[0].start_sample + words[-1].end_sample,
                    2,
                )
                if not ownership_start <= cue_midpoint < ownership_end:
                    continue
                if (
                    segment.average_log_probability < _MINIMUM_AVERAGE_LOG_PROBABILITY
                    or _semantic_character_count(text)
                    < _MINIMUM_SEMANTIC_CHARACTER_COUNT
                ):
                    rejected_diagnostics.append(
                        _rejected_speech_diagnostic(
                            scan,
                            stream,
                            chunk,
                            words,
                            text,
                            segment.average_log_probability,
                            segment.no_speech_probability,
                            provenance,
                        )
                    )
                    continue
                cues.append(
                    _speech_cue(
                        source,
                        scan,
                        stream,
                        chunk,
                        words[0].start_sample,
                        words[-1].end_sample,
                        text,
                        recognition.detected_language or stream.language,
                        diagnostics,
                        provenance,
                    )
                )
    outcome = _speech_outcome(
        stream.index,
        cues,
        rejected_diagnostics,
        detected_speech,
        processed_chunk_count,
    )
    return ContextStageResult(
        cues=tuple(cues),
        outcomes=(outcome,),
        rejected_speech_diagnostics=tuple(rejected_diagnostics),
    )


def _speech_outcome(
    stream_index: int,
    cues: list[ContextCue],
    rejected_diagnostics: list[RejectedSpeechDiagnostic],
    detected_speech: bool,
    processed_chunk_count: int,
) -> ContextSourceOutcome:
    if cues:
        return ContextSourceOutcome(
            source_kind="speech_to_text",
            stream_index=stream_index,
            status="available",
            reason_code=(
                "context_extracted_with_rejections"
                if rejected_diagnostics
                else "context_extracted"
            ),
            cue_count=len(cues),
            rejected_count=len(rejected_diagnostics),
            processed_chunk_count=processed_chunk_count,
        )
    if rejected_diagnostics:
        return ContextSourceOutcome(
            source_kind="speech_to_text",
            stream_index=stream_index,
            status="low_reliability",
            reason_code="asr_below_policy_threshold",
            rejected_count=len(rejected_diagnostics),
            processed_chunk_count=processed_chunk_count,
        )
    return ContextSourceOutcome(
        source_kind="speech_to_text",
        stream_index=stream_index,
        status="no_speech",
        reason_code=("asr_no_speech" if detected_speech else "vad_no_speech"),
        processed_chunk_count=processed_chunk_count,
    )


def _speech_cue(
    source: VideoSource,
    scan: VideoScanResult,
    stream: MediaStream,
    chunk: PcmAudioChunk,
    local_start_sample: int,
    local_end_sample: int,
    text: str,
    language: str | None,
    diagnostics: ContextCueDiagnostics,
    provenance: ContextCueProvenance,
) -> ContextCue:
    start, end = _video_time_interval(
        scan,
        chunk,
        local_start_sample,
        local_end_sample,
    )
    return ContextCue(
        identifier=build_context_cue_id(
            video_fingerprint=source.fingerprint,
            source_kind="speech_to_text",
            stream_index=stream.index,
            start=start,
            end=end,
            text=text,
        ),
        video_fingerprint=source.fingerprint,
        source_kind="speech_to_text",
        stream_index=stream.index,
        start=start,
        end=end,
        timestamp_basis="asr_sample_grid_estimate",
        text=text,
        language=language,
        diagnostics=diagnostics,
        provenance=provenance,
    )


def _rejected_speech_diagnostic(
    scan: VideoScanResult,
    stream: MediaStream,
    chunk: PcmAudioChunk,
    words: tuple[SpeechWord, ...],
    text: str,
    average_log_probability: float,
    no_speech_probability: float | None,
    provenance: ContextCueProvenance,
) -> RejectedSpeechDiagnostic:
    first_word = words[0]
    last_word = words[-1]
    start, end = _video_time_interval(
        scan,
        chunk,
        first_word.start_sample,
        last_word.end_sample,
    )
    return RejectedSpeechDiagnostic(
        stream_index=stream.index,
        start=start,
        end=end,
        text=text,
        average_log_probability=average_log_probability,
        no_speech_probability=no_speech_probability,
        word_probabilities=tuple(word.probability for word in words),
        provenance=provenance,
    )


def _speech_provenance(
    stream: MediaStream,
    chunk: PcmAudioChunk,
    speech_runtime: SpeechRuntime,
    configuration: EffectiveConfiguration,
    *,
    detected_language: str | None,
) -> ContextCueProvenance:
    return ContextCueProvenance(
        codec_name=stream.codec_name,
        source_pts=chunk.pts,
        source_time_base=chunk.time_base,
        stream_language=stream.language,
        is_default=stream.is_default,
        is_forced=stream.is_forced,
        language_source=(
            "speech_recognition" if detected_language is not None else "stream_metadata"
        ),
        chunk_sample_start=chunk.sample_start,
        chunk_sample_end=chunk.sample_start + chunk.sample_count,
        speech_runtime_identity=speech_runtime.runtime_identity,
        resolved_model_identity=speech_runtime.resolved_model_identity,
        device=configuration.speech_to_text_device,
        compute_type=configuration.speech_to_text_compute_type,
    )


def _video_time_interval(
    scan: VideoScanResult,
    chunk: PcmAudioChunk,
    local_start_sample: int,
    local_end_sample: int,
) -> tuple[Fraction, Fraction]:
    if (
        local_start_sample < 0
        or local_end_sample <= local_start_sample
        or local_end_sample > chunk.sample_count
    ):
        msg = "timestamp_drift"
        raise ValueError(msg)
    video_origin = scan.timeline.origin_pts * scan.timeline.time_base
    chunk_origin = chunk.pts * chunk.time_base - video_origin
    start = chunk_origin + Fraction(local_start_sample, chunk.sample_rate)
    end = chunk_origin + Fraction(local_end_sample, chunk.sample_rate)
    if start < 0 or end <= start or end > scan.timeline.duration.seconds:
        msg = "timestamp_drift"
        raise ValueError(msg)
    return start, end


def _group_words(
    words: tuple[SpeechWord, ...],
    sample_rate: int,
) -> tuple[tuple[SpeechWord, ...], ...]:
    if not words:
        return ()
    groups: list[list[SpeechWord]] = [[words[0]]]
    maximum_gap_samples = _WORD_GAP_SECONDS * sample_rate
    for word in words[1:]:
        if word.start_sample - groups[-1][-1].end_sample > maximum_gap_samples:
            groups.append([word])
        else:
            groups[-1].append(word)
    return tuple(tuple(group) for group in groups)


def _iter_validated_pcm_chunks(
    chunks: Iterable[PcmAudioChunk],
    stream: MediaStream,
) -> Iterator[PcmAudioChunk]:
    """観測PCM originが選択streamのsample gridと一致するchunkだけを返す。"""
    if stream.start_pts is None or stream.time_base is None:
        msg = "timestamp_drift"
        raise ValueError(msg)
    stream_origin = stream.start_pts * stream.time_base
    for chunk in chunks:
        expected_origin = stream_origin + Fraction(
            chunk.sample_start,
            chunk.sample_rate,
        )
        observed_origin = chunk.pts * chunk.time_base
        if (
            chunk.stream_index != stream.index
            or chunk.sample_rate != _AUDIO_SAMPLE_RATE
            or abs(observed_origin - expected_origin) > Fraction(1, chunk.sample_rate)
        ):
            msg = "timestamp_drift"
            raise ValueError(msg)
        yield chunk


def _semantic_character_count(text: str) -> int:
    return sum(
        1
        for character in text
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def _iter_chunk_ownership(
    chunks: Iterable[PcmAudioChunk],
) -> Iterator[tuple[PcmAudioChunk, Fraction, Fraction]]:
    iterator = iter(chunks)
    try:
        current = next(iterator)
    except StopIteration:
        return
    previous_end: int | None = None
    while True:
        try:
            following = next(iterator)
        except StopIteration:
            following = None
        start = (
            Fraction(current.sample_start)
            if previous_end is None
            else Fraction(previous_end + current.sample_start, 2)
        )
        end = (
            Fraction(current.sample_start + current.sample_count)
            if following is None
            else Fraction(
                current.sample_start + current.sample_count + following.sample_start,
                2,
            )
        )
        yield current, start, end
        if following is None:
            return
        previous_end = current.sample_start + current.sample_count
        current = following
