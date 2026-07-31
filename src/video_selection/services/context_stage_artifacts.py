"""Context Stage ResultとCompleted Stage JSONの相互変換。"""

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Literal, cast

from ..models.context_cue import (
    ContextCue,
    ContextCueReliability,
    ContextSourceKind,
    ContextTimestampBasis,
)
from ..models.context_cue_diagnostics import ContextCueDiagnostics
from ..models.context_cue_equivalence_group import ContextCueEquivalenceGroup
from ..models.context_cue_provenance import (
    ContextCueProvenance,
    ContextLanguageSource,
)
from ..models.context_source_outcome import (
    ContextSourceOutcome,
    ContextSourceStatus,
)
from ..models.context_stage_result import ContextStageResult
from ..models.rejected_speech_diagnostic import RejectedSpeechDiagnostic
from .build_context_cue_equivalence_groups import (
    build_context_cue_equivalence_groups,
)
from .build_context_cue_id import build_context_cue_id

_SCHEMA = "game-screen-pick/context-collection@1.0.0"
_SOURCE_KINDS = frozenset({"embedded_subtitle", "speech_to_text"})
_TIMESTAMP_BASES = frozenset(
    {"source_pts", "container_text_ms", "asr_sample_grid_estimate"}
)
_OUTCOME_CONTRACTS = {
    ("embedded_subtitle", "available", "context_extracted"),
    ("embedded_subtitle", "absent", "no_subtitle_stream"),
    ("embedded_subtitle", "no_context", "no_subtitle_events"),
    ("speech_to_text", "available", "context_extracted"),
    ("speech_to_text", "available", "context_extracted_with_rejections"),
    ("speech_to_text", "absent", "no_audio_stream"),
    ("speech_to_text", "low_reliability", "asr_below_policy_threshold"),
    ("speech_to_text", "low_reliability", "asr_zero_duration"),
    ("speech_to_text", "no_speech", "asr_no_speech"),
    ("speech_to_text", "no_speech", "vad_no_speech"),
}


def serialize_context_stage(result: ContextStageResult) -> dict[str, object]:
    """Context Stage Resultをpath非依存artifactへ変換する。"""
    return {
        "schema": _SCHEMA,
        "cues": [
            {
                "id": cue.identifier,
                "video_fingerprint": cue.video_fingerprint,
                "source_kind": cue.source_kind,
                "stream_index": cue.stream_index,
                "start": _serialize_fraction(cue.start),
                "end": _serialize_fraction(cue.end),
                "timestamp_basis": cue.timestamp_basis,
                "text": cue.text,
                "language": cue.language,
                "reliability": cue.reliability,
                "diagnostics": _serialize_diagnostics(cue.diagnostics),
                "provenance": _serialize_provenance(cue.provenance),
            }
            for cue in result.cues
        ],
        "outcomes": [
            {
                "source_kind": outcome.source_kind,
                "stream_index": outcome.stream_index,
                "status": outcome.status,
                "reason_code": outcome.reason_code,
                "cue_count": outcome.cue_count,
                "rejected_count": outcome.rejected_count,
                "processed_chunk_count": outcome.processed_chunk_count,
            }
            for outcome in result.outcomes
        ],
        "rejected_speech_diagnostics": [
            {
                "stream_index": diagnostic.stream_index,
                "start": _serialize_fraction(diagnostic.start),
                "end": _serialize_fraction(diagnostic.end),
                "text": diagnostic.text,
                "average_log_probability": diagnostic.average_log_probability,
                "no_speech_probability": diagnostic.no_speech_probability,
                "word_probabilities": list(diagnostic.word_probabilities),
                "reason_code": diagnostic.reason_code,
                "reliability": diagnostic.reliability,
                "provenance": _serialize_provenance(diagnostic.provenance),
            }
            for diagnostic in result.rejected_speech_diagnostics
        ],
        "equivalence_groups": [
            {
                "representative_cue_id": group.representative_cue_id,
                "cue_ids": list(group.cue_ids),
            }
            for group in result.equivalence_groups
        ],
    }


def restore_context_stage(
    artifact: Mapping[str, object],
    *,
    expected_video_fingerprint: str | None = None,
    video_duration: Fraction | None = None,
) -> ContextStageResult:
    """検証済みartifactからContext Stage Resultを復元する。"""
    if artifact.get("schema") != _SCHEMA:
        msg = "Context Collection artifact schemaが不正です"
        raise ValueError(msg)
    result = ContextStageResult(
        cues=tuple(_restore_cue(item) for item in _mapping_list(artifact.get("cues"))),
        outcomes=tuple(
            _restore_outcome(item) for item in _mapping_list(artifact.get("outcomes"))
        ),
        rejected_speech_diagnostics=tuple(
            _restore_rejected_speech_diagnostic(item)
            for item in _mapping_list(artifact.get("rejected_speech_diagnostics"))
        ),
        equivalence_groups=tuple(
            ContextCueEquivalenceGroup(
                representative_cue_id=_string(item.get("representative_cue_id")),
                cue_ids=tuple(_string(cue_id) for cue_id in _list(item.get("cue_ids"))),
            )
            for item in _mapping_list(artifact.get("equivalence_groups"))
        ),
    )
    _validate_context_stage_result(
        result,
        expected_video_fingerprint=expected_video_fingerprint,
        video_duration=video_duration,
    )
    return result


def _restore_cue(value: Mapping[str, object]) -> ContextCue:
    return ContextCue(
        identifier=_string(value.get("id")),
        video_fingerprint=_string(value.get("video_fingerprint")),
        source_kind=cast(ContextSourceKind, _string(value.get("source_kind"))),
        stream_index=_integer(value.get("stream_index")),
        start=_fraction(value.get("start")),
        end=_fraction(value.get("end")),
        timestamp_basis=cast(
            ContextTimestampBasis,
            _string(value.get("timestamp_basis")),
        ),
        text=_string(value.get("text")),
        language=_optional_string(value.get("language")),
        reliability=cast(
            ContextCueReliability,
            _string(value.get("reliability")),
        ),
        diagnostics=_restore_diagnostics(value.get("diagnostics")),
        provenance=_restore_provenance(value.get("provenance")),
    )


def _restore_outcome(value: Mapping[str, object]) -> ContextSourceOutcome:
    stream_index = value.get("stream_index")
    return ContextSourceOutcome(
        source_kind=cast(ContextSourceKind, _string(value.get("source_kind"))),
        stream_index=None if stream_index is None else _integer(stream_index),
        status=cast(ContextSourceStatus, _string(value.get("status"))),
        reason_code=_string(value.get("reason_code")),
        cue_count=_integer(value.get("cue_count")),
        rejected_count=_integer(value.get("rejected_count")),
        processed_chunk_count=_integer(value.get("processed_chunk_count")),
    )


def _restore_rejected_speech_diagnostic(
    value: Mapping[str, object],
) -> RejectedSpeechDiagnostic:
    no_speech_probability = value.get("no_speech_probability")
    word_probabilities = _list(value.get("word_probabilities"))
    return RejectedSpeechDiagnostic(
        stream_index=_integer(value.get("stream_index")),
        start=_fraction(value.get("start")),
        end=_fraction(value.get("end")),
        text=_string(value.get("text")),
        average_log_probability=_number(value.get("average_log_probability")),
        no_speech_probability=(
            None if no_speech_probability is None else _number(no_speech_probability)
        ),
        word_probabilities=tuple(
            None if item is None else _number(item) for item in word_probabilities
        ),
        reason_code=_string(value.get("reason_code")),
        reliability=cast(Literal["low"], _string(value.get("reliability"))),
        provenance=_restore_provenance(value.get("provenance")),
    )


def _serialize_diagnostics(
    diagnostics: ContextCueDiagnostics | None,
) -> dict[str, object] | None:
    if diagnostics is None:
        return None
    return {
        "average_log_probability": diagnostics.average_log_probability,
        "no_speech_probability": diagnostics.no_speech_probability,
        "word_probabilities": list(diagnostics.word_probabilities),
    }


def _restore_diagnostics(value: object) -> ContextCueDiagnostics | None:
    if value is None:
        return None
    mapping = _mapping(value)
    no_speech_probability = mapping.get("no_speech_probability")
    return ContextCueDiagnostics(
        average_log_probability=_number(mapping.get("average_log_probability")),
        no_speech_probability=(
            None if no_speech_probability is None else _number(no_speech_probability)
        ),
        word_probabilities=tuple(
            None if item is None else _number(item)
            for item in _list(mapping.get("word_probabilities"))
        ),
    )


def _serialize_provenance(
    provenance: ContextCueProvenance | None,
) -> dict[str, object] | None:
    if provenance is None:
        return None
    return {
        "codec_name": provenance.codec_name,
        "source_pts": provenance.source_pts,
        "source_time_base": _serialize_fraction(provenance.source_time_base),
        "stream_language": provenance.stream_language,
        "is_default": provenance.is_default,
        "is_forced": provenance.is_forced,
        "language_source": provenance.language_source,
        "chunk_sample_start": provenance.chunk_sample_start,
        "chunk_sample_end": provenance.chunk_sample_end,
        "speech_runtime_identity": provenance.speech_runtime_identity,
        "resolved_model_identity": provenance.resolved_model_identity,
        "device": provenance.device,
        "compute_type": provenance.compute_type,
    }


def _restore_provenance(value: object) -> ContextCueProvenance | None:
    if value is None:
        return None
    mapping = _mapping(value)
    return ContextCueProvenance(
        codec_name=_string(mapping.get("codec_name")),
        source_pts=_integer(mapping.get("source_pts")),
        source_time_base=_fraction(mapping.get("source_time_base")),
        stream_language=_optional_string(mapping.get("stream_language")),
        is_default=_boolean(mapping.get("is_default")),
        is_forced=_boolean(mapping.get("is_forced")),
        language_source=cast(
            ContextLanguageSource,
            _string(mapping.get("language_source")),
        ),
        chunk_sample_start=_optional_integer(mapping.get("chunk_sample_start")),
        chunk_sample_end=_optional_integer(mapping.get("chunk_sample_end")),
        speech_runtime_identity=_optional_string(
            mapping.get("speech_runtime_identity")
        ),
        resolved_model_identity=_optional_string(
            mapping.get("resolved_model_identity")
        ),
        device=_optional_string(mapping.get("device")),
        compute_type=_optional_string(mapping.get("compute_type")),
    )


def _serialize_fraction(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction(value: object) -> Fraction:
    mapping = _mapping(value)
    return Fraction(
        _integer(mapping.get("numerator")),
        _integer(mapping.get("denominator")),
    )


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = "Context artifact objectが不正です"
        raise ValueError(msg)
    return cast(Mapping[str, object], value)


def _mapping_list(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        msg = "Context artifact listが不正です"
        raise ValueError(msg)
    return tuple(_mapping(item) for item in value)


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        msg = "Context artifact listが不正です"
        raise ValueError(msg)
    return value


def _string(value: object) -> str:
    if not isinstance(value, str):
        msg = "Context artifact stringが不正です"
        raise ValueError(msg)
    return value


def _optional_string(value: object) -> str | None:
    return None if value is None else _string(value)


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = "Context artifact integerが不正です"
        raise ValueError(msg)
    return value


def _optional_integer(value: object) -> int | None:
    return None if value is None else _integer(value)


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        msg = "Context artifact booleanが不正です"
        raise ValueError(msg)
    return value


def _number(value: object) -> float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        msg = "Context artifact numberが不正です"
        raise ValueError(msg)
    return float(value)


def _validate_context_stage_result(
    result: ContextStageResult,
    *,
    expected_video_fingerprint: str | None,
    video_duration: Fraction | None,
) -> None:
    """Cue、outcome、diagnostic、equivalenceのcross-referenceを検証する。"""
    if expected_video_fingerprint is not None and not _is_sha256(
        expected_video_fingerprint
    ):
        raise ValueError("Context artifactのexpected Video Fingerprintが不正です")
    if video_duration is not None and video_duration <= 0:
        raise ValueError("Context artifactのVideo Durationが不正です")
    cue_ids: set[str] = set()
    cue_counts = dict.fromkeys(_SOURCE_KINDS, 0)
    for cue in result.cues:
        _validate_cue(
            cue,
            expected_video_fingerprint=expected_video_fingerprint,
            video_duration=video_duration,
        )
        if cue.identifier in cue_ids:
            raise ValueError("Context artifactのCue IDが重複しています")
        cue_ids.add(cue.identifier)
        cue_counts[cue.source_kind] += 1

    outcomes_by_source: dict[str, ContextSourceOutcome] = {}
    for outcome in result.outcomes:
        _validate_outcome(outcome)
        if outcome.source_kind in outcomes_by_source:
            raise ValueError("Context artifactのsource outcomeが重複しています")
        outcomes_by_source[outcome.source_kind] = outcome
    if "embedded_subtitle" not in outcomes_by_source:
        raise ValueError("Context artifactにsubtitle outcomeがありません")
    for source_kind, cue_count in cue_counts.items():
        source_outcome = outcomes_by_source.get(source_kind)
        if cue_count > 0 and source_outcome is None:
            raise ValueError("Context artifactのCueに対応するoutcomeがありません")
        if source_outcome is not None and source_outcome.cue_count != cue_count:
            raise ValueError("Context artifactのCue countが一致しません")

    for diagnostic in result.rejected_speech_diagnostics:
        _validate_rejected_speech_diagnostic(
            diagnostic,
            video_duration=video_duration,
        )
    speech_outcome = outcomes_by_source.get("speech_to_text")
    rejected_count = len(result.rejected_speech_diagnostics)
    if (rejected_count > 0 and speech_outcome is None) or (
        speech_outcome is not None and speech_outcome.rejected_count != rejected_count
    ):
        raise ValueError("Context artifactのrejected countが一致しません")
    if speech_outcome is not None and speech_outcome.status == "low_reliability":
        reasons = {item.reason_code for item in result.rejected_speech_diagnostics}
        reasons_are_consistent = (
            reasons == {"asr_zero_duration"}
            if speech_outcome.reason_code == "asr_zero_duration"
            else "asr_below_policy_threshold" in reasons
            and reasons <= {"asr_below_policy_threshold", "asr_zero_duration"}
        )
        if not reasons_are_consistent:
            raise ValueError("Context artifactのrejected reasonが一致しません")

    expected_groups = build_context_cue_equivalence_groups(result.cues)
    if result.equivalence_groups != expected_groups:
        raise ValueError("Context artifactのequivalence groupが不正です")


def _validate_cue(
    cue: ContextCue,
    *,
    expected_video_fingerprint: str | None,
    video_duration: Fraction | None,
) -> None:
    if (
        cue.source_kind not in _SOURCE_KINDS
        or cue.timestamp_basis not in _TIMESTAMP_BASES
        or cue.reliability != "usable"
        or not _is_sha256(cue.video_fingerprint)
        or (
            expected_video_fingerprint is not None
            and cue.video_fingerprint != expected_video_fingerprint
        )
        or cue.stream_index < 0
        or cue.start < 0
        or cue.end <= cue.start
        or (video_duration is not None and cue.end > video_duration)
        or not cue.text
        or cue.text != cue.text.strip()
        or cue.identifier
        != build_context_cue_id(
            video_fingerprint=cue.video_fingerprint,
            source_kind=cue.source_kind,
            stream_index=cue.stream_index,
            start=cue.start,
            end=cue.end,
            text=cue.text,
        )
    ):
        raise ValueError("Context artifactのCue domainが不正です")
    _validate_cue_diagnostics(cue)
    _validate_provenance(cue.provenance, cue.source_kind)


def _validate_cue_diagnostics(cue: ContextCue) -> None:
    diagnostics = cue.diagnostics
    if cue.source_kind == "embedded_subtitle":
        if diagnostics is not None or cue.timestamp_basis not in {
            "source_pts",
            "container_text_ms",
        }:
            raise ValueError("Context artifactのsubtitle Cueが不正です")
        return
    if diagnostics is None or cue.timestamp_basis != "asr_sample_grid_estimate":
        raise ValueError("Context artifactのspeech Cueが不正です")
    if not math.isfinite(diagnostics.average_log_probability):
        raise ValueError("Context artifactのspeech scoreが不正です")
    _validate_probability(diagnostics.no_speech_probability)
    for value in diagnostics.word_probabilities:
        _validate_probability(value)


def _validate_provenance(
    provenance: ContextCueProvenance | None,
    source_kind: str,
) -> None:
    if (
        provenance is None
        or not provenance.codec_name
        or provenance.source_time_base <= 0
        or provenance.language_source not in {"stream_metadata", "speech_recognition"}
    ):
        raise ValueError("Context artifactのCue provenanceが不正です")
    if source_kind == "embedded_subtitle":
        if (
            provenance.language_source != "stream_metadata"
            or provenance.chunk_sample_start is not None
            or provenance.chunk_sample_end is not None
            or provenance.speech_runtime_identity is not None
            or provenance.resolved_model_identity is not None
            or provenance.device is not None
            or provenance.compute_type is not None
        ):
            raise ValueError("Context artifactのsubtitle provenanceが不正です")
        return
    if (
        provenance.language_source not in {"stream_metadata", "speech_recognition"}
        or provenance.chunk_sample_start is None
        or provenance.chunk_sample_end is None
        or provenance.chunk_sample_start < 0
        or provenance.chunk_sample_end <= provenance.chunk_sample_start
        or not provenance.speech_runtime_identity
        or not provenance.resolved_model_identity
        or not provenance.device
        or not provenance.compute_type
    ):
        raise ValueError("Context artifactのspeech provenanceが不正です")


def _validate_outcome(outcome: ContextSourceOutcome) -> None:
    counts = (
        outcome.cue_count,
        outcome.rejected_count,
        outcome.processed_chunk_count,
    )
    if (
        (
            outcome.source_kind,
            outcome.status,
            outcome.reason_code,
        )
        not in _OUTCOME_CONTRACTS
        or (outcome.stream_index is not None and outcome.stream_index < 0)
        or any(value < 0 for value in counts)
        or (
            outcome.status == "absent"
            and (
                outcome.stream_index is not None
                or outcome.cue_count != 0
                or outcome.rejected_count != 0
                or outcome.processed_chunk_count != 0
            )
        )
        or (outcome.status != "absent" and outcome.stream_index is None)
        or (
            outcome.status in {"no_context", "no_speech", "low_reliability"}
            and outcome.cue_count != 0
        )
        or (outcome.status == "available" and outcome.cue_count < 1)
        or (
            outcome.source_kind == "embedded_subtitle"
            and (outcome.rejected_count != 0 or outcome.processed_chunk_count != 0)
        )
        or (
            outcome.source_kind == "speech_to_text"
            and outcome.reason_code == "context_extracted"
            and outcome.rejected_count != 0
        )
        or (
            outcome.source_kind == "speech_to_text"
            and outcome.reason_code == "context_extracted_with_rejections"
            and outcome.rejected_count < 1
        )
        or (
            outcome.source_kind == "speech_to_text"
            and outcome.status == "low_reliability"
            and outcome.rejected_count < 1
        )
        or (
            outcome.source_kind == "speech_to_text"
            and outcome.status == "no_speech"
            and outcome.rejected_count != 0
        )
    ):
        raise ValueError("Context artifactのsource outcomeが不正です")


def _validate_rejected_speech_diagnostic(
    diagnostic: RejectedSpeechDiagnostic,
    *,
    video_duration: Fraction | None,
) -> None:
    if (
        diagnostic.stream_index < 0
        or diagnostic.start < 0
        or diagnostic.end < diagnostic.start
        or (video_duration is not None and diagnostic.end > video_duration)
        or diagnostic.reason_code
        not in {"asr_below_policy_threshold", "asr_zero_duration"}
        or diagnostic.reliability != "low"
        or not math.isfinite(diagnostic.average_log_probability)
        or (
            diagnostic.reason_code == "asr_zero_duration"
            and diagnostic.end != diagnostic.start
        )
        or (
            diagnostic.reason_code == "asr_below_policy_threshold"
            and diagnostic.end <= diagnostic.start
        )
    ):
        raise ValueError("Context artifactのrejected diagnosticが不正です")
    _validate_probability(diagnostic.no_speech_probability)
    for value in diagnostic.word_probabilities:
        _validate_probability(value)
    _validate_provenance(diagnostic.provenance, "speech_to_text")


def _validate_probability(value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Context artifactのprobabilityが不正です")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
