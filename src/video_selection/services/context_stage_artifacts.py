"""Context Stage ResultとCompleted Stage JSONの相互変換。"""

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

_SCHEMA = "game-screen-pick/context-collection@1.0.0"


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


def restore_context_stage(artifact: Mapping[str, object]) -> ContextStageResult:
    """検証済みartifactからContext Stage Resultを復元する。"""
    if artifact.get("schema") != _SCHEMA:
        msg = "Context Collection artifact schemaが不正です"
        raise ValueError(msg)
    return ContextStageResult(
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
    if not isinstance(value, int | float) or isinstance(value, bool):
        msg = "Context artifact numberが不正です"
        raise ValueError(msg)
    return float(value)
