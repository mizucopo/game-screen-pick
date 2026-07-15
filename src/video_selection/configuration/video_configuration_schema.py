"""動画入力TOML v1のstrict schema。"""

import math
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from .configuration_error import ConfigurationError

CONFIG_VERSION = "1.0.0"

CONFIG_KEYS = (
    "config_version",
    "input.recursive",
    "selection.image_count",
    "selection.scene_hint",
    "selection.spoiler_sensitivity",
    "selection.similarity_threshold",
    "frame_extraction.heartbeat_interval_seconds",
    "frame_extraction.scene_change_threshold",
    "frame_extraction.scene_min_interval_seconds",
    "frame_extraction.decode_backend",
    "frame_extraction.refinement_radius_seconds",
    "frame_extraction.max_frame_candidates",
    "candidate_moments.density_per_minute",
    "context.language",
    "context.subtitle_stream_index",
    "context.audio_stream_index",
    "ollama.host",
    "ollama.timeout_seconds",
    "ollama.max_parallel_requests",
    "models.auto_upgrade",
    "models.scene_catalog.name",
    "models.scene_catalog.num_ctx",
    "models.candidate_annotation.name",
    "models.candidate_annotation.num_ctx",
    "models.speech_to_text.name",
    "models.speech_to_text.device",
    "models.speech_to_text.compute_type",
    "models.speech_to_text.beam_size",
    "speech_to_text.vad_filter",
    "speech_to_text.chunk_seconds",
    "speech_to_text.overlap_seconds",
)

DEFAULT_VALUES: dict[str, object] = {
    "config_version": CONFIG_VERSION,
    "input.recursive": False,
    "selection.image_count": 100,
    "selection.scene_hint": None,
    "selection.spoiler_sensitivity": "medium",
    "selection.similarity_threshold": 0.72,
    "frame_extraction.heartbeat_interval_seconds": 1.0,
    "frame_extraction.scene_change_threshold": 0.25,
    "frame_extraction.scene_min_interval_seconds": 0.5,
    "frame_extraction.decode_backend": "cpu",
    "frame_extraction.refinement_radius_seconds": 1.0,
    "frame_extraction.max_frame_candidates": 3,
    "candidate_moments.density_per_minute": 2.0,
    "context.language": "ja",
    "context.subtitle_stream_index": None,
    "context.audio_stream_index": None,
    "ollama.host": "http://localhost:11434",
    "ollama.timeout_seconds": 60.0,
    "ollama.max_parallel_requests": 1,
    "models.auto_upgrade": True,
    "models.scene_catalog.name": "qwen3-vl:8b-instruct",
    "models.scene_catalog.num_ctx": 32768,
    "models.candidate_annotation.name": "qwen3-vl:8b-instruct",
    "models.candidate_annotation.num_ctx": 32768,
    "models.speech_to_text.name": ("dropbox-dash/faster-whisper-large-v3-turbo"),
    "models.speech_to_text.device": "cuda",
    "models.speech_to_text.compute_type": "float16",
    "models.speech_to_text.beam_size": 5,
    "speech_to_text.vad_filter": True,
    "speech_to_text.chunk_seconds": 600.0,
    "speech_to_text.overlap_seconds": 5.0,
}

_SECTION_KEYS = {
    "input": frozenset({"recursive"}),
    "selection": frozenset(
        {
            "image_count",
            "scene_hint",
            "spoiler_sensitivity",
            "similarity_threshold",
        }
    ),
    "frame_extraction": frozenset(
        {
            "heartbeat_interval_seconds",
            "scene_change_threshold",
            "scene_min_interval_seconds",
            "decode_backend",
            "refinement_radius_seconds",
            "max_frame_candidates",
        }
    ),
    "candidate_moments": frozenset({"density_per_minute"}),
    "context": frozenset({"language", "subtitle_stream_index", "audio_stream_index"}),
    "ollama": frozenset({"host", "timeout_seconds", "max_parallel_requests"}),
    "speech_to_text": frozenset({"vad_filter", "chunk_seconds", "overlap_seconds"}),
}
_MODEL_KEYS = frozenset(
    {"auto_upgrade", "scene_catalog", "candidate_annotation", "speech_to_text"}
)
_OLLAMA_MODEL_KEYS = frozenset({"name", "num_ctx"})
_SPEECH_MODEL_KEYS = frozenset({"name", "device", "compute_type", "beam_size"})
_BOOL_KEYS = frozenset(
    {
        "input.recursive",
        "models.auto_upgrade",
        "speech_to_text.vad_filter",
    }
)
_POSITIVE_INTEGER_KEYS = frozenset(
    {
        "selection.image_count",
        "ollama.max_parallel_requests",
        "models.speech_to_text.beam_size",
    }
)
_OPTIONAL_NON_NEGATIVE_INTEGER_KEYS = frozenset(
    {"context.subtitle_stream_index", "context.audio_stream_index"}
)
_POSITIVE_NUMBER_KEYS = frozenset(
    {
        "frame_extraction.heartbeat_interval_seconds",
        "frame_extraction.scene_min_interval_seconds",
        "candidate_moments.density_per_minute",
        "ollama.timeout_seconds",
        "speech_to_text.chunk_seconds",
    }
)
_NON_NEGATIVE_NUMBER_KEYS = frozenset(
    {
        "frame_extraction.refinement_radius_seconds",
        "speech_to_text.overlap_seconds",
    }
)
_NONEMPTY_STRING_KEYS = frozenset(
    {
        "selection.scene_hint",
        "models.speech_to_text.device",
        "models.speech_to_text.compute_type",
    }
)
_OLLAMA_MODEL_NAME_KEYS = frozenset(
    {"models.scene_catalog.name", "models.candidate_annotation.name"}
)
_LANGUAGE_PATTERN = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")
_HUGGING_FACE_REPO_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?"
)


def load_video_configuration(path: Path) -> dict[str, object]:
    """明示されたTOMLを読み、全keyをstrictに検証する。"""
    try:
        with path.open("rb") as file:
            raw_data = cast(dict[str, object], tomllib.load(file))
    except tomllib.TOMLDecodeError:
        raise ConfigurationError(
            "CONFIG_TOML_INVALID",
            "TOMLの構文が不正です",
        ) from None
    except OSError:
        raise ConfigurationError(
            "CONFIG_FILE_UNREADABLE",
            "指定された設定ファイルを読み込めません",
        ) from None

    _validate_root_keys(raw_data)
    version = raw_data.get("config_version")
    if version is None:
        raise ConfigurationError(
            "CONFIG_VERSION_REQUIRED",
            "config_versionが必要です",
        )
    validate_configuration_value("config_version", version)

    flattened: dict[str, object] = {"config_version": CONFIG_VERSION}
    for section_name, known_keys in _SECTION_KEYS.items():
        if section_name not in raw_data:
            continue
        section = _require_table(raw_data[section_name], section_name)
        _reject_unknown_keys(section_name, section, known_keys)
        for key, value in section.items():
            canonical_key = f"{section_name}.{key}"
            flattened[canonical_key] = validate_configuration_value(
                canonical_key,
                value,
            )

    if "models" in raw_data:
        flattened.update(_validate_models(raw_data["models"]))

    document_values = dict(DEFAULT_VALUES)
    document_values.update(flattened)
    validate_cross_constraints(document_values)
    return flattened


def validate_configuration_value(key: str, value: object) -> object:
    """一つのcanonical keyを型・enum・範囲に従って検証する。"""
    if key == "config_version":
        version = _require_string(key, value)
        if version != CONFIG_VERSION:
            raise _invalid_value(key, f"{CONFIG_VERSION}である必要があります")
        return version

    if key in _BOOL_KEYS:
        if type(value) is not bool:
            raise _invalid_type(key, "boolean")
        return value

    if key in _POSITIVE_INTEGER_KEYS:
        integer = _require_integer(key, value)
        if integer < 1:
            raise _invalid_value(key, "1以上である必要があります")
        return integer

    if key in _OPTIONAL_NON_NEGATIVE_INTEGER_KEYS:
        integer = _require_integer(key, value)
        if integer < 0:
            raise _invalid_value(key, "0以上である必要があります")
        return integer

    if key in _POSITIVE_NUMBER_KEYS:
        number = _require_number(key, value)
        if number <= 0:
            raise _invalid_value(key, "0より大きい必要があります")
        return number

    if key in _NON_NEGATIVE_NUMBER_KEYS:
        number = _require_number(key, value)
        if number < 0:
            raise _invalid_value(key, "0以上である必要があります")
        return number

    if key == "selection.similarity_threshold":
        number = _require_number(key, value)
        if not 0 <= number <= 0.98:
            raise _invalid_value(key, "0以上0.98以下である必要があります")
        return number

    if key == "frame_extraction.scene_change_threshold":
        number = _require_number(key, value)
        if not 0 <= number <= 1:
            raise _invalid_value(key, "0以上1以下である必要があります")
        return number

    if key == "frame_extraction.max_frame_candidates":
        integer = _require_integer(key, value)
        if not 1 <= integer <= 3:
            raise _invalid_value(key, "1以上3以下である必要があります")
        return integer

    if key in {
        "models.scene_catalog.num_ctx",
        "models.candidate_annotation.num_ctx",
    }:
        integer = _require_integer(key, value)
        if integer < 32768:
            raise _invalid_value(key, "32768以上である必要があります")
        return integer

    if key == "selection.spoiler_sensitivity":
        sensitivity = _require_string(key, value)
        if sensitivity not in {"low", "medium", "high"}:
            raise _invalid_value(key, "low、medium、highのいずれかが必要です")
        return sensitivity

    if key == "frame_extraction.decode_backend":
        backend = _require_string(key, value)
        if backend not in {"cpu", "nvdec"}:
            raise _invalid_value(key, "cpuまたはnvdecである必要があります")
        return backend

    if key in _NONEMPTY_STRING_KEYS:
        return _require_nonempty_string(key, value)

    if key == "context.language":
        language = _require_nonempty_string(key, value)
        if _LANGUAGE_PATTERN.fullmatch(language) is None:
            raise _invalid_value(key, "BCP 47相当のlanguage tagが必要です")
        return language

    if key == "ollama.host":
        host = _require_nonempty_string(key, value)
        _validate_http_url(key, host)
        return host

    if key in _OLLAMA_MODEL_NAME_KEYS:
        model_name = _require_nonempty_string(key, value)
        if any(character.isspace() for character in model_name):
            raise _invalid_value(key, "空白を含まないOllama tagが必要です")
        return model_name

    if key == "models.speech_to_text.name":
        repo_id = _require_nonempty_string(key, value)
        if _HUGGING_FACE_REPO_PATTERN.fullmatch(repo_id) is None:
            raise _invalid_value(key, "Hugging Face repo IDが必要です")
        return repo_id

    raise ConfigurationError(
        "CONFIG_UNKNOWN_KEY",
        f"未対応の設定keyです: {key}",
    )


def validate_cross_constraints(values: Mapping[str, object]) -> None:
    """複数keyにまたがるv1制約を検証する。"""
    chunk_seconds = cast(float, values["speech_to_text.chunk_seconds"])
    overlap_seconds = cast(float, values["speech_to_text.overlap_seconds"])
    if overlap_seconds >= chunk_seconds:
        raise ConfigurationError(
            "CONFIG_INVALID_RELATION",
            "speech_to_text.overlap_secondsはchunk_seconds未満である必要があります",
        )


def _validate_root_keys(raw_data: Mapping[str, object]) -> None:
    known_keys = {"config_version", "models", *_SECTION_KEYS}
    for key in raw_data:
        if key not in known_keys:
            raise ConfigurationError(
                "CONFIG_UNKNOWN_SECTION",
                f"未知のsectionまたはroot keyです: {key}",
            )


def _validate_models(raw_models: object) -> dict[str, object]:
    models = _require_table(raw_models, "models")
    _reject_unknown_keys("models", models, _MODEL_KEYS)
    flattened: dict[str, object] = {}
    if "auto_upgrade" in models:
        flattened["models.auto_upgrade"] = validate_configuration_value(
            "models.auto_upgrade",
            models["auto_upgrade"],
        )

    role_keys = {
        "scene_catalog": _OLLAMA_MODEL_KEYS,
        "candidate_annotation": _OLLAMA_MODEL_KEYS,
        "speech_to_text": _SPEECH_MODEL_KEYS,
    }
    for role, known_keys in role_keys.items():
        if role not in models:
            continue
        table_name = f"models.{role}"
        role_values = _require_table(models[role], table_name)
        _reject_unknown_keys(table_name, role_values, known_keys)
        for key, value in role_values.items():
            canonical_key = f"{table_name}.{key}"
            flattened[canonical_key] = validate_configuration_value(
                canonical_key,
                value,
            )
    return flattened


def _require_table(value: object, key: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _invalid_type(key, "table")
    return cast(dict[str, object], value)


def _reject_unknown_keys(
    section_name: str,
    values: Mapping[str, object],
    known_keys: frozenset[str],
) -> None:
    for key in values:
        if key not in known_keys:
            raise ConfigurationError(
                "CONFIG_UNKNOWN_KEY",
                f"未知の設定keyです: {section_name}.{key}",
            )


def _require_string(key: str, value: object) -> str:
    if type(value) is not str:
        raise _invalid_type(key, "string")
    return value


def _require_nonempty_string(key: str, value: object) -> str:
    text = _require_string(key, value)
    if not text.strip():
        raise _invalid_value(key, "空でないstringが必要です")
    return text


def _require_integer(key: str, value: object) -> int:
    if type(value) is not int:
        raise _invalid_type(key, "integer")
    return value


def _require_number(key: str, value: object) -> float:
    if type(value) not in {int, float}:
        raise _invalid_type(key, "number")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        raise _invalid_value(key, "有限のnumberが必要です")
    return number


def _validate_http_url(key: str, value: str) -> None:
    if any(character.isspace() for character in value):
        raise _invalid_value(key, "absolute HTTP(S) URLが必要です")
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        raise _invalid_value(key, "absolute HTTP(S) URLが必要です") from None
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise _invalid_value(key, "absolute HTTP(S) URLが必要です")


def _invalid_type(key: str, expected_type: str) -> ConfigurationError:
    return ConfigurationError(
        "CONFIG_INVALID_TYPE",
        f"{key}は{expected_type}である必要があります",
    )


def _invalid_value(key: str, constraint: str) -> ConfigurationError:
    return ConfigurationError(
        "CONFIG_INVALID_VALUE",
        f"{key}は{constraint}",
    )
