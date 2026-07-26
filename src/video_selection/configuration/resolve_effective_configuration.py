"""動画入力Effective Configurationのsource別解決。"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from ..models.effective_configuration import EffectiveConfiguration
from .configuration_error import ConfigurationError
from .configuration_source import ConfigurationSource
from .video_configuration_schema import (
    CONFIG_KEYS,
    DEFAULT_VALUES,
    load_video_configuration,
    validate_configuration_value,
    validate_cross_constraints,
)


def resolve_effective_configuration(
    *,
    video_input_folder: Path,
    output_folder: Path,
    config_path: Path | None = None,
    image_count: int | None = None,
    recursive: bool | None = None,
    scene_hint: str | None = None,
    spoiler_sensitivity: str | None = None,
    similarity_threshold: float | None = None,
    video_scan_workers: str | int | None = None,
    video_scan_auto_max_workers: int | None = None,
    ollama_host: str | None = None,
    reset_cache: bool | None = None,
    debug: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> EffectiveConfiguration:
    """明示CLI、TOML、公開環境変数、既定値をkeyごとに解決する。"""
    _validate_path_argument("video_input_folder", video_input_folder)
    _validate_path_argument("output_folder", output_folder)
    if config_path is not None:
        _validate_path_argument("config_path", config_path)

    toml_values = (
        load_video_configuration(config_path) if config_path is not None else {}
    )
    values = dict(DEFAULT_VALUES)
    sources = dict.fromkeys(CONFIG_KEYS, ConfigurationSource.BUILT_IN)
    public_environment = os.environ if environ is None else environ

    cli_values: dict[str, object | None] = {
        "selection.image_count": image_count,
        "input.recursive": recursive,
        "selection.scene_hint": scene_hint,
        "selection.spoiler_sensitivity": spoiler_sensitivity,
        "selection.similarity_threshold": similarity_threshold,
        "video_scan.workers": video_scan_workers,
        "video_scan.auto_max_workers": video_scan_auto_max_workers,
        "ollama.host": ollama_host,
    }
    environment_keys = {
        "ollama.host": "OLLAMA_HOST",
        "video_scan.workers": "GAME_SCREEN_PICK_VIDEO_SCAN_WORKERS",
        "video_scan.auto_max_workers": ("GAME_SCREEN_PICK_VIDEO_SCAN_AUTO_MAX_WORKERS"),
    }
    for key, environment_key in environment_keys.items():
        if cli_values[key] is not None or key in toml_values:
            continue
        environment_value = public_environment.get(environment_key)
        if environment_value is None:
            continue
        values[key] = validate_configuration_value(
            key,
            _parse_environment_value(key, environment_value),
        )
        sources[key] = ConfigurationSource.ENVIRONMENT

    for key, value in toml_values.items():
        values[key] = value
        sources[key] = ConfigurationSource.TOML

    for key, value in cli_values.items():
        if value is None:
            continue
        values[key] = validate_configuration_value(key, value)
        sources[key] = ConfigurationSource.CLI

    validate_cross_constraints(values)
    validated_reset_cache = _resolve_action_flag("reset_cache", reset_cache)
    validated_debug = _resolve_action_flag("debug", debug)
    sources["video_input_folder"] = ConfigurationSource.CLI
    sources["output_folder"] = ConfigurationSource.CLI
    sources["reset_cache"] = (
        ConfigurationSource.CLI
        if reset_cache is not None
        else ConfigurationSource.BUILT_IN
    )
    sources["debug"] = (
        ConfigurationSource.CLI if debug is not None else ConfigurationSource.BUILT_IN
    )

    return EffectiveConfiguration(
        video_input_folder=video_input_folder,
        output_folder=output_folder,
        config_version=cast(str, values["config_version"]),
        recursive=cast(bool, values["input.recursive"]),
        image_count=cast(int, values["selection.image_count"]),
        scene_hint=cast(str | None, values["selection.scene_hint"]),
        spoiler_sensitivity=cast(str, values["selection.spoiler_sensitivity"]),
        similarity_threshold=cast(
            float,
            values["selection.similarity_threshold"],
        ),
        heartbeat_interval_seconds=cast(
            float,
            values["frame_extraction.heartbeat_interval_seconds"],
        ),
        scene_change_threshold=cast(
            float,
            values["frame_extraction.scene_change_threshold"],
        ),
        scene_min_interval_seconds=cast(
            float,
            values["frame_extraction.scene_min_interval_seconds"],
        ),
        decode_backend=cast(str, values["frame_extraction.decode_backend"]),
        refinement_radius_seconds=cast(
            float,
            values["frame_extraction.refinement_radius_seconds"],
        ),
        max_frame_candidates=cast(
            int,
            values["frame_extraction.max_frame_candidates"],
        ),
        video_scan_workers=cast(str | int, values["video_scan.workers"]),
        video_scan_auto_max_workers=cast(
            int,
            values["video_scan.auto_max_workers"],
        ),
        candidate_density_per_minute=cast(
            float,
            values["candidate_moments.density_per_minute"],
        ),
        language=cast(str, values["context.language"]),
        subtitle_stream_index=cast(
            int | None,
            values["context.subtitle_stream_index"],
        ),
        audio_stream_index=cast(
            int | None,
            values["context.audio_stream_index"],
        ),
        ollama_host=cast(str, values["ollama.host"]),
        ollama_timeout_seconds=cast(float, values["ollama.timeout_seconds"]),
        ollama_max_parallel_requests=cast(
            int,
            values["ollama.max_parallel_requests"],
        ),
        models_auto_upgrade=cast(bool, values["models.auto_upgrade"]),
        scene_catalog_model=cast(str, values["models.scene_catalog.name"]),
        scene_catalog_num_ctx=cast(int, values["models.scene_catalog.num_ctx"]),
        candidate_annotation_model=cast(
            str,
            values["models.candidate_annotation.name"],
        ),
        candidate_annotation_num_ctx=cast(
            int,
            values["models.candidate_annotation.num_ctx"],
        ),
        speech_to_text_model=cast(str, values["models.speech_to_text.name"]),
        speech_to_text_device=cast(str, values["models.speech_to_text.device"]),
        speech_to_text_compute_type=cast(
            str,
            values["models.speech_to_text.compute_type"],
        ),
        speech_to_text_beam_size=cast(
            int,
            values["models.speech_to_text.beam_size"],
        ),
        speech_vad_filter=cast(bool, values["speech_to_text.vad_filter"]),
        speech_chunk_seconds=cast(float, values["speech_to_text.chunk_seconds"]),
        speech_overlap_seconds=cast(
            float,
            values["speech_to_text.overlap_seconds"],
        ),
        reset_cache=validated_reset_cache,
        debug=validated_debug,
        provenance=tuple(sources.items()),
    )


def _validate_path_argument(key: str, value: object) -> None:
    if not isinstance(value, Path):
        raise ConfigurationError(
            "CONFIG_INVALID_TYPE",
            f"{key}はpathである必要があります",
        )


def _resolve_action_flag(key: str, value: bool | None) -> bool:
    if value is None:
        return False
    if type(value) is not bool:
        raise ConfigurationError(
            "CONFIG_INVALID_TYPE",
            f"{key}はbooleanである必要があります",
        )
    return value


def _parse_environment_value(key: str, value: str) -> object:
    """公開環境変数の文字列をcanonical設定型へ変換する。"""
    if key == "video_scan.workers":
        if value == "auto":
            return value
        try:
            return int(value)
        except ValueError:
            raise ConfigurationError(
                "CONFIG_INVALID_TYPE",
                "video_scan.workers環境変数はautoまたは整数である必要があります",
            ) from None
    if key == "video_scan.auto_max_workers":
        try:
            return int(value)
        except ValueError:
            raise ConfigurationError(
                "CONFIG_INVALID_TYPE",
                "video_scan.auto_max_workers環境変数は整数である必要があります",
            ) from None
    return value
