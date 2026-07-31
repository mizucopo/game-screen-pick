"""public cutover前の動画入力internal CLI adapter。"""

from collections.abc import Callable, Mapping
from pathlib import Path

from ..models.effective_configuration import EffectiveConfiguration
from ..models.run_outcome import RunOutcome
from .resolve_effective_configuration import resolve_effective_configuration


def run_internal_cli_adapter(
    application_run: Callable[[EffectiveConfiguration], RunOutcome],
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
) -> RunOutcome:
    """設定解決後だけapplicationのrun境界を呼び出す。"""
    configuration = resolve_effective_configuration(
        video_input_folder=video_input_folder,
        output_folder=output_folder,
        config_path=config_path,
        image_count=image_count,
        recursive=recursive,
        scene_hint=scene_hint,
        spoiler_sensitivity=spoiler_sensitivity,
        similarity_threshold=similarity_threshold,
        video_scan_workers=video_scan_workers,
        video_scan_auto_max_workers=video_scan_auto_max_workers,
        ollama_host=ollama_host,
        reset_cache=reset_cache,
        debug=debug,
        environ=environ,
    )
    return application_run(configuration)
