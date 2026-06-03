"""application実行リクエスト."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ApplicationRunRequest:
    """CLIからapplication実行層へ渡すリクエスト."""

    num: int
    similarity: float | None
    recursive: bool
    profile: str | None
    config_path: str | None
    ollama_model: str | None
    ollama_host: str | None
    ollama_timeout: float | None
    ollama_max_workers: int | None
    ollama_cache_enabled: bool
    scene_hint: str | None
    report_json: str | None
    rename: bool
    batch_size: int | None
    result_max_workers: int | None
    max_dim: int
    max_memory_gb: int
    debug: bool
    input_dir: str
    output_dir: str
