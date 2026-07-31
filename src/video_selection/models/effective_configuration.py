"""Video Set選定のEffective Configuration。"""

from dataclasses import dataclass, field
from pathlib import Path

from ..configuration.configuration_source import ConfigurationSource


@dataclass(frozen=True)
class EffectiveConfiguration:
    """source別解決とvalidationが完了した一回の実行設定。"""

    video_input_folder: Path
    output_folder: Path
    image_count: int = 100
    config_version: str = "1.0.0"
    recursive: bool = False
    scene_hint: str | None = None
    spoiler_sensitivity: str = "medium"
    similarity_threshold: float = 0.72
    heartbeat_interval_seconds: float = 1.0
    scene_change_threshold: float = 0.25
    scene_min_interval_seconds: float = 0.5
    decode_backend: str = "cpu"
    refinement_radius_seconds: float = 1.0
    max_frame_candidates: int = 3
    video_scan_workers: str | int = "auto"
    video_scan_auto_max_workers: int = 6
    candidate_density_per_minute: float = 2.0
    language: str = "ja"
    subtitle_stream_index: int | None = None
    audio_stream_index: int | None = None
    ollama_host: str = field(default="http://localhost:11434", repr=False)
    ollama_timeout_seconds: float = 60.0
    ollama_max_parallel_requests: int = 1
    models_auto_upgrade: bool = True
    scene_catalog_model: str = "qwen3-vl:8b-instruct"
    scene_catalog_num_ctx: int = 32768
    candidate_annotation_model: str = "qwen3-vl:8b-instruct"
    candidate_annotation_num_ctx: int = 32768
    speech_to_text_model: str = "dropbox-dash/faster-whisper-large-v3-turbo"
    speech_to_text_device: str = "cuda"
    speech_to_text_compute_type: str = "float16"
    speech_to_text_beam_size: int = 5
    speech_vad_filter: bool = True
    speech_chunk_seconds: float = 600.0
    speech_overlap_seconds: float = 5.0
    reset_cache: bool = False
    debug: bool = False
    video_identity_cache_folder: Path | None = field(
        default=None,
        repr=False,
    )
    provenance: tuple[tuple[str, ConfigurationSource], ...] = field(
        default=(),
        repr=False,
    )

    def __post_init__(self) -> None:
        """要求画像枚数を検証する。"""
        if self.image_count < 1:
            msg = "image_countは正の整数である必要があります"
            raise ValueError(msg)

    @property
    def processing_cache_folder(self) -> Path:
        """Video Input Folderが所有するprocessing cacheを返す。"""
        return self.video_input_folder / ".game-screen-pick" / "cache"

    @property
    def durable_video_identity_cache_folder(self) -> Path:
        """Processing Stage cacheと寿命を分離したidentity cacheを返す。"""
        return self.video_identity_cache_folder or (
            self.video_input_folder / ".game-screen-pick" / "video-identities"
        )

    def source_for(self, key: str) -> ConfigurationSource:
        """canonical keyに採用された設定sourceを返す。"""
        for configured_key, source in self.provenance:
            if configured_key == key:
                return source
        raise KeyError(key)
