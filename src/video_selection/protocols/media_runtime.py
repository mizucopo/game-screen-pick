"""MediaRuntimeのsemantic port。"""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..models.decoded_video_frame import DecodedVideoFrame
from ..models.embedded_subtitle import EmbeddedSubtitle
from ..models.media_probe import MediaProbe
from ..models.media_runtime_identity import MediaRuntimeIdentity
from ..models.media_stream import MediaStream
from ..models.native_video_scan import NativeVideoScan
from ..models.pcm_audio_chunk import PcmAudioChunk


@runtime_checkable
class MediaRuntime(Protocol):
    """external process detailを隠すsemantic media operation境界。"""

    def preflight(self) -> MediaRuntimeIdentity:
        """system media toolsのversionと能力を検証する。"""

    def probe(self, media_path: Path) -> MediaProbe:
        """containerとordered stream metadataを返す。"""

    def scan_video_frames(
        self,
        media_path: Path,
        stream_index: int,
        max_dimension: int,
    ) -> Iterator[DecodedVideoFrame]:
        """一回のdecodeからnative PTS順にproxy frameを返す。"""

    def scan_video_frame_ranges(
        self,
        media_path: Path,
        stream_index: int,
        pts_ranges: tuple[tuple[int, int], ...],
        max_dimension: int,
    ) -> Iterator[DecodedVideoFrame]:
        """複数の半開PTS range内にあるnative frameを一回で返す。"""

    def scan_video(
        self,
        media_path: Path,
        stream: MediaStream,
        artifact_folder: Path,
        *,
        heartbeat_interval_seconds: float,
        scene_change_threshold: float,
        scene_min_interval_seconds: float,
        decode_backend: str,
    ) -> NativeVideoScan:
        """一回のdecodeからheartbeat、scene、timeline端点を返す。"""

    def extract_video_frame(
        self,
        media_path: Path,
        stream_index: int,
        pts: int,
        max_dimension: int,
    ) -> DecodedVideoFrame:
        """指定source PTSの一つのframeを返す。"""

    def extract_original_video_frame(
        self,
        media_path: Path,
        stream_index: int,
        pts: int,
    ) -> DecodedVideoFrame:
        """指定source PTSの一つの元寸法frameを返す。"""

    def write_mjpeg_proxy(
        self,
        frame: DecodedVideoFrame,
        output_path: Path,
        *,
        quality: int,
    ) -> None:
        """RGB frameをmetadataなしMJPEG proxyへ保存する。"""

    def scan_pcm_audio(
        self,
        media_path: Path,
        stream_index: int,
        sample_rate: int,
        frame_sample_count: int,
    ) -> Iterator[PcmAudioChunk]:
        """選択audioを連続PCM sample gridとして返す。"""

    def read_embedded_subtitles(
        self,
        media_path: Path,
        stream_index: int,
    ) -> tuple[EmbeddedSubtitle, ...]:
        """選択text subtitleの元packet timingと本文を返す。"""
