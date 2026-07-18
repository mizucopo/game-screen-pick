"""Video Stageが必要とするMediaRuntime port。"""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from ..models.decoded_video_frame import DecodedVideoFrame
from ..models.embedded_subtitle import EmbeddedSubtitle
from ..models.media_probe import MediaProbe
from ..models.media_runtime_identity import MediaRuntimeIdentity
from ..models.media_stream import MediaStream
from ..models.native_video_scan import NativeVideoScan
from ..models.pcm_audio_chunk import PcmAudioChunk


class VideoStageMediaRuntime(Protocol):
    """Video ScanとFrame Refinementに限定したexternal media境界。"""

    def preflight(self) -> MediaRuntimeIdentity:
        """system media runtime identityを解決する。"""

    def probe(self, media_path: Path) -> MediaProbe:
        """media stream metadataを返す。"""

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
        """一回のdecodeからVideo Scan artifactを返す。"""

    def cancel_video_scans(self) -> None:
        """実行中のVideo Scan subprocessを終了させる。"""

    def scan_video_frame_ranges(
        self,
        media_path: Path,
        stream_index: int,
        pts_ranges: tuple[tuple[int, int], ...],
        max_dimension: int,
    ) -> Iterator[DecodedVideoFrame]:
        """refinement range内のnative frameを返す。"""

    def write_mjpeg_proxy(
        self,
        frame: DecodedVideoFrame,
        output_path: Path,
        *,
        quality: int,
    ) -> None:
        """選抜済みframeをMJPEG proxyへ保存する。"""

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
