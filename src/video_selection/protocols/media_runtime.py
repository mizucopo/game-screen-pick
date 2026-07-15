"""MediaRuntimeのsemantic port。"""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..models.decoded_video_frame import DecodedVideoFrame
from ..models.embedded_subtitle import EmbeddedSubtitle
from ..models.media_probe import MediaProbe
from ..models.media_runtime_identity import MediaRuntimeIdentity
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

    def extract_video_frame(
        self,
        media_path: Path,
        stream_index: int,
        pts: int,
        max_dimension: int,
    ) -> DecodedVideoFrame:
        """指定source PTSの一つのframeを返す。"""

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
