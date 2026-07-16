"""Selected Image公開に必要なMediaRuntime port。"""

from pathlib import Path
from typing import Protocol

from ..models.decoded_video_frame import DecodedVideoFrame


class SelectedFrameMediaRuntime(Protocol):
    """exact PTSの元解像度frame再抽出だけを公開する境界。"""

    def extract_video_frame(
        self,
        media_path: Path,
        stream_index: int,
        pts: int,
        max_dimension: int,
    ) -> DecodedVideoFrame:
        """指定source PTSの一つのRGB24 frameを返す。"""
