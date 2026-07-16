"""Selected Image公開に必要なMediaRuntime port。"""

from pathlib import Path
from typing import Protocol

from ..models.decoded_video_frame import DecodedVideoFrame


class SelectedFrameMediaRuntime(Protocol):
    """exact PTSの元解像度frame再抽出だけを公開する境界。"""

    def extract_original_video_frame(
        self,
        media_path: Path,
        stream_index: int,
        pts: int,
    ) -> DecodedVideoFrame:
        """指定source PTSの一つの元寸法RGB24 frameを返す。"""
