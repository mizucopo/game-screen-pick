"""Selected Image Encoding v1。"""

import hashlib
from pathlib import Path

from PIL import Image

from ..models.decoded_video_frame import DecodedVideoFrame
from ..models.selected_image_artifact import SelectedImageArtifact

SELECTED_WEBP_QUALITY = 95


def encode_selected_webp(
    image_id: str,
    frame: DecodedVideoFrame,
    output_path: Path,
    relative_path: str,
) -> SelectedImageArtifact:
    """RGB24 frameを元寸法のmetadataなしlossy WebPへ保存する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.frombytes("RGB", (frame.width, frame.height), frame.pixels) as image:
        image.save(
            output_path,
            format="WEBP",
            lossless=False,
            quality=SELECTED_WEBP_QUALITY,
            method=6,
            exif=b"",
            icc_profile=None,
            xmp=b"",
        )
    content = output_path.read_bytes()
    return SelectedImageArtifact(
        image_id=image_id,
        relative_path=relative_path,
        sha256=hashlib.sha256(content).hexdigest(),
        width=frame.width,
        height=frame.height,
        size_bytes=len(content),
    )
