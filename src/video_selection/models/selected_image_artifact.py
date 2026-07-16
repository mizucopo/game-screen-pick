"""staging済みSelected Image artifact。"""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

_FRAME_ID = re.compile(r"frm_[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class SelectedImageArtifact:
    """選択frameと公開WebPのidentity、path、内容診断を結ぶ。"""

    image_id: str
    relative_path: str
    sha256: str
    width: int
    height: int
    size_bytes: int

    def __post_init__(self) -> None:
        """stable ID、relative WebP path、hash、寸法を検証する。"""
        path = PurePosixPath(self.relative_path)
        if (
            _FRAME_ID.fullmatch(self.image_id) is None
            or _SHA256.fullmatch(self.sha256) is None
            or path.is_absolute()
            or ".." in path.parts
            or path.parts[:1] != ("images",)
            or path.suffix != ".webp"
            or self.relative_path != path.as_posix()
            or self.width < 1
            or self.height < 1
            or self.size_bytes < 1
        ):
            msg = "Selected Image artifactのID、path、hash、寸法が不正です"
            raise ValueError(msg)
