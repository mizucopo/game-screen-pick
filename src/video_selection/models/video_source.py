"""Video Set内の一つのVideo Source。"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class VideoSource:
    """content identityと発見時filesystem snapshotを持つvideo。"""

    path: Path
    relative_path: str
    fingerprint: str
    size_bytes: int
    modified_at_ns: int
    device: int
    inode: int

    def __post_init__(self) -> None:
        """安全な相対pathとSHA-256 fingerprintを検証する。"""
        relative_path = PurePosixPath(self.relative_path)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or self.relative_path != relative_path.as_posix()
        ):
            msg = "Video Sourceには正規化済み相対pathが必要です"
            raise ValueError(msg)
        if len(self.fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.fingerprint
        ):
            msg = "Video Fingerprintには64桁のSHA-256が必要です"
            raise ValueError(msg)

    @property
    def stat_signature(self) -> tuple[int, int, int, int]:
        """実行中の内容変更検知に使う発見時statを返す。"""
        return (self.device, self.inode, self.size_bytes, self.modified_at_ns)
