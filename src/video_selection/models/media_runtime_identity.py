"""system media toolの解決済みidentity。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MediaRuntimeIdentity:
    """同一buildとして検証されたFFmpeg/ffprobeの解決済みidentity。"""

    ffmpeg_version: str
    ffprobe_version: str
    build_capability_sha256: str

    def __post_init__(self) -> None:
        """build/capability digestが完全SHA-256であることを検証する。"""
        if len(self.build_capability_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.build_capability_sha256
        ):
            msg = "Media Runtime Identityには完全SHA-256が必要です"
            raise ValueError(msg)
