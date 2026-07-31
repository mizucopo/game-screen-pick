"""suite materializationをsystem Media Runtime Identityへ固定する。"""

from collections.abc import Callable

from ..media.ffmpeg_media_runtime import FfmpegMediaRuntime
from ..models.media_runtime_identity import MediaRuntimeIdentity

MediaRuntimeProbe = Callable[[], MediaRuntimeIdentity]


def probe_media_runtime_identity() -> MediaRuntimeIdentity:
    """現在のsystem FFmpeg/ffprobe identityを返す。"""
    return FfmpegMediaRuntime().preflight()


def media_runtime_identity_record(
    identity: MediaRuntimeIdentity,
) -> dict[str, str]:
    """materialization manifest用のprivacy-safe identityを返す。"""
    return {
        "ffmpeg_version": identity.ffmpeg_version,
        "ffprobe_version": identity.ffprobe_version,
        "build_capability_sha256": identity.build_capability_sha256,
    }


def parse_media_runtime_identity_record(
    value: object,
) -> dict[str, str] | None:
    """checkpointのMedia Runtime Identityを完全な形だけ復元する。"""
    if not isinstance(value, dict) or set(value) != {
        "ffmpeg_version",
        "ffprobe_version",
        "build_capability_sha256",
    }:
        return None
    ffmpeg_version = value.get("ffmpeg_version")
    ffprobe_version = value.get("ffprobe_version")
    build_digest = value.get("build_capability_sha256")
    if (
        not isinstance(ffmpeg_version, str)
        or not ffmpeg_version
        or not isinstance(ffprobe_version, str)
        or not ffprobe_version
        or not isinstance(build_digest, str)
        or len(build_digest) != 64
        or any(character not in "0123456789abcdef" for character in build_digest)
    ):
        return None
    return {
        "ffmpeg_version": ffmpeg_version,
        "ffprobe_version": ffprobe_version,
        "build_capability_sha256": build_digest,
    }
