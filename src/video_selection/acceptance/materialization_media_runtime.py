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
