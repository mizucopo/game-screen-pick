"""Native Video Scan partitionをprivate checkpointへ変換する。"""

import stat
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import cast

from PIL import Image, UnidentifiedImageError

from ..models.empty_video_scan_partition import EmptyVideoScanPartition
from ..models.native_video_scan import NativeVideoScan
from ..models.scanned_video_frame import ScannedVideoFrame

_SCHEMA = "game-screen-pick/video-scan-partition@2.0.0"


def serialize_video_scan_partition(
    scan: NativeVideoScan | EmptyVideoScanPartition,
    checkpoint_root: Path,
) -> dict[str, object]:
    """一つのscan partitionをpath非依存artifactへ変換する。"""
    metrics = {
        "wall_seconds": scan.wall_seconds,
        "cpu_seconds": scan.cpu_seconds,
        "decode_pass_count": scan.decode_pass_count,
    }
    if isinstance(scan, EmptyVideoScanPartition):
        return {
            "schema": _SCHEMA,
            "status": "empty",
            "stream_index": scan.stream_index,
            "start_pts": scan.start_pts,
            "end_pts": scan.end_pts,
            "time_base": _fraction_value(scan.time_base),
            "metrics": metrics,
        }
    return {
        "schema": _SCHEMA,
        "status": "frames",
        "stream_index": scan.stream_index,
        "origin_pts": scan.origin_pts,
        "last_frame_pts": scan.last_frame_pts,
        "last_frame_duration_ts": scan.last_frame_duration_ts,
        "minimum_frame_delta_ts": scan.minimum_frame_delta_ts,
        "maximum_frame_count_per_pts": scan.maximum_frame_count_per_pts,
        "time_base": _fraction_value(scan.time_base),
        "heartbeats": [
            _frame_value(frame, checkpoint_root) for frame in scan.heartbeats
        ],
        "scene_frames": [
            _frame_value(frame, checkpoint_root) for frame in scan.scene_frames
        ],
        "metrics": metrics,
    }


def restore_video_scan_partition(
    artifact: Mapping[str, object],
    checkpoint_root: Path,
) -> NativeVideoScan | EmptyVideoScanPartition:
    """検証済みcheckpointから一つのscan partitionを復元する。"""
    if artifact.get("schema") != _SCHEMA:
        msg = "Video Scan partition artifact schemaが不正です"
        raise ValueError(msg)
    metrics = _mapping(artifact.get("metrics"))
    status = _string(artifact.get("status"))
    if status == "empty":
        return EmptyVideoScanPartition(
            stream_index=_integer(artifact.get("stream_index")),
            start_pts=_integer(artifact.get("start_pts")),
            end_pts=_optional_integer(artifact.get("end_pts")),
            time_base=_fraction(artifact.get("time_base")),
            wall_seconds=_number(metrics.get("wall_seconds")),
            cpu_seconds=_number(metrics.get("cpu_seconds")),
            decode_pass_count=_integer(metrics.get("decode_pass_count")),
        )
    if status != "frames":
        msg = "Video Scan partition artifact statusが不正です"
        raise ValueError(msg)
    return NativeVideoScan(
        stream_index=_integer(artifact.get("stream_index")),
        origin_pts=_integer(artifact.get("origin_pts")),
        last_frame_pts=_integer(artifact.get("last_frame_pts")),
        last_frame_duration_ts=_optional_integer(
            artifact.get("last_frame_duration_ts")
        ),
        minimum_frame_delta_ts=_optional_integer(
            artifact.get("minimum_frame_delta_ts")
        ),
        maximum_frame_count_per_pts=_optional_integer(
            artifact.get("maximum_frame_count_per_pts")
        ),
        time_base=_fraction(artifact.get("time_base")),
        heartbeats=tuple(
            _restore_frame(item, checkpoint_root)
            for item in _mapping_list(artifact.get("heartbeats"))
        ),
        scene_frames=tuple(
            _restore_frame(item, checkpoint_root)
            for item in _mapping_list(artifact.get("scene_frames"))
        ),
        wall_seconds=_number(metrics.get("wall_seconds")),
        cpu_seconds=_number(metrics.get("cpu_seconds")),
        decode_pass_count=_integer(metrics.get("decode_pass_count")),
    )


def _frame_value(frame: ScannedVideoFrame, root: Path) -> dict[str, object]:
    return {
        "source_pts": frame.source_pts,
        "duration_ts": frame.duration_ts,
        "time_base": _fraction_value(frame.time_base),
        "width": frame.width,
        "height": frame.height,
        "image_path": _relative_path(frame.image_path, root),
    }


def _restore_frame(
    value: Mapping[str, object],
    root: Path,
) -> ScannedVideoFrame:
    frame = ScannedVideoFrame(
        source_pts=_integer(value.get("source_pts")),
        duration_ts=_optional_integer(value.get("duration_ts")),
        time_base=_fraction(value.get("time_base")),
        width=_integer(value.get("width")),
        height=_integer(value.get("height")),
        image_path=_artifact_path(root, value.get("image_path")),
    )
    _validate_jpeg_proxy(frame.image_path, frame.width, frame.height)
    return frame


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        msg = "Video Scan partition proxyはcheckpoint root配下に必要です"
        raise ValueError(msg) from error


def _artifact_path(root: Path, value: object) -> Path:
    relative = PurePosixPath(_string(value))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        msg = "Video Scan partition proxy pathが不正です"
        raise ValueError(msg)
    path = root.joinpath(*relative.parts)
    if not _is_regular_file(path):
        msg = "Video Scan partition proxy artifactがありません"
        raise ValueError(msg)
    return path


def _fraction_value(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _fraction(value: object) -> Fraction:
    items = _list(value)
    if len(items) != 2:
        msg = "Video Scan partition time baseが不正です"
        raise ValueError(msg)
    denominator = _integer(items[1])
    if denominator == 0:
        msg = "Video Scan partition time base denominatorは0にできません"
        raise ValueError(msg)
    return Fraction(_integer(items[0]), denominator)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        msg = "Video Scan partition artifact objectが不正です"
        raise ValueError(msg)
    return cast(Mapping[str, object], value)


def _mapping_list(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(item) for item in _list(value))


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        msg = "Video Scan partition artifact listが不正です"
        raise ValueError(msg)
    return value


def _string(value: object) -> str:
    if not isinstance(value, str):
        msg = "Video Scan partition artifact stringが不正です"
        raise ValueError(msg)
    return value


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = "Video Scan partition artifact integerが不正です"
        raise ValueError(msg)
    return value


def _optional_integer(value: object) -> int | None:
    return None if value is None else _integer(value)


def _number(value: object) -> float:
    if type(value) not in {int, float}:
        msg = "Video Scan partition artifact numberが不正です"
        raise ValueError(msg)
    return float(cast(int | float, value))


def _validate_jpeg_proxy(path: Path, width: int, height: int) -> None:
    """checkpoint proxyが記録寸法の完全なJPEGか検証する。"""
    try:
        with Image.open(path) as image:
            valid = image.format == "JPEG" and image.size == (width, height)
            image.verify()
    except PermissionError:
        raise
    except (OSError, UnidentifiedImageError):
        valid = False
    if not valid:
        raise ValueError("Video Scan partition proxy JPEGが不正です")


def _is_regular_file(path: Path) -> bool:
    """欠損だけをFalseとし、access failureをcorruptionへ変換しない。"""
    try:
        mode = path.lstat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return False
    return stat.S_ISREG(mode)
