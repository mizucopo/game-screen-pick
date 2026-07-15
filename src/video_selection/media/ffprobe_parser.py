"""ffprobe JSONをMediaRuntimeのsemantic modelへ変換する。"""

from collections.abc import Mapping
from fractions import Fraction
from typing import cast

from ..models.media_probe import MediaProbe
from ..models.media_stream import MediaStream, MediaStreamKind

_SUPPORTED_STREAM_KINDS = frozenset(
    {"video", "audio", "subtitle", "data", "attachment"}
)


def parse_media_probe(document: object) -> MediaProbe:
    """ffprobe documentを検証してMediaProbeへ変換する。"""
    if not isinstance(document, Mapping):
        msg = "ffprobe resultはobjectである必要があります"
        raise ValueError(msg)
    raw_format = document.get("format")
    raw_streams = document.get("streams")
    if not isinstance(raw_format, Mapping) or not isinstance(raw_streams, list):
        msg = "ffprobe resultにformatとstreamsが必要です"
        raise ValueError(msg)
    format_name = raw_format.get("format_name")
    if not isinstance(format_name, str):
        msg = "ffprobe format_nameが不正です"
        raise ValueError(msg)
    return MediaProbe(
        format_names=tuple(format_name.split(",")),
        streams=tuple(_parse_stream(item) for item in raw_streams),
    )


def _parse_stream(raw_stream: object) -> MediaStream:
    if not isinstance(raw_stream, Mapping):
        msg = "ffprobe streamはobjectである必要があります"
        raise ValueError(msg)
    kind_value = raw_stream.get("codec_type")
    if kind_value not in _SUPPORTED_STREAM_KINDS:
        msg = "対応していないffprobe stream種別です"
        raise ValueError(msg)
    codec_name = raw_stream.get("codec_name")
    if not isinstance(codec_name, str):
        msg = "ffprobe codec_nameが不正です"
        raise ValueError(msg)
    tags = raw_stream.get("tags", {})
    disposition = raw_stream.get("disposition", {})
    if not isinstance(tags, Mapping) or not isinstance(disposition, Mapping):
        msg = "ffprobe stream metadataが不正です"
        raise ValueError(msg)
    language_value = tags.get("language")
    language = language_value if isinstance(language_value, str) else None
    return MediaStream(
        index=_required_int(raw_stream.get("index")),
        kind=cast(MediaStreamKind, kind_value),
        codec_name=codec_name,
        time_base=_optional_fraction(raw_stream.get("time_base")),
        start_pts=_optional_int(raw_stream.get("start_pts")),
        duration_ts=_optional_int(raw_stream.get("duration_ts")),
        width=_optional_int(raw_stream.get("width")),
        height=_optional_int(raw_stream.get("height")),
        sample_rate=_optional_int(raw_stream.get("sample_rate")),
        channels=_optional_int(raw_stream.get("channels")),
        language=language,
        is_default=disposition.get("default") == 1,
        is_forced=disposition.get("forced") == 1,
        is_attached_picture=disposition.get("attached_pic") == 1,
    )


def _required_int(value: object) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        msg = "ffprobe integer fieldが必要です"
        raise ValueError(msg)
    return parsed


def _optional_int(value: object) -> int | None:
    if value is None or value == "N/A":
        return None
    if isinstance(value, bool):
        msg = "ffprobe integer fieldが不正です"
        raise ValueError(msg)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as error:
            msg = "ffprobe integer fieldが不正です"
            raise ValueError(msg) from error
    msg = "ffprobe integer fieldが不正です"
    raise ValueError(msg)


def _optional_fraction(value: object) -> Fraction | None:
    if value is None or value in {"N/A", "0/0"}:
        return None
    if not isinstance(value, str):
        msg = "ffprobe time_baseが不正です"
        raise ValueError(msg)
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        msg = "ffprobe time_baseが不正です"
        raise ValueError(msg) from error
    if parsed <= 0:
        msg = "ffprobe time_baseは正である必要があります"
        raise ValueError(msg)
    return parsed
