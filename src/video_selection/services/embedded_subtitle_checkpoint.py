"""Embedded Subtitle抽出を選択stream単位で再利用する。"""

from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import cast

from ..models.checkpoint_operation import CheckpointOperation
from ..models.embedded_subtitle import EmbeddedSubtitle
from ..protocols.run_observer import RunObserver
from .durable_work_unit_cache import DurableWorkUnitCache

_SCHEMA = "game-screen-pick/embedded-subtitle-stream@1.0.0"


class EmbeddedSubtitleCheckpoint:
    """完了済みの選択subtitle streamを再抽出せず復元する。"""

    def __init__(
        self,
        cache_folder: Path,
        *,
        source_fingerprint: str,
        stream_index: int,
        extraction_semantic_input: Mapping[str, object],
        validate_source: Callable[[], None] | None = None,
        observer: RunObserver | None = None,
    ) -> None:
        self._stream_index = stream_index
        self._semantic_input = {
            "subtitle_extraction_semantic_input": dict(extraction_semantic_input),
            "stream_index": stream_index,
        }
        self._validate_source = validate_source or _skip_validation
        self._cache = DurableWorkUnitCache(
            cache_folder,
            subject_fingerprint=source_fingerprint,
            operation=CheckpointOperation.EMBEDDED_SUBTITLE_STREAM,
            observer=observer,
        )

    def resolve(
        self,
        extract: Callable[[], tuple[EmbeddedSubtitle, ...]],
    ) -> tuple[EmbeddedSubtitle, ...]:
        """stream checkpointを復元し、miss時だけMedia Runtimeを呼ぶ。"""
        work_unit_key = f"stream-{self._stream_index}"
        bundle, _reused = self._cache.resolve(
            work_unit_key,
            self._semantic_input,
            lambda _checkpoint_root: self._produce(extract),
            validate_bundle=lambda value: _restore_events(
                value.artifact,
                self._stream_index,
            ),
        )
        events = _restore_events(bundle.artifact, self._stream_index)
        self._validate_source()
        return events

    def _produce(
        self,
        extract: Callable[[], tuple[EmbeddedSubtitle, ...]],
    ) -> dict[str, object]:
        events = extract()
        if any(event.stream_index != self._stream_index for event in events):
            raise ValueError("Embedded Subtitle streamが選択streamと一致しません")
        self._validate_source()
        return {
            "schema": _SCHEMA,
            "stream_index": self._stream_index,
            "events": [
                {
                    "pts": event.pts,
                    "duration_ts": event.duration_ts,
                    "time_base": [
                        event.time_base.numerator,
                        event.time_base.denominator,
                    ],
                    "text": event.text,
                }
                for event in events
            ],
        }


def _restore_events(
    artifact: Mapping[str, object],
    stream_index: int,
) -> tuple[EmbeddedSubtitle, ...]:
    if (
        artifact.get("schema") != _SCHEMA
        or artifact.get("stream_index") != stream_index
    ):
        raise ValueError("Embedded Subtitle checkpoint artifactが不正です")
    values = artifact.get("events")
    if not isinstance(values, list) or not all(
        isinstance(item, dict) and all(isinstance(key, str) for key in item)
        for item in values
    ):
        raise ValueError("Embedded Subtitle checkpoint event列が不正です")
    events = tuple(
        EmbeddedSubtitle(
            stream_index=stream_index,
            pts=_integer(value.get("pts")),
            duration_ts=_integer(value.get("duration_ts")),
            time_base=_fraction(value.get("time_base")),
            text=_string(value.get("text")),
        )
        for value in cast(list[dict[str, object]], values)
    )
    if any(
        current.pts < previous.pts
        for previous, current in zip(events, events[1:], strict=False)
    ):
        raise ValueError("Embedded Subtitle checkpoint event順が不正です")
    return events


def _fraction(value: object) -> Fraction:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )
        or value[1] == 0
    ):
        raise ValueError("Embedded Subtitle checkpoint time baseが不正です")
    return Fraction(value[0], value[1])


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Embedded Subtitle checkpoint integerが不正です")
    return value


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Embedded Subtitle checkpoint textが不正です")
    return value


def _skip_validation() -> None:
    return
