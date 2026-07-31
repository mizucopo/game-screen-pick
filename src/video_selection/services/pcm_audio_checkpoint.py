"""PCM extractionをcanonical sample range単位で再利用する。"""

from collections.abc import Callable, Iterator, Mapping
from fractions import Fraction
from pathlib import Path

from ..models.checkpoint_operation import CheckpointOperation
from ..models.durable_work_unit_bundle import DurableWorkUnitBundle
from ..models.pcm_audio_chunk import PcmAudioChunk
from ..protocols.run_observer import RunObserver
from .durable_work_unit_cache import DurableWorkUnitCache

PcmChunkExtractor = Callable[[int, int], PcmAudioChunk | None]

_SCHEMA = "game-screen-pick/pcm-audio-range@1.0.0"


class PcmAudioCheckpoint:
    """完了済みPCM sample rangeを再抽出せず順序どおり復元する。"""

    def __init__(
        self,
        cache_folder: Path,
        *,
        source_fingerprint: str,
        stream_index: int,
        sample_rate: int,
        frame_sample_count: int,
        extraction_semantic_input: Mapping[str, object],
        validate_source: Callable[[], None] | None = None,
        observer: RunObserver | None = None,
    ) -> None:
        if sample_rate < 1 or frame_sample_count < 1:
            msg = "PCM checkpointのsample設定は正の整数が必要です"
            raise ValueError(msg)
        self._stream_index = stream_index
        self._sample_rate = sample_rate
        self._frame_sample_count = frame_sample_count
        self._extraction_semantic_input = dict(extraction_semantic_input)
        self._validate_source = validate_source or _skip_validation
        self._cache = DurableWorkUnitCache(
            cache_folder,
            subject_fingerprint=source_fingerprint,
            operation=CheckpointOperation.PCM_AUDIO_CHUNK,
            observer=observer,
        )

    def resolve(self, extract: PcmChunkExtractor) -> Iterator[PcmAudioChunk]:
        """range hitを復元し、最初の未確定rangeだけを順次抽出する。"""
        sample_start = 0
        previous: PcmAudioChunk | None = None
        while True:
            semantic_input = self._semantic_input(sample_start)
            work_unit_key = (
                f"stream-{self._stream_index}-samples-"
                f"{sample_start}-{self._frame_sample_count}"
            )

            def produce(
                checkpoint_root: Path,
                sample_start: int = sample_start,
                previous: PcmAudioChunk | None = previous,
            ) -> dict[str, object]:
                return self._produce(
                    checkpoint_root,
                    extract,
                    sample_start,
                    previous,
                )

            def validate_bundle(
                value: DurableWorkUnitBundle,
                sample_start: int = sample_start,
                previous: PcmAudioChunk | None = previous,
            ) -> None:
                chunk = self._restore(value, sample_start)
                if chunk is not None:
                    _validate_chunk_contract(
                        chunk,
                        stream_index=self._stream_index,
                        sample_rate=self._sample_rate,
                        expected_sample_start=sample_start,
                        maximum_sample_count=self._frame_sample_count,
                        previous=previous,
                    )

            bundle, _reused = self._cache.resolve(
                work_unit_key,
                semantic_input,
                produce,
                validate_bundle=validate_bundle,
            )
            chunk = self._restore(bundle, sample_start)
            self._validate_source()
            if chunk is None:
                return
            _validate_chunk_contract(
                chunk,
                stream_index=self._stream_index,
                sample_rate=self._sample_rate,
                expected_sample_start=sample_start,
                maximum_sample_count=self._frame_sample_count,
                previous=previous,
            )
            yield chunk
            if chunk.sample_count < self._frame_sample_count:
                return
            sample_start += self._frame_sample_count
            previous = chunk

    def _produce(
        self,
        checkpoint_root: Path,
        extract: PcmChunkExtractor,
        sample_start: int,
        previous: PcmAudioChunk | None,
    ) -> dict[str, object]:
        """一つのrangeまたはEOF markerをsource検証後に確定する。"""
        chunk = extract(sample_start, self._frame_sample_count)
        if chunk is None:
            self._validate_source()
            return {
                "schema": _SCHEMA,
                "kind": "end",
                "stream_index": self._stream_index,
                "sample_start": sample_start,
                "maximum_sample_count": self._frame_sample_count,
                "sample_rate": self._sample_rate,
            }
        _validate_chunk_contract(
            chunk,
            stream_index=self._stream_index,
            sample_rate=self._sample_rate,
            expected_sample_start=sample_start,
            maximum_sample_count=self._frame_sample_count,
            previous=previous,
        )
        (checkpoint_root / "audio.pcm").write_bytes(chunk.pcm_bytes)
        self._validate_source()
        return {
            "schema": _SCHEMA,
            "kind": "chunk",
            "artifact_path": "audio.pcm",
            "stream_index": chunk.stream_index,
            "sample_start": chunk.sample_start,
            "sample_count": chunk.sample_count,
            "maximum_sample_count": self._frame_sample_count,
            "sample_rate": chunk.sample_rate,
            "channel_count": chunk.channel_count,
            "sample_format": chunk.sample_format,
            "pts": chunk.pts,
            "time_base": [chunk.time_base.numerator, chunk.time_base.denominator],
        }

    def _restore(
        self,
        bundle: DurableWorkUnitBundle,
        sample_start: int,
    ) -> PcmAudioChunk | None:
        artifact = bundle.artifact
        common_is_valid = (
            artifact.get("schema") == _SCHEMA
            and artifact.get("stream_index") == self._stream_index
            and artifact.get("sample_start") == sample_start
            and artifact.get("maximum_sample_count") == self._frame_sample_count
            and artifact.get("sample_rate") == self._sample_rate
        )
        if not common_is_valid:
            raise ValueError("PCM checkpoint artifactが不正です")
        if artifact.get("kind") == "end":
            return None
        if (
            artifact.get("kind") != "chunk"
            or artifact.get("artifact_path") != "audio.pcm"
            or artifact.get("channel_count") != 1
            or artifact.get("sample_format") != "s16le"
        ):
            raise ValueError("PCM checkpoint artifactが不正です")
        return PcmAudioChunk(
            stream_index=self._stream_index,
            sample_start=sample_start,
            sample_count=_integer(artifact.get("sample_count")),
            sample_rate=self._sample_rate,
            channel_count=1,
            sample_format="s16le",
            pts=_integer(artifact.get("pts")),
            time_base=_fraction(artifact.get("time_base")),
            pcm_bytes=(bundle.root / "audio.pcm").read_bytes(),
        )

    def _semantic_input(self, sample_start: int) -> dict[str, object]:
        return {
            "pcm_extraction_semantic_input": self._extraction_semantic_input,
            "stream_index": self._stream_index,
            "sample_rate": self._sample_rate,
            "sample_start": sample_start,
            "maximum_sample_count": self._frame_sample_count,
        }


def _validate_chunk_contract(
    chunk: PcmAudioChunk,
    *,
    stream_index: int,
    sample_rate: int,
    expected_sample_start: int,
    maximum_sample_count: int,
    previous: PcmAudioChunk | None,
) -> None:
    if (
        chunk.stream_index != stream_index
        or chunk.sample_rate != sample_rate
        or chunk.sample_start != expected_sample_start
        or chunk.sample_count > maximum_sample_count
        or chunk.channel_count != 1
        or chunk.sample_format != "s16le"
    ):
        raise ValueError("timestamp_drift")
    if previous is None:
        return
    expected_time = previous.pts * previous.time_base + Fraction(
        previous.sample_count,
        previous.sample_rate,
    )
    actual_time = chunk.pts * chunk.time_base
    if abs(actual_time - expected_time) > Fraction(1, sample_rate):
        raise ValueError("timestamp_drift")


def _fraction(value: object) -> Fraction:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )
        or value[1] == 0
    ):
        raise ValueError("PCM checkpoint time baseが不正です")
    return Fraction(value[0], value[1])


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("PCM checkpoint integerが不正です")
    return value


def _skip_validation() -> None:
    return
