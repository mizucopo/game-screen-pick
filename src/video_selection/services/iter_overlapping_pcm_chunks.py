"""連続PCM sample gridからoverlap付きSTT chunkを構築する。"""

from collections import deque
from collections.abc import Iterable, Iterator
from fractions import Fraction
from itertools import islice

from ..models.pcm_audio_chunk import PcmAudioChunk


def iter_overlapping_pcm_chunks(
    source_chunks: Iterable[PcmAudioChunk],
    overlap_samples: int,
) -> Iterator[PcmAudioChunk]:
    """step幅のsource chunkへ後続sampleを加えたoverlap windowを返す。"""
    if overlap_samples < 0:
        msg = "overlap sample数は0以上である必要があります"
        raise ValueError(msg)
    iterator = iter(source_chunks)
    try:
        first = next(iterator)
    except StopIteration:
        return
    if overlap_samples == 0:
        yield first
        previous = first
        for chunk in iterator:
            _validate_continuity(previous, chunk)
            yield chunk
            previous = chunk
        return

    buffered = deque((first,))
    exhausted = False
    last_yielded_end: int | None = None
    while buffered:
        current = buffered[0]
        following_sample_count = sum(
            chunk.sample_count for chunk in islice(buffered, 1, None)
        )
        while following_sample_count < overlap_samples and not exhausted:
            try:
                following = next(iterator)
            except StopIteration:
                exhausted = True
                break
            _validate_continuity(buffered[-1], following)
            buffered.append(following)
            following_sample_count += following.sample_count

        included_sample_count = min(overlap_samples, following_sample_count)
        window_end = current.sample_start + current.sample_count + included_sample_count
        if last_yielded_end is None or window_end > last_yielded_end:
            yield _build_window(
                current,
                islice(buffered, 1, None),
                included_sample_count,
            )
            last_yielded_end = window_end
        buffered.popleft()


def _build_window(
    current: PcmAudioChunk,
    following_chunks: Iterable[PcmAudioChunk],
    included_sample_count: int,
) -> PcmAudioChunk:
    pcm_bytes = bytearray(current.pcm_bytes)
    remaining = included_sample_count
    for chunk in following_chunks:
        if remaining == 0:
            break
        used_sample_count = min(remaining, chunk.sample_count)
        pcm_bytes.extend(chunk.pcm_bytes[: used_sample_count * 2])
        remaining -= used_sample_count
    return PcmAudioChunk(
        stream_index=current.stream_index,
        sample_start=current.sample_start,
        sample_count=current.sample_count + included_sample_count,
        sample_rate=current.sample_rate,
        channel_count=1,
        sample_format="s16le",
        pts=current.pts,
        time_base=current.time_base,
        pcm_bytes=bytes(pcm_bytes),
    )


def _validate_continuity(
    current: PcmAudioChunk,
    following: PcmAudioChunk,
) -> None:
    if (
        following.stream_index != current.stream_index
        or following.sample_rate != current.sample_rate
        or following.sample_start != current.sample_start + current.sample_count
        or following.channel_count != 1
        or following.sample_format != "s16le"
    ):
        msg = "timestamp_drift"
        raise ValueError(msg)
    expected_time = current.pts * current.time_base + Fraction(
        current.sample_count,
        current.sample_rate,
    )
    actual_time = following.pts * following.time_base
    if abs(actual_time - expected_time) > Fraction(1, current.sample_rate):
        msg = "timestamp_drift"
        raise ValueError(msg)
