"""Refinement Window Groupの共有resource worker数を解決する。"""

from fractions import Fraction

from .resolve_frame_range_worker_count import resolve_frame_range_worker_count

_MAX_REFINEMENT_DIMENSION = 960
_ESTIMATED_MAX_FRAMES_PER_SECOND = 240
_RETAINED_FRAME_BYTES_PER_PIXEL = 4
_AVAILABLE_MEMORY_BUDGET_DIVISOR = 4


def resolve_refinement_group_worker_count(
    pts_ranges: tuple[tuple[int, int], ...],
    *,
    time_base: Fraction,
    source_width: int | None,
    source_height: int | None,
    logical_cpu_count: int,
    available_memory_bytes: int | None,
) -> int:
    """Group、CPU、RGB保持memoryからbounded worker上限を返す。"""
    _validate_inputs(
        pts_ranges,
        time_base,
        source_width,
        source_height,
        logical_cpu_count,
        available_memory_bytes,
    )
    cpu_limit = resolve_frame_range_worker_count(
        len(pts_ranges),
        logical_cpu_count=logical_cpu_count,
    )
    if available_memory_bytes is None or source_width is None or source_height is None:
        return 1
    memory_budget = available_memory_bytes // _AVAILABLE_MEMORY_BUDGET_DIVISOR
    group_bytes = sorted(
        (
            _estimated_group_bytes(
                start_pts,
                end_pts,
                time_base,
                source_width,
                source_height,
            )
            for start_pts, end_pts in pts_ranges
        ),
        reverse=True,
    )
    memory_limit = 0
    reserved_bytes = 0
    for estimated_bytes in group_bytes:
        if reserved_bytes + estimated_bytes > memory_budget:
            break
        reserved_bytes += estimated_bytes
        memory_limit += 1
    return min(cpu_limit, max(1, memory_limit))


def _estimated_group_bytes(
    start_pts: int,
    end_pts: int,
    time_base: Fraction,
    source_width: int,
    source_height: int,
) -> int:
    """一Groupが保持し得るscaled frameの基礎memoryを保守的に返す。"""
    width, height = _scaled_dimensions(source_width, source_height)
    duration = (end_pts - start_pts) * time_base
    estimated_frame_count = max(
        1,
        _ceiling(duration * _ESTIMATED_MAX_FRAMES_PER_SECOND),
    )
    return width * height * _RETAINED_FRAME_BYTES_PER_PIXEL * estimated_frame_count


def _scaled_dimensions(width: int, height: int) -> tuple[int, int]:
    """960px上限で偶数へ切り上げた保守的な寸法を返す。"""
    longest = max(width, height)
    if longest <= _MAX_REFINEMENT_DIMENSION:
        return (_even_ceiling(width), _even_ceiling(height))
    scale = Fraction(_MAX_REFINEMENT_DIMENSION, longest)
    return (
        _even_ceiling(_ceiling(width * scale)),
        _even_ceiling(_ceiling(height * scale)),
    )


def _ceiling(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _even_ceiling(value: int) -> int:
    return max(2, value + value % 2)


def _validate_inputs(
    pts_ranges: tuple[tuple[int, int], ...],
    time_base: Fraction,
    source_width: int | None,
    source_height: int | None,
    logical_cpu_count: int,
    available_memory_bytes: int | None,
) -> None:
    if not pts_ranges or any(start >= end for start, end in pts_ranges):
        raise ValueError("Refinement Group PTS rangeが不正です")
    if time_base <= 0 or logical_cpu_count < 1:
        raise ValueError("Refinement Groupのtime baseとCPU数が不正です")
    if (source_width is None) != (source_height is None) or (
        source_width is not None
        and source_height is not None
        and (source_width < 1 or source_height < 1)
    ):
        raise ValueError("Refinement Groupのsource寸法が不正です")
    if available_memory_bytes is not None and available_memory_bytes < 0:
        raise ValueError("available memoryは非負である必要があります")
