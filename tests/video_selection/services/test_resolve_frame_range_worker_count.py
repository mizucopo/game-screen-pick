"""Frame Range worker解決policyのtest。"""

import pytest

from src.video_selection.services.resolve_frame_range_worker_count import (
    resolve_frame_range_worker_count,
)


@pytest.mark.parametrize(
    ("range_count", "logical_cpu_count", "expected"),
    [
        (1, 24, 1),
        (8, 4, 1),
        (8, 8, 2),
        (8, 16, 4),
        (8, 64, 4),
    ],
)
def test_worker_count_uses_range_cpu_and_safe_limits(
    range_count: int,
    logical_cpu_count: int,
    expected: int,
) -> None:
    """range件数、CPU容量、safe capの最小値が選択されること。

    Arrange:
        - range件数とlogical CPU数が用意される
    Act:
        - Frame Range worker数が解決される
    Assert:
        - 三つの上限を超えないworker数が返されること
    """
    # Arrange
    # Act
    actual = resolve_frame_range_worker_count(
        range_count,
        logical_cpu_count=logical_cpu_count,
    )

    # Assert
    assert actual == expected


@pytest.mark.parametrize(
    ("range_count", "logical_cpu_count"),
    [(0, 4), (1, 0)],
)
def test_worker_count_rejects_non_positive_inputs(
    range_count: int,
    logical_cpu_count: int,
) -> None:
    """正でないrange件数またはCPU数が拒否されること。

    Arrange:
        - 正でない入力値が用意される
    Act:
        - worker解決が試行される
    Assert:
        - 明確な入力errorが返されること
    """
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError):
        resolve_frame_range_worker_count(
            range_count,
            logical_cpu_count=logical_cpu_count,
        )
