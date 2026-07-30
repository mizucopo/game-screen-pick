"""EmptyVideoScanPartition modelのtest。"""

from fractions import Fraction

import pytest

from src.video_selection.models.empty_video_scan_partition import (
    EmptyVideoScanPartition,
)


def test_empty_partition_requires_a_valid_half_open_range() -> None:
    """終了PTSが開始PTS以下の空partitionが拒否されること。

    Arrange:
        - 開始PTSと同じ終了PTSを持つ空partition値が用意される
    Act:
        - EmptyVideoScanPartitionが構築される
    Assert:
        - 不正rangeとしてValueErrorが返されること
    """
    # Arrange

    # Act
    with pytest.raises(ValueError) as caught:
        EmptyVideoScanPartition(
            stream_index=0,
            start_pts=10,
            end_pts=10,
            time_base=Fraction(1, 10),
            wall_seconds=0.1,
            cpu_seconds=0.05,
            decode_pass_count=1,
        )

    # Assert
    assert "range" in str(caught.value)
