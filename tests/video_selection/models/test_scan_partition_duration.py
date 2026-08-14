"""ScanPartitionDurationのtest。"""

import pytest

from src.video_selection.models.scan_partition_duration import ScanPartitionDuration


@pytest.mark.parametrize(
    ("is_exact", "expected_source"),
    [
        pytest.param(True, "stream", id="exact-stream"),
        pytest.param(False, "container", id="approximate-container"),
    ],
)
def test_duration_source_is_derived_from_precision(
    is_exact: bool,
    expected_source: str,
) -> None:
    """durationの精度からcache provenance sourceが導出されること。

    Arrange:
        - exactまたは近似のdurationが用意される
    Act:
        - cache provenance用のsourceが取得される
    Assert:
        - exactはstream、近似はcontainerとして返されること
    """
    # Arrange
    duration = ScanPartitionDuration(duration_ts=900, is_exact=is_exact)

    # Act
    source = duration.source

    # Assert
    assert source == expected_source


def test_non_positive_duration_is_rejected() -> None:
    """0 tickのdurationが拒否されること。

    Arrange:
        - 0 tickのduration値が用意される
    Act:
        - ScanPartitionDurationが構築される
    Assert:
        - ValueErrorが送出されること
    """
    # Arrange
    duration_ts = 0

    # Act
    with pytest.raises(ValueError) as captured:
        ScanPartitionDuration(duration_ts=duration_ts, is_exact=True)

    # Assert
    assert "正のtick数" in str(captured.value)
