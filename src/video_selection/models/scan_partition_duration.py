"""Video Scan partition構築用のduration。"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ScanPartitionDuration:
    """stream tickとその終端精度を一体で保持する。"""

    duration_ts: int
    is_exact: bool

    def __post_init__(self) -> None:
        """durationが正のstream tickであることを検証する。"""
        if self.duration_ts <= 0:
            raise ValueError("Scan Partition durationには正のtick数が必要です")

    @property
    def source(self) -> Literal["stream", "container"]:
        """cache provenance用のduration sourceを返す。"""
        return "stream" if self.is_exact else "container"
