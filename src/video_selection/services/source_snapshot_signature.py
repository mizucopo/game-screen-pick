"""Video Sourceの軽量snapshot signature。"""

import os

SourceSnapshotSignature = tuple[int, int]


def source_snapshot_signature(stat: os.stat_result) -> SourceSnapshotSignature:
    """run間とrun中の照合に使うsizeとmtimeを返す。"""
    return stat.st_size, stat.st_mtime_ns
