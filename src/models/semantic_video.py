"""動画理解のchunk条件と選定候補の根拠."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SemanticVideoOptions:
    """全編を理解するための重複chunk条件."""

    chunk_seconds: float = 30.0
    overlap_seconds: float = 2.0

    def __post_init__(self) -> None:
        """chunkが有限で、次のchunkへ必ず進むことを確認する."""
        values = (self.chunk_seconds, self.overlap_seconds)
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or (isinstance(value, float) and not math.isfinite(value))
                for value in values
            )
            or not 0 <= self.overlap_seconds < self.chunk_seconds
        ):
            raise ValueError(
                "動画理解は有限の chunk_seconds > overlap_seconds >= 0 が必要です"
            )
        if not 1 <= self.chunk_seconds <= 120:
            raise ValueError("動画理解のchunk_secondsは1から120秒で指定してください")
        if self.chunk_seconds - self.overlap_seconds < 1:
            raise ValueError("動画理解のchunkは1秒以上ずつ進む必要があります")


@dataclass(frozen=True)
class SemanticVideoPlan:
    """動画内容から導いた候補時刻と原動画時刻での根拠."""

    timestamps: tuple[float, ...]
    cache_key: str
    evidence: dict[str, Any]

    def provenance(self, timestamp: float) -> list[dict[str, Any]]:
        """候補時刻を含む、動画理解が返したeventを返す."""
        return [
            dict(event)
            for event in self.evidence["events"]
            if event["start_seconds"] <= timestamp <= event["end_seconds"]
        ]

    def importance(self, timestamp: float) -> float:
        """該当eventの最大importanceを0から100で返す."""
        return max(
            (float(event["importance"]) for event in self.provenance(timestamp)),
            default=0.0,
        )
