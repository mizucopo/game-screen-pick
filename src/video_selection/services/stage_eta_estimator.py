"""Comparable Work SeriesからStage ETAを求める。"""

import math
from statistics import fmean
from typing import Literal

from ..models.processing_stage import ProcessingStage
from ..models.progress_event import EstimationState

WorkDisposition = Literal["reuse", "recompute"]
EtaEstimate = tuple[EstimationState, float | None]
_MINIMUM_SAMPLE_COUNT = 5
_MINIMUM_STAGE_SECONDS = 30.0
_MAXIMUM_STABLE_SWING = 0.5


class StageEtaEstimator:
    """一回のrun内で比較可能なwork sampleだけを集計する。"""

    def __init__(self) -> None:
        self._samples: dict[
            tuple[ProcessingStage, str, WorkDisposition],
            list[float],
        ] = {}

    def record_sample(
        self,
        stage: ProcessingStage,
        work_unit_kind: str,
        disposition: WorkDisposition,
        duration_seconds: float,
    ) -> None:
        """完了した一つのwork unitを現在のrunへ記録する。"""
        if not work_unit_kind:
            msg = "work unit kindが必要です"
            raise ValueError(msg)
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            msg = "work unit durationは0より大きい有限値である必要があります"
            raise ValueError(msg)
        key = (stage, work_unit_kind, disposition)
        samples = self._samples.setdefault(key, [])
        if len(samples) < _MINIMUM_SAMPLE_COUNT:
            samples.append(duration_seconds)
            return
        previous_mean = fmean(samples)
        candidate_samples = [*samples[1:], duration_seconds]
        candidate_mean = fmean(candidate_samples)
        if abs(candidate_mean - previous_mean) / previous_mean > _MAXIMUM_STABLE_SWING:
            samples[:] = [duration_seconds]
            return
        samples[:] = candidate_samples

    def estimate(
        self,
        stage: ProcessingStage,
        work_unit_kind: str,
        *,
        remaining_reuse_count: int | None,
        remaining_recompute_count: int | None,
        stage_elapsed_seconds: float,
    ) -> EtaEstimate:
        """系列別の残件数が既知な場合だけStage ETAを返す。"""
        if remaining_reuse_count is None or remaining_recompute_count is None:
            return ("unavailable", None)
        if remaining_reuse_count < 0 or remaining_recompute_count < 0:
            msg = "remaining work countは0以上である必要があります"
            raise ValueError(msg)
        if stage_elapsed_seconds < _MINIMUM_STAGE_SECONDS:
            return ("estimating", None)
        remaining: tuple[tuple[WorkDisposition, int], ...] = (
            ("reuse", remaining_reuse_count),
            ("recompute", remaining_recompute_count),
        )
        if sum(count for _, count in remaining) == 0:
            return ("unavailable", None)
        eta_seconds = 0.0
        for disposition, count in remaining:
            if count == 0:
                continue
            samples = self._samples.get(
                (stage, work_unit_kind, disposition),
                [],
            )
            if len(samples) < _MINIMUM_SAMPLE_COUNT:
                return ("estimating", None)
            eta_seconds += fmean(samples) * count
        return ("available", eta_seconds)
