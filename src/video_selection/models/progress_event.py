"""renderer非依存のProgress Event。"""

import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from .processing_stage import ProcessingStage

ProgressEventKind = Literal[
    "run_started",
    "stage_started",
    "progress",
    "cache",
    "external_work_started",
    "heartbeat",
    "retrying",
    "warning",
    "stage_completed",
    "run_completed",
    "run_failed",
    "run_interrupted",
]
ProgressSeverity = Literal["info", "warning", "error"]
EstimationState = Literal["unavailable", "estimating", "available"]

_STABLE_CODE = re.compile(r"[a-z][a-z0-9_-]*\Z")
_STAGE_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProgressEvent:
    """runとProcessing Stageの安全な観測値を保持する。"""

    kind: ProgressEventKind
    severity: ProgressSeverity
    stage: ProcessingStage | None = None
    stage_fingerprint: str | None = None
    stage_index: int | None = None
    stage_count: int | None = None
    video_order: int | None = None
    video_count: int | None = None
    video_relative_path: str | None = None
    processed_count: int | None = None
    total_count: int | None = None
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    reuse_count: int = 0
    recompute_count: int = 0
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    estimation_state: EstimationState = "unavailable"
    work_unit_kind: str | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        """path、count、時間、stable codeの不変条件を検証する。"""
        _validate_position("stage", self.stage_index, self.stage_count)
        _validate_position("video", self.video_order, self.video_count)
        _validate_progress_count(self.processed_count, self.total_count)
        for label, value in (
            ("cache hit", self.cache_hit_count),
            ("cache miss", self.cache_miss_count),
            ("reuse", self.reuse_count),
            ("recompute", self.recompute_count),
        ):
            if value < 0:
                msg = f"{label} countは0以上である必要があります"
                raise ValueError(msg)
        _validate_seconds("elapsed", self.elapsed_seconds, allow_zero=True)
        _validate_seconds("ETA", self.eta_seconds, allow_zero=False)
        if (self.eta_seconds is None) == (self.estimation_state == "available"):
            msg = "ETA stateとETA secondsが一致していません"
            raise ValueError(msg)
        if self.video_relative_path is not None:
            _validate_relative_path(self.video_relative_path)
        _validate_stable_code("work unit", self.work_unit_kind)
        _validate_stable_code("reason", self.reason_code)
        if self.stage_fingerprint is not None and (
            self.stage is None
            or _STAGE_FINGERPRINT.fullmatch(self.stage_fingerprint) is None
        ):
            msg = "Stage fingerprintはStage付きの完全SHA-256である必要があります"
            raise ValueError(msg)


def _validate_position(label: str, index: int | None, total: int | None) -> None:
    if index is not None and index < 1:
        msg = f"{label} indexは1以上である必要があります"
        raise ValueError(msg)
    if total is not None and total < 1:
        msg = f"{label} countは1以上である必要があります"
        raise ValueError(msg)
    if index is not None and total is not None and index > total:
        msg = f"{label} indexはcount以下である必要があります"
        raise ValueError(msg)


def _validate_progress_count(processed: int | None, total: int | None) -> None:
    if processed is not None and processed < 0:
        msg = "processed countは0以上である必要があります"
        raise ValueError(msg)
    if total is not None and total < 0:
        msg = "total countは0以上である必要があります"
        raise ValueError(msg)
    if processed is not None and total is not None and processed > total:
        msg = "processed countはtotal count以下である必要があります"
        raise ValueError(msg)


def _validate_seconds(label: str, value: float | None, *, allow_zero: bool) -> None:
    if value is None:
        return
    minimum_is_valid = value >= 0 if allow_zero else value > 0
    if not math.isfinite(value) or not minimum_is_valid:
        qualifier = "0以上" if allow_zero else "0より大きい有限値"
        msg = f"{label} secondsは{qualifier}である必要があります"
        raise ValueError(msg)


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\0" in value:
        msg = "video relative pathにabsolute pathまたは親参照は指定できません"
        raise ValueError(msg)


def _validate_stable_code(label: str, value: str | None) -> None:
    if value is not None and _STABLE_CODE.fullmatch(value) is None:
        msg = f"{label} codeはstable code形式である必要があります"
        raise ValueError(msg)
