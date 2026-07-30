"""Process killからAcceptance attemptのwork量を回復するjournal。"""

import json
import math
import stat
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import cast

from .atomic_json import read_json_object, write_atomic_json

_SCHEMA = "game-screen-pick/acceptance-attempt-journal@1.0.0"


class AcceptanceAttemptJournal:
    """active attemptのprivacy-safe metricsとcontextをatomic保存する。"""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = Lock()

    @property
    def exists(self) -> bool:
        """確定済みjournalが存在するか返す。"""
        return _regular_file_exists(self._path)

    def start(
        self,
        *,
        attempt_id: str,
        step_kind: str,
        step_name: str,
        started_at_epoch_seconds: float,
        execution_context: Mapping[str, object],
    ) -> None:
        """pipeline開始前のattempt identityを確定する。"""
        if _regular_file_exists(self._path):
            raise ValueError("Acceptance attempt journalが既に存在します")
        write_atomic_json(
            self._path,
            {
                "schema": _SCHEMA,
                "attempt_id": attempt_id,
                "step_kind": step_kind,
                "step_name": step_name,
                "started_at_epoch_seconds": started_at_epoch_seconds,
                "execution_context": dict(execution_context),
                "metrics": _empty_metrics(),
                "work_unit_resolutions": {},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def record_snapshot(
        self,
        metrics: Mapping[str, object],
        work_unit_resolutions: Mapping[str, str],
    ) -> None:
        """observer snapshotを既存attempt identityへ上書き確定する。"""
        with self._lock:
            journal = self._read()
            if journal is None:
                return
            journal["metrics"] = _merge_metrics(
                _validated_metrics(
                    _mapping(journal.get("metrics"), "metrics"),
                ),
                _validated_metrics(metrics),
            )
            journal["work_unit_resolutions"] = _merge_resolutions(
                _validated_resolutions(
                    _mapping(
                        journal.get("work_unit_resolutions"),
                        "work_unit_resolutions",
                    )
                ),
                _validated_resolutions(work_unit_resolutions),
            )
            journal["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_atomic_json(self._path, journal)

    def recover(
        self,
        *,
        attempt_id: str,
        step_kind: str,
        step_name: str,
        processing_cache_folder: Path,
        video_identity_cache_folder: Path | None = None,
    ) -> tuple[dict[str, object], dict[str, object]] | None:
        """journalと確定manifestからkill直前までのmetrics/contextを返す。"""
        with self._lock:
            journal = self._read()
        if journal is None:
            return None
        if (
            journal.get("attempt_id") != attempt_id
            or journal.get("step_kind") != step_kind
            or journal.get("step_name") != step_name
        ):
            raise ValueError("Acceptance attempt journalがactive markerと一致しません")
        metrics = _validated_metrics(_mapping(journal.get("metrics"), "metrics"))
        resolutions = _validated_resolutions(
            _mapping(
                journal.get("work_unit_resolutions"),
                "work_unit_resolutions",
            )
        )
        started_at = _number(
            journal.get("started_at_epoch_seconds"),
            "started_at_epoch_seconds",
        )
        recovered_checkpoints = _completed_checkpoints_since(
            processing_cache_folder,
            started_at,
            video_identity_cache_folder,
        )
        recovered_recomputes = {
            fingerprint
            for fingerprint in recovered_checkpoints
            if resolutions.get(fingerprint) not in {"reused", "recomputed"}
        }
        recovered_misses = {
            fingerprint
            for fingerprint in recovered_recomputes
            if fingerprint not in resolutions
        }
        metrics["cache_miss_count"] = _integer(
            metrics.get("cache_miss_count"),
            "cache_miss_count",
        ) + len(recovered_misses)
        metrics["unexpected_recompute_count"] = _integer(
            metrics.get("unexpected_recompute_count"),
            "unexpected_recompute_count",
        ) + len(recovered_recomputes)
        return (
            metrics,
            dict(
                _mapping(
                    journal.get("execution_context"),
                    "execution_context",
                )
            ),
        )

    def clear(self) -> None:
        """active marker消去後のjournalを削除する。"""
        self._path.unlink(missing_ok=True)

    def _read(self) -> dict[str, object] | None:
        journal = read_json_object(self._path)
        if journal is None:
            return None
        if journal.get("schema") != _SCHEMA:
            raise ValueError("Acceptance attempt journal schemaが不正です")
        return journal


def _empty_metrics() -> dict[str, object]:
    return {
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "reuse_count": 0,
        "unexpected_recompute_count": 0,
        "stage_durations_seconds": {},
        "completed_stage_counts": {},
    }


def _validated_metrics(value: Mapping[str, object]) -> dict[str, object]:
    result = _empty_metrics()
    for key in (
        "cache_hit_count",
        "cache_miss_count",
        "reuse_count",
        "unexpected_recompute_count",
    ):
        result[key] = _integer(value.get(key, 0), key)
    result["stage_durations_seconds"] = _numeric_mapping(
        value.get("stage_durations_seconds", {})
    )
    result["completed_stage_counts"] = _integer_mapping(
        value.get("completed_stage_counts", {})
    )
    return result


def _validated_resolutions(
    value: Mapping[str, object] | Mapping[str, str],
) -> dict[str, str]:
    if any(
        not _is_sha256(key) or item not in {"miss_started", "reused", "recomputed"}
        for key, item in value.items()
    ):
        raise ValueError("Acceptance attempt work unit resolutionが不正です")
    return {key: cast(str, item) for key, item in value.items()}


def _completed_checkpoints_since(
    cache_folder: Path,
    started_at_epoch_seconds: float,
    video_identity_cache_folder: Path | None,
) -> set[str]:
    roots_and_patterns = [
        (cache_folder / "work-units", "*/*/*/manifest.json"),
        (cache_folder / "videos", "*/*/*/manifest.json"),
        (cache_folder / "video-sets", "*/*/*/manifest.json"),
    ]
    if video_identity_cache_folder is not None:
        roots_and_patterns.append((video_identity_cache_folder, "*.json"))
    result: set[str] = set()
    for root, pattern in roots_and_patterns:
        if not root.is_dir() or root.is_symlink():
            continue
        for path in root.glob(pattern):
            if path.is_symlink():
                continue
            try:
                value: object = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    continue
                fingerprint = value.get(
                    "work_unit_fingerprint",
                    value.get("stage_fingerprint"),
                )
                completed_at = value.get("completed_at")
                completed_epoch = (
                    datetime.fromisoformat(completed_at).timestamp()
                    if isinstance(completed_at, str)
                    else -1.0
                )
            except (OSError, ValueError):
                continue
            if _is_sha256(fingerprint) and completed_epoch >= started_at_epoch_seconds:
                result.add(cast(str, fingerprint))
    return result


def _merge_metrics(
    previous: Mapping[str, object],
    current: Mapping[str, object],
) -> dict[str, object]:
    """並行snapshotが逆順に到着しても累積値を後退させない。"""
    result = _empty_metrics()
    for key in (
        "cache_hit_count",
        "cache_miss_count",
        "reuse_count",
        "unexpected_recompute_count",
    ):
        result[key] = max(
            _integer(previous.get(key), key),
            _integer(current.get(key), key),
        )
    result["stage_durations_seconds"] = _maximum_numeric_mappings(
        previous.get("stage_durations_seconds"),
        current.get("stage_durations_seconds"),
    )
    result["completed_stage_counts"] = _maximum_integer_mappings(
        previous.get("completed_stage_counts"),
        current.get("completed_stage_counts"),
    )
    return result


def _merge_resolutions(
    previous: Mapping[str, str],
    current: Mapping[str, str],
) -> dict[str, str]:
    """同じcheckpointの確定状態を並行snapshotで後退させない。"""
    priority = {"miss_started": 0, "reused": 1, "recomputed": 2}
    result = dict(previous)
    for fingerprint, status in current.items():
        prior = result.get(fingerprint)
        if prior is None or priority[status] >= priority[prior]:
            result[fingerprint] = status
    return result


def _maximum_numeric_mappings(
    first: object,
    second: object,
) -> dict[str, float]:
    first_values = _numeric_mapping(first)
    second_values = _numeric_mapping(second)
    return {
        key: max(first_values.get(key, 0.0), second_values.get(key, 0.0))
        for key in first_values.keys() | second_values.keys()
    }


def _maximum_integer_mappings(
    first: object,
    second: object,
) -> dict[str, int]:
    first_values = _integer_mapping(first)
    second_values = _integer_mapping(second)
    return {
        key: max(first_values.get(key, 0), second_values.get(key, 0))
        for key in first_values.keys() | second_values.keys()
    }


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Acceptance attempt journal {label}が不正です")
    return cast(dict[str, object], value)


def _numeric_mapping(value: object) -> dict[str, float]:
    mapping = _mapping(value, "numeric mapping")
    result: dict[str, float] = {}
    for key, item in mapping.items():
        number = _number(item, key)
        if number < 0:
            raise ValueError("Acceptance attempt journal durationが不正です")
        result[key] = number
    return result


def _integer_mapping(value: object) -> dict[str, int]:
    mapping = _mapping(value, "integer mapping")
    return {key: _integer(item, key) for key, item in mapping.items()}


def _number(value: object, label: str) -> float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or float(value) < 0
    ):
        raise ValueError(f"Acceptance attempt journal {label}が不正です")
    return float(value)


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Acceptance attempt journal {label}が不正です")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _regular_file_exists(path: Path) -> bool:
    """欠損だけを不存在とし、access障害や非通常fileを隠さない。"""
    try:
        mode = path.lstat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return False
    if not stat.S_ISREG(mode):
        raise ValueError("Acceptance attempt journalが通常fileではありません")
    return True
