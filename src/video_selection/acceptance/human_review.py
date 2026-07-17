"""target-only review worksheetの生成、検証、aggregate gate。"""

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from ..models.selection_rejection_reason import SelectionRejectionReason
from .atomic_json import read_json_object, write_atomic_json

_WORKSHEET_SCHEMA = "game-screen-pick/human-review-worksheet@1.0.0"
_VISUAL_QUALITY_VALUES = {
    "pass",
    "broken",
    "black",
    "white",
    "transition",
    "near_duplicate",
    "pending",
}
_BOOLEAN_REVIEW_VALUES = {"yes", "no", "pending"}
_CONSISTENCY_VALUES = {"consistent", "contradictory", "pending"}
_MONOTONICITY_VALUES = {"pass", "fail", "pending"}


def ensure_review_worksheet(
    path: Path,
    *,
    suite: str,
    suite_fingerprint: str,
    canonical_report: Mapping[str, object],
    selection_artifact: Mapping[str, object],
) -> dict[str, object]:
    """既存worksheetを保持し、なければcandidate IDとstable enum欄を生成する。"""
    existing = read_json_object(path)
    if existing is not None:
        _require_worksheet_identity(existing, suite, suite_fingerprint)
        return existing
    selected_value = canonical_report.get("selected")
    rejected_value = selection_artifact.get("rejected")
    if not isinstance(selected_value, list) or not isinstance(rejected_value, list):
        raise ValueError("Review worksheet sourceのcandidate集合が不正です")
    selected = [_selected_review_entry(item) for item in selected_value]
    rejected = [_rejected_review_entry(item) for item in rejected_value]
    worksheet: dict[str, object] = {
        "schema": _WORKSHEET_SCHEMA,
        "suite": suite,
        "suite_fingerprint": suite_fingerprint,
        "instructions": {
            "visual_quality": ("pass|broken|black|white|transition|near_duplicate"),
            "blog_usable": "yes|no",
            "annotation_consistency": "consistent|contradictory",
            "context_overrode_visual_invalidity": "yes|no",
            "spoiler_monotonicity": "pass|fail",
        },
        "reviewer": "",
        "completed_at": None,
        "selected": selected,
        "rejected": rejected,
        "suite_checks": {"spoiler_monotonicity": "pending"},
    }
    write_atomic_json(path, worksheet)
    return worksheet


def evaluate_human_review(
    worksheet: Mapping[str, object],
    *,
    suite: str,
    suite_fingerprint: str,
) -> dict[str, object]:
    """worksheetをstable aggregateへ変換しpending/pass/failを返す。"""
    _require_worksheet_identity(worksheet, suite, suite_fingerprint)
    selected = _mapping_list(worksheet.get("selected"), "selected")
    rejected = _mapping_list(worksheet.get("rejected"), "rejected")
    suite_checks = _mapping(worksheet.get("suite_checks"), "suite_checks")
    spoiler_monotonicity = _enum_value(
        suite_checks.get("spoiler_monotonicity"),
        _MONOTONICITY_VALUES,
        "spoiler_monotonicity",
    )
    visual_values: list[str] = []
    usable_values: list[str] = []
    consistency_values: list[str] = []
    context_values: list[str] = []
    for item in selected:
        visual_values.append(
            _enum_value(
                item.get("visual_quality"),
                _VISUAL_QUALITY_VALUES,
                "visual_quality",
            )
        )
        usable_values.append(
            _enum_value(item.get("blog_usable"), _BOOLEAN_REVIEW_VALUES, "blog_usable")
        )
        consistency_values.append(
            _enum_value(
                item.get("annotation_consistency"),
                _CONSISTENCY_VALUES,
                "annotation_consistency",
            )
        )
        context_values.append(
            _enum_value(
                item.get("context_overrode_visual_invalidity"),
                _BOOLEAN_REVIEW_VALUES,
                "context_overrode_visual_invalidity",
            )
        )
    stable_rejection_count = sum(_rejection_is_stable(item) for item in rejected)
    pending = (
        not isinstance(worksheet.get("reviewer"), str)
        or not cast(str, worksheet.get("reviewer")).strip()
        or worksheet.get("completed_at") is None
        or spoiler_monotonicity == "pending"
        or any(
            value == "pending"
            for value in (
                *visual_values,
                *usable_values,
                *consistency_values,
                *context_values,
            )
        )
    )
    selected_count = len(selected)
    invalid_visual_count = sum(value != "pass" for value in visual_values)
    usable_count = sum(value == "yes" for value in usable_values)
    contradiction_count = sum(value == "contradictory" for value in consistency_values)
    context_override_count = sum(value == "yes" for value in context_values)
    usable_ratio = usable_count / selected_count if selected_count else 0.0
    contradiction_ratio = (
        contradiction_count / selected_count if selected_count else 1.0
    )
    gates = {
        "invalid_visual_selected_zero": invalid_visual_count == 0
        and selected_count > 0,
        "blog_usable_at_least_90_percent": usable_ratio >= 0.9,
        "annotation_contradiction_below_10_percent": contradiction_ratio < 0.1,
        "context_override_zero": context_override_count == 0,
        "spoiler_monotonicity": spoiler_monotonicity == "pass",
        "stable_rejection_reason": stable_rejection_count == len(rejected),
    }
    return {
        "status": (
            "pending_human_review"
            if pending
            else "passed"
            if all(gates.values())
            else "failed"
        ),
        "selected_count": selected_count,
        "rejected_count": len(rejected),
        "invalid_visual_count": invalid_visual_count,
        "blog_usable_count": usable_count,
        "blog_usable_ratio": usable_ratio,
        "annotation_contradiction_count": contradiction_count,
        "annotation_contradiction_ratio": contradiction_ratio,
        "context_override_count": context_override_count,
        "stable_rejection_reason_count": stable_rejection_count,
        "gates": gates,
    }


def complete_review_metadata(worksheet: dict[str, object]) -> None:
    """test/tooling用にreview完了時刻をUTCで設定する。"""
    worksheet["completed_at"] = datetime.now(timezone.utc).isoformat()


def _selected_review_entry(value: object) -> dict[str, object]:
    item = _mapping(value, "selected candidate")
    output = _mapping(item.get("output"), "selected output")
    candidate_id = _candidate_id(item.get("image_id"))
    relative_output = output.get("relative_path")
    if not isinstance(relative_output, str):
        raise ValueError("Selected candidate outputが不正です")
    return {
        "candidate_id": candidate_id,
        "output_relative_path": relative_output,
        "visual_quality": "pending",
        "blog_usable": "pending",
        "annotation_consistency": "pending",
        "context_overrode_visual_invalidity": "pending",
    }


def _rejected_review_entry(value: object) -> dict[str, object]:
    item = _mapping(value, "rejected candidate")
    candidate_id = _candidate_id(item.get("candidate_id"))
    reason_code = item.get("reason_code")
    if not isinstance(reason_code, str):
        raise ValueError("Rejected candidate reasonが不正です")
    try:
        SelectionRejectionReason(reason_code)
    except ValueError:
        msg = "Rejected candidate reasonがstable enumではありません"
        raise ValueError(msg) from None
    return {"candidate_id": candidate_id, "reason_code": reason_code}


def _rejection_is_stable(value: Mapping[str, object]) -> bool:
    try:
        _candidate_id(value.get("candidate_id"))
        reason = value.get("reason_code")
        if not isinstance(reason, str):
            return False
        SelectionRejectionReason(reason)
    except ValueError:
        return False
    return True


def _require_worksheet_identity(
    worksheet: Mapping[str, object],
    suite: str,
    suite_fingerprint: str,
) -> None:
    if (
        worksheet.get("schema") != _WORKSHEET_SCHEMA
        or worksheet.get("suite") != suite
        or worksheet.get("suite_fingerprint") != suite_fingerprint
    ):
        raise ValueError("Human review worksheetが対象suiteと一致しません")


def _candidate_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("frm_")
        or len(value) != 68
        or any(character not in "0123456789abcdef" for character in value[4:])
    ):
        raise ValueError("Human review candidate IDが不正です")
    return value


def _enum_value(value: object, allowed: set[str], location: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"Human review {location}がstable enumではありません")
    return value


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Human review {location}にはobjectが必要です")
    return cast(dict[str, object], value)


def _mapping_list(value: object, location: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"Human review {location}にはarrayが必要です")
    return [_mapping(item, location) for item in value]
