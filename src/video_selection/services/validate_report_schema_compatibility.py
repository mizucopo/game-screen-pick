"""Canonical Selection Report readerのschema compatibility gate。"""

import re
from typing import cast

from .build_canonical_selection_report import REPORT_SCHEMA_NAME

SUPPORTED_REPORT_SCHEMA_MAJORS = frozenset({1, 2})
_SEMANTIC_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def validate_report_schema_compatibility(report: dict[str, object]) -> None:
    """対応majorを受理し未知fieldやenumを解釈せず保持可能にする。"""
    schema_value = report.get("schema")
    if not isinstance(schema_value, dict):
        raise ValueError("Canonical Selection Reportにschema objectがありません")
    schema = cast(dict[object, object], schema_value)
    name = schema.get("name")
    version = schema.get("version")
    match = _SEMANTIC_VERSION.fullmatch(version) if isinstance(version, str) else None
    if name != REPORT_SCHEMA_NAME or match is None:
        raise ValueError("Canonical Selection Report schema identityが不正です")
    if int(match.group(1)) not in SUPPORTED_REPORT_SCHEMA_MAJORS:
        raise ValueError(
            f"未対応major versionのCanonical Selection Reportです: {version}"
        )
