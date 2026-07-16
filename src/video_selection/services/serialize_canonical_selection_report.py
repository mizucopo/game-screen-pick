"""Canonical Selection Reportの決定的なJSON serialization。"""

import json


def serialize_canonical_selection_report(report: dict[str, object]) -> str:
    """JSON key順を契約化せずproducer出力だけを安定化して返す。"""
    return (
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
