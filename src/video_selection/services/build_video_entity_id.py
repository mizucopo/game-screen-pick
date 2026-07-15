"""Video Stage entityの安定IDを導出する。"""

import hashlib
import json
from fractions import Fraction
from typing import Literal

EntityPrefix = Literal["seg", "mom", "frm"]


def build_video_entity_id(
    prefix: EntityPrefix,
    algorithm: str,
    video_fingerprint: str,
    *times: Fraction,
) -> str:
    """canonical JSONからprefix付きSHA-256 IDを返す。"""
    if len(video_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in video_fingerprint
    ):
        msg = "Video Fingerprintには64桁のSHA-256が必要です"
        raise ValueError(msg)
    normalized = json.dumps(
        {
            "algorithm": algorithm,
            "times": [
                {"denominator": item.denominator, "numerator": item.numerator}
                for item in times
            ],
            "video_fingerprint": video_fingerprint,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"{prefix}_{hashlib.sha256(normalized).hexdigest()}"
