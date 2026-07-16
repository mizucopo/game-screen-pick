"""report provenanceへ安全に含められるJSON値の検証。"""

import json
import math
import re
from collections.abc import Mapping

_WINDOWS_ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/]")
_SAFE_REFERENCE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_PRIVATE_KEY_PARTS = (
    "credential",
    "environment_variable",
    "host",
    "password",
    "path",
    "prompt_body",
    "raw_response",
    "secret",
    "stack_trace",
    "token_value",
)


def validate_privacy_safe_mapping(
    value: Mapping[str, object],
    *,
    field_name: str,
) -> None:
    """pathやsecret-bearing keyを含まないJSON mappingを検証する。"""
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        msg = f"{field_name}には有限なJSON値が必要です"
        raise ValueError(msg) from error
    _validate_value(value, field_name)


def validate_reference(value: str, *, field_name: str) -> None:
    """registry参照に使えるprivacy-safeな短い名前を検証する。"""
    if _SAFE_REFERENCE.fullmatch(value) is None:
        msg = f"{field_name}にはcanonical reference nameが必要です"
        raise ValueError(msg)


def string_looks_private(value: str) -> bool:
    """絶対path、endpoint、改行を含む文字列かを返す。"""
    return (
        (value != "/" and value.startswith(("/", "\\")))
        or _WINDOWS_ABSOLUTE_PATH.match(value) is not None
        or "://" in value
        or "\n" in value
        or "\r" in value
    )


def _validate_value(value: object, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"{location}のkeyは文字列である必要があります"
                raise ValueError(msg)
            normalized_key = key.casefold()
            if any(part in normalized_key for part in _PRIVATE_KEY_PARTS):
                msg = f"{location}に非公開keyは指定できません: {key}"
                raise ValueError(msg)
            _validate_value(item, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_value(item, f"{location}[{index}]")
        return
    if isinstance(value, str) and string_looks_private(value):
        msg = f"{location}に絶対pathまたはendpointは指定できません"
        raise ValueError(msg)
    if isinstance(value, float) and not math.isfinite(value):
        msg = f"{location}には有限値が必要です"
        raise ValueError(msg)
