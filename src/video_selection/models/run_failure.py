"""利用者へ安全に通知できるrun terminal failure。"""

import math
import re
from dataclasses import dataclass, field
from typing import Literal

RunFailureExitCode = Literal[1, 2, 130]
ResumeGuidance = Literal[
    "completed_stages_reusable",
    "run_not_started",
]
SafeObservedValue = str | int | float | bool

_STABLE_CODE = re.compile(r"[a-z][a-z0-9_-]*\Z")


@dataclass(frozen=True, slots=True, kw_only=True)
class RunFailure:
    """stable codeとallowlist値だけを公開する失敗結果。"""

    reason_code: str
    exit_code: RunFailureExitCode
    remediation_code: str
    resume_guidance: ResumeGuidance
    observed_values: tuple[tuple[str, SafeObservedValue], ...] = ()
    cause: BaseException | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """公開値がstableかつ秘密を持ち込めない形式であることを検証する。"""
        _validate_code("reason", self.reason_code)
        _validate_code("remediation", self.remediation_code)
        keys: set[str] = set()
        for key, value in self.observed_values:
            _validate_code("observed value", key)
            if key in keys:
                msg = "observed value keyは重複できません"
                raise ValueError(msg)
            keys.add(key)
            _validate_observed_value(value)

    def __str__(self) -> str:
        """raw causeを含めずstable reasonだけを返す。"""
        return self.reason_code


def _validate_code(label: str, value: str) -> None:
    if _STABLE_CODE.fullmatch(value) is None:
        msg = f"{label} codeはstable code形式である必要があります"
        raise ValueError(msg)


def _validate_observed_value(value: SafeObservedValue) -> None:
    if isinstance(value, str):
        _validate_code("observed string", value)
        return
    if isinstance(value, float) and not math.isfinite(value):
        msg = "observed numberは有限値である必要があります"
        raise ValueError(msg)
