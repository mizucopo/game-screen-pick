"""一回のVision推論のprivacy-safe診断。"""

import re
from dataclasses import dataclass

from .model_runtime_identity import ModelRuntimeIdentity
from .resolved_model_identity import ResolvedModelIdentity

_SAFE_VALUE_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z._:+/-]{0,255}")


@dataclass(frozen=True)
class VisionInferenceDiagnostics:
    """再現に必要なidentity、logical operationの回数、token、durationを保持する。"""

    request_fingerprint: str
    model_name: str
    model_identity: str
    runtime_identity: str
    prompt_version: str
    schema_version: str
    stage_contract_version: str
    retry_policy_version: str
    cache_hit: bool
    attempt_count: int
    validation_code: str | None
    image_count: int
    context_cue_count: int
    duration_seconds: float
    prompt_eval_count: int | None
    eval_count: int | None
    done_reason: str | None

    def __post_init__(self) -> None:
        """pathや自由形式detailを持たない診断だけを受理する。"""
        safe_values = (
            self.prompt_version,
            self.schema_version,
            self.stage_contract_version,
            self.retry_policy_version,
        )
        optional_safe_values = (self.validation_code, self.done_reason)
        counts = (
            self.attempt_count,
            self.image_count,
            self.context_cue_count,
        )
        optional_counts = (self.prompt_eval_count, self.eval_count)
        try:
            model_identity = ResolvedModelIdentity.from_identifier(self.model_identity)
            runtime_identity = ModelRuntimeIdentity.from_identifier(
                self.runtime_identity
            )
            canonical_identity_pair = (
                model_identity.store_kind is runtime_identity.store_kind
            )
        except ValueError:
            canonical_identity_pair = False
        if (
            len(self.request_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.request_fingerprint
            )
            or any(
                _SAFE_VALUE_PATTERN.fullmatch(value) is None for value in safe_values
            )
            or any(
                value is not None and _SAFE_VALUE_PATTERN.fullmatch(value) is None
                for value in optional_safe_values
            )
            or not 1 <= self.attempt_count <= 4
            or any(value < 0 for value in counts)
            or any(value is not None and value < 0 for value in optional_counts)
            or self.duration_seconds < 0
            or _SAFE_VALUE_PATTERN.fullmatch(self.model_name) is None
            or not canonical_identity_pair
        ):
            msg = "Vision inference diagnosticsが不正です"
            raise ValueError(msg)
