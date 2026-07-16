"""Human/Canonical reportへ公開するStage provenance。"""

import re
from collections.abc import Mapping
from dataclasses import dataclass

from .report_value import validate_privacy_safe_mapping, validate_reference

_STAGE_FINGERPRINT = re.compile(r"stg_[0-9a-f]{64}")


@dataclass(frozen=True)
class ReportStageProvenance:
    """一つのProcessing Stageのprivacy-safeな再現診断。"""

    name: str
    fingerprint: str
    upstream_fingerprints: tuple[str, ...]
    cache_hits: int
    cache_misses: int
    recomputed_items: int
    attempt_count: int
    validation_failures: int
    effective_settings: Mapping[str, object]
    tool_refs: tuple[str, ...]
    model_refs: tuple[str, ...]
    contract_refs: tuple[str, ...]
    duration_ms: int
    prompt_eval_tokens: int | None = None
    eval_tokens: int | None = None

    def __post_init__(self) -> None:
        """fingerprint、件数、registry参照、設定の公開安全性を検証する。"""
        validate_reference(self.name, field_name="Report Stage name")
        fingerprints = (self.fingerprint, *self.upstream_fingerprints)
        counts = (
            self.cache_hits,
            self.cache_misses,
            self.recomputed_items,
            self.attempt_count,
            self.validation_failures,
            self.duration_ms,
        )
        optional_counts = (self.prompt_eval_tokens, self.eval_tokens)
        reference_groups = (self.tool_refs, self.model_refs, self.contract_refs)
        if (
            any(_STAGE_FINGERPRINT.fullmatch(item) is None for item in fingerprints)
            or len(self.upstream_fingerprints) != len(set(self.upstream_fingerprints))
            or any(item < 0 for item in counts)
            or any(item is not None and item < 0 for item in optional_counts)
            or any(len(group) != len(set(group)) for group in reference_groups)
        ):
            msg = "Report Stage provenanceのfingerprintまたは件数が不正です"
            raise ValueError(msg)
        for group_name, group in (
            ("tool_refs", self.tool_refs),
            ("model_refs", self.model_refs),
            ("contract_refs", self.contract_refs),
        ):
            for item in group:
                validate_reference(item, field_name=group_name)
        validate_privacy_safe_mapping(
            self.effective_settings,
            field_name="Report Stage effective settings",
        )
