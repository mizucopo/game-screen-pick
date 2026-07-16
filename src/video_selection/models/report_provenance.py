"""Canonical Selection Reportのrun-level provenance。"""

from collections.abc import Mapping
from dataclasses import dataclass

from .model_role import ModelRole
from .report_stage_provenance import ReportStageProvenance
from .report_value import validate_privacy_safe_mapping, validate_reference


@dataclass(frozen=True)
class ReportProvenance:
    """runtime、tool、contract registryとStage診断を保持する。"""

    runtime: Mapping[str, object]
    tools: Mapping[str, str]
    contracts: Mapping[str, str]
    stages: tuple[ReportStageProvenance, ...]

    def __post_init__(self) -> None:
        """registry参照とprivacy-safeな値だけを受理する。"""
        if not self.tools or not self.contracts or not self.stages:
            msg = "Report provenanceにはtool、contract、Stageが必要です"
            raise ValueError(msg)
        validate_privacy_safe_mapping(self.runtime, field_name="Report runtime")
        validate_privacy_safe_mapping(self.tools, field_name="Report tools")
        validate_privacy_safe_mapping(self.contracts, field_name="Report contracts")
        for registry_name, registry in (
            ("tools", self.tools),
            ("contracts", self.contracts),
        ):
            for key in registry:
                validate_reference(key, field_name=f"Report {registry_name}")
        names = tuple(item.name for item in self.stages)
        fingerprints = {item.fingerprint for item in self.stages}
        if len(names) != len(set(names)) or len(fingerprints) != len(self.stages):
            msg = "Report Stageのnameとfingerprintは一意である必要があります"
            raise ValueError(msg)
        model_refs = {role.value for role in ModelRole}
        for stage in self.stages:
            if (
                not set(stage.upstream_fingerprints) <= fingerprints
                or not set(stage.tool_refs) <= set(self.tools)
                or not set(stage.model_refs) <= model_refs
                or not set(stage.contract_refs) <= set(self.contracts)
            ):
                msg = "Report Stage provenanceのregistry参照が解決できません"
                raise ValueError(msg)
