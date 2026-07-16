"""Report Provenance modelのtest。"""

import pytest

from src.video_selection.models.report_provenance import ReportProvenance
from src.video_selection.models.report_stage_provenance import ReportStageProvenance


def _stage(**overrides: object) -> ReportStageProvenance:
    values: dict[str, object] = {
        "name": "final_selection",
        "fingerprint": "stg_" + "1" * 64,
        "upstream_fingerprints": (),
        "cache_hits": 0,
        "cache_misses": 1,
        "recomputed_items": 1,
        "attempt_count": 1,
        "validation_failures": 0,
        "effective_settings": {},
        "tool_refs": ("ffmpeg",),
        "model_refs": ("candidate_annotation",),
        "contract_refs": ("selection_policy",),
        "duration_ms": 12,
    }
    values.update(overrides)
    return ReportStageProvenance(**values)  # type: ignore[arg-type]


def test_registry_references_are_validated_as_one_provenance_graph() -> None:
    """解決可能なtool、model、contract参照が受理されること。

    Arrange:
        - runtime、tool、contract registryと一つのStageが用意される
    Act:
        - Report Provenanceが構築される
    Assert:
        - Stage graphとregistry値が保持されること
    """
    # Arrange / Act
    provenance = ReportProvenance(
        runtime={"os": "linux"},
        tools={"ffmpeg": "6.1.1"},
        contracts={"selection_policy": "v1"},
        stages=(_stage(),),
    )

    # Assert
    assert provenance.stages[0].name == "final_selection"
    assert provenance.tools == {"ffmpeg": "6.1.1"}


def test_unresolved_tool_reference_is_rejected() -> None:
    """Stageから未登録toolへの参照が拒否されること。

    Arrange:
        - registryに存在しないtoolを参照するStageが用意される
    Act:
        - Report Provenanceが構築される
    Assert:
        - 解決不能なregistry参照として拒否されること
    """
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="registry参照"):
        ReportProvenance(
            runtime={"os": "linux"},
            tools={"ffprobe": "6.1.1"},
            contracts={"selection_policy": "v1"},
            stages=(_stage(),),
        )
