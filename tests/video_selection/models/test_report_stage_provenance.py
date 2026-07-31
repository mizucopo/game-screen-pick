"""Report Stage Provenance modelのtest。"""

import pytest

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
        "effective_settings": {"requested_image_count": 10},
        "tool_refs": (),
        "model_refs": ("candidate_annotation",),
        "contract_refs": ("selection_policy",),
        "duration_ms": 12,
    }
    values.update(overrides)
    return ReportStageProvenance(**values)  # type: ignore[arg-type]


def test_privacy_safe_stage_diagnostics_are_accepted() -> None:
    """再現に必要な有限値とregistry参照が保持されること。

    Arrange:
        - 完全fingerprintとprivacy-safeなStage診断が用意される
    Act:
        - Report Stage Provenanceが構築される
    Assert:
        - 設定とtoken件数が変更されず保持されること
    """
    # Arrange
    prompt_eval_tokens = 42

    # Act
    stage = _stage(prompt_eval_tokens=prompt_eval_tokens, eval_tokens=7)

    # Assert
    assert stage.effective_settings == {"requested_image_count": 10}
    assert stage.prompt_eval_tokens == 42
    assert stage.eval_tokens == 7


def test_absolute_path_in_effective_settings_is_rejected() -> None:
    """Stage effective settingsの絶対pathが公開前に拒否されること。

    Arrange:
        - absolute cache pathを含むStage設定が用意される
    Act:
        - Report Stage Provenanceが構築される
    Assert:
        - 非公開pathを含む設定として拒否されること
    """
    # Arrange
    effective_settings = {"cache_root": "/private/cache"}

    # Act
    with pytest.raises(ValueError) as error:
        _stage(effective_settings=effective_settings)

    # Assert
    assert "絶対path" in str(error.value)
