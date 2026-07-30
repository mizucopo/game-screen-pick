"""FrozenModelRuntimeのtest。"""

from pathlib import Path

import pytest

from src.video_selection.acceptance.frozen_model_runtime import FrozenModelRuntime
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from tests.video_selection.fakes.fake_model_runtime import FakeModelRuntime


def test_same_model_selectors_return_exact_pre_resolved_identities(
    tmp_path: Path,
) -> None:
    """同じmodel selectorでは事前解決済みidentityがそのまま返されること。

    Arrange:
        - Effective Configurationと解決済み3 role modelが用意される
    Act:
        - FrozenModelRuntimeからmodelが解決される
    Assert:
        - 同一ResolvedModels instanceが返されること
    """
    # Arrange
    configuration = _configuration(tmp_path)
    models = FakeModelRuntime("frozen").resolve_models(configuration)

    # Act
    actual = FrozenModelRuntime(models).resolve_models(configuration)

    # Assert
    assert actual is models


def test_changed_model_selector_is_rejected(tmp_path: Path) -> None:
    """phase間でmodel selectorが変化するとfreeze違反になること。

    Arrange:
        - 事前解決後とは異なるScene Catalog model設定が用意される
    Act:
        - FrozenModelRuntimeからmodel解決が試行される
    Assert:
        - model設定変化としてValueErrorになること
    """
    # Arrange
    configuration = _configuration(tmp_path)
    models = FakeModelRuntime("frozen").resolve_models(configuration)
    changed = EffectiveConfiguration(
        video_input_folder=configuration.video_input_folder,
        output_folder=configuration.output_folder,
        scene_catalog_model="changed:latest",
    )

    # Act
    # Assert
    with pytest.raises(ValueError, match="freeze"):
        FrozenModelRuntime(models).resolve_models(changed)


def _configuration(tmp_path: Path) -> EffectiveConfiguration:
    """model selectorを持つ最小Effective Configurationを返す。"""
    return EffectiveConfiguration(
        video_input_folder=tmp_path / "input",
        output_folder=tmp_path / "output",
    )
