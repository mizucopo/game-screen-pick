"""DurableWorkUnitBundleの契約test。"""

from pathlib import Path

import pytest

from src.video_selection.models.durable_work_unit_bundle import (
    DurableWorkUnitBundle,
)


def test_bundle_preserves_validated_artifact_root(tmp_path: Path) -> None:
    """検証済みartifactとrootが変更不能なbundleで保持されること。

    Arrange:
        - artifact metadataとcheckpoint rootが用意される
    Act:
        - Durable Work Unit Bundleが構築される
    Assert:
        - 値が保持されfield自体は変更できないこと
    """
    # Arrange
    artifact = {"schema": "example@1", "value": 1}

    # Act
    bundle = DurableWorkUnitBundle(artifact=artifact, root=tmp_path)

    # Assert
    assert bundle.artifact == artifact
    assert bundle.root == tmp_path
    with pytest.raises(AttributeError):
        bundle.root = tmp_path / "changed"  # type: ignore[misc]
