"""Durable checkpoint engine version registryのtest。"""

import pytest

from src.video_selection.models.checkpoint_operation import CheckpointOperation
from src.video_selection.services.checkpoint_version import checkpoint_version


@pytest.mark.parametrize("operation", tuple(CheckpointOperation))
def test_every_checkpoint_operation_has_an_explicit_version(
    operation: CheckpointOperation,
) -> None:
    """全checkpoint operationにsilent fallbackでないversionが登録されること。

    Arrange:
        - 定義済みの各Checkpoint Operationが用意される
    Act:
        - operation固有versionが解決される
    Assert:
        - 空値や共通fallbackが返されないこと
    """
    # Arrange

    # Act
    version = checkpoint_version(operation)

    # Assert
    assert version
    assert version != "walking-skeleton-0"
