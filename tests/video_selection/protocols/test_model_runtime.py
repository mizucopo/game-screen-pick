"""ModelRuntime protocolのcontract test。"""

from src.video_selection.model_runtime.model_lifecycle_runtime import (
    ModelLifecycleRuntime,
)
from src.video_selection.protocols.model_runtime import ModelRuntime


def test_model_lifecycle_runtime_satisfies_model_runtime_protocol() -> None:
    """production lifecycle runtimeがModelRuntime portとして使用できること。

    Arrange:
        - external operationをまだ開始していないproduction runtimeが用意される
    Act:
        - runtimeがModelRuntime型の変数へ割り当てられる
    Assert:
        - 同じruntime instanceが保持されること
    """
    # Arrange
    runtime = ModelLifecycleRuntime()

    # Act
    port: ModelRuntime = runtime

    # Assert
    assert port is runtime
