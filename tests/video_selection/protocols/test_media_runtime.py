"""MediaRuntime semantic portのcontract test。"""

from src.video_selection.media.ffmpeg_media_runtime import FfmpegMediaRuntime
from src.video_selection.protocols.media_runtime import MediaRuntime


def test_ffmpeg_adapter_satisfies_media_runtime_port_without_running_tools() -> None:
    """FFmpeg adapterが全semantic operationを公開すること。

    Arrange:
        - system toolをまだ実行していないFFmpeg adapterが用意される
    Act:
        - MediaRuntime portへの適合が確認される
    Assert:
        - adapterがportを満たすこと
    """
    # Arrange
    runtime: MediaRuntime = FfmpegMediaRuntime()

    # Act
    satisfies_port = isinstance(runtime, MediaRuntime)

    # Assert
    assert satisfies_port
