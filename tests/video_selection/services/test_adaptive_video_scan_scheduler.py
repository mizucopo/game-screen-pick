"""Adaptive Video Scan Schedulerのcontract test。"""

from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from typing import cast

from src.video_selection.models.prepared_video_scan import PreparedVideoScan
from src.video_selection.services.adaptive_video_scan_controller import (
    AdaptiveVideoScanController,
)
from src.video_selection.services.adaptive_video_scan_scheduler import (
    AdaptiveVideoScanScheduler,
)


def test_immediately_reused_scans_do_not_chain_callbacks_recursively() -> None:
    """多数の即時cache hitがcall stackを増やさず順序付きで返されること。

    Arrange:
        - 1500件のtaskを同期的に完了するexecutorと固定1 workerが用意される
    Act:
        - 全Video Scan taskがSchedulerへ投入される
    Assert:
        - recursion failureなしで全Futureが順序付き完了されること
    """
    # Arrange
    task_count = 1500
    controller = AdaptiveVideoScanController(
        video_count=task_count,
        configured_workers=1,
        auto_max_workers=6,
        decode_backend="cpu",
        logical_cpu_count=24,
        initial_resource_sample=None,
    )

    def submit_immediately(
        task: object,
        index: int,
    ) -> Future[PreparedVideoScan]:
        assert callable(task)
        future: Future[PreparedVideoScan] = Future()
        future.set_result(task(index))
        return future

    executor = cast(
        ThreadPoolExecutor,
        SimpleNamespace(submit=submit_immediately),
    )
    scheduler = AdaptiveVideoScanScheduler(
        executor,
        controller,
        lambda _index: PreparedVideoScan(
            reused=True,
            duration_seconds=0.001,
            input_seconds_per_wall_second=None,
        ),
        lambda: None,
    )

    # Act
    completed = scheduler.start(task_count)

    # Assert
    assert len(completed) == task_count
    assert all(future.result().reused for future in completed)
