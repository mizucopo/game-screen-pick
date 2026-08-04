"""Adaptive Video Scan Schedulerのcontract test。"""

from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from threading import Condition, Event
from types import SimpleNamespace
from typing import cast

from src.video_selection.models.prepared_video_scan import PreparedVideoScan
from src.video_selection.models.video_scan_resource_sample import (
    VideoScanResourceSample,
)
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


def test_refinement_reservation_caps_growth_and_interrupt_refill() -> None:
    """Refinement予約がscan増員と割り込み後の補充を制限すること。

    Arrange:
        - 24 logical CPUで3から6 workerへ増加できるNVDEC Schedulerが用意される
        - 3 Refinement worker分の12 CPUが予約される
    Act:
        - rolling余力を持つ4件のscanが予約中に順次完了される
    Assert:
        - Controllerが4 workerへ増加してもactive scanは3件に保たれること
        - 予約解放後に4件目のactive scanが投入されること
        - 次の予約中断とscan完了が競合しても5件目が投入されないこと
    """
    # Arrange
    task_count = 12
    sample = VideoScanResourceSample(
        cpu_percent=45.0,
        memory_percent=50.0,
        decoder_percent=40.0,
        gpu_percent=20.0,
        vram_percent=22.0,
        disk_busy_percent=40.0,
        disk_read_mib_per_second=300.0,
    )
    controller = AdaptiveVideoScanController(
        video_count=task_count,
        configured_workers="auto",
        auto_max_workers=6,
        decode_backend="nvdec",
        logical_cpu_count=24,
        initial_resource_sample=sample,
    )
    releases = tuple(Event() for _ in range(task_count))
    task_condition = Condition()
    active_count = 0
    peak_count = 0
    started_count = 0

    def run_scan(index: int) -> PreparedVideoScan:
        nonlocal active_count, peak_count, started_count
        with task_condition:
            active_count += 1
            peak_count = max(peak_count, active_count)
            started_count += 1
            task_condition.notify_all()
        try:
            assert releases[index].wait(timeout=2)
            return PreparedVideoScan(
                reused=False,
                duration_seconds=0.1,
                input_seconds_per_wall_second=1.1,
            )
        finally:
            with task_condition:
                active_count -= 1
                task_condition.notify_all()

    def has_started(expected_count: int) -> bool:
        return started_count >= expected_count

    # Act
    with ThreadPoolExecutor(max_workers=6) as executor:
        scheduler = AdaptiveVideoScanScheduler(
            executor,
            controller,
            run_scan,
            lambda: sample,
        )
        completed = scheduler.start(task_count)
        with task_condition:
            assert task_condition.wait_for(lambda: started_count == 3, timeout=2)
        with scheduler.reserve_refinement_workers(
            desired_workers=3,
            logical_cpu_count=24,
        ) as refinement_workers:
            for completed_index in range(4):
                releases[completed_index].set()
                expected_started_count = completed_index + 4
                with task_condition:
                    assert task_condition.wait_for(
                        partial(has_started, expected_started_count),
                        timeout=2,
                    )
            with task_condition:
                assert active_count == 3
                assert peak_count == 3
            assert refinement_workers == 3
            assert controller.current_workers == 4
        with task_condition:
            assert task_condition.wait_for(lambda: active_count == 4, timeout=2)
            assert peak_count == 4
        try:
            with scheduler.reserve_refinement_workers(
                desired_workers=2,
                logical_cpu_count=24,
            ):
                releases[4].set()
                with task_condition:
                    assert task_condition.wait_for(
                        partial(has_started, 9),
                        timeout=2,
                    )
                assert controller.current_workers == 5
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
        releases[5].set()
        with task_condition:
            assert task_condition.wait_for(lambda: active_count == 3, timeout=2)
            assert peak_count == 4
            assert started_count == 9
        scheduler.cancel_pending()
        for release in releases:
            release.set()

    # Assert
    assert len(completed) == task_count
