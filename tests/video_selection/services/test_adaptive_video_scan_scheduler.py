"""Adaptive Video Scan Schedulerのcontract test。"""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from threading import Condition, Event
from types import SimpleNamespace
from typing import cast

import pytest

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


def test_waiting_refinement_is_prioritized_before_scan_refill() -> None:
    """待機中のRefinement容量が完了したscanの補充より先に確保されること。

    Arrange:
        - 16 logical CPUを使い切る2件のCPU scanと後続taskが用意される
        - 次に処理するVideoのRefinement要求がscan完了前に登録される
    Act:
        - 先頭scanが完了されRefinement workerが予約される
    Assert:
        - 後続scanが補充される前にRefinementへCPU容量が渡されること
        - Refinement予約の解放後に後続scanが補充されること
    """
    # Arrange
    task_count = 4
    controller = AdaptiveVideoScanController(
        video_count=task_count,
        configured_workers=2,
        auto_max_workers=6,
        decode_backend="cpu",
        logical_cpu_count=16,
        initial_resource_sample=None,
    )
    releases = tuple(Event() for _ in range(task_count))
    task_condition = Condition()
    started_count = 0

    def run_scan(index: int) -> PreparedVideoScan:
        nonlocal started_count
        with task_condition:
            started_count += 1
            task_condition.notify_all()
        assert releases[index].wait(timeout=2)
        return PreparedVideoScan(
            reused=False,
            duration_seconds=0.1,
            input_seconds_per_wall_second=1.0,
        )

    # Act
    with ThreadPoolExecutor(max_workers=2) as executor:
        scheduler = AdaptiveVideoScanScheduler(
            executor,
            controller,
            run_scan,
            lambda: None,
        )
        completed = scheduler.start(task_count)
        with task_condition:
            assert task_condition.wait_for(lambda: started_count == 2, timeout=2)
        with scheduler.prioritize_refinement_capacity(logical_cpu_count=16):
            releases[0].set()
            assert completed[0].result(timeout=2) is not None
            with scheduler.reserve_refinement_workers(
                desired_workers=1,
                logical_cpu_count=16,
            ) as refinement_workers:
                assert refinement_workers == 1
                with task_condition:
                    assert started_count == 2
        with task_condition:
            assert task_condition.wait_for(lambda: started_count == 3, timeout=2)
        scheduler.cancel_pending()
        for release in releases:
            release.set()

    # Assert
    assert completed[0].done()


def test_refinement_reservation_wait_aborts_when_background_scan_fails() -> None:
    """CPU容量待機中にbackground scanが失敗すると予約が中止されること。

    Arrange:
        - 16 logical CPUを使い切る2件のCPU scanが用意される
        - 一方のscanがRefinement予約開始後に失敗するよう調整される
    Act:
        - Refinement workerの予約がCPU容量を待機する
    Assert:
        - scan失敗がそのまま返されRefinement処理が開始されないこと
    """
    # Arrange
    controller = AdaptiveVideoScanController(
        video_count=2,
        configured_workers=2,
        auto_max_workers=6,
        decode_backend="cpu",
        logical_cpu_count=16,
        initial_resource_sample=None,
    )
    scans_started = Condition()
    started_count = 0
    release_first = Event()
    reservation_started = Event()

    def run_scan(index: int) -> PreparedVideoScan:
        nonlocal started_count
        with scans_started:
            started_count += 1
            scans_started.notify_all()
        if index == 0:
            assert release_first.wait(timeout=2)
            return PreparedVideoScan(
                reused=False,
                duration_seconds=0.1,
                input_seconds_per_wall_second=1.0,
            )
        assert reservation_started.wait(timeout=2)
        time.sleep(0.05)
        raise OSError("injected background scan failure")

    # Act
    with ThreadPoolExecutor(max_workers=2) as executor:
        scheduler = AdaptiveVideoScanScheduler(
            executor,
            controller,
            run_scan,
            lambda: None,
        )
        completed = scheduler.start(2)
        with scans_started:
            assert scans_started.wait_for(lambda: started_count == 2, timeout=2)
        reservation_started.set()

        # Assert
        with (
            pytest.raises(OSError, match="injected background scan failure"),
            scheduler.reserve_refinement_workers(
                desired_workers=1,
                logical_cpu_count=16,
            ),
        ):
            raise AssertionError("Refinement処理は開始されません")
        release_first.set()
        scheduler.cancel_pending()
        with pytest.raises(OSError, match="injected background scan failure"):
            completed[1].result(timeout=2)
