"""Refinement Group schedulerのtest。"""

import os
import signal
import threading
import time
from functools import partial

import pytest

from src.video_selection.services.refinement_group_scheduler import (
    RefinementGroupScheduler,
)


def test_scheduler_runs_up_to_the_worker_limit() -> None:
    """taskが指定worker上限まで同時実行されること。

    Arrange:
        - 4件の独立taskと2 workerのschedulerが用意される
    Act:
        - 全taskが解決される
    Assert:
        - 最大2件が同時実行されること
        - 結果が入力順で返されること
    """
    # Arrange
    active_count = 0
    peak_count = 0
    active_lock = threading.Lock()
    first_pair_started = threading.Event()

    def task(index: int) -> int:
        nonlocal active_count, peak_count
        with active_lock:
            active_count += 1
            peak_count = max(peak_count, active_count)
            if active_count == 2:
                first_pair_started.set()
        try:
            assert first_pair_started.wait(timeout=1)
            return index
        finally:
            with active_lock:
                active_count -= 1

    scheduler = RefinementGroupScheduler(max_workers=2)

    # Act
    results = scheduler.resolve(tuple(partial(task, index) for index in range(4)))

    # Assert
    assert peak_count == 2
    assert results == (0, 1, 2, 3)


def test_scheduler_restores_results_to_input_order() -> None:
    """task完了順が異なっても結果が入力順へ復元されること。

    Arrange:
        - 後順位taskが先に完了する2件のtaskが用意される
    Act:
        - 2 workerでtaskが解決される
    Assert:
        - 戻り値が完了順ではなく入力順になること
    """
    # Arrange
    later_completed = threading.Event()

    def first() -> str:
        assert later_completed.wait(timeout=1)
        return "first"

    def second() -> str:
        later_completed.set()
        return "second"

    # Act
    results = RefinementGroupScheduler(max_workers=2).resolve((first, second))

    # Assert
    assert results == ("first", "second")


def test_scheduler_propagates_interrupt_without_starting_serial_siblings() -> None:
    """直列workerの割り込み後に後続taskが開始されないこと。

    Arrange:
        - KeyboardInterruptを送出する先頭taskと後続taskが用意される
    Act:
        - 1 workerでtask解決が試行される
    Assert:
        - KeyboardInterruptが維持されること
        - 後続taskが開始されないこと
    """
    # Arrange
    sibling_started = False

    def interrupt() -> None:
        raise KeyboardInterrupt

    def sibling() -> None:
        nonlocal sibling_started
        sibling_started = True

    # Act
    # Assert
    with pytest.raises(KeyboardInterrupt):
        RefinementGroupScheduler(max_workers=1).resolve((interrupt, sibling))
    assert not sibling_started


def test_scheduler_cancels_queued_tasks_after_user_interrupt() -> None:
    """並列実行中のuser interrupt後にqueued taskが開始されないこと。

    Arrange:
        - 2 workerを占有するtaskと2件のqueued taskが用意される
    Act:
        - main threadへSIGINTが送られる
    Assert:
        - KeyboardInterruptが維持されること
        - 実行中だった2件だけが開始済みになること
    """
    # Arrange
    started_indexes: list[int] = []
    started_lock = threading.Lock()
    first_pair_started = threading.Event()
    release_running = threading.Event()

    def task(index: int) -> int:
        with started_lock:
            started_indexes.append(index)
            if len(started_indexes) == 2:
                first_pair_started.set()
        assert release_running.wait(timeout=1)
        return index

    def interrupt_main_thread() -> None:
        assert first_pair_started.wait(timeout=1)
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.05)
        release_running.set()

    interrupter = threading.Thread(target=interrupt_main_thread)
    interrupter.start()

    # Act
    # Assert
    with pytest.raises(KeyboardInterrupt):
        RefinementGroupScheduler(max_workers=2).resolve(
            tuple(partial(task, index) for index in range(4))
        )
    interrupter.join(timeout=1)
    assert not interrupter.is_alive()
    assert sorted(started_indexes) == [0, 1]


def test_scheduler_does_not_start_waiting_tasks_after_worker_failure() -> None:
    """worker failure後に待機中taskが開始されないこと。

    Arrange:
        - 2 workerを同時開始するtaskと後続の待機taskが用意される
        - 先頭taskが失敗し、もう一方のtaskは短時間実行を続ける
    Act:
        - schedulerによるtask解決が試行される
    Assert:
        - 先頭の失敗が維持されること
        - failure検知前に実行中だった2件だけが開始されること
    """
    # Arrange
    started_indexes: list[int] = []
    started_lock = threading.Lock()
    first_pair_started = threading.Barrier(2)

    def task(index: int) -> int:
        with started_lock:
            started_indexes.append(index)
        if index < 2:
            first_pair_started.wait(timeout=1)
        if index == 0:
            raise RuntimeError("injected refinement group failure")
        if index == 1:
            time.sleep(0.05)
        return index

    # Act
    # Assert
    with pytest.raises(RuntimeError, match="injected refinement group failure"):
        RefinementGroupScheduler(max_workers=2).resolve(
            tuple(partial(task, index) for index in range(20))
        )
    assert sorted(started_indexes) == [0, 1]


@pytest.mark.parametrize("max_workers", [0, -1, True])
def test_scheduler_rejects_invalid_worker_limits(max_workers: int) -> None:
    """不正なworker上限が拒否されること。

    Arrange:
        - 1未満またはbooleanのworker値が用意される
    Act:
        - schedulerが構築される
    Assert:
        - 明確な設定errorが返されること
    """
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError, match="worker数"):
        RefinementGroupScheduler(max_workers=max_workers)
