"""Refinement Window Group taskを上限付きで並列実行する。"""

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TypeVar

GroupResult = TypeVar("GroupResult")


class RefinementGroupScheduler:
    """完了順に依存せずtask結果を入力順へ戻すscheduler。"""

    def __init__(self, *, max_workers: int) -> None:
        if (
            not isinstance(max_workers, int)
            or isinstance(max_workers, bool)
            or max_workers < 1
        ):
            raise ValueError("Refinement Group worker数は1以上である必要があります")
        self._max_workers = max_workers

    def resolve(
        self,
        tasks: tuple[Callable[[], GroupResult], ...],
    ) -> tuple[GroupResult, ...]:
        """taskをbounded実行し入力順の結果だけを返す。"""
        if len(tasks) < 2 or self._max_workers == 1:
            return tuple(task() for task in tasks)
        worker_count = min(self._max_workers, len(tasks))
        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="refinement-group",
        )
        in_flight: dict[Future[GroupResult], int] = {}
        next_task_index = 0

        def fill_available_workers() -> None:
            nonlocal next_task_index
            while next_task_index < len(tasks) and len(in_flight) < worker_count:
                in_flight[executor.submit(tasks[next_task_index])] = next_task_index
                next_task_index += 1

        try:
            fill_available_workers()
            results: dict[int, GroupResult] = {}
            while in_flight:
                completed, _pending = wait(
                    tuple(in_flight),
                    return_when=FIRST_COMPLETED,
                )
                completed_in_input_order = sorted(
                    completed,
                    key=in_flight.__getitem__,
                )
                for future in completed_in_input_order:
                    result = future.result()
                    results[in_flight.pop(future)] = result
                fill_available_workers()
            return tuple(results[index] for index in range(len(tasks)))
        except BaseException:
            for future in in_flight:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
