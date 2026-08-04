"""Refinement Window Group taskを上限付きで並列実行する。"""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TypeVar, cast

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
        executor = ThreadPoolExecutor(
            max_workers=min(self._max_workers, len(tasks)),
            thread_name_prefix="refinement-group",
        )
        futures: list[Future[GroupResult]] = []
        try:
            for task in tasks:
                futures.append(executor.submit(task))
            future_indexes = {future: index for index, future in enumerate(futures)}
            results: list[GroupResult | None] = [None] * len(futures)
            for future in as_completed(futures):
                results[future_indexes[future]] = future.result()
            return tuple(cast(GroupResult, result) for result in results)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
