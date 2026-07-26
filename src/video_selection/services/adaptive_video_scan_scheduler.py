"""Video Scan taskを動的worker上限に従って遅延投入する。"""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock

from ..models.prepared_video_scan import PreparedVideoScan
from ..models.video_scan_resource_sample import VideoScanResourceSample
from .adaptive_video_scan_controller import AdaptiveVideoScanController
from .sample_video_scan_resources_safely import sample_video_scan_resources_safely

ScanTask = Callable[[int], PreparedVideoScan]
ResourceSampler = Callable[[], VideoScanResourceSample | None]


class AdaptiveVideoScanScheduler:
    """active taskを止めずscan完了境界でadmissionだけを変更する。"""

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        controller: AdaptiveVideoScanController,
        task: ScanTask,
        resource_sampler: ResourceSampler,
    ) -> None:
        self._executor = executor
        self._controller = controller
        self._task = task
        self._resource_sampler = resource_sampler
        self._lock = RLock()
        self._slots: tuple[Future[PreparedVideoScan], ...] = ()
        self._submitted: dict[int, Future[PreparedVideoScan]] = {}
        self._next_index = 0
        self._active_count = 0
        self._stopped = False
        self._filling = False

    def start(self, task_count: int) -> tuple[Future[PreparedVideoScan], ...]:
        """初期worker上限までtaskを投入し順序付きFuture列を返す。"""
        if task_count < 1:
            raise ValueError("Video Scan task件数は正である必要があります")
        with self._lock:
            if self._slots:
                raise RuntimeError("Video Scan Schedulerは再利用できません")
            self._slots = tuple(Future() for _ in range(task_count))
            self._fill_available_slots()
            return self._slots

    def cancel_pending(self) -> None:
        """未開始taskを止めactive taskの完了を待てる状態にする。"""
        with self._lock:
            self._stopped = True
            for future in self._submitted.values():
                future.cancel()
            for index in range(self._next_index, len(self._slots)):
                self._slots[index].cancel()

    def _fill_available_slots(self) -> None:
        if self._filling:
            return
        self._filling = True
        try:
            while (
                not self._stopped
                and self._active_count < self._controller.current_workers
                and self._next_index < len(self._slots)
            ):
                index = self._next_index
                self._next_index += 1
                self._active_count += 1
                actual = self._executor.submit(self._task, index)
                self._submitted[index] = actual

                def complete_task(
                    completed: Future[PreparedVideoScan],
                    task_index: int = index,
                ) -> None:
                    self._complete(task_index, completed)

                actual.add_done_callback(complete_task)
        finally:
            self._filling = False

    def _complete(
        self,
        index: int,
        actual: Future[PreparedVideoScan],
    ) -> None:
        try:
            result = actual.result()
            error: BaseException | None = None
        except BaseException as caught:
            result = None
            error = caught
        resource_sample = (
            None
            if (
                result is None
                or result.reused
                or not self._controller.resource_sampling_enabled
            )
            else self._safe_resource_sample()
        )
        with self._lock:
            self._active_count -= 1
            slot = self._slots[index]
            if error is not None:
                if not slot.done():
                    slot.set_exception(error)
                self._stopped = True
                for task_index, future in self._submitted.items():
                    if task_index != index:
                        future.cancel()
                for task_index in range(self._next_index, len(self._slots)):
                    self._slots[task_index].cancel()
                return
            if result is None:
                raise AssertionError("完了したVideo Scan resultがありません")
            if not slot.done():
                slot.set_result(result)
            if self._stopped:
                return
            self._controller.observe_scan_completion(
                reused=result.reused,
                input_seconds_per_wall_second=(result.input_seconds_per_wall_second),
                resource_sample=resource_sample,
            )
            self._fill_available_slots()

    def _safe_resource_sample(self) -> VideoScanResourceSample | None:
        return sample_video_scan_resources_safely(self._resource_sampler)
