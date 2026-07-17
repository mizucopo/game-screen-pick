from threading import Event

from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.services.external_work_monitor import ExternalWorkMonitor
from src.video_selection.services.run_progress_tracker import RunProgressTracker
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver


def test_external_work_monitor_emits_heartbeat_every_thirty_seconds() -> None:
    """blocking external work中に開始eventと30秒heartbeatが発行されること。

    Arrange:
        - fake clockと30秒進めるwait boundaryが用意される
    Act:
        - active Stage内でblocking external operationが実行される
    Assert:
        - 開始直後と30秒後に根拠のない進捗率なしでeventが発行されること
    """
    # Arrange
    observer = RecordingRunObserver()
    current_time = [0.0]
    second_wait_started = Event()
    waits: list[float] = []
    wait_count = 0

    def wait_for_stop(stop: Event, timeout_seconds: float) -> bool:
        nonlocal wait_count
        waits.append(timeout_seconds)
        wait_count += 1
        if wait_count == 1:
            current_time[0] += timeout_seconds
            return False
        second_wait_started.set()
        return stop.wait(timeout=1.0)

    tracker = RunProgressTracker(observer, clock=lambda: current_time[0])
    tracker.start_run()
    tracker.start_stage(
        ProcessingStage.COLLECT_CONTEXT,
        work_unit_kind="audio_chunk",
    )
    monitor = ExternalWorkMonitor(tracker, wait_for_stop=wait_for_stop)

    def external_operation() -> str:
        if not second_wait_started.wait(timeout=1.0):
            msg = "heartbeatが発行されませんでした"
            raise RuntimeError(msg)
        return "completed"

    # Act
    result = monitor.run(
        external_operation,
        reason_code="speech_recognition_started",
    )

    # Assert
    external_events = observer.progress_events[-2:]
    assert (
        result,
        waits,
        tuple(event.kind for event in external_events),
        tuple(event.elapsed_seconds for event in external_events),
        tuple(event.processed_count for event in external_events),
        tuple(event.eta_seconds for event in external_events),
    ) == (
        "completed",
        [30.0, 30.0],
        ("external_work_started", "heartbeat"),
        (0.0, 30.0),
        (None, None),
        (None, None),
    )
