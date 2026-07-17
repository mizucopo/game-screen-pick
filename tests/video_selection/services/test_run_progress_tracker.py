from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.services.run_progress_tracker import RunProgressTracker
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver


def test_run_progress_tracker_emits_serial_typed_lifecycle() -> None:
    """runとStageの直列ライフサイクルが型付きeventで通知されること。

    Arrange:
        - Recording Run Observerと制御可能なmonotonic clockが用意される
    Act:
        - run、Stage、進捗、Stage完了、run完了が順番に記録される
    Assert:
        - event kind、Stage番号、Stage経過時間が契約順で通知されること
    """
    # Arrange
    observer = RecordingRunObserver()
    now = 10.0
    tracker = RunProgressTracker(observer, clock=lambda: now)

    # Act
    tracker.start_run()
    tracker.start_stage(
        ProcessingStage.SCAN_VIDEO,
        stage_count=2,
        video_order=1,
        video_count=1,
        video_relative_path="chapter-01.mkv",
        work_unit_kind="video",
    )
    now = 15.0
    tracker.progress(processed_count=1, total_count=1)
    tracker.complete_stage()
    tracker.start_stage(
        ProcessingStage.EXTRACT_FRAME_CANDIDATES,
        stage_count=2,
        video_order=1,
        video_count=1,
        video_relative_path="chapter-01.mkv",
        work_unit_kind="candidate",
    )
    now = 19.0
    tracker.complete_stage()
    tracker.complete_run()

    # Assert
    assert tuple(
        (event.kind, event.stage_index, event.elapsed_seconds)
        for event in observer.progress_events
    ) == (
        ("run_started", None, None),
        ("stage_started", 1, 0.0),
        ("progress", 1, 5.0),
        ("stage_completed", 1, 5.0),
        ("stage_started", 2, 0.0),
        ("stage_completed", 2, 4.0),
        ("run_completed", None, None),
    )


def test_run_progress_tracker_emits_eta_from_known_work_series() -> None:
    """系列別残件数が既知の場合だけsampleからETA付きeventが通知されること。

    Arrange:
        - active Annotation Stageへ10秒のrecompute sampleが5件記録される
    Act:
        - Stage開始50秒後に残りrecompute 5件の進捗が通知される
    Assert:
        - available状態と50秒のETAがProgress Eventへ含まれること
    """
    # Arrange
    observer = RecordingRunObserver()
    current_time = [0.0]
    tracker = RunProgressTracker(observer, clock=lambda: current_time[0])
    tracker.start_run()
    tracker.start_stage(
        ProcessingStage.ANNOTATE_CANDIDATE,
        work_unit_kind="candidate",
    )
    for _ in range(5):
        tracker.record_work_sample("recompute", 10.0)
    current_time[0] = 50.0

    # Act
    tracker.progress(
        processed_count=5,
        total_count=10,
        remaining_reuse_count=0,
        remaining_recompute_count=5,
    )

    # Assert
    event = observer.progress_events[-1]
    assert (
        event.kind,
        event.estimation_state,
        event.eta_seconds,
        event.elapsed_seconds,
    ) == ("progress", "available", 50.0, 50.0)


def test_run_progress_tracker_excludes_zero_duration_sample() -> None:
    """0秒の完了sampleがComparable Work Seriesから除外されること。

    Arrange:
        - 0秒のcurrent sampleと4件の10秒sampleが記録される
    Act:
        - 5件目の有効sampleの前後でETAが要求される
    Assert:
        - 0秒は件数に含まれず、有効sampleが5件になった後だけETAが返されること
    """
    # Arrange
    observer = RecordingRunObserver()
    current_time = [0.0]
    tracker = RunProgressTracker(observer, clock=lambda: current_time[0])
    tracker.start_run()
    tracker.start_stage(
        ProcessingStage.ANNOTATE_CANDIDATE,
        work_unit_kind="candidate",
    )
    tracker.record_work_sample("recompute")
    for _ in range(4):
        tracker.record_work_sample("recompute", 10.0)
    current_time[0] = 30.0

    # Act
    tracker.progress(
        remaining_reuse_count=0,
        remaining_recompute_count=1,
    )
    before_fifth_sample = observer.progress_events[-1]
    tracker.record_work_sample("recompute", 10.0)
    tracker.progress(
        remaining_reuse_count=0,
        remaining_recompute_count=1,
    )
    after_fifth_sample = observer.progress_events[-1]

    # Assert
    assert (
        before_fifth_sample.estimation_state,
        before_fifth_sample.eta_seconds,
        after_fifth_sample.estimation_state,
        after_fifth_sample.eta_seconds,
    ) == ("estimating", None, "available", 10.0)


def test_progress_without_comparable_series_omits_eta() -> None:
    """比較可能なwork系列がなくても件数進捗だけが通知されること。

    Arrange:
        - work unit種別を持たないactive Stageが用意される
    Act:
        - 30秒経過後に既知totalなしの処理済み件数が通知される
    Assert:
        - progress eventは発行され、ETAと根拠のないtotalが省略されること
    """
    # Arrange
    observer = RecordingRunObserver()
    current_time = [0.0]
    tracker = RunProgressTracker(observer, clock=lambda: current_time[0])
    tracker.start_run()
    tracker.start_stage(ProcessingStage.SELECT_IMAGES)
    current_time[0] = 31.0

    # Act
    tracker.progress(processed_count=1)

    # Assert
    event = observer.progress_events[-1]
    assert (
        event.kind,
        event.processed_count,
        event.total_count,
        event.work_unit_kind,
        event.estimation_state,
        event.eta_seconds,
    ) == ("progress", 1, None, None, "unavailable", None)
