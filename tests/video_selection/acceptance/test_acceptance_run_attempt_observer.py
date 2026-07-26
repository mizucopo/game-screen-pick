"""Acceptance Run Attempt observerのtest。"""

from src.video_selection.acceptance.acceptance_run_attempt_observer import (
    AcceptanceRunAttemptObserver,
)
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.progress_event import ProgressEvent


def test_attempt_metrics_aggregate_cache_recompute_and_stage_duration() -> None:
    """Progress Eventがpathなしのattempt metricへ集計されること。

    Arrange:
        - stage開始、cache、完了eventが用意される
    Act:
        - acceptance observerがeventを収集して集計する
    Assert:
        - hit/miss/reuse/recomputeとdurationが合計されること
        - active Stageが完了後に解放されること
    """
    # Arrange
    observer = AcceptanceRunAttemptObserver()

    # Act
    observer.observe(
        ProgressEvent(
            kind="stage_started",
            severity="info",
            stage=ProcessingStage.SCAN_VIDEO,
            reason_code="stage_started",
        )
    )
    observer.observe(
        ProgressEvent(
            kind="cache",
            severity="info",
            stage=ProcessingStage.SCAN_VIDEO,
            cache_hit_count=1,
            cache_miss_count=1,
            reuse_count=1,
            recompute_count=1,
            reason_code="cache_observed",
        )
    )
    observer.observe(
        ProgressEvent(
            kind="stage_completed",
            severity="info",
            stage=ProcessingStage.SCAN_VIDEO,
            cache_hit_count=1,
            cache_miss_count=1,
            reuse_count=1,
            recompute_count=1,
            elapsed_seconds=3.5,
            reason_code="stage_completed",
        )
    )
    metrics = observer.attempt_metrics()

    # Assert
    assert observer.current_stage is None
    assert metrics["cache_hit_count"] == 1
    assert metrics["cache_miss_count"] == 1
    assert metrics["reuse_count"] == 1
    assert metrics["unexpected_recompute_count"] == 1
    assert metrics["stage_durations_seconds"] == {"scan-video": 3.5}


def test_attempt_metrics_include_interrupted_active_stage_duration() -> None:
    """中断時に実行中だったStageの経過時間も集計されること。

    Arrange:
        - scan-video開始後2.5秒でrunが中断されたevent列が用意される
    Act:
        - acceptance observerがeventを収集して集計する
    Assert:
        - 未完了scan-videoの2.5秒もStage時間へ含まれること
    """
    # Arrange
    observer = AcceptanceRunAttemptObserver()
    events = (
        ProgressEvent(
            kind="stage_started",
            severity="info",
            stage=ProcessingStage.SCAN_VIDEO,
            elapsed_seconds=0.0,
            reason_code="stage_started",
        ),
        ProgressEvent(
            kind="run_interrupted",
            severity="warning",
            stage=ProcessingStage.SCAN_VIDEO,
            elapsed_seconds=2.5,
            reason_code="user_interrupt",
        ),
    )

    # Act
    for event in events:
        observer.observe(event)
    metrics = observer.attempt_metrics()

    # Assert
    assert observer.current_stage is None
    assert metrics["stage_durations_seconds"] == {"scan-video": 2.5}
