import pytest

from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.progress_event import ProgressEvent


def test_progress_event_carries_safe_typed_stage_observation() -> None:
    """安全なStage進捗が表示文でなく型付き値として保持されること。

    Arrange:
        - Stage、Video、count、cache、経過時間、ETAが用意される
    Act:
        - renderer非依存のProgress Eventが生成される
    Assert:
        - 入力した型付き観測値がそのまま公開されること
    """
    # Arrange
    stage = ProcessingStage.SCAN_VIDEO

    # Act
    event = ProgressEvent(
        kind="progress",
        severity="info",
        stage=stage,
        stage_index=2,
        video_order=1,
        video_count=3,
        video_relative_path="chapter-01/movie.mkv",
        processed_count=5,
        total_count=10,
        cache_hit_count=1,
        cache_miss_count=4,
        reuse_count=1,
        recompute_count=4,
        elapsed_seconds=31.0,
        eta_seconds=45.0,
        estimation_state="available",
        work_unit_kind="video",
        reason_code="stage_progress",
    )

    # Assert
    assert event == ProgressEvent(
        kind="progress",
        severity="info",
        stage=stage,
        stage_index=2,
        video_order=1,
        video_count=3,
        video_relative_path="chapter-01/movie.mkv",
        processed_count=5,
        total_count=10,
        cache_hit_count=1,
        cache_miss_count=4,
        reuse_count=1,
        recompute_count=4,
        elapsed_seconds=31.0,
        eta_seconds=45.0,
        estimation_state="available",
        work_unit_kind="video",
        reason_code="stage_progress",
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"video_relative_path": "/private/movie.mkv"}, "absolute path"),
        ({"processed_count": 6, "total_count": 5}, "processed count"),
        ({"eta_seconds": 1.0, "estimation_state": "estimating"}, "ETA state"),
    ],
)
def test_progress_event_rejects_unsafe_or_inconsistent_observation(
    overrides: dict[str, object],
    message: str,
) -> None:
    """安全でないpathまたは矛盾する進捗値が拒否されること。

    Arrange:
        - 基本となるStage進捗と不正な上書き値が用意される
    Act:
        - Progress Eventの生成が試行される
    Assert:
        - 不正な観測値がValueErrorで拒否されること
    """
    # Arrange
    values: dict[str, object] = {
        "kind": "progress",
        "severity": "info",
        "stage": ProcessingStage.SCAN_VIDEO,
        "stage_index": 1,
        "processed_count": 5,
        "total_count": 5,
        "elapsed_seconds": 30.0,
        "estimation_state": "estimating",
    }
    values.update(overrides)

    # Act / Assert
    with pytest.raises(ValueError, match=message):
        ProgressEvent(**values)  # type: ignore[arg-type]
