"""Video Set選定のRun Status test。"""

import pytest

from src.video_selection.models.run_status import RunStatus


@pytest.mark.parametrize(
    ("requested_count", "selected_count", "expected"),
    (
        (1, 1, RunStatus.COMPLETED),
        (2, 1, RunStatus.COMPLETED_WITH_WARNINGS),
    ),
)
def test_status_is_derived_from_selection_counts(
    requested_count: int,
    selected_count: int,
    expected: RunStatus,
) -> None:
    """選定枚数不足だけがwarning付き正常終了として判定されること。

    Arrange:
        - 要求枚数、選定枚数、期待statusが用意される
    Act:
        - selection countからRun Statusが導出される
    Assert:
        - 完全選定またはwarning付き選定が返されること
    """
    # Arrange
    selection_counts = (requested_count, selected_count)

    # Act
    actual = RunStatus.from_selection_counts(*selection_counts)

    # Assert
    assert actual is expected


def test_other_warning_marks_complete_selection_as_warning() -> None:
    """選定枚数以外のwarningでもwarning付き正常終了になること。

    Arrange:
        - 要求枚数を満たす選定結果と他のwarningが用意される
    Act:
        - selection countとwarningからRun Statusが導出される
    Assert:
        - completed_with_warningsが返されること
    """
    # Arrange
    selection_counts = (1, 1)

    # Act
    actual = RunStatus.from_selection_counts(
        *selection_counts,
        has_other_warnings=True,
    )

    # Assert
    assert actual is RunStatus.COMPLETED_WITH_WARNINGS
