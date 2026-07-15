"""Video Set選定applicationの正常終了status。"""

from enum import StrEnum


class RunStatus(StrEnum):
    """完全選定またはwarning付き選定を表す。"""

    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"

    @classmethod
    def from_selection_counts(
        cls,
        requested_count: int,
        selected_count: int,
    ) -> "RunStatus":
        """要求枚数と選定枚数から正常終了statusを返す。"""
        if selected_count < requested_count:
            return cls.COMPLETED_WITH_WARNINGS
        return cls.COMPLETED
