"""model同期のrun別結果。"""

from enum import StrEnum


class ModelUpdateStatus(StrEnum):
    """実行identityとは分離して記録する更新結果。"""

    NOT_REQUESTED = "not_requested"
    UNCHANGED = "unchanged"
    UPDATED = "updated"
    BOOTSTRAPPED = "bootstrapped"
    UNAVAILABLE = "unavailable"
