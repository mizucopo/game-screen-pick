"""Migration state transition result。"""

from dataclasses import dataclass

from migration_state import MigrationState


@dataclass(frozen=True)
class Transition:
    """state transitionの結果。"""

    state: MigrationState
    accepted: bool
    message: str
    blockers: tuple[str, ...] = ()
