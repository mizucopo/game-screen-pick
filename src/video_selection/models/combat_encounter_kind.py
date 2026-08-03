"""画像内で観測された戦闘種別。"""

from typing import Literal, cast, get_args

CombatEncounterKind = Literal[
    "not_combat",
    "ordinary",
    "major",
    "uncertain",
]

COMBAT_ENCOUNTER_KINDS = cast(
    tuple[CombatEncounterKind, ...],
    get_args(CombatEncounterKind),
)
