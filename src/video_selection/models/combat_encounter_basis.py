"""Combat Encounter Kindを支持する画像内根拠。"""

from typing import Literal, cast, get_args

from .combat_encounter_kind import CombatEncounterKind

CombatEncounterBasis = Literal[
    "none",
    "ordinary_opponent_presentation",
    "ordinary_encounter_presentation",
    "major_opponent_presentation",
    "major_encounter_presentation",
    "ambiguous",
]

COMBAT_ENCOUNTER_BASES = cast(
    tuple[CombatEncounterBasis, ...],
    get_args(CombatEncounterBasis),
)
ORDINARY_COMBAT_ENCOUNTER_BASES = frozenset(
    {
        "ordinary_opponent_presentation",
        "ordinary_encounter_presentation",
    }
)
MAJOR_COMBAT_ENCOUNTER_BASES = frozenset(
    {
        "major_opponent_presentation",
        "major_encounter_presentation",
    }
)


def combat_encounter_classification_is_valid(
    combat_encounter_kind: CombatEncounterKind,
    combat_encounter_basis: CombatEncounterBasis,
) -> bool:
    """戦闘種別と積極的な画像内根拠の関係が整合するかを返す。"""
    if combat_encounter_kind == "not_combat":
        return combat_encounter_basis == "none"
    if combat_encounter_kind == "ordinary":
        return combat_encounter_basis in ORDINARY_COMBAT_ENCOUNTER_BASES
    if combat_encounter_kind == "major":
        return combat_encounter_basis in MAJOR_COMBAT_ENCOUNTER_BASES
    return combat_encounter_basis == "ambiguous"
