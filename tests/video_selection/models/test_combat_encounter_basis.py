"""Combat Encounter Kindと画像内根拠の関係を検証する。"""

import pytest

from src.video_selection.models.combat_encounter_basis import (
    CombatEncounterBasis,
    combat_encounter_classification_is_valid,
)
from src.video_selection.models.combat_encounter_kind import CombatEncounterKind


@pytest.mark.parametrize(
    ("combat_encounter_kind", "combat_encounter_basis", "expected"),
    (
        ("not_combat", "none", True),
        ("ordinary", "ordinary_opponent_presentation", True),
        ("ordinary", "ordinary_encounter_presentation", True),
        ("major", "major_opponent_presentation", True),
        ("major", "major_encounter_presentation", True),
        ("uncertain", "ambiguous", True),
        ("ordinary", "ambiguous", False),
        ("ordinary", "none", False),
        ("ordinary", "major_opponent_presentation", False),
        ("major", "ordinary_opponent_presentation", False),
        ("uncertain", "ordinary_encounter_presentation", False),
        ("not_combat", "ambiguous", False),
    ),
)
def test_combat_encounter_kind_requires_matching_positive_basis(
    combat_encounter_kind: CombatEncounterKind,
    combat_encounter_basis: CombatEncounterBasis,
    expected: bool,
) -> None:
    """戦闘種別が対応する積極的な画像内根拠だけで支持されること。

    Arrange:
        - 戦闘種別と通常・主要・曖昧な画像内根拠の組が用意される
    Act:
        - 戦闘種別と根拠の関係が検証される
    Assert:
        - ordinaryが消極的理由や曖昧な根拠では成立しないこと
    """
    # Arrange
    kind = combat_encounter_kind
    basis = combat_encounter_basis

    # Act
    actual = combat_encounter_classification_is_valid(kind, basis)

    # Assert
    assert actual is expected
