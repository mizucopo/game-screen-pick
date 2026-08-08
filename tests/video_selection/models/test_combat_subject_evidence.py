"""Combat Subject Evidenceの公開domain contractのtest。"""

import pytest

from src.video_selection.models.combat_subject_evidence import CombatSubjectEvidence


def test_distinctive_evidence_requires_complete_visual_features() -> None:
    """distinctiveな外見根拠には比較可能な全特徴が要求されること。

    Arrange:
        - 色と固有特徴がないdistinctiveな外見観測が用意される
    Act:
        - Combat Subject Evidenceの構築が試行される
    Assert:
        - 不完全な画像内根拠として拒否されること
    """
    # Arrange
    colors = ()
    traits = ()

    # Act
    # Assert
    with pytest.raises(ValueError, match="Combat Subject Evidence"):
        CombatSubjectEvidence(
            body_plan="quadruped",
            scale="large",
            surface="organic",
            colors=colors,
            traits=traits,
            distinctiveness="distinctive",
        )
