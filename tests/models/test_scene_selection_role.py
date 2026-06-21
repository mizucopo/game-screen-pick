"""scene_selection_role.py の単体テスト."""

import pytest

from src.models.scene_selection_role import SceneSelectionRole


@pytest.mark.parametrize(
    "value,expected_role",
    [
        ("ordinary", SceneSelectionRole.ORDINARY),
        ("cinematic", SceneSelectionRole.CINEMATIC),
        ("recurring_gameplay", SceneSelectionRole.RECURRING_GAMEPLAY),
        (" cinematic ", SceneSelectionRole.CINEMATIC),
        (SceneSelectionRole.RECURRING_GAMEPLAY, SceneSelectionRole.RECURRING_GAMEPLAY),
    ],
)
def test_from_value_returns_known_scene_selection_role(
    value: object,
    expected_role: SceneSelectionRole,
) -> None:
    """既知のrole値がSceneSelectionRoleとして返されること.

    Arrange:
        - 既知のroleを表す値がある
    Act:
        - SceneSelectionRoleへ正規化される
    Assert:
        - 対応するSceneSelectionRoleが返されること
    """
    # Arrange
    # (パラメータ化されたvalueを使用)

    # Act
    role = SceneSelectionRole.from_value(value)

    # Assert
    assert role == expected_role


@pytest.mark.parametrize("value", ["gameplay", "", None, 1])
def test_from_value_normalizes_unknown_value_to_ordinary(value: object) -> None:
    """未知のrole値がordinaryとして扱われること.

    Arrange:
        - 未知または文字列以外のrole値がある
    Act:
        - SceneSelectionRoleへ正規化される
    Assert:
        - ordinaryが返されること
    """
    # Arrange
    # (パラメータ化されたvalueを使用)

    # Act
    role = SceneSelectionRole.from_value(value)

    # Assert
    assert role == SceneSelectionRole.ORDINARY
