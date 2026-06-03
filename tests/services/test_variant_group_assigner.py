"""VariantGroupAssignerの単体テスト."""

from src.services.variant_group_assigner import VariantGroupAssigner
from tests.conftest import _feature, _near_duplicate
from tests.services.test_dynamic_scene_selector import build_dynamic_candidate


def test_assign_groups_near_duplicates_within_same_scene() -> None:
    """同一scene内の近い画像が同じvariant groupへ割り当てられること.

    Arrange:
        - 同じsceneに近い特徴の候補と離れた特徴の候補がある
    Act:
        - variant groupが割り当てられる
    Assert:
        - 近い候補だけが同じgroupになること
    """
    # Arrange
    base = _feature(1)
    near = _near_duplicate(base, 2)
    candidates = [
        build_dynamic_candidate("/tmp/a.jpg", "conversation", base),
        build_dynamic_candidate("/tmp/b.jpg", "conversation", near),
        build_dynamic_candidate("/tmp/c.jpg", "conversation", _feature(30)),
    ]
    assigner = VariantGroupAssigner(similarity_threshold=0.95)

    # Act
    groups = assigner.assign(candidates)

    # Assert
    assert groups["/tmp/a.jpg"] == "conversation_001"
    assert groups["/tmp/b.jpg"] == "conversation_001"
    assert groups["/tmp/c.jpg"] == "conversation_002"
