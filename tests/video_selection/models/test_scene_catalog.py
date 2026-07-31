import pytest

from src.video_selection.models.scene_catalog import SceneCatalog
from src.video_selection.models.scene_catalog_entry import SceneCatalogEntry


def test_catalog_requires_unique_scenes_and_ordinary_other() -> None:
    """共有Scene Catalogが3〜8件と通常扱いのotherを要求すること。

    Arrange:
        - 重複しない3件のsceneとordinary roleのotherが用意される
    Act:
        - Scene Catalogが構築される
    Assert:
        - scene順とslug lookupが保持されること
    """
    # Arrange
    scenes = (
        SceneCatalogEntry(
            "exploration", "探索", "フィールド探索", "exploration", "ordinary"
        ),
        SceneCatalogEntry("battle", "戦闘", "通常戦闘", "combat", "recurring_gameplay"),
        SceneCatalogEntry("other", "その他", "分類不能", "other", "ordinary"),
    )

    # Act
    catalog = SceneCatalog(scenes)

    # Assert
    assert catalog.slugs == ("exploration", "battle", "other")
    assert catalog.for_slug("battle").display_name == "戦闘"
    assert catalog.for_slug("battle").scene_kind == "combat"


@pytest.mark.parametrize(
    "scenes",
    [
        (
            SceneCatalogEntry("battle", "戦闘", "通常戦闘", "combat", "ordinary"),
            SceneCatalogEntry("battle", "戦闘2", "別の戦闘", "combat", "cinematic"),
            SceneCatalogEntry("other", "その他", "分類不能", "other", "ordinary"),
        ),
        (
            SceneCatalogEntry(
                "exploration", "探索", "フィールド探索", "exploration", "ordinary"
            ),
            SceneCatalogEntry("battle", "戦闘", "通常戦闘", "combat", "ordinary"),
            SceneCatalogEntry("other", "その他", "分類不能", "other", "cinematic"),
        ),
        (
            SceneCatalogEntry(
                "exploration", "探索", "フィールド探索", "exploration", "ordinary"
            ),
            SceneCatalogEntry("battle", "戦闘", "通常戦闘", "combat", "ordinary"),
            SceneCatalogEntry("other", "その他", "分類不能", "exploration", "ordinary"),
        ),
    ],
)
def test_invalid_catalog_domain_is_rejected(
    scenes: tuple[SceneCatalogEntry, ...],
) -> None:
    """slug重複またはkind・roleが不正なotherが拒否されること。

    Arrange:
        - domain contractに違反するscene列が用意される
    Act:
        - Scene Catalogの構築が試行される
    Assert:
        - domain validation errorが返されること
    """
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError, match="Scene Catalog"):
        SceneCatalog(scenes)
