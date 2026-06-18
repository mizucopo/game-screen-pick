"""Ollama応答parserの単体テスト."""

import pytest

from src.models.scene_catalog_entry import SceneCatalogEntry
from src.models.scene_selection_role import SceneSelectionRole
from src.services.ollama_response_parser import OllamaResponseParser


def test_parse_catalog_response_returns_scene_catalog() -> None:
    """scene catalog のJSON応答が解析されること.

    Arrange:
        - 3個以上のsceneとotherを含むJSON応答がある
    Act:
        - catalog応答が解析される
    Assert:
        - scene slug、日本語表示名、説明が保持されること
    """
    # Arrange
    content = """
    {
      "scenes": [
        {
          "slug": "battle",
          "display_name": "戦闘",
          "description": "敵と戦う場面",
          "selection_role": "ordinary"
        },
        {
          "slug": "conversation",
          "display_name": "会話",
          "description": "人物同士の会話",
          "selection_role": "ordinary"
        },
        {"slug": "other", "display_name": "その他", "description": "分類しにくい場面"}
      ]
    }
    """

    # Act
    result = OllamaResponseParser.parse_catalog_response(content)

    # Assert
    assert result == [
        SceneCatalogEntry("battle", "戦闘", "敵と戦う場面"),
        SceneCatalogEntry("conversation", "会話", "人物同士の会話"),
        SceneCatalogEntry("other", "その他", "分類しにくい場面"),
    ]


def test_parse_catalog_response_assigns_scene_selection_roles() -> None:
    """catalog応答のselection roleがsceneへ反映されること.

    Arrange:
        - recurring gameplay、cinematic、otherを含むcatalog応答がある
        - otherにはcinematic roleが返されている
    Act:
        - catalog応答が解析される
    Assert:
        - roleがscene catalog entryへ反映されること
        - other sceneはordinaryへ正規化されること
    """
    # Arrange
    content = """
    {
      "scenes": [
        {
          "slug": "battle",
          "display_name": "戦闘",
          "description": "敵と戦う場面",
          "selection_role": "recurring_gameplay"
        },
        {
          "slug": "event",
          "display_name": "イベント",
          "description": "演出中心の場面",
          "selection_role": "cinematic"
        },
        {
          "slug": "other",
          "display_name": "その他",
          "description": "分類しにくい場面",
          "selection_role": "cinematic"
        }
      ]
    }
    """

    # Act
    result = OllamaResponseParser.parse_catalog_response(content)

    # Assert
    assert [scene.selection_role for scene in result] == [
        SceneSelectionRole.RECURRING_GAMEPLAY,
        SceneSelectionRole.CINEMATIC,
        SceneSelectionRole.ORDINARY,
    ]


def test_parse_catalog_response_rejects_missing_scene_selection_role() -> None:
    """other以外のsceneでselection role欠落が不正応答として扱われること.

    Arrange:
        - other以外のsceneにselection_roleがないcatalog応答がある
    Act:
        - catalog応答が解析される
    Assert:
        - selection_role欠落として失敗すること
    """
    # Arrange
    content = """
    {
      "scenes": [
        {"slug": "battle", "display_name": "戦闘", "description": "敵と戦う場面"},
        {
          "slug": "event",
          "display_name": "イベント",
          "description": "演出中心の場面",
          "selection_role": "cinematic"
        },
        {"slug": "other", "display_name": "その他", "description": "分類しにくい場面"}
      ]
    }
    """

    # Act / Assert
    with pytest.raises(ValueError, match="selection_role"):
        OllamaResponseParser.parse_catalog_response(content)


def test_parse_catalog_response_rejects_path_like_scene_slug() -> None:
    """pathとして危険なscene slugが拒否されること.

    Arrange:
        - path separatorを含むscene slugを返すJSON応答がある
    Act:
        - catalog応答が解析される
    Assert:
        - 不正なslugとして失敗すること
    """
    # Arrange
    content = """
    {
      "scenes": [
        {
          "slug": "../battle",
          "display_name": "戦闘",
          "description": "敵と戦う場面",
          "selection_role": "ordinary"
        },
        {
          "slug": "conversation",
          "display_name": "会話",
          "description": "人物同士の会話",
          "selection_role": "ordinary"
        },
        {"slug": "other", "display_name": "その他", "description": "分類しにくい場面"}
      ]
    }
    """

    # Act / Assert
    with pytest.raises(ValueError, match="scene slug"):
        OllamaResponseParser.parse_catalog_response(content)


def test_parse_catalog_response_rejects_duplicate_scene_slug() -> None:
    """重複したscene slugが拒否されること.

    Arrange:
        - 同じslugを持つ複数sceneを返すJSON応答がある
    Act:
        - catalog応答が解析される
    Assert:
        - 重複slugとして失敗すること
    """
    # Arrange
    content = """
    {
      "scenes": [
        {
          "slug": "battle",
          "display_name": "戦闘",
          "description": "敵と戦う場面",
          "selection_role": "ordinary"
        },
        {
          "slug": "battle",
          "display_name": "バトル",
          "description": "派手な戦闘",
          "selection_role": "ordinary"
        },
        {
          "slug": "conversation",
          "display_name": "会話",
          "description": "人物の会話",
          "selection_role": "ordinary"
        },
        {"slug": "other", "display_name": "その他", "description": "分類しにくい場面"}
      ]
    }
    """

    # Act / Assert
    with pytest.raises(ValueError, match="重複"):
        OllamaResponseParser.parse_catalog_response(content)


def test_parse_classification_response_uses_catalog_display_name() -> None:
    """classification応答がcatalogの表示名と対応付けられること.

    Arrange:
        - catalogに含まれるscene slugを返すJSON応答がある
    Act:
        - classification応答が解析される
    Assert:
        - scene slug、表示名、説明、信頼度が返されること
    """
    # Arrange
    catalog = [
        SceneCatalogEntry("battle", "戦闘", "敵と戦う場面"),
        SceneCatalogEntry("other", "その他", "分類しにくい場面"),
    ]
    content = """
    {
      "scene_slug": "battle",
      "confidence": 0.82,
      "description": "敵との戦闘が中央に写っている"
    }
    """

    # Act
    result = OllamaResponseParser.parse_classification_response(content, catalog)

    # Assert
    assert result.scene_slug == "battle"
    assert result.scene_display_name == "戦闘"
    assert result.scene_description == "敵との戦闘が中央に写っている"
    assert result.confidence == 0.82


def test_parse_classification_response_rejects_unknown_scene() -> None:
    """catalogにないscene slugは拒否されること.

    Arrange:
        - catalogに含まれないscene slugを返すJSON応答がある
    Act:
        - classification応答が解析される
    Assert:
        - 不正応答として失敗すること
    """
    # Arrange
    catalog = [SceneCatalogEntry("other", "その他", "分類しにくい場面")]
    content = '{"scene_slug": "battle", "confidence": 0.5, "description": "戦闘"}'

    # Act / Assert
    with pytest.raises(ValueError, match="catalog"):
        OllamaResponseParser.parse_classification_response(content, catalog)


@pytest.mark.parametrize(
    "content",
    [
        '{"scene_slug": "battle", "description": "戦闘"}',
        '{"scene_slug": "battle", "confidence": null, "description": "戦闘"}',
    ],
)
def test_parse_classification_response_rejects_missing_confidence(
    content: str,
) -> None:
    """classification応答でconfidence欠落が不正応答として扱われること.

    Arrange:
        - confidenceが欠落またはnullのJSON応答がある
    Act:
        - classification応答が解析される
    Assert:
        - ValueErrorとして失敗すること
    """
    # Arrange
    catalog = [
        SceneCatalogEntry("battle", "戦闘", "敵と戦う場面"),
        SceneCatalogEntry("other", "その他", "分類しにくい場面"),
    ]

    # Act / Assert
    with pytest.raises(ValueError, match="confidence"):
        OllamaResponseParser.parse_classification_response(content, catalog)
