"""Ollama応答parserの単体テスト."""

import pytest

from src.models.scene_catalog_entry import SceneCatalogEntry
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
        {"slug": "battle", "display_name": "戦闘", "description": "敵と戦う場面"},
        {
          "slug": "conversation",
          "display_name": "会話",
          "description": "人物同士の会話"
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
