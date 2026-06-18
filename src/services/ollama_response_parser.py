"""Ollama scene分析応答のparser."""

import json
import re
from typing import Any

from ..models.scene_catalog_entry import SceneCatalogEntry
from ..models.scene_classification import SceneClassification
from ..models.scene_selection_role import SceneSelectionRole


class OllamaResponseParser:
    """OllamaのJSON応答をdomain modelへ変換する."""

    @staticmethod
    def parse_catalog_response(content: str) -> list[SceneCatalogEntry]:
        """scene catalog応答を解析する."""
        payload = OllamaResponseParser._load_json_object(content)
        raw_scenes = payload.get("scenes")
        if not isinstance(raw_scenes, list):
            msg = "catalog応答にはscenes配列が必要です"
            raise ValueError(msg)
        scenes = [
            SceneCatalogEntry(
                slug=OllamaResponseParser._required_string(item, "slug"),
                display_name=OllamaResponseParser._required_string(
                    item, "display_name"
                ),
                description=OllamaResponseParser._required_string(item, "description"),
                selection_role=OllamaResponseParser._selection_role(
                    slug=OllamaResponseParser._required_string(item, "slug"),
                    payload=item,
                ),
            )
            for item in raw_scenes
            if isinstance(item, dict)
        ]
        if not 3 <= len(scenes) <= 8:
            msg = "scene catalogは3から8個のsceneである必要があります"
            raise ValueError(msg)
        scene_slugs = [scene.slug for scene in scenes]
        if len(scene_slugs) != len(set(scene_slugs)):
            msg = "scene catalogのslugが重複しています"
            raise ValueError(msg)
        if "other" not in {scene.slug for scene in scenes}:
            msg = "scene catalogにはotherが必要です"
            raise ValueError(msg)
        return scenes

    @staticmethod
    def parse_classification_response(
        content: str,
        catalog: list[SceneCatalogEntry],
    ) -> SceneClassification:
        """classification応答を解析する."""
        payload = OllamaResponseParser._load_json_object(content)
        scene_slug = OllamaResponseParser._required_string(payload, "scene_slug")
        scene_by_slug = {scene.slug: scene for scene in catalog}
        if scene_slug not in scene_by_slug:
            msg = f"catalogにないscene_slugです: {scene_slug}"
            raise ValueError(msg)
        scene = scene_by_slug[scene_slug]
        return SceneClassification(
            scene_slug=scene_slug,
            scene_display_name=scene.display_name,
            scene_description=OllamaResponseParser._required_string(
                payload, "description"
            ),
            confidence=OllamaResponseParser._required_float(payload, "confidence"),
        )

    @staticmethod
    def _load_json_object(content: str) -> dict[str, Any]:
        """文字列からJSON objectを読み取る."""
        stripped = content.strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if match is None:
                raise
            payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            msg = "Ollama応答はJSON objectである必要があります"
            raise ValueError(msg)
        return payload

    @staticmethod
    def _required_string(payload: dict[str, Any], key: str) -> str:
        """必須文字列を取り出す."""
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            msg = f"{key}は必須です"
            raise ValueError(msg)
        return value.strip()

    @staticmethod
    def _required_float(payload: dict[str, Any], key: str) -> float:
        """必須数値を取り出す."""
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            msg = f"{key}は数値である必要があります"
            raise ValueError(msg)
        try:
            return float(value)
        except ValueError as exc:
            msg = f"{key}は数値である必要があります"
            raise ValueError(msg) from exc

    @staticmethod
    def _selection_role(
        *,
        slug: str,
        payload: dict[str, Any],
    ) -> SceneSelectionRole:
        """catalog payloadからselection roleを取り出す."""
        if slug == "other":
            return SceneSelectionRole.ORDINARY
        value = payload.get("selection_role")
        if value is None:
            value = payload.get("scene_selection_role")
        if isinstance(value, SceneSelectionRole):
            return value
        if not isinstance(value, str) or not value.strip():
            msg = "selection_roleは必須です"
            raise ValueError(msg)
        return SceneSelectionRole.from_value(value)
