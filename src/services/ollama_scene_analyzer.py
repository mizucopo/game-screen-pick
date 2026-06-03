"""Ollama chat API を使うscene analyzer."""

import base64
import hashlib
import json
from pathlib import Path
from threading import Lock
from urllib.request import Request, urlopen

from ..models.ollama_config import OllamaConfig
from ..models.scene_catalog_entry import SceneCatalogEntry
from ..models.scene_classification import SceneClassification
from .ollama_response_parser import OllamaResponseParser


class OllamaSceneAnalyzer:
    """Ollamaでscene catalog作成と画像分類を行う."""

    def __init__(self, config: OllamaConfig) -> None:
        """analyzerを初期化する."""
        self.config = config
        self._cache_lock = Lock()

    def generate_scene_catalog(
        self,
        representative_paths: list[str],
        scene_hint: str | None,
    ) -> list[SceneCatalogEntry]:
        """代表画像からscene catalogを作成する."""
        content = self._post_chat(
            prompt=self._build_catalog_prompt(scene_hint),
            image_paths=representative_paths,
        )
        return OllamaResponseParser.parse_catalog_response(content)

    def classify_image(
        self,
        image_path: str,
        catalog: list[SceneCatalogEntry],
    ) -> SceneClassification | None:
        """画像をscene catalogのsceneへ分類する."""
        cache_key = self._build_cache_key(image_path, catalog)
        cached_classification = self._get_cached_classification(
            image_path,
            cache_key,
        )
        if cached_classification is not None:
            return cached_classification

        prompt = self._build_classification_prompt(catalog, retry=False)
        retry_prompt = self._build_classification_prompt(catalog, retry=True)
        for current_prompt in (prompt, retry_prompt):
            try:
                content = self._post_chat(
                    prompt=current_prompt,
                    image_paths=[image_path],
                )
                classification = OllamaResponseParser.parse_classification_response(
                    content,
                    catalog,
                )
                if self.config.cache_enabled:
                    self._write_classification_cache(
                        image_path,
                        cache_key,
                        classification,
                    )
                return classification
            except (OSError, ValueError):
                continue
        return None

    def _post_chat(self, prompt: str, image_paths: list[str]) -> str:
        """Ollama chat APIへJSON requestを送信する."""
        payload = {
            "model": self.config.model,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [self._encode_image(path) for path in image_paths],
                }
            ],
        }
        request = Request(
            url=f"{self.config.host.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.config.timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        message = response_payload.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            msg = "Ollama応答にmessage.contentがありません"
            raise ValueError(msg)
        return content

    @staticmethod
    def _encode_image(path: str) -> str:
        """画像ファイルをbase64文字列へ変換する."""
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("ascii")

    def _get_cached_classification(
        self,
        image_path: str,
        cache_key: str,
    ) -> SceneClassification | None:
        """cache済み分類結果を返す."""
        if not self.config.cache_enabled:
            return None
        with self._cache_lock:
            cached = self._read_classification_cache(image_path).get(cache_key)
        if not isinstance(cached, dict):
            return None
        return self._classification_from_cache(cached)

    @staticmethod
    def _classification_from_cache(
        cached: dict[object, object],
    ) -> SceneClassification | None:
        """cache payloadを分類結果へ変換する."""
        try:
            confidence = cached["confidence"]
            if not isinstance(confidence, int | float | str):
                return None
            return SceneClassification(
                scene_slug=str(cached["scene_slug"]),
                scene_display_name=str(cached["scene_display_name"]),
                scene_description=str(cached["scene_description"]),
                confidence=float(confidence),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _build_cache_key(
        self,
        image_path: str,
        catalog: list[SceneCatalogEntry],
    ) -> str:
        """分類cache keyを作る."""
        path = Path(image_path)
        stat = path.stat()
        catalog_key = json.dumps(
            self._catalog_to_cache_key_payload(catalog),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        raw_key = (
            f"{self.config.model}|{path.resolve()}|{stat.st_mtime_ns}|"
            f"{stat.st_size}|{catalog_key}"
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @staticmethod
    def _catalog_to_cache_key_payload(
        catalog: list[SceneCatalogEntry],
    ) -> list[dict[str, str]]:
        """分類cache keyへ含めるcatalog情報を返す."""
        return [
            {
                "slug": scene.slug,
                "display_name": scene.display_name,
                "description": scene.description,
            }
            for scene in catalog
        ]

    @staticmethod
    def _cache_path_for_image(image_path: str) -> Path:
        """画像pathからcache file pathを返す."""
        return (
            Path(image_path).parent
            / ".game-screen-pick"
            / "cache"
            / "ollama-scenes.json"
        )

    def _read_classification_cache(
        self,
        image_path: str,
    ) -> dict[str, object]:
        """分類cacheを読み込む."""
        cache_path = self._cache_path_for_image(image_path)
        if not cache_path.exists():
            return {}
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        cache = payload.get("classifications", {})
        return cache if isinstance(cache, dict) else {}

    def _write_classification_cache(
        self,
        image_path: str,
        cache_key: str,
        classification: SceneClassification,
    ) -> None:
        """分類cacheを書き込む."""
        cache_path = self._cache_path_for_image(image_path)
        with self._cache_lock:
            cache = self._read_classification_cache(image_path)
            cache[cache_key] = self._classification_to_cache(classification)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"classifications": cache}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    @staticmethod
    def _classification_to_cache(
        classification: SceneClassification,
    ) -> dict[str, str | float]:
        """分類結果をcache payloadへ変換する."""
        return {
            "scene_slug": classification.scene_slug,
            "scene_display_name": classification.scene_display_name,
            "scene_description": classification.scene_description,
            "confidence": classification.confidence,
        }

    @staticmethod
    def _build_catalog_prompt(scene_hint: str | None) -> str:
        """scene catalog作成promptを構築する."""
        hint = scene_hint or "指定なし"
        return (
            "ゲームスクリーンショット群から、ブログ画像選択に役立つscene catalogを"
            "3から8個作ってください。必ずotherを含めてください。"
            "各sceneは英語のslug、日本語display_name、日本語descriptionを持ち、"
            "JSONのみで返してください。"
            f"\nヒント: {hint}"
            '\n形式: {"scenes":[{"slug":"battle","display_name":"戦闘",'
            '"description":"敵と戦う場面"}]}'
        )

    @staticmethod
    def _build_classification_prompt(
        catalog: list[SceneCatalogEntry],
        retry: bool,
    ) -> str:
        """画像分類promptを構築する."""
        scenes = [
            {
                "slug": scene.slug,
                "display_name": scene.display_name,
                "description": scene.description,
            }
            for scene in catalog
        ]
        retry_note = (
            "前回の応答は不正でした。必ず指定形式のJSONのみを返してください。"
            if retry
            else ""
        )
        return (
            f"{retry_note}"
            "画像を次のscene catalogのいずれか1つに分類してください。"
            "confidenceは0から1の数値、descriptionはブログ画像選択に役立つ短い日本語説明です。"
            f"\nscene catalog: {json.dumps(scenes, ensure_ascii=False)}"
            '\n形式: {"scene_slug":"battle","confidence":0.82,'
            '"description":"敵との戦闘が中央に写っている"}'
        )
