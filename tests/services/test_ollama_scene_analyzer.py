"""OllamaSceneAnalyzerの単体テスト."""

import json
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest

from src.models.ollama_config import OllamaConfig
from src.models.scene_catalog_entry import SceneCatalogEntry
from src.services.ollama_scene_analyzer import OllamaSceneAnalyzer


def test_generate_scene_catalog_posts_images_to_chat_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """scene catalog作成で/api/chatへ画像付きリクエストが送信されること.

    Arrange:
        - 入力画像とfakeのurlopenがある
    Act:
        - scene catalogを作成する
    Assert:
        - Ollama chat APIへモデル名、ヒント、base64画像が送信されること
    """
    # Arrange
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"image-bytes")
    captured_requests: list[Request] = []

    def fake_urlopen(request: Request, timeout: float) -> Any:
        captured_requests.append(request)
        assert timeout == 30.0
        return _FakeResponse(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "scenes": [
                                {
                                    "slug": "battle",
                                    "display_name": "戦闘",
                                    "description": "敵と戦う場面",
                                },
                                {
                                    "slug": "conversation",
                                    "display_name": "会話",
                                    "description": "人物同士の会話",
                                },
                                {
                                    "slug": "other",
                                    "display_name": "その他",
                                    "description": "分類しにくい場面",
                                },
                            ]
                        }
                    )
                }
            }
        )

    monkeypatch.setattr("src.services.ollama_scene_analyzer.urlopen", fake_urlopen)
    analyzer = OllamaSceneAnalyzer(
        OllamaConfig(model="gemma4", host="http://ollama:11434", timeout=30.0)
    )

    # Act
    result = analyzer.generate_scene_catalog(
        representative_paths=[str(image_path)],
        scene_hint="RPG。戦闘と探索が混在している",
    )

    # Assert
    request = captured_requests[0]
    assert request.full_url == "http://ollama:11434/api/chat"
    payload = json.loads(request.data.decode("utf-8"))  # type: ignore[union-attr]
    assert payload["model"] == "gemma4"
    assert "RPG" in payload["messages"][0]["content"]
    assert payload["messages"][0]["images"] == ["aW1hZ2UtYnl0ZXM="]
    assert result[0].slug == "battle"


def test_classify_image_retries_once_when_response_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """分類応答が不正な場合は1回だけ再試行されること.

    Arrange:
        - 1回目は不正JSON、2回目は正常JSONを返すfake urlopenがある
    Act:
        - 画像が分類される
    Assert:
        - 2回目の応答でclassificationが返されること
    """
    # Arrange
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"image-bytes")
    responses = [
        {"message": {"content": "not json"}},
        {
            "message": {
                "content": json.dumps(
                    {
                        "scene_slug": "battle",
                        "confidence": 0.9,
                        "description": "敵との戦闘場面",
                    }
                )
            }
        },
    ]

    def fake_urlopen(_request: Request, timeout: float) -> Any:
        assert timeout == 60.0
        return _FakeResponse(responses.pop(0))

    monkeypatch.setattr("src.services.ollama_scene_analyzer.urlopen", fake_urlopen)
    analyzer = OllamaSceneAnalyzer(OllamaConfig(model="gemma4"))
    catalog = [
        SceneCatalogEntry("battle", "戦闘", "敵と戦う場面"),
        SceneCatalogEntry("other", "その他", "分類しにくい場面"),
    ]

    # Act
    result = analyzer.classify_image(str(image_path), catalog)

    # Assert
    assert result is not None
    assert result.scene_slug == "battle"
    assert len(responses) == 0


def test_classify_image_uses_file_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """同じ画像分類ではファイルキャッシュが使用されること.

    Arrange:
        - cache有効のOllamaSceneAnalyzerがある
        - 同じ画像を2回分類する
    Act:
        - classify_imageが2回呼ばれる
    Assert:
        - Ollama API呼び出しは1回だけで、2回目はcacheから返されること
    """
    # Arrange
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"image-bytes")
    called_count = 0

    def fake_urlopen(_request: Request, timeout: float) -> Any:
        nonlocal called_count
        assert timeout == 60.0
        called_count += 1
        return _FakeResponse(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "scene_slug": "battle",
                            "confidence": 0.9,
                            "description": "敵との戦闘場面",
                        }
                    )
                }
            }
        )

    monkeypatch.setattr("src.services.ollama_scene_analyzer.urlopen", fake_urlopen)
    analyzer = OllamaSceneAnalyzer(OllamaConfig(model="gemma4"))
    catalog = [
        SceneCatalogEntry("battle", "戦闘", "敵と戦う場面"),
        SceneCatalogEntry("other", "その他", "分類しにくい場面"),
    ]

    # Act
    first = analyzer.classify_image(str(image_path), catalog)
    second = analyzer.classify_image(str(image_path), catalog)

    # Assert
    assert first == second
    assert called_count == 1
    assert (tmp_path / ".game-screen-pick" / "cache" / "ollama-scenes.json").exists()


class _FakeResponse:
    """urllib response互換のfake."""

    def __init__(self, payload: object) -> None:
        """fake responseを初期化する."""
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        """context managerに入る."""
        return self

    def __exit__(self, *_args: object) -> None:
        """context managerを抜ける."""

    def read(self) -> bytes:
        """JSON bytesを返す."""
        return json.dumps(self._payload).encode("utf-8")
