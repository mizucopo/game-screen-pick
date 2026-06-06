"""OllamaSceneAnalyzerの単体テスト."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
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
        - Ollama chat APIへモデル名、thinking無効、ヒント、base64画像が送信されること
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
    assert payload["think"] is False
    assert "RPG" in payload["messages"][0]["content"]
    assert payload["messages"][0]["images"] == ["aW1hZ2UtYnl0ZXM="]
    assert result[0].slug == "battle"


def test_generate_scene_catalog_accepts_host_without_url_scheme(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """schemeなしのOllama hostでもHTTPリクエストが生成されること.

    Arrange:
        - schemeなしのOllama hostと入力画像がある
    Act:
        - scene catalogが作成される
    Assert:
        - http schemeが補完されたOllama chat APIへ送信されること
    """
    # Arrange
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"image-bytes")
    captured_requests: list[Request] = []

    def fake_urlopen(request: Request, timeout: float) -> Any:
        captured_requests.append(request)
        assert timeout == 60.0
        return _FakeResponse(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "scenes": [
                                {
                                    "slug": "conversation",
                                    "display_name": "会話",
                                    "description": "人物同士の会話",
                                },
                                {
                                    "slug": "background",
                                    "display_name": "背景",
                                    "description": "場所や背景が分かる場面",
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
    analyzer = OllamaSceneAnalyzer(OllamaConfig(model="gemma4", host="192.168.1.31"))

    # Act
    result = analyzer.generate_scene_catalog(
        representative_paths=[str(image_path)],
        scene_hint=None,
    )

    # Assert
    assert captured_requests[0].full_url == "http://192.168.1.31:11434/api/chat"
    assert result[0].slug == "conversation"


def test_generate_scene_catalog_retries_with_fewer_images_when_response_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """catalog応答が不正な場合に画像数を減らして再試行されること.

    Arrange:
        - 複数の代表画像と1回目だけ不正応答するfake urlopenがある
    Act:
        - scene catalogが作成される
    Assert:
        - 2回目のリクエストは画像数が減り、catalogが返されること
    """
    # Arrange
    image_paths = [tmp_path / f"screen_{index}.png" for index in range(4)]
    for image_path in image_paths:
        image_path.write_bytes(f"image-{image_path.name}".encode("utf-8"))
    image_counts: list[int] = []

    def fake_urlopen(request: Request, timeout: float) -> Any:
        assert timeout == 60.0
        payload = json.loads(request.data.decode("utf-8"))  # type: ignore[union-attr]
        image_counts.append(len(payload["messages"][0]["images"]))
        if len(image_counts) == 1:
            return _FakeResponse({"message": {"content": "not json"}})
        return _FakeResponse(_catalog_chat_payload())

    monkeypatch.setattr("src.services.ollama_scene_analyzer.urlopen", fake_urlopen)
    analyzer = OllamaSceneAnalyzer(OllamaConfig(model="gemma4"))

    # Act
    result = analyzer.generate_scene_catalog(
        representative_paths=[str(image_path) for image_path in image_paths],
        scene_hint=None,
    )

    # Assert
    assert image_counts == [4, 2]
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
        _classification_chat_payload(),
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


def test_classify_image_returns_none_when_ollama_call_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """画像ごとのOllama呼び出し失敗が分類失敗として扱われること.

    Arrange:
        - Ollama API呼び出しがtimeout相当の例外を返す
    Act:
        - 画像が分類される
    Assert:
        - 例外を送出せずNoneが返されること
    """
    # Arrange
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"image-bytes")

    def fake_urlopen(_request: Request, timeout: float) -> Any:
        assert timeout == 60.0
        raise TimeoutError

    monkeypatch.setattr("src.services.ollama_scene_analyzer.urlopen", fake_urlopen)
    analyzer = OllamaSceneAnalyzer(OllamaConfig(model="gemma4"))
    catalog = [
        SceneCatalogEntry("battle", "戦闘", "敵と戦う場面"),
        SceneCatalogEntry("other", "その他", "分類しにくい場面"),
    ]

    # Act
    result = analyzer.classify_image(str(image_path), catalog)

    # Assert
    assert result is None


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
        return _FakeResponse(_classification_chat_payload())

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


def test_classify_image_cache_key_includes_catalog_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """同じslugでもcatalog内容が異なる場合は別cacheとして扱われること.

    Arrange:
        - 同じ画像と同じslugで説明が異なる2つのcatalogがある
    Act:
        - それぞれのcatalogで画像が分類される
    Assert:
        - 2回ともOllama APIが呼ばれ、cacheが混同されないこと
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
            _classification_chat_payload(description=f"分類結果{called_count}")
        )

    monkeypatch.setattr("src.services.ollama_scene_analyzer.urlopen", fake_urlopen)
    analyzer = OllamaSceneAnalyzer(OllamaConfig(model="gemma4"))
    first_catalog = [
        SceneCatalogEntry("battle", "戦闘", "敵と戦う場面"),
        SceneCatalogEntry("other", "その他", "分類しにくい場面"),
    ]
    second_catalog = [
        SceneCatalogEntry("battle", "バトル", "敵と戦う派手な場面"),
        SceneCatalogEntry("other", "その他", "分類しにくい場面"),
    ]

    # Act
    first = analyzer.classify_image(str(image_path), first_catalog)
    second = analyzer.classify_image(str(image_path), second_catalog)

    # Assert
    assert first is not None
    assert second is not None
    assert first.scene_display_name == "戦闘"
    assert second.scene_display_name == "バトル"
    assert called_count == 2


def test_classify_image_returns_result_when_cache_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """cache書き込みに失敗しても分類結果が返されること.

    Arrange:
        - 分類応答は正常に返る
        - cache directoryを作れない入力ディレクトリがある
    Act:
        - 画像が分類される
    Assert:
        - cache失敗で再試行せず、正常なclassificationが返されること
    """
    # Arrange
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"image-bytes")
    (tmp_path / ".game-screen-pick").write_text("not a directory", encoding="utf-8")
    called_count = 0

    def fake_urlopen(_request: Request, timeout: float) -> Any:
        nonlocal called_count
        assert timeout == 60.0
        called_count += 1
        return _FakeResponse(_classification_chat_payload())

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
    assert called_count == 1


def test_classify_image_preserves_parallel_cache_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """並列分類で同じcache fileへ書き込んでも全entryが保持されること.

    Arrange:
        - 同じディレクトリにある複数画像が並列分類される
    Act:
        - それぞれの画像分類が同時に完了する
    Assert:
        - cache fileに全画像分の分類entryが残ること
    """
    # Arrange
    image_paths = [tmp_path / f"screen_{index}.png" for index in range(6)]
    for image_path in image_paths:
        image_path.write_bytes(f"image-{image_path.name}".encode("utf-8"))

    barrier = threading.Barrier(len(image_paths))

    def fake_urlopen(_request: Request, timeout: float) -> Any:
        assert timeout == 60.0
        barrier.wait(timeout=2.0)
        return _FakeResponse(_classification_chat_payload())

    monkeypatch.setattr("src.services.ollama_scene_analyzer.urlopen", fake_urlopen)
    analyzer = OllamaSceneAnalyzer(OllamaConfig(model="gemma4"))
    catalog = [
        SceneCatalogEntry("battle", "戦闘", "敵と戦う場面"),
        SceneCatalogEntry("other", "その他", "分類しにくい場面"),
    ]

    # Act
    with ThreadPoolExecutor(max_workers=len(image_paths)) as executor:
        results = list(
            executor.map(
                lambda image_path: analyzer.classify_image(str(image_path), catalog),
                image_paths,
            )
        )

    # Assert
    assert all(result is not None for result in results)
    cache_path = tmp_path / ".game-screen-pick" / "cache" / "ollama-scenes.json"
    cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(cache_payload["classifications"]) == len(image_paths)


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


def _catalog_chat_payload() -> dict[str, object]:
    """正常なcatalog chat応答payloadを返す."""
    return {
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


def _classification_chat_payload(
    description: str = "敵との戦闘場面",
) -> dict[str, object]:
    """正常なclassification chat応答payloadを返す."""
    return {
        "message": {
            "content": json.dumps(
                {
                    "scene_slug": "battle",
                    "confidence": 0.9,
                    "description": description,
                }
            )
        }
    }
