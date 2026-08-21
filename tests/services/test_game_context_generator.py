"""Web検索を使うGame Context生成境界のテスト."""

import json
from typing import Any

import pytest

from src.services.game_context_generator import (
    GameContextGenerationError,
    GameContextGenerator,
)

VALID_RESULT = {
    "status": "ok",
    "identified_title": "ドラゴンクエストXI 過ぎ去りし時を求めて",
    "game_context": (
        "ジャンル: ロールプレイングゲーム\n"
        "基本的なゲーム進行と主なプレイ要素: 世界を探索し、会話と戦闘を進める。\n"
        "代表的な画面や場面: フィールド探索、コマンド戦闘、町での会話。\n"
        "画像選定で重視する視覚的要素: 景観、仲間、戦闘状況が明瞭な画面。"
    ),
}


class RecordingRequester:
    """固定応答を返し、送信内容を記録するfake transport."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def __call__(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.calls.append((url, headers, payload, timeout_seconds))
        return self.responses.pop(0)


@pytest.mark.parametrize(
    ("provider", "api_key_name", "endpoint", "tool_type", "response"),
    [
        (
            "openai",
            "OPENAI_API_KEY",
            "https://api.openai.com/v1/responses",
            "web_search",
            {
                "model": "openai-used",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": json.dumps(VALID_RESULT)}
                        ],
                    }
                ],
            },
        ),
        (
            "gemini",
            "GEMINI_API_KEY",
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            "google_search",
            {
                "model": "gemini-used",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": json.dumps(VALID_RESULT)}],
                    }
                ],
            },
        ),
        (
            "xai",
            "XAI_API_KEY",
            "https://api.x.ai/v1/responses",
            "web_search",
            {
                "model": "xai-used",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": json.dumps(VALID_RESULT)}
                        ],
                    }
                ],
            },
        ),
    ],
)
def test_integrated_provider_uses_only_selected_web_search_api(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    api_key_name: str,
    endpoint: str,
    tool_type: str,
    response: dict[str, Any],
) -> None:
    """選択providerだけへ検索付き生成requestを送り、共通contextを得ること."""
    monkeypatch.setenv(api_key_name, "secret")
    requester = RecordingRequester([response])
    generator = GameContextGenerator(requester=requester)

    result = generator.generate(
        game_title="ドラクエ11",
        provider=provider,
        model=f"{provider}-requested",
        ollama_host="127.0.0.1:11434",
        timeout_seconds=42.0,
    )

    assert result.game_context == VALID_RESULT["game_context"]
    assert result.provider == provider
    assert result.model == f"{provider}-used"
    assert len(requester.calls) == 1
    url, headers, payload, timeout = requester.calls[0]
    assert url == endpoint
    assert timeout == 42.0
    assert any(value.endswith("secret") for value in headers.values())
    assert payload["model"] == f"{provider}-requested"
    assert payload["tools"] == [{"type": tool_type}]
    assert "ドラクエ11" in json.dumps(payload, ensure_ascii=False)
    assert "信頼できない外部データ" in json.dumps(payload, ensure_ascii=False)


def test_ollama_searches_cloud_then_generates_with_selected_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OllamaだけがWeb Search API結果を指定local modelへ渡すこと."""
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-secret")
    requester = RecordingRequester(
        [
            {
                "results": [
                    {
                        "title": "公式サイト",
                        "url": "https://example.test/official",
                        "content": "ゲーム紹介",
                    }
                ]
            },
            {
                "model": "qwen-context:latest",
                "message": {"content": json.dumps(VALID_RESULT)},
            },
        ]
    )

    result = GameContextGenerator(requester=requester).generate(
        game_title="ドラクエ11",
        provider="ollama",
        model="qwen-context",
        ollama_host="ollama.internal:11434",
        timeout_seconds=30.0,
    )

    assert result.model == "qwen-context:latest"
    assert [call[0] for call in requester.calls] == [
        "https://ollama.com/api/web_search",
        "http://ollama.internal:11434/api/chat",
    ]
    assert requester.calls[0][1]["Authorization"] == "Bearer ollama-secret"
    assert requester.calls[0][2]["max_results"] == 10
    assert requester.calls[1][2]["model"] == "qwen-context"
    assert "公式サイト" in json.dumps(requester.calls[1][2], ensure_ascii=False)


@pytest.mark.parametrize("status", ["ambiguous", "insufficient", "conflict"])
def test_generation_rejects_unresolved_game_identity_or_sources(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    """作品特定不能・情報不足・矛盾をcontextとして受け入れないこと."""
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(
                            {
                                "status": status,
                                "error": "対象を一意に特定できません",
                            }
                        ),
                    }
                ],
            }
        ]
    }

    with pytest.raises(GameContextGenerationError, match=f"openai.*{status}"):
        GameContextGenerator(requester=RecordingRequester([response])).generate(
            game_title="曖昧な名前",
            provider="openai",
            model="gpt-test",
            ollama_host="127.0.0.1:11434",
            timeout_seconds=30.0,
        )


def test_generation_reports_missing_provider_key_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """認証未設定をprovider別に示し、他providerを呼び出さないこと."""
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    requester = RecordingRequester([])

    with pytest.raises(GameContextGenerationError, match="xai.*XAI_API_KEY"):
        GameContextGenerator(requester=requester).generate(
            game_title="Game",
            provider="xai",
            model="grok-test",
            ollama_host="127.0.0.1:11434",
            timeout_seconds=30.0,
        )

    assert requester.calls == []
