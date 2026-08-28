"""Web検索を使うGame Context生成境界のテスト."""

import io
import json
import traceback
from email.message import Message
from typing import Any
from urllib.error import HTTPError

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
                    {"type": "web_search_call", "status": "completed"},
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": json.dumps(VALID_RESULT)}
                        ],
                    },
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
                "outputs": [
                    {
                        "type": "google_search_call",
                        "id": "search-call",
                        "arguments": {"queries": ["ドラクエ11"]},
                    },
                    {
                        "type": "google_search_result",
                        "call_id": "search-call",
                        "result": {"url": "https://example.test"},
                    },
                    {"type": "text", "text": json.dumps(VALID_RESULT)},
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
                    {"type": "web_search_call", "status": "completed"},
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": json.dumps(VALID_RESULT)}
                        ],
                    },
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
    assert payload["tool_choice"] == ("any" if provider == "gemini" else "required")
    assert "ドラクエ11" in json.dumps(payload, ensure_ascii=False)
    assert "信頼できない外部データ" in json.dumps(payload, ensure_ascii=False)


def test_configured_api_key_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """設定ファイル由来の非空値を環境変数より優先すること."""
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    requester = RecordingRequester(
        [
            {
                "output": [
                    {"type": "web_search_call", "status": "completed"},
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(VALID_RESULT),
                            }
                        ],
                    },
                ]
            }
        ]
    )

    GameContextGenerator(
        requester=requester,
        api_key="configured-secret",
    ).generate(
        game_title="ドラクエ11",
        provider="openai",
        model="gpt-test",
        ollama_host="127.0.0.1:11434",
        timeout_seconds=42.0,
    )

    assert requester.calls[0][1]["Authorization"] == "Bearer configured-secret"


def test_api_key_is_redacted_from_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """外部応答errorに認証値が混ざっても例外messageへ残さないこと."""
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    def raise_secret_error(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del url, headers, payload, timeout_seconds
        raise ValueError("provider echoed configured-secret")

    with pytest.raises(GameContextGenerationError) as error_info:
        GameContextGenerator(
            requester=raise_secret_error,
            api_key="configured-secret",
        ).generate(
            game_title="ドラクエ11",
            provider="openai",
            model="gpt-test",
            ollama_host="127.0.0.1:11434",
            timeout_seconds=42.0,
        )

    assert "configured-secret" not in str(error_info.value)
    assert "<redacted>" in str(error_info.value)
    assert error_info.value.__cause__ is None
    formatted = "".join(
        traceback.format_exception(
            error_info.type,
            error_info.value,
            error_info.tb,
        )
    )
    assert "configured-secret" not in formatted


def test_http_error_redacts_api_key_before_detail_truncation() -> None:
    """500文字境界をまたぐ認証値も部分文字列を残さず伏せること."""
    api_key = "credential-fragment-that-is-secret"

    def raise_http_error(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del headers, payload, timeout_seconds
        body = f"{'x' * 490}{api_key}tail".encode()
        raise HTTPError(url, 401, "Unauthorized", Message(), io.BytesIO(body))

    with pytest.raises(GameContextGenerationError) as error_info:
        GameContextGenerator(
            requester=raise_http_error,
            api_key=api_key,
        ).generate(
            game_title="ドラクエ11",
            provider="openai",
            model="gpt-test",
            ollama_host="127.0.0.1:11434",
            timeout_seconds=42.0,
        )

    message = str(error_info.value)
    assert api_key not in message
    assert "credential" not in message
    assert "<redacted>" in message


def test_gemini_accepts_current_steps_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini Interactions APIの現行steps形式も受け入れること."""
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    response = {
        "model": "gemini-used",
        "steps": [
            {
                "type": "google_search_call",
                "id": "search-call",
                "arguments": {"queries": ["ドラクエ11"]},
            },
            {
                "type": "google_search_result",
                "call_id": "search-call",
                "result": [{"url": "https://example.test"}],
            },
            {
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps(VALID_RESULT)}],
            },
        ],
    }

    result = GameContextGenerator(requester=RecordingRequester([response])).generate(
        game_title="ドラクエ11",
        provider="gemini",
        model="gemini-requested",
        ollama_host="127.0.0.1:11434",
        timeout_seconds=42.0,
    )

    assert result.game_context == VALID_RESULT["game_context"]


@pytest.mark.parametrize(
    ("provider", "api_key_name", "response"),
    [
        (
            "openai",
            "OPENAI_API_KEY",
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": json.dumps(VALID_RESULT)}
                        ],
                    }
                ]
            },
        ),
        (
            "gemini",
            "GEMINI_API_KEY",
            {"outputs": [{"type": "text", "text": json.dumps(VALID_RESULT)}]},
        ),
        (
            "xai",
            "XAI_API_KEY",
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": json.dumps(VALID_RESULT)}
                        ],
                    }
                ]
            },
        ),
    ],
)
def test_integrated_provider_rejects_context_without_search_call(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    api_key_name: str,
    response: dict[str, Any],
) -> None:
    """検索toolの実行記録がないcontextを保存対象にしないこと."""
    monkeypatch.setenv(api_key_name, "secret")

    with pytest.raises(GameContextGenerationError, match=f"{provider}.*検索tool"):
        GameContextGenerator(requester=RecordingRequester([response])).generate(
            game_title="ドラクエ11",
            provider=provider,
            model=f"{provider}-requested",
            ollama_host="127.0.0.1:11434",
            timeout_seconds=42.0,
        )


@pytest.mark.parametrize("provider", ["openai", "xai"])
def test_responses_provider_rejects_unsuccessful_search_call(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    """Responses APIの未完了検索callを成功証拠として扱わないこと."""
    monkeypatch.setenv(
        "OPENAI_API_KEY" if provider == "openai" else "XAI_API_KEY",
        "secret",
    )
    response = {
        "output": [
            {"type": "web_search_call", "status": "incomplete"},
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(VALID_RESULT)}],
            },
        ]
    }

    with pytest.raises(GameContextGenerationError, match=f"{provider}.*検索tool"):
        GameContextGenerator(requester=RecordingRequester([response])).generate(
            game_title="ドラクエ11",
            provider=provider,
            model=f"{provider}-requested",
            ollama_host="127.0.0.1:11434",
            timeout_seconds=42.0,
        )


def test_gemini_rejects_search_call_without_matching_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini検索callに対応するresultがない応答を拒否すること."""
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    response = {
        "outputs": [
            {"type": "google_search_call", "id": "search-call"},
            {
                "type": "google_search_result",
                "call_id": "different-call",
                "result": {"url": "https://example.test"},
            },
            {"type": "text", "text": json.dumps(VALID_RESULT)},
        ]
    }

    with pytest.raises(GameContextGenerationError, match="gemini.*検索tool"):
        GameContextGenerator(requester=RecordingRequester([response])).generate(
            game_title="ドラクエ11",
            provider="gemini",
            model="gemini-requested",
            ollama_host="127.0.0.1:11434",
            timeout_seconds=42.0,
        )


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
            {"type": "web_search_call", "status": "completed"},
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
            },
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


def test_generation_reports_invalid_context_with_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """共通項目が欠けたmodel応答もprovider別errorに変換すること."""
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    response = {
        "output": [
            {"type": "web_search_call", "status": "completed"},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(
                            {
                                "status": "ok",
                                "identified_title": "Game",
                                "game_context": "ジャンル: RPG",
                            }
                        ),
                    }
                ],
            },
        ]
    }

    with pytest.raises(GameContextGenerationError, match="openai.*共通項目"):
        GameContextGenerator(requester=RecordingRequester([response])).generate(
            game_title="Game",
            provider="openai",
            model="gpt-test",
            ollama_host="127.0.0.1:11434",
            timeout_seconds=30.0,
        )
