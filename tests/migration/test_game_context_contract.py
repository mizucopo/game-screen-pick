"""固定AI応答からGame Contextの正規化・停止条件を比較する."""

import json
from pathlib import Path
from typing import Any, cast

import pytest

from src.services.game_context_generator import (
    GameContextGenerationError,
    GameContextGenerator,
)

ROOT = Path(__file__).parents[1] / "fixtures" / "rust_migration"
EXPECTED = json.loads(
    (ROOT / "game_context_expectations.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", EXPECTED["cases"], ids=lambda case: case["name"])
def test_game_context_response_contract(case: dict[str, Any]) -> None:
    """旧providerはmockのみ。4項目契約はBrave+vLLMにも維持する."""
    calls: list[str] = []
    if "unicode_codepoints" in case:
        response_text = case["http_response"]["output"][1]["content"][0]["text"]
        raw_context = json.loads(response_text)["game_context"]
        normalized = raw_context.strip()
        assert len(normalized) == case["unicode_codepoints"]
        assert len(normalized.encode("utf-8")) == case["utf8_byte_length"]

    def requester(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        assert url == "https://api.openai.com/v1/responses"
        assert headers["Authorization"] == "Bearer fixture-only-token"
        assert payload["model"] == "fixture-context-model"
        assert timeout_seconds == 1.0
        calls.append(url)
        return cast(dict[str, Any], case["http_response"])

    generator = GameContextGenerator(requester=requester, api_key="fixture-only-token")

    def generate() -> str:
        return generator.generate(
            game_title="合成 fixture game",
            provider="openai",
            model="fixture-context-model",
            ollama_host="http://unused.invalid",
            timeout_seconds=1.0,
        ).game_context

    if "expected_error" in case:
        with pytest.raises(GameContextGenerationError, match=case["expected_error"]):
            generate()
    else:
        assert generate() == case["expected_context"]
    assert len(calls) == 1
