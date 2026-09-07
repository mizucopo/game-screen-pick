"""vLLM接続とOpenAI互換HTTP API境界のテスト."""

import io
import json
from dataclasses import FrozenInstanceError
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from src.models.video_selection import FrameCandidate
from src.models.vllm_config import VllmConfig
from src.services.vllm_client import VllmClient


def chat_response(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(payload),
                },
            }
        ]
    }


def fake_http(
    monkeypatch: pytest.MonkeyPatch,
    response: object,
) -> list[Request]:
    requests: list[Request] = []

    def urlopen(request: Request, *, timeout: float) -> io.BytesIO:
        assert timeout > 0
        requests.append(request)
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("src.services.vllm_client.urlopen", urlopen)
    return requests


def test_config_normalizes_endpoint_and_keeps_api_key_private() -> None:
    config = VllmConfig(
        model=" game-model ",
        base_url=" https://inference.example/proxy/ ",
        api_key="secret-key",
        cache_revision=" weights-2 ",
    )

    assert config.model == "game-model"
    assert config.base_url == "https://inference.example/proxy/v1"
    assert config.cache_revision == "weights-2"
    assert config.timeout_seconds == 900.0
    assert "secret-key" not in repr(config)
    with pytest.raises(FrozenInstanceError):
        config.model = "changed"  # type: ignore[misc]


def test_runtime_is_acquired_only_before_live_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def http(request: Request, *, timeout: float) -> io.BytesIO:
        assert timeout > 0
        assert events[-1] == "ready"
        events.append(request.full_url)
        return io.BytesIO(b'{"data": [{"id": "video-model"}]}')

    monkeypatch.setattr("src.services.vllm_client.urlopen", http)
    client = VllmClient(
        VllmConfig(model="video-model"),
        before_request=lambda: events.append("ready"),
    )
    assert client.model_metadata()["resolved_name"] == "video-model"
    assert events == []
    client.fetch_model_metadata({"video-model"})
    assert events == ["ready", "http://127.0.0.1:8000/v1/models"]


@pytest.mark.parametrize(
    "values",
    [
        {"model": " "},
        {"model": "game-model", "cache_revision": " "},
        {"model": "game-model", "timeout_seconds": 0},
        {"model": "game-model", "timeout_seconds": float("nan")},
        {"model": "game-model", "timeout_seconds": float("inf")},
        {"model": "game-model", "base_url": "file:///tmp/server"},
        {"model": "game-model", "base_url": "http://user:secret@server/v1"},
        {"model": "game-model", "base_url": "http://server/v1?api_key=secret"},
        {"model": "game-model", "base_url": "http://"},
    ],
)
def test_config_rejects_invalid_or_secret_bearing_settings(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        VllmConfig(**values)  # type: ignore[arg-type]


def test_complete_json_sends_ordered_video_with_temporal_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = fake_http(monkeypatch, chat_response({"events": []}))
    client = VllmClient(VllmConfig(model="game-model", api_key="secret-key"))
    media = {
        "type": "video_url",
        "video_url": {"url": "data:video/jpeg;base64,anBlZw==,anBlZw=="},
    }
    metadata = {
        "video": {
            "fps": 30.0,
            "frames_indices": [0, 300],
            "total_num_frames": 600,
            "duration": 20.0,
            "do_sample_frames": False,
        }
    }
    schema = {"type": "object", "properties": {"events": {"type": "array"}}}

    result = client.complete_json(
        prompt="Describe events in order",
        media=media,
        schema=schema,
        schema_name="video_events",
        media_io_kwargs=metadata,
    )

    assert result == {"events": []}
    assert len(requests) == 1
    request = requests[0]
    assert request.full_url == "http://127.0.0.1:8000/v1/chat/completions"
    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer secret-key"
    assert isinstance(request.data, bytes)
    body = json.loads(request.data)
    assert body["model"] == "game-model"
    assert body["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Describe events in order"}, media],
        }
    ]
    assert body["media_io_kwargs"] == metadata
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "video_events", "strict": True, "schema": schema},
    }
    assert body["temperature"] == 0
    assert body["seed"] == 271
    assert 0 < body["max_tokens"] <= 8192
    assert body["stream"] is False


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"choices": []},
        {"choices": [None]},
        {"choices": "secret-key"},
        {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
        {"choices": [{"finish_reason": "stop", "message": None}]},
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "{}",
                        "refusal": "secret-key",
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "secret-key"},
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "[]"},
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "{} trailing"},
                }
            ]
        },
        {"error": {"message": "secret-key"}},
    ],
)
def test_complete_json_rejects_unfinished_refused_or_invalid_responses(
    monkeypatch: pytest.MonkeyPatch,
    response: object,
) -> None:
    fake_http(monkeypatch, response)
    client = VllmClient(VllmConfig(model="game-model", api_key="secret-key"))

    with pytest.raises(ValueError, match="vLLM") as error:
        client.complete_json(
            prompt="inspect",
            media={"type": "video_url", "video_url": {"url": "unused"}},
            schema={"type": "object"},
            schema_name="test",
        )

    assert "secret-key" not in str(error.value)


@pytest.mark.parametrize(
    "failure",
    [
        HTTPError("http://server", 401, "secret-key", Message(), None),
        URLError("secret-key"),
        TimeoutError("secret-key"),
    ],
)
def test_request_failure_does_not_expose_response_or_credentials(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    def failed_urlopen(request: Request, *, timeout: float) -> io.BytesIO:
        del request, timeout
        raise failure

    monkeypatch.setattr("src.services.vllm_client.urlopen", failed_urlopen)
    client = VllmClient(VllmConfig(model="game-model", api_key="secret-key"))

    with pytest.raises(RuntimeError, match="vLLM") as error:
        client.complete_json(
            prompt="inspect",
            media={"type": "video_url", "video_url": {"url": "unused"}},
            schema={"type": "object"},
            schema_name="test",
        )

    assert "secret-key" not in str(error.value)
    assert error.value.__suppress_context__


def test_model_identity_is_local_revision_bound_and_verifies_served_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = fake_http(
        monkeypatch,
        {"data": [{"id": "game-model"}, {"id": "other-model"}]},
    )
    client = VllmClient(VllmConfig(model="game-model", api_key="secret-key"))

    local = client.model_metadata()

    assert requests == []
    assert local["resolved_name"] == "game-model"
    assert local["identity_source"] == "configured_cache_revision"
    assert local["cache_revision"] == "1"
    assert "secret-key" not in json.dumps(local)
    assert len(local["digest"]) == 64
    assert client.gpu_evidence == {}
    assert client.fetch_model_metadata({"game-model"}) == {"game-model": local}
    assert requests[0].full_url == "http://127.0.0.1:8000/v1/models"
    assert requests[0].method == "GET"
    assert requests[0].data is None

    changed_revision = VllmClient(VllmConfig(model="game-model", cache_revision="2"))
    changed_endpoint = VllmClient(
        VllmConfig(model="game-model", base_url="http://other:8000/v1")
    )
    changed_key = VllmClient(VllmConfig(model="game-model", api_key="other-key"))
    assert changed_revision.model_metadata()["digest"] != local["digest"]
    assert changed_endpoint.model_metadata()["digest"] != local["digest"]
    assert changed_key.model_metadata() == local


@pytest.mark.parametrize(
    "response",
    [{}, {"data": []}, {"data": [{"id": "game-model:latest"}]}],
)
def test_model_validation_requires_exact_served_model(
    monkeypatch: pytest.MonkeyPatch,
    response: object,
) -> None:
    fake_http(monkeypatch, response)
    client = VllmClient(VllmConfig(model="game-model"))

    with pytest.raises(ValueError, match="vLLM"):
        client.fetch_model_metadata({"game-model"})


def test_model_validation_rejects_unconfigured_model_without_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = fake_http(monkeypatch, {"data": [{"id": "other-model"}]})
    client = VllmClient(VllmConfig(model="game-model"))

    with pytest.raises(ValueError, match="vLLM"):
        client.fetch_model_metadata({"other-model"})

    assert requests == []


def frame_response(frame_id: str, score: float = 80) -> dict[str, Any]:
    return {
        "id": frame_id,
        "blog_score": score,
        "transition": False,
        "scene": "探索",
        "reason": "景観と状況が伝わる",
    }


def test_assess_sends_contact_sheet_and_maps_ids_back_in_candidate_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requests = fake_http(
        monkeypatch,
        chat_response({"frames": [frame_response("A02", 70), frame_response("A01")]}),
    )
    client = VllmClient(VllmConfig(model="game-model"))
    contact_sheet = tmp_path / "sheet.jpg"
    contact_sheet.write_bytes(b"jpeg")
    candidates = [
        FrameCandidate("f101", 1, "unused"),
        FrameCandidate("f303", 3, "unused"),
    ]

    assessments = client.assess(
        model="game-model",
        model_digest=client.model_metadata()["digest"],
        prompt="evaluate",
        candidates=candidates,
        contact_sheet=contact_sheet,
    )

    assert [assessment.frame_id for assessment in assessments] == ["f101", "f303"]
    assert [assessment.blog_score for assessment in assessments] == [80.0, 70.0]
    assert assessments[0].scene == "探索"
    assert not assessments[0].is_transition
    assert len(requests) == 1
    assert isinstance(requests[0].data, bytes)
    body = json.loads(requests[0].data)
    assert body["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,anBlZw=="},
    }
    fields = body["response_format"]["json_schema"]["schema"]["properties"]["frames"]
    assert fields["minItems"] == fields["maxItems"] == 2
    assert fields["items"]["properties"]["id"]["enum"] == ["A01", "A02"]
    assert "media_io_kwargs" not in body


@pytest.mark.parametrize(
    "frames",
    [
        [],
        [frame_response("f101")],
        [frame_response("A01"), frame_response("A01")],
        [frame_response("A01"), frame_response("A02")],
        [frame_response("A01", -1)],
        [frame_response("A01", 101)],
        [frame_response("A01", float("nan"))],
        [frame_response("A01", float("inf"))],
        [{**frame_response("A01"), "blog_score": "80"}],
        [{**frame_response("A01"), "blog_score": True}],
        [{**frame_response("A01"), "transition": "false"}],
        [{**frame_response("A01"), "scene": None}],
        [{**frame_response("A01"), "reason": 5}],
        [{**frame_response("A01"), "extra": "unexpected"}],
    ],
)
def test_assess_rejects_invalid_scores_fields_and_nonunique_display_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    frames: list[dict[str, Any]],
) -> None:
    fake_http(monkeypatch, chat_response({"frames": frames}))
    client = VllmClient(VllmConfig(model="game-model"))
    contact_sheet = tmp_path / "sheet.jpg"
    contact_sheet.write_bytes(b"jpeg")

    with pytest.raises(ValueError, match="vLLM"):
        client.assess(
            model="game-model",
            model_digest=client.model_metadata()["digest"],
            prompt="evaluate",
            candidates=[FrameCandidate("f101", 1, "unused")],
            contact_sheet=contact_sheet,
        )
