"""画像評価だけのcache missとrun再開におけるGPU runtimeの契約."""

import io
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
from urllib.request import Request

import pytest

from src.services.video_selector import VideoSelector
from tests.services.test_semantic_video_pipeline import semantic_plan, semantic_request
from tests.services.test_video_pipeline import FakeFrameExtractor
from tests.services.test_vllm_runtime_session import RuntimeTransport, managed_config


@pytest.mark.parametrize(
    "failure", [RuntimeError("inference failed"), KeyboardInterrupt()]
)
def test_image_only_resume_closes_runtime_and_restarts_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    transport = RuntimeTransport(monkeypatch)
    transport.served_model = "video-model"
    fail_inference = True

    def inference(request: Request, *, timeout: float) -> io.BytesIO:
        assert transport.live
        if request.full_url.endswith("/models"):
            return transport.http(request, timeout=timeout)
        if fail_inference:
            raise failure
        assert isinstance(request.data, bytes)
        body = json.loads(request.data)
        assert body["messages"][0]["content"][1]["type"] == "image_url"
        schema = body["response_format"]["json_schema"]["schema"]
        item_properties = schema["properties"]["frames"]["items"]["properties"]
        display_ids = item_properties["id"]["enum"]
        result = {
            "frames": [
                {
                    "id": frame_id,
                    "blog_score": 95,
                    "transition": False,
                    "scene": "ボス戦",
                    "reason": "動画の重要場面",
                }
                for frame_id in display_ids
            ]
        }
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(result),
                            },
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr("src.services.vllm_client.urlopen", inference)
    request = replace(semantic_request(tmp_path), vllm_runtime_config=managed_config())
    planner = Mock()
    planner.plan.return_value = semantic_plan()
    selector = VideoSelector(
        request, frame_extractor=FakeFrameExtractor(), semantic_planner=planner
    )

    with pytest.raises(type(failure)) as caught:
        selector.run()
    assert caught.value is failure
    assert transport.commands == [["start"], ["stop"]]
    assert not transport.live

    fail_inference = False
    assert selector.run().is_file()
    assert transport.commands == [["start"], ["stop"], ["start"], ["stop"]]
    # Ollamaの解放を指定していないので、未使用の不正hostも無視する。
    assert all("/api/" not in request.full_url for request in transport.requests)
    history = list(transport.history)
    assert selector.run().is_file()
    assert transport.history == history


def test_existing_managed_server_blocks_uncached_ollama_context_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RuntimeTransport(monkeypatch)
    transport.live = True
    request = replace(
        semantic_request(tmp_path),
        game_title="test game",
        game_context="",
        game_context_provider="ollama",
        game_context_model="context-model",
        ollama_host="http://configured-ollama",
        vllm_runtime_config=managed_config(unload_ollama=True),
    )
    context_generator = Mock()
    planner = Mock()
    selector = VideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        semantic_planner=planner,
        context_generator=context_generator,
    )

    with pytest.raises(RuntimeError, match="既に応答"):
        selector.run()

    context_generator.generate.assert_not_called()
    planner.plan.assert_not_called()
    assert transport.commands == []
    assert transport.loaded_models == ["ollama-model"]
