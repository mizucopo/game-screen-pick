"""実動画の変換からCLI最終画像までをHTTP境界だけ置換して検証する."""

import base64
import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request

import pytest
from PIL import Image, ImageChops, ImageStat

from src.main import run


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="実動画の統合テストにはffmpegとffprobeが必要です",
)
@pytest.mark.parametrize("managed_runtime", [False, True])
def test_semantic_video_cli_decodes_video_and_extracts_event_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    managed_runtime: bool,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    original_video = input_dir / "game.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=6:duration=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "2",
            str(original_video),
        ],
        check=True,
        capture_output=True,
    )
    config = tmp_path / "config.toml"
    config.write_text(
        '[run]\nselection_method = "semantic_video"\n'
        'vllm_model = "video-model"\nvllm_base_url = "http://inference.invalid/v1"\n'
        'vllm_api_key = "test-only-token"\n',
        encoding="utf-8",
    )
    if managed_runtime:
        with config.open("a", encoding="utf-8") as file:
            file.write(
                'ollama_host = "http://configured-ollama.invalid:11434"\n'
                'ollama_api_key = "ollama-test-token"\n'
                "ollama_unload_before_vllm = true\n"
                'vllm_start_command = ["/fake/runtime-start", "{model}"]\n'
                'vllm_stop_command = ["/fake/runtime-stop"]\n'
            )
    event = {
        "start_seconds": 2.4,
        "end_seconds": 2.6,
        "timestamp_seconds": 2.5,
        "summary": "移動していた対象が画面中央に到着する",
        "importance": 90,
    }
    requests: list[Request] = []
    videos: list[tuple[Path, dict[str, Any]]] = []
    images: list[bytes] = []
    lifecycle: list[str] = []
    server_running = not managed_runtime
    loaded_models = ["old-model"]
    original_run = subprocess.run

    def command_run(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[Any]:
        nonlocal server_running
        if command[0] == "/fake/runtime-start":
            assert loaded_models == []
            assert command == ["/fake/runtime-start", "video-model"]
            lifecycle.append("start")
            server_running = True
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[0] == "/fake/runtime-stop":
            lifecycle.append("stop")
            server_running = False
            return subprocess.CompletedProcess(command, 0, "", "")
        return original_run(command, **kwargs)

    def runtime_http(request: Request, *, timeout: float) -> io.BytesIO:
        assert managed_runtime
        assert timeout > 0
        if request.full_url.startswith("http://configured-ollama.invalid:11434/"):
            assert request.get_header("Authorization") == "Bearer ollama-test-token"
            if request.full_url.endswith("/api/ps"):
                return io.BytesIO(
                    json.dumps(
                        {"models": [{"name": name} for name in loaded_models]}
                    ).encode()
                )
            assert request.full_url.endswith("/api/generate")
            assert isinstance(request.data, bytes)
            body = json.loads(request.data)
            assert body["model"] == "old-model"
            assert body["keep_alive"] == 0
            loaded_models.clear()
            lifecycle.append("unload")
            return io.BytesIO(b'{"done": true, "done_reason": "unload"}')
        assert request.full_url.startswith("http://inference.invalid/")
        if not server_running:
            raise URLError(ConnectionRefusedError("stopped test server"))
        if request.full_url.endswith("/health"):
            return io.BytesIO(b"")
        assert request.full_url.endswith("/v1/models")
        return io.BytesIO(b'{"data": [{"id": "video-model"}]}')

    def http_response(request: Request, *, timeout: float) -> io.BytesIO:
        assert timeout > 0
        assert server_running
        lifecycle.append("inference")
        requests.append(request)
        assert request.get_header("Authorization") == "Bearer test-only-token"
        if request.full_url == "http://inference.invalid/v1/models":
            return io.BytesIO(json.dumps({"data": [{"id": "video-model"}]}).encode())
        assert request.full_url == "http://inference.invalid/v1/chat/completions"
        assert isinstance(request.data, bytes)
        body = json.loads(request.data)
        assert body["model"] == "video-model"
        content = body["messages"][0]["content"]
        media = content[1]
        if media["type"] == "video_url":
            prefix, encoded = media["video_url"]["url"].split(",", maxsplit=1)
            assert prefix == "data:video/mp4;base64"
            received = tmp_path / f"received-{len(videos)}.mp4"
            received.write_bytes(base64.b64decode(encoded, validate=True))
            videos.append((received, body["media_io_kwargs"]))
            result = {"events": [event], "excluded_intervals": []}
        else:
            assert media["type"] == "image_url"
            prefix, encoded = media["image_url"]["url"].split(",", maxsplit=1)
            assert prefix == "data:image/jpeg;base64"
            images.append(base64.b64decode(encoded, validate=True))
            schema = body["response_format"]["json_schema"]["schema"]
            display_ids = schema["properties"]["frames"]["items"]["properties"]["id"]
            result = {
                "frames": [
                    {
                        "id": frame_id,
                        "blog_score": 95,
                        "transition": False,
                        "scene": "移動",
                        "reason": "動画で確認した到着場面",
                    }
                    for frame_id in display_ids["enum"]
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

    monkeypatch.setattr("src.services.vllm_client.urlopen", http_response)
    monkeypatch.setattr("src.services.ollama_frame_assessor.urlopen", http_response)
    if managed_runtime:
        monkeypatch.setattr("src.services.vllm_runtime_session.urlopen", runtime_http)
        monkeypatch.setattr(subprocess, "run", command_run)
    output_dir = tmp_path / "selected"

    run(
        [
            "--config",
            str(config),
            "--num",
            "1",
            "--game-context",
            "移動する対象の記録",
            str(input_dir),
            str(output_dir),
        ]
    )

    assert all("/v1/" in request.full_url for request in requests)
    if managed_runtime:
        assert lifecycle[:2] == ["unload", "start"]
        assert lifecycle[-1] == "stop"
        assert lifecycle.count("start") == lifecycle.count("stop") == 1
    else:
        assert set(lifecycle) == {"inference"}
    assert len(videos) == 1
    assert len(images) == 2
    received_video, media_options = videos[0]
    assert media_options == {"video": {"fps": 1.0, "num_frames": 4}}
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_frames",
            "-of",
            "json",
            str(received_video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    decoded = json.loads(probe.stdout)
    stream = decoded["streams"][0]
    assert (stream["width"], stream["height"]) == (512, 288)
    assert stream["avg_frame_rate"] == "1/1"
    assert float(stream["duration"]) == pytest.approx(4.0)
    frame_times = [
        float(frame["best_effort_timestamp_time"]) for frame in decoded["frames"]
    ]
    assert frame_times == [0.0, 1.0, 2.0, 3.0]
    for jpeg in images:
        with Image.open(io.BytesIO(jpeg)) as image:
            image.verify()

    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["selection_method"] == "semantic_video"
    assert report["sample_count"] == 1
    assert report["output_count"] == 1
    selected = report["selected"][0]
    assert selected["timestamp_seconds"] == 2.5
    assert selected["semantic_provenance"] == [{**event, "chunk_index": 0}]
    chunk = report["videos"][0]["semantic_analysis"]["chunks"][0]
    assert chunk["start_seconds"] == 0.0
    assert 23 / 6 <= chunk["end_seconds"] <= 4.0
    assert "test-only-token" not in json.dumps(report)
    assert (output_dir / "selected-contact-sheet.jpg").is_file()

    reference_path = tmp_path / "reference.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            "2.5",
            "-i",
            str(original_video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(reference_path),
        ],
        check=True,
        capture_output=True,
    )
    with (
        Image.open(output_dir / selected["output_path"]) as actual,
        Image.open(reference_path) as reference,
    ):
        assert actual.size == (640, 360)
        difference = ImageChops.difference(
            actual.convert("RGB"), reference.convert("RGB")
        )
        assert max(ImageStat.Stat(difference).mean) < 1.0

    # 完了cacheの再利用は、起動コマンドを変更してもruntimeへ触れない。
    previous_events = list(lifecycle)
    if managed_runtime:
        config.write_text(
            config.read_text().replace("/fake/runtime-start", "/fake/changed-start"),
            encoding="utf-8",
        )
    run(
        [
            "--config",
            str(config),
            "--num",
            "1",
            "--game-context",
            "移動する対象の記録",
            str(input_dir),
            str(output_dir),
        ]
    )
    assert lifecycle == previous_events
