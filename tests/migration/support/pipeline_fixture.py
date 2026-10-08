"""Replay public media and fixed HTTP replies through the production selectors."""

from __future__ import annotations

import base64
import io
import json
import math
import shutil
from pathlib import Path
from typing import Any
from urllib.request import Request

import numpy as np
import pytest
from PIL import Image

from src.models.video_selection import VideoMetadata
from src.models.video_selection_request import VideoSelectionRequest
from src.models.vllm_config import VllmConfig
from src.services.video_frame_extractor import VideoFrameExtractor
from src.services.video_phase_cache import CACHE_DIRECTORY_NAME
from src.services.video_selector import image_difference_hash
from src.utils.video_selection_files import file_sha256, json_digest

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures/rust_migration/pipeline"


def load_json(path: Path) -> dict[str, Any]:
    """Load a fixture object without deriving expectations from production code."""
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


class RecordingExtractor(VideoFrameExtractor):
    """Record native decoding calls while preserving FFmpeg and ffprobe behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.probes = 0
        self.extractions: list[tuple[float, int | None]] = []

    def probe(self, video: Path) -> VideoMetadata:
        self.probes += 1
        return super().probe(video)

    def extract_frame(
        self,
        video: Path,
        timestamp_seconds: float,
        output_path: Path,
        *,
        max_width: int | None,
        video_stream_index: int = 0,
    ) -> None:
        self.extractions.append((timestamp_seconds, max_width))
        super().extract_frame(
            video,
            timestamp_seconds,
            output_path,
            max_width=max_width,
            video_stream_index=video_stream_index,
        )


class FixtureHttp:
    """Substitute only HTTP I/O with committed Ollama and vLLM JSON responses."""

    def __init__(self, method: str, *, interrupt_secondary: bool = False) -> None:
        self.responses = load_json(FIXTURE_ROOT / "responses" / f"{method}.json")
        self.interrupt_secondary = interrupt_secondary
        self.calls: list[str] = []
        self.forbid_calls = False

    def __call__(self, request: Request, *, timeout: float) -> io.BytesIO:
        assert not self.forbid_calls, "warm/stored cache must not contact inference"
        assert timeout > 0
        assert request.full_url.startswith(
            ("http://fixture-ollama.invalid:11434/", "http://fixture-vllm.invalid/v1/")
        )
        if request.data is None:
            body = {}
        else:
            assert isinstance(request.data, bytes)
            body = json.loads(request.data)
        if request.full_url.endswith("/api/tags"):
            kind = "tags"
        elif request.full_url.endswith("/api/show"):
            assert body["model"] in {"fixture-primary", "fixture-secondary"}
            kind = "show"
        elif request.full_url.endswith("/api/ps"):
            kind = "ps"
        elif request.full_url.endswith("/models"):
            kind = "models"
        else:
            assert request.full_url.endswith(("/api/chat", "/chat/completions"))
            schema = body.get("format")
            if schema is not None:
                prompt = body["messages"][0]["content"]
                media = body["messages"][0]["images"][0]
                decoded = base64.b64decode(media, validate=True)
                with Image.open(io.BytesIO(decoded)) as image:
                    image.verify()
                kind = "secondary" if "3コマ" in prompt else "primary"
                assert body["model"] == f"fixture-{kind}"
            else:
                assert body["model"] == "fixture-video"
                content = body["messages"][0]["content"]
                prompt = content[0]["text"]
                media = content[1]
                prefix, encoded = media[f"{media['type']}"]["url"].split(",", 1)
                decoded = base64.b64decode(encoded, validate=True)
                schema = body["response_format"]["json_schema"]["schema"]
                if media["type"] == "video_url":
                    assert prefix == "data:video/mp4;base64"
                    assert len(decoded) > 100
                    assert body["media_io_kwargs"] == {
                        "video": {"fps": 1.0, "num_frames": 6}
                    }
                    kind = "video"
                else:
                    assert prefix == "data:image/jpeg;base64"
                    with Image.open(io.BytesIO(decoded)) as image:
                        image.verify()
                    kind = "secondary" if "3コマ" in prompt else "primary"
            if kind != "video":
                response = self.responses[kind]
                message = (
                    response["message"]
                    if "message" in response
                    else response["choices"][0]["message"]
                )
                frames = json.loads(message["content"])["frames"]
                expected_ids = schema["properties"]["frames"]["items"]["properties"][
                    "id"
                ]["enum"]
                assert [frame["id"] for frame in frames] == expected_ids
        self.calls.append(kind)
        if kind == "secondary" and self.interrupt_secondary:
            raise KeyboardInterrupt("fixture interruption after durable primary batch")
        return io.BytesIO(json.dumps(self.responses[kind]).encode("utf-8"))

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.services.ollama_frame_assessor.urlopen", self)
        monkeypatch.setattr("src.services.vllm_client.urlopen", self)


def request_for(root: Path, method: str) -> VideoSelectionRequest:
    """Copy the fixed video and return identical public settings for each replay."""
    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    video = input_dir / "synthetic-game.mkv"
    shutil.copyfile(FIXTURE_ROOT / video.name, video)
    return VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(root / "selected"),
        output_count=2,
        game_title=None,
        game_context="公開合成fixture: 探索・戦闘・会話・イベントを記録する。",
        primary_model="fixture-primary",
        secondary_model="fixture-secondary",
        ollama_host="http://fixture-ollama.invalid:11434",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=1,
        sample_interval_seconds=1.0 if method == "sampled_frames" else None,
        debug=False,
        selection_method=method,
        vllm_config=(
            VllmConfig(
                base_url="http://fixture-vllm.invalid/v1",
                model="fixture-video",
                cache_revision="fixture-338-v1",
                timeout_seconds=1.0,
            )
            if method == "semantic_video"
            else None
        ),
    )


def cache_root(request: VideoSelectionRequest) -> Path:
    return Path(request.input_videos[0]).parent / CACHE_DIRECTORY_NAME


def restore_stored_cache(
    request: VideoSelectionRequest, method: str, state: str
) -> None:
    """Replay old cache bytes, then apply the committed partial/corrupt recipe."""
    root = cache_root(request)
    shutil.copytree(FIXTURE_ROOT / "stored-cache" / method, root)
    recipes = load_json(FIXTURE_ROOT / "cache-replay.json")
    for relative_glob in recipes[state].get("delete", []):
        matches = list(root.glob(relative_glob))
        assert matches, f"fixture replay glob matched nothing: {relative_glob}"
        for path in matches:
            path.unlink()
    for operation in recipes[state].get("replace", []):
        matches = list(root.glob(operation["glob"]))
        assert matches, f"fixture replay glob matched nothing: {operation['glob']}"
        for path in matches:
            path.write_text(operation["content"], encoding="utf-8")


def schema(value: Any) -> Any:
    """Record all report field names and nested types independent of byte encoding."""
    if isinstance(value, dict):
        return {key: schema(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        variants = {json.dumps(schema(item), sort_keys=True) for item in value}
        return [json.loads(item) for item in sorted(variants)]
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    assert isinstance(value, str)
    return "string"


def normalized_report(report: dict[str, Any]) -> dict[str, Any]:
    """Normalize only path and content-digest fields; preserve selection semantics."""
    result: dict[str, Any] = json.loads(json.dumps(report))
    result["manifest_digest"] = "<manifest-digest>"
    for video in result["videos"]:
        video["path"] = Path(video["path"]).name
    for selected in result["selected"]:
        selected["video"] = Path(selected["video"]).name
    for item in result["artifact_integrity"]:
        item["size"] = "<jpeg-size>"
        item["sha256"] = "<jpeg-sha256>"
    return result


def pipeline_contract(request: VideoSelectionRequest) -> dict[str, Any]:
    """Capture fixed decisions and durable cache contracts without temporary paths."""
    root = cache_root(request)
    manifest_paths = list((root / "runs").glob("*/run-manifest.json"))
    assert len(manifest_paths) == 1
    manifest = load_json(manifest_paths[0])
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_digest"
    }
    assert json_digest(manifest_body) == manifest["manifest_digest"]
    report = load_json(Path(request.output_dir) / "report.json")
    assert report["manifest_digest"] == manifest["manifest_digest"]
    mechanical: list[dict[str, Any]] = []
    assessments: dict[str, Any] = {}
    assessment_envelopes: list[dict[str, Any]] = []
    envelopes: list[dict[str, Any]] = []
    for path in sorted((root / "videos").rglob("*.json")):
        value = load_json(path)
        assert value["cache_key"] == (
            path.parent.name
            if path.name in {"manifest.json", "frames.json"}
            else path.stem
        ) or path.name.startswith("chunk-")
        if "phase" in value:
            data = value["data"]
            phase = value["phase"]
            if "payload_digest" in data:
                payload = {
                    key: item for key, item in data.items() if key != "payload_digest"
                }
                assert json_digest(payload) == data["payload_digest"]
            if phase == "video-probe":
                assert (
                    json_digest(
                        {"identity": data["identity"], "metadata": data["metadata"]}
                    )
                    == data["probe_payload_digest"]
                )
            if phase == "semantic_video_chunk":
                assert json_digest(data["result"]) == data["result_digest"]
            envelopes.append(
                {
                    "phase": phase,
                    "cache_schema_version": value["cache_schema_version"],
                    "phase_version": value["phase_version"],
                    "cache_key": value["cache_key"],
                    "data_schema": schema(data),
                }
            )
            if phase == "mechanical-analysis":
                mechanical.append(
                    {
                        "candidates": data["candidates"],
                        "rejected_frame_ids": data["rejected_frame_ids"],
                    }
                )
        elif "assessments" in value:
            assert (
                json_digest(
                    {
                        "cache_key": value["cache_key"],
                        "assessments": value["assessments"],
                    }
                )
                == value["payload_digest"]
            )
            stage = path.parent.name
            assessments[stage] = value["assessments"]
            assessment_envelopes.append(
                {
                    "phase": stage,
                    "cache_key": value["cache_key"],
                    "payload_schema": schema(value),
                }
            )
    completion_paths = list(manifest_paths[0].parent.glob("completion-*.json"))
    assert len(completion_paths) == 1
    completion = load_json(completion_paths[0])
    assert completion["manifest_digest"] == manifest["manifest_digest"]
    assert completion["input_directory"] == str(
        Path(request.input_videos[0]).resolve().parent
    )
    assert {item["path"] for item in completion["artifacts"]} == {
        "report.json",
        "selected-contact-sheet.jpg",
        "selected-01.jpg",
        "selected-02.jpg",
    }
    for item in [*report["artifact_integrity"], *completion["artifacts"]]:
        artifact = Path(request.output_dir) / item["path"]
        assert artifact.stat().st_size == item["size"]
        assert file_sha256(artifact) == item["sha256"]
    images = []
    for name in ("selected-01.jpg", "selected-02.jpg", "selected-contact-sheet.jpg"):
        with Image.open(Path(request.output_dir) / name) as image:
            image.load()
            images.append(
                {
                    "path": name,
                    "format": image.format,
                    "mode": image.mode,
                    "width": image.width,
                    "height": image.height,
                }
            )
    return {
        "report_schema": schema(report),
        "report": normalized_report(report),
        "run_key": manifest["run_key"],
        "run_manifest_schema_version": manifest["schema_version"],
        "algorithm_version": manifest["algorithm_version"],
        "prompt_version": manifest["prompt_version"],
        "phase_versions": manifest["phase_versions"],
        "inputs": manifest["inputs"],
        "mechanical": mechanical,
        "assessments": assessments,
        "assessment_envelopes": assessment_envelopes,
        "phase_envelopes": sorted(envelopes, key=lambda item: item["phase"]),
        "images": images,
    }


def _assert_values(actual: Any, expected: Any, *, field: str = "") -> None:
    """Allow numerical roundoff only for unrounded computed mechanical quality."""
    if field == "quality_score":
        assert actual == pytest.approx(expected, abs=1e-6, rel=1e-8)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_values(actual[key], value, field=key)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_values(actual_item, expected_item)
    else:
        assert actual == expected, f"{field}: {actual!r} != {expected!r}"


def assert_reference_images(request: VideoSelectionRequest, method: str) -> None:
    """Enforce decoded RGB tolerances and strict dHash against checked-in PNGs."""
    for name in ("selected-01", "selected-02"):
        with (
            Image.open(Path(request.output_dir) / f"{name}.jpg") as actual,
            Image.open(
                FIXTURE_ROOT / "reference-images" / method / f"{name}.png"
            ) as reference,
        ):
            assert actual.size == reference.size
            assert image_difference_hash(actual) == image_difference_hash(reference)
            pixels = np.asarray(actual.convert("RGB"), dtype=np.int16)
            reference_pixels = np.asarray(reference.convert("RGB"), dtype=np.int16)
            difference = np.abs(pixels - reference_pixels)
            assert float(difference.mean()) <= 1.0
            assert int(difference.max()) <= 16
            mse = float(np.mean(np.square(difference.astype(np.float64))))
            assert mse == 0 or 10 * math.log10(255**2 / mse) >= 40.0


def assert_contract(
    request: VideoSelectionRequest, method: str, *, exact_assessment_keys: bool = False
) -> None:
    expected = load_json(FIXTURE_ROOT / "expected" / f"{method}.json")
    actual = pipeline_contract(request)
    if not exact_assessment_keys:
        # Their fixed values remain in the golden and stored cache. Across permitted
        # JPEG encoders only these keys depend on native JPEG bytes and raw floats.
        for contract in (actual, expected):
            for envelope in contract["assessment_envelopes"]:
                envelope["cache_key"] = "<jpeg-content-dependent>"
    _assert_values(actual, expected)
    assert_reference_images(request, method)
