"""Replay public media and fixed HTTP replies through the production selectors."""

from __future__ import annotations

import base64
import io
import json
import math
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
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
from tests.migration.support.sheet_contract import assert_sheet_pixels, sheet_contract

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures/rust_migration/pipeline"


def load_json(path: Path) -> dict[str, Any]:
    """Load a fixture object without deriving expectations from production code."""
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _request_without_media(body: dict[str, Any]) -> dict[str, Any]:
    """Retain prompt, schema, labels and clip options, replacing only inline bytes."""
    result: dict[str, Any] = json.loads(json.dumps(body))
    message = result["messages"][0]
    if "images" in message:
        assert len(message["images"]) == 1
        message["images"][0] = "<inference-media>"
    else:
        assert len(message["content"]) == 2
        media = message["content"][1]
        media[media["type"]]["url"] = "<inference-media>"
    return result


def _decoded_video_media(
    encoded: bytes,
) -> tuple[dict[str, Any], list[Image.Image]]:
    """Probe PTS and decode every submitted video frame with native FFmpeg."""
    with TemporaryDirectory(prefix="migration-inference-media-") as directory:
        video = Path(directory) / "submitted.mp4"
        video.write_bytes(encoded)
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_frames",
                "-show_format",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(probe.stdout)
        assert len(payload["streams"]) == 1, "video media must contain one video stream"
        stream = payload["streams"][0]
        assert stream["codec_type"] == "video"
        fields = (
            "codec_type",
            "codec_name",
            "width",
            "height",
            "pix_fmt",
            "avg_frame_rate",
        )
        metadata = {field: stream[field] for field in fields}
        metadata["color_range"] = stream.get("color_range", "unspecified")
        metadata.update(
            {
                "start_seconds": float(payload["format"].get("start_time", 0)),
                "duration_seconds": float(payload["format"]["duration"]),
                "frame_timestamps": [
                    float(frame["best_effort_timestamp_time"])
                    for frame in payload["frames"]
                ],
            }
        )
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
        ).stdout
    dimensions = (stream["width"], stream["height"])
    frame_size = dimensions[0] * dimensions[1] * 3
    assert len(decoded) == frame_size * len(metadata["frame_timestamps"])
    images = [
        Image.frombytes("RGB", dimensions, decoded[offset : offset + frame_size])
        for offset in range(0, len(decoded), frame_size)
    ]
    return metadata, images


def _video_semantic_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Compare pixel layout; range encodings must also pass the decoded RGB gate."""
    result = dict(metadata)
    if result["pix_fmt"] == "yuvj420p":
        result["pix_fmt"] = "yuv420p"
    # Full/limited encodings can preserve the same decoded content. Retain their
    # raw tags in the recording, but reject incorrect interpretation by pixels.
    result.pop("color_range")
    return result


def _assert_media_pixels(
    actual: Image.Image, reference: Image.Image, label: str
) -> None:
    """Apply the unchanged RGB/dHash gate to every submitted image pixel."""
    assert actual.size == reference.size, f"{label}: dimensions changed"
    assert image_difference_hash(actual) == image_difference_hash(reference), (
        f"{label}: dHash changed"
    )
    actual_pixels = np.asarray(actual.convert("RGB"), dtype=np.int16)
    expected_pixels = np.asarray(reference.convert("RGB"), dtype=np.int16)
    difference = np.abs(actual_pixels - expected_pixels)
    assert float(difference.mean(axis=(0, 1)).max()) <= 1.0, f"{label}: channel MAE"
    assert int(difference.max()) <= 16, f"{label}: maximum pixel difference"
    mse = float(np.mean(np.square(difference.astype(np.float64))))
    assert mse == 0 or 10 * math.log10(255**2 / mse) >= 40, f"{label}: PSNR"


def assert_inference_media(
    method: str, kind: str, encoded: bytes, body: dict[str, Any]
) -> None:
    """Compare every AI input pixel and video PTS with reviewed, stored references."""
    directory = FIXTURE_ROOT / "inference-media" / method
    expected = load_json(directory / f"{kind}.json")
    assert _request_without_media(body) == expected["request"], (
        f"{method}/{kind}: inference prompt/schema/media options changed"
    )
    if kind == "video":
        metadata, frames = _decoded_video_media(encoded)
        assert _video_semantic_metadata(metadata) == _video_semantic_metadata(
            expected["video_metadata"]
        ), "video timing/stream changed"
        assert len(frames) == len(expected["decoded_frames"])
        for index, (frame, name) in enumerate(
            zip(frames, expected["decoded_frames"], strict=True)
        ):
            with frame, Image.open(directory / name) as reference:
                _assert_media_pixels(frame, reference, f"video frame {index}")
    else:
        with (
            Image.open(io.BytesIO(encoded)) as actual,
            Image.open(directory / expected["decoded_frames"][0]) as reference,
        ):
            assert actual.format == "JPEG", "inference sheet must be JPEG"
            assert_sheet_pixels(
                actual,
                reference,
                f"{method}/{kind} sheet",
                sheet_contract(method, kind),
                _assert_media_pixels,
            )


def record_inference_media(
    directory: Path, kind: str, encoded: bytes, body: dict[str, Any]
) -> None:
    """Record public AI input only during the explicit reviewed baseline command."""
    directory.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "source_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "request": _request_without_media(body),
    }
    if kind == "video":
        (directory / "video.mp4").write_bytes(encoded)
        video_metadata, frames = _decoded_video_media(encoded)
        metadata["video_metadata"] = video_metadata
        names = [f"video-frame-{index:03d}.png" for index in range(len(frames))]
    else:
        (directory / f"{kind}.jpg").write_bytes(encoded)
        frames = [Image.open(io.BytesIO(encoded)).convert("RGB")]
        names = [f"{kind}.png"]
    metadata["decoded_frames"] = names
    for frame, name in zip(frames, names, strict=True):
        with frame:
            frame.save(directory / name)
    (directory / f"{kind}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


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

    def __init__(
        self,
        method: str,
        *,
        interrupt_secondary: bool = False,
        record_media_directory: Path | None = None,
    ) -> None:
        self.method = method
        self.record_media_directory = record_media_directory
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
            if self.record_media_directory is None:
                assert_inference_media(self.method, kind, decoded, body)
            else:
                record_inference_media(self.record_media_directory, kind, decoded, body)
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


def assert_cold_inference_calls(http: FixtureHttp) -> None:
    """Require one call per inference stage without fixing metadata HTTP probes."""
    expected = ["primary", "secondary"]
    if http.method == "semantic_video":
        expected.insert(0, "video")
    actual = [call for call in http.calls if call in {"primary", "secondary", "video"}]
    assert actual == expected, f"cold inference sequence: {actual!r} != {expected!r}"


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
    for operation in recipes[state].get("change_payload_digest", []):
        matches = list(root.glob(operation["glob"]))
        assert matches, f"fixture replay glob matched nothing: {operation['glob']}"
        for path in matches:
            value = load_json(path)
            value["data"]["payload_digest"] = operation["value"]
            path.write_text(
                json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8"
            )


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


def _assert_report_input_paths(
    report: dict[str, Any], request: VideoSelectionRequest
) -> None:
    """Prove source provenance before removing only the permitted temporary root."""
    inputs = {
        index: str(Path(path).resolve())
        for index, path in enumerate(request.input_videos, start=1)
    }
    for collection, field in (("videos", "path"), ("selected", "video")):
        for record in report[collection]:
            assert record[field] == inputs.get(record["video_index"]), (
                f"report absolute input path: {collection}.{field}"
            )


def _assert_jpeg_receipts(
    path: Path,
    data: dict[str, Any],
    *,
    candidate: bool,
    required_names: set[str] | None = None,
) -> list[str]:
    """Keep encoder-dependent hashes truthful instead of comparing native bytes."""
    records = data["source_frames" if candidate else "frames"]
    label = "candidate JPEG receipt" if candidate else "context JPEG receipt"
    names: list[str] = []
    for record in records:
        name = f"{record['frame_id']}.jpg" if candidate else record["name"]
        assert (
            isinstance(name, str) and Path(name).name == name and name not in names
        ), f"{label}: name"
        if not candidate:
            assert name.endswith(("-before.jpg", "-after.jpg")), f"{label}: name"
        names.append(name)
        digest = record["image_sha256" if candidate else "sha256"]
        assert (
            isinstance(digest, str)
            and len(digest) == 64
            and set(digest) <= set("0123456789abcdef")
        ), f"{label}: SHA-256 format"
        # Prior unused context receipts survive Python resume even when their
        # JPEGs are gone. Truth-check only images requested by this fixed run.
        if required_names is not None and name not in required_names:
            continue
        image = path.parent / "frames" / name
        assert image.is_file() and not image.is_symlink(), f"{label}: missing image"
        if candidate:
            assert image.stat().st_size == record["file_size"], f"{label}: size"
        assert file_sha256(image) == digest, f"{label}: SHA-256"
    if required_names is not None:
        assert required_names <= set(names), "context record names missing"
        return [name for name in names if name in required_names]
    return names


def _stored_phase_data(request: VideoSelectionRequest, path: Path) -> dict[str, Any]:
    """Read the unchanged Python recipe, independent of the submitted records."""
    reference = (
        FIXTURE_ROOT
        / "stored-cache"
        / request.selection_method
        / path.relative_to(cache_root(request))
    )
    data: dict[str, Any] = load_json(reference)["data"]
    return data


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
    _assert_report_input_paths(report, request)
    mechanical: list[dict[str, Any]] = []
    candidate_manifest_digests: dict[str, str] = {}
    context_record_names: list[dict[str, Any]] = []
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
            video_identity = path.relative_to(root / "videos").parts[0]
            if phase == "candidate-extraction":
                _assert_jpeg_receipts(path, data, candidate=True)
                # Python's reader requires the complete ordered ID/time vector.
                # Keep encoder-dependent size/SHA tied to this run's own JPEGs.
                expected_records = [
                    (record["frame_id"], record["timestamp_seconds"])
                    for record in _stored_phase_data(request, path)["source_frames"]
                ]
                assert [
                    (record["frame_id"], record["timestamp_seconds"])
                    for record in data["source_frames"]
                ] == expected_records, "candidate record names/order/time changed"
                assert video_identity not in candidate_manifest_digests
                candidate_manifest_digests[video_identity] = data["payload_digest"]
            if phase == "secondary-context":
                # The unchanged stored Python recipe independently fixes the
                # required names; do not infer completeness from the new writer.
                required_names = {
                    record["name"]
                    for record in _stored_phase_data(request, path)["frames"]
                }
                context_record_names.append(
                    {
                        "video_identity_key": video_identity,
                        "cache_key": value["cache_key"],
                        "frame_names": _assert_jpeg_receipts(
                            path, data, candidate=False, required_names=required_names
                        ),
                    }
                )
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
                assert data["source_frames_digest"] == candidate_manifest_digests.get(
                    video_identity
                ), "mechanical candidate-manifest link"
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
    expected_artifacts = {
        "report.json",
        "selected-contact-sheet.jpg",
        "selected-01.jpg",
        "selected-02.jpg",
    }
    assert len(completion["artifacts"]) == len(expected_artifacts), (
        "completion artifact count"
    )
    assert {item["path"] for item in completion["artifacts"]} == expected_artifacts
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
        "run_manifest": manifest,
        "run_manifest_schema": schema(manifest),
        "run_key": manifest["run_key"],
        "run_manifest_schema_version": manifest["schema_version"],
        "algorithm_version": manifest["algorithm_version"],
        "prompt_version": manifest["prompt_version"],
        "phase_versions": manifest["phase_versions"],
        "inputs": manifest["inputs"],
        "mechanical": mechanical,
        "context_record_names": context_record_names,
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
    for name in ("selected-01", "selected-02", "selected-contact-sheet"):
        with (
            Image.open(Path(request.output_dir) / f"{name}.jpg") as actual,
            Image.open(
                FIXTURE_ROOT / "reference-images" / method / f"{name}.png"
            ) as reference,
        ):
            label = f"{method}/{name} output"
            if name == "selected-contact-sheet":
                assert_sheet_pixels(
                    actual,
                    reference,
                    label,
                    sheet_contract(method, "selected"),
                    _assert_media_pixels,
                )
            else:
                _assert_media_pixels(actual, reference, label)


def assert_contract(
    request: VideoSelectionRequest, method: str, *, exact_assessment_keys: bool = False
) -> None:
    expected = load_json(FIXTURE_ROOT / "expected" / f"{method}.json")
    actual = pipeline_contract(request)
    # The current fixture manifest has only relative input paths and fixed model
    # metadata, with no native JPEG digest dependencies. Normalize no field here.
    assert actual["run_manifest_schema"] == expected["run_manifest_schema"], (
        "run manifest schema changed"
    )
    assert actual["run_manifest"] == expected["run_manifest"], (
        "run manifest values changed"
    )
    assert actual["context_record_names"] == expected["context_record_names"], (
        "context record names changed"
    )
    if not exact_assessment_keys:
        # Their fixed values remain in the golden and stored cache. Across permitted
        # JPEG encoders only these keys depend on native JPEG bytes and raw floats.
        for contract in (actual, expected):
            for envelope in contract["assessment_envelopes"]:
                envelope["cache_key"] = "<jpeg-content-dependent>"
    _assert_values(actual, expected)
    assert_reference_images(request, method)
