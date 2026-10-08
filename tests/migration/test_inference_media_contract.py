"""Reject AI requests whose parseable media has wrong pixels, order or timing."""

import base64
import io
import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest
from PIL import Image

from tests.migration.support.pipeline_fixture import (
    FIXTURE_ROOT,
    FixtureHttp,
    _decoded_video_media,
    load_json,
)

SHEETS = (
    ("sampled_frames", "primary"),
    ("sampled_frames", "secondary"),
    ("semantic_video", "primary"),
    ("semantic_video", "secondary"),
)


def _request(method: str, kind: str, encoded: bytes) -> Request:
    """Reconstruct the stored request, substituting only its inline media bytes."""
    body: dict[str, Any] = load_json(
        FIXTURE_ROOT / "inference-media" / method / f"{kind}.json"
    )["request"]
    content = body["messages"][0]
    value = base64.b64encode(encoded).decode("ascii")
    if method == "sampled_frames":
        content["images"][0] = value
        url = "http://fixture-ollama.invalid:11434/api/chat"
    else:
        media = content["content"][1]
        prefix = "data:video/mp4" if kind == "video" else "data:image/jpeg"
        media[media["type"]]["url"] = f"{prefix};base64,{value}"
        url = "http://fixture-vllm.invalid/v1/chat/completions"
    return Request(url, data=json.dumps(body).encode("utf-8"))


def _sheet(method: str, kind: str) -> Image.Image:
    with Image.open(FIXTURE_ROOT / "inference-media" / method / f"{kind}.png") as image:
        return image.convert("RGB")


def _jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=100, subsampling=0)
    return buffer.getvalue()


@pytest.mark.parametrize(("method", "kind"), (*SHEETS, ("semantic_video", "video")))
def test_fixture_http_accepts_recorded_media(method: str, kind: str) -> None:
    """The committed real HTTP media and its full request are accepted unchanged."""
    extension = "mp4" if kind == "video" else "jpg"
    media = (
        FIXTURE_ROOT / "inference-media" / method / f"{kind}.{extension}"
    ).read_bytes()
    response = FixtureHttp(method)(_request(method, kind, media), timeout=1)
    assert isinstance(json.loads(response.read()), dict)


@pytest.mark.parametrize(("method", "kind"), SHEETS)
def test_fixture_http_accepts_reencoded_sheet(method: str, kind: str) -> None:
    """Negative cases below fail for changed content, not merely JPEG reencoding."""
    with _sheet(method, kind) as image:
        FixtureHttp(method)(_request(method, kind, _jpeg(image)), timeout=1)


@pytest.mark.parametrize(("method", "kind"), SHEETS)
@pytest.mark.parametrize("mutation", ("blank", "reordered", "missing_label"))
def test_fixture_http_rejects_changed_sheet(
    method: str, kind: str, mutation: str
) -> None:
    """Correct schema, IDs and image dimensions cannot hide an incorrect AI sheet."""
    with _sheet(method, kind) as image:
        contextual = kind == "secondary"
        cell_width, label_height, row_height = (
            (960, 42, 222) if contextual else (480, 32, 302)
        )
        if mutation == "blank":
            image.paste("black", (0, 0, image.width, image.height))
        elif mutation == "missing_label":
            # No font region is masked: a missing A01/time/source label must fail.
            image.paste("black", (0, 0, cell_width, label_height))
        else:
            first = (0, label_height, cell_width, row_height)
            second = (cell_width, label_height, cell_width * 2, row_height)
            first_pixels, second_pixels = image.crop(first), image.crop(second)
            image.paste(second_pixels, first)
            image.paste(first_pixels, second)
            first_pixels.close()
            second_pixels.close()
        request = _request(method, kind, _jpeg(image))
    with pytest.raises(AssertionError, match="sheet"):
        FixtureHttp(method)(request, timeout=1)


@pytest.mark.parametrize("method", ("sampled_frames", "semantic_video"))
def test_fixture_http_rejects_swapped_context_panels(method: str) -> None:
    """Swapping before/after while keeping labels and the central candidate fails."""
    with _sheet(method, "secondary") as image:
        before = (0, 42, 320, 222)
        after = (640, 42, 960, 222)
        before_pixels, after_pixels = image.crop(before), image.crop(after)
        image.paste(after_pixels, before)
        image.paste(before_pixels, after)
        before_pixels.close()
        after_pixels.close()
        request = _request(method, "secondary", _jpeg(image))
    with pytest.raises(AssertionError, match="sheet"):
        FixtureHttp(method)(request, timeout=1)


@pytest.mark.parametrize("mutation", ("reversed_frames", "wrong_timing"))
def test_fixture_http_rejects_wrong_video(tmp_path: Path, mutation: str) -> None:
    """A valid MP4 must contain the right frame content and ordered PTS."""
    directory = FIXTURE_ROOT / "inference-media" / "semantic_video"
    output = tmp_path / "changed.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(directory / "video.mp4"),
            "-vf",
            "reverse" if mutation == "reversed_frames" else "setpts=2*PTS",
            "-fps_mode",
            "passthrough",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "pc",
            "-crf",
            "28",
            "-threads",
            "2",
            "-y",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    encoded = output.read_bytes()
    if mutation == "reversed_frames":
        # Same codec/dimensions/frame count/PTS: rejection must inspect video pixels.
        metadata, frames = _decoded_video_media(encoded)
        for frame in frames:
            frame.close()
        assert metadata == load_json(directory / "video.json")["video_metadata"]
    with pytest.raises(AssertionError, match="video"):
        FixtureHttp("semantic_video")(
            _request("semantic_video", "video", encoded), timeout=1
        )


def test_fixture_http_rejects_wrong_declared_chunk_interval() -> None:
    """Correct clip pixels do not excuse a prompt using the wrong original interval."""
    encoded = (FIXTURE_ROOT / "inference-media/semantic_video/video.mp4").read_bytes()
    request = _request("semantic_video", "video", encoded)
    assert isinstance(request.data, bytes)
    body = json.loads(request.data)
    content = body["messages"][0]["content"][0]
    original = "original video interval: 0.000000 - 5.800000 seconds"
    assert original in content["text"]
    content["text"] = content["text"].replace(
        original, "original video interval: 1.000000 - 6.800000 seconds"
    )
    changed = Request(request.full_url, data=json.dumps(body).encode("utf-8"))
    with pytest.raises(AssertionError, match="prompt/schema/media options"):
        FixtureHttp("semantic_video")(changed, timeout=1)


def test_fixture_http_accepts_equivalent_limited_range_video(tmp_path: Path) -> None:
    """Correct limited-range reencoding preserves decoded content and timing."""
    original = FIXTURE_ROOT / "inference-media/semantic_video/video.mp4"
    output = tmp_path / "limited-range.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(original),
            "-vf",
            "scale=in_range=pc:out_range=tv",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-bsf:v",
            "h264_metadata=video_full_range_flag=0",
            "-crf",
            "18",
            "-threads",
            "2",
            "-an",
            "-y",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    encoded = output.read_bytes()
    metadata, frames = _decoded_video_media(encoded)
    for frame in frames:
        frame.close()
    assert metadata["pix_fmt"] == "yuv420p"
    assert metadata["color_range"] == "tv"
    FixtureHttp("semantic_video")(
        _request("semantic_video", "video", encoded), timeout=1
    )


def test_fixture_http_rejects_incorrect_range_flag(tmp_path: Path) -> None:
    """Changing only the SPS range flag must fail even with the same layout and PTS."""
    original = FIXTURE_ROOT / "inference-media/semantic_video/video.mp4"
    output = tmp_path / "incorrect-range-flag.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(original),
            "-c:v",
            "copy",
            "-bsf:v",
            "h264_metadata=video_full_range_flag=0",
            "-an",
            "-y",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    encoded = output.read_bytes()
    metadata, frames = _decoded_video_media(encoded)
    for frame in frames:
        frame.close()
    expected = load_json(original.with_suffix(".json"))["video_metadata"]
    assert metadata["frame_timestamps"] == expected["frame_timestamps"]
    assert metadata["color_range"] == "tv"
    with pytest.raises(AssertionError, match="video frame"):
        FixtureHttp("semantic_video")(
            _request("semantic_video", "video", encoded), timeout=1
        )
