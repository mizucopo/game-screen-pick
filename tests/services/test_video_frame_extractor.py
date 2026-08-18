"""単一動画向けffprobe metadata境界のテスト."""

from pathlib import Path
from typing import Any

import pytest

from src.services.video_frame_extractor import VideoFrameExtractor


def _probe_with_payload(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> tuple[VideoFrameExtractor, list[list[str]]]:
    """外部commandを使わず指定payloadを返すextractorを作る."""
    commands: list[list[str]] = []
    extractor = VideoFrameExtractor.__new__(VideoFrameExtractor)

    def run_json(command: list[str]) -> dict[str, Any]:
        commands.append(command)
        return payload

    monkeypatch.setattr(extractor, "_run_json", run_json)
    return extractor, commands


def test_probe_prefers_selected_video_stream_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """containerより短い先頭video streamの時間をsamplingへ使うこと."""
    extractor, commands = _probe_with_payload(
        monkeypatch,
        {
            "format": {"duration": "12.0"},
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "60/1",
                    "duration": "8.25",
                },
                {"index": 1, "codec_type": "audio", "duration": "12.0"},
            ],
        },
    )

    metadata = extractor.probe(Path("sample.mp4"))

    assert metadata.duration_seconds == 8.25
    assert "duration" in commands[0][commands[0].index("-show_entries") + 1]


def test_probe_falls_back_to_container_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """video stream時間が利用不能ならcontainer時間を使うこと."""
    extractor, _commands = _probe_with_payload(
        monkeypatch,
        {
            "format": {"duration": "12.0"},
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "60/1",
                    "duration": "N/A",
                }
            ],
        },
    )

    metadata = extractor.probe(Path("sample.mp4"))

    assert metadata.duration_seconds == 12.0


def test_probe_skips_attached_picture_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cover artを除外して実際のvideo streamを選ぶこと."""
    extractor, _commands = _probe_with_payload(
        monkeypatch,
        {
            "format": {"duration": "12.0"},
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "mjpeg",
                    "width": 600,
                    "height": 600,
                    "avg_frame_rate": "0/0",
                    "duration": "12.0",
                    "disposition": {"attached_pic": 1},
                },
                {
                    "index": 2,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "60/1",
                    "duration": "8.0",
                    "disposition": {"attached_pic": 0},
                },
            ],
        },
    )

    metadata = extractor.probe(Path("sample.mp4"))

    assert metadata.codec_name == "h264"
    assert metadata.duration_seconds == 8.0
    assert metadata.video_stream_index == 2


def test_extract_frame_maps_the_selected_video_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """probeで選んだglobal stream indexをffmpegへ渡すこと."""
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> None:
        commands.append(command)
        temporary = Path(command[-1])
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(b"jpeg")

    monkeypatch.setattr("src.services.video_frame_extractor.subprocess.run", run)
    monkeypatch.setattr(
        "src.services.video_frame_extractor.is_valid_image",
        lambda _path: True,
    )
    extractor = VideoFrameExtractor.__new__(VideoFrameExtractor)
    output = tmp_path / "frame.jpg"

    extractor.extract_frame(
        Path("sample.mp4"),
        1.0,
        output,
        max_width=None,
        video_stream_index=2,
    )

    mapping_index = commands[0].index("-map")
    assert commands[0][mapping_index + 1] == "0:2"
    assert output.is_file()
