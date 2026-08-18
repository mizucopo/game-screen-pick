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
