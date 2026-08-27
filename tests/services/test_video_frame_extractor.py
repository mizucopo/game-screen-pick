"""単一動画向けffprobe metadata境界のテスト."""

import stat
from pathlib import Path
from typing import Any

import pytest

from src.services.video_frame_extractor import VideoFrameExtractor


def _probe_with_payload(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    *,
    packet_payload: dict[str, Any] | None = None,
) -> tuple[VideoFrameExtractor, list[list[str]]]:
    """外部commandを使わず指定payloadを返すextractorを作る."""
    commands: list[list[str]] = []
    extractor = VideoFrameExtractor.__new__(VideoFrameExtractor)

    def run_json(command: list[str]) -> dict[str, Any]:
        commands.append(command)
        if "-show_packets" in command:
            return packet_payload if packet_payload is not None else {"packets": []}
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


def test_probe_offsets_delayed_video_and_limits_fallback_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """遅れて始まるvideo streamの位置と残り時間を返すこと."""
    extractor, commands = _probe_with_payload(
        monkeypatch,
        {
            "format": {"start_time": "0.0", "duration": "8.0"},
            "streams": [
                {"index": 0, "codec_type": "audio", "duration": "8.0"},
                {
                    "index": 1,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "1/1",
                    "start_time": "5.0",
                    "duration": "N/A",
                },
            ],
        },
    )

    metadata = extractor.probe(Path("sample.mp4"))

    assert metadata.start_time_seconds == 5.0
    assert metadata.duration_seconds == 3.0
    assert "start_time" in commands[0][commands[0].index("-show_entries") + 1]


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


def test_probe_reads_actual_last_video_packet_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """可変frame間隔でも最後にdecode可能な位置をmetadataへ含めること."""
    extractor, commands = _probe_with_payload(
        monkeypatch,
        {
            "format": {"start_time": "0.0", "duration": "8.0"},
            "streams": [
                {
                    "index": 2,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "30/1",
                    "duration": "8.0",
                }
            ],
        },
        packet_payload={
            "packets": [
                {"pts_time": "0.0", "dts_time": "-0.033"},
                {"pts_time": "5.0", "dts_time": "4.967"},
                {"pts_time": "6.5", "dts_time": "6.467"},
            ]
        },
    )

    metadata = extractor.probe(Path("sample.mp4"))

    assert metadata.last_frame_timestamp_seconds == 6.5
    packet_command = next(command for command in commands if "-show_packets" in command)
    assert packet_command[packet_command.index("-select_streams") + 1] == "2"


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


def test_extract_frame_does_not_reuse_fixed_temporary_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """予測可能な旧temporary pathを削除せずsymlinkも辿らないこと."""

    def run(command: list[str], **_kwargs: Any) -> None:
        Path(command[-1]).write_bytes(b"jpeg")

    monkeypatch.setattr("src.services.video_frame_extractor.subprocess.run", run)
    monkeypatch.setattr(
        "src.services.video_frame_extractor.is_valid_image",
        lambda _path: True,
    )
    extractor = VideoFrameExtractor.__new__(VideoFrameExtractor)
    output = tmp_path / "frame.jpg"
    legacy_temporary = tmp_path / ".frame.partial.jpg"
    external = tmp_path / "external.txt"
    external.write_text("user-owned", encoding="utf-8")
    legacy_temporary.symlink_to(external)
    mode_probe = tmp_path / "mode-probe"
    mode_probe.touch()
    expected_mode = stat.S_IMODE(mode_probe.stat().st_mode)

    extractor.extract_frame(Path("sample.mp4"), 1.0, output, max_width=None)

    assert output.read_bytes() == b"jpeg"
    assert stat.S_IMODE(output.stat().st_mode) == expected_mode
    assert legacy_temporary.is_symlink()
    assert external.read_text(encoding="utf-8") == "user-owned"
