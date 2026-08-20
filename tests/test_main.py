"""単一動画CLI adapterの単体テスト."""

from pathlib import Path

import pytest

from src.main import run
from src.models.video_selection_request import VideoSelectionRequest


def test_cli_translates_multiple_inputs_to_video_selection_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLIオプションが単一動画requestへ変換されること."""
    input_video = tmp_path / "game-part-1.mp4"
    second_input_video = tmp_path / "game-part-2.mp4"
    output_dir = tmp_path / "selected"
    input_video.write_bytes(b"video")
    second_input_video.write_bytes(b"video")
    captured_requests: list[VideoSelectionRequest] = []

    def capture_request(request: VideoSelectionRequest) -> None:
        captured_requests.append(request)

    monkeypatch.setattr("src.main.run_video_application", capture_request)

    run(
        [
            "-n",
            "12",
            "--game-title",
            "ゲーム名",
            "--game-context",
            "探索を含む",
            "--primary-model",
            "primary:latest",
            "--secondary-model",
            "secondary:latest",
            "--ollama-host",
            "192.168.1.31:11434",
            "--ollama-timeout",
            "120",
            "--allow-cpu",
            "--ffmpeg-workers",
            "4",
            "--sample-interval-seconds",
            "2.5",
            "--debug",
            str(input_video),
            str(second_input_video),
            str(output_dir),
        ]
    )

    assert captured_requests == [
        VideoSelectionRequest(
            input_videos=(str(input_video), str(second_input_video)),
            output_dir=str(output_dir),
            output_count=12,
            game_title="ゲーム名",
            game_context="探索を含む",
            primary_model="primary:latest",
            secondary_model="secondary:latest",
            ollama_host="192.168.1.31:11434",
            ollama_timeout=120.0,
            allow_cpu=True,
            ffmpeg_workers=4,
            sample_interval_seconds=2.5,
            debug=True,
        )
    ]


@pytest.mark.parametrize(
    "args,error_pattern",
    [
        (["-n", "0"], "正の整数"),
        (["-n", "-1"], "正の整数"),
        (["--ollama-timeout", "0"], "正の数"),
        (["--sample-interval-seconds", "-1"], "正の数"),
        (["--sample-interval-seconds", "0.1"], "0.25以上"),
        (["--ffmpeg-workers", "0"], "正の整数"),
        (["--ffmpeg-workers", "5"], "1から4"),
        (["-n", "601"], "600以下"),
    ],
)
def test_cli_rejects_invalid_numeric_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    error_pattern: str,
) -> None:
    """不正な数値をapplicationへ渡さないこと."""
    input_video = tmp_path / "game.mp4"
    input_video.write_bytes(b"video")
    monkeypatch.setattr(
        "src.main.run_video_application",
        lambda _request: pytest.fail("applicationは呼ばれないこと"),
    )

    with pytest.raises(SystemExit):
        run([*args, str(input_video), str(tmp_path / "selected")])

    assert error_pattern in capsys.readouterr().err


def test_cli_requires_a_file_as_input(tmp_path: Path) -> None:
    """入力folderを動画として受け入れないこと."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(SystemExit):
        run([str(input_dir), str(tmp_path / "selected")])
