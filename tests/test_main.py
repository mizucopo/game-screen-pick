"""動画ディレクトリCLI adapterの単体テスト."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from src.main import execute, run
from src.models.video_selection_request import VideoSelectionRequest


def test_video_selection_request_preserves_legacy_positional_constructor() -> None:
    """旧単一動画requestの13引数位置指定を同じ順序で受け付けること."""
    request = VideoSelectionRequest(
        "input.mp4",
        "output",
        12,
        "ゲーム名",
        "探索を含む",
        "primary",
        "secondary",
        "127.0.0.1:11434",
        120.0,
        True,
        4,
        2.5,
        True,
    )

    assert request.input_videos == ("input.mp4",)
    assert request.input_video == "input.mp4"
    assert request.output_dir == "output"
    assert request.output_count == 12
    assert request.debug is True


def test_cli_translates_sorted_directory_videos_to_video_selection_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLIオプションと名前順の対象動画がrequestへ変換されること."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    input_video = input_dir / "game-part-1.mp4"
    second_input_video = input_dir / "game-part-2.MKV"
    output_dir = tmp_path / "selected"
    second_input_video.write_bytes(b"video")
    input_video.write_bytes(b"video")
    (input_dir / "notes.txt").write_text("not a video", encoding="utf-8")
    nested_dir = input_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "game-part-3.mp4").write_bytes(b"video")
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
            str(input_dir),
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
    assert captured_requests[0].input_video is None


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
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    (input_dir / "game.mp4").write_bytes(b"video")
    monkeypatch.setattr(
        "src.main.run_video_application",
        lambda _request: pytest.fail("applicationは呼ばれないこと"),
    )

    with pytest.raises(SystemExit):
        run([*args, str(input_dir), str(tmp_path / "selected")])

    assert error_pattern in capsys.readouterr().err


def test_cli_rejects_a_file_as_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """単一動画ファイルを公開CLIの入力として受け入れないこと."""
    input_video = tmp_path / "game.mp4"
    input_video.write_bytes(b"video")
    monkeypatch.setattr(
        "src.main.run_video_application",
        lambda _request: pytest.fail("applicationは呼ばれないこと"),
    )

    with pytest.raises(SystemExit):
        run([str(input_video), str(tmp_path / "selected")])

    assert "入力動画ディレクトリが見つかりません" in capsys.readouterr().err


def test_cli_rejects_directory_without_supported_videos(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """対象動画がない入力ディレクトリを明確に拒否すること."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    (input_dir / "notes.txt").write_text("not a video", encoding="utf-8")
    nested_dir = input_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "nested.mp4").write_bytes(b"video")
    monkeypatch.setattr(
        "src.main.run_video_application",
        lambda _request: pytest.fail("applicationは呼ばれないこと"),
    )

    with pytest.raises(SystemExit):
        run([str(input_dir), str(tmp_path / "selected")])

    assert "処理対象の動画が見つかりません" in capsys.readouterr().err


def test_cli_help_describes_directory_input() -> None:
    """helpが入力動画ディレクトリと出力フォルダを案内すること."""
    result = CliRunner().invoke(execute, ["--help"])

    assert result.exit_code == 0
    assert "INPUT_VIDEO_DIR OUTPUT_DIR" in result.output
    assert "INPUT_VIDEO..." not in result.output
