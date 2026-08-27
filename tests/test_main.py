"""動画ディレクトリCLI adapterの単体テスト."""

import json
import logging
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.main import execute, run
from src.models.video_selection_request import VideoSelectionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated_default_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """各CLI testへ秘密値を持たない既定configと隔離済み環境を用意する."""
    for name in (
        "OLLAMA_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "XAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.toml").write_text(
        """[run]
game_context_provider = "ollama"
game_context_model = "qwen3.8:27b"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


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
    external_video = tmp_path / "external.mp4"
    external_video.write_bytes(b"video")
    (input_dir / "linked.mp4").symlink_to(external_video)
    captured_requests: list[VideoSelectionRequest] = []
    config_path = tmp_path / "picker.toml"
    config_path.write_text(
        """[run]
game_context_provider = "openai"
game_context_model = "gpt-context"
openai_api_key = "configured-secret"
primary_model = "primary:latest"
secondary_model = "secondary:latest"
ollama_host = "192.168.1.31:11434"
ollama_timeout = 120
allow_cpu = true
ffmpeg_workers = 4
sample_interval_seconds = 2.5
debug = true
""",
        encoding="utf-8",
    )

    def capture_request(request: VideoSelectionRequest) -> None:
        captured_requests.append(request)

    monkeypatch.setattr("src.main.run_video_application", capture_request)

    run(
        [
            "-n",
            "12",
            "--game-title",
            "ゲーム名",
            "--config",
            str(config_path),
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
            game_context="",
            game_context_provider="openai",
            game_context_model="gpt-context",
            game_context_api_key="configured-secret",
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


def test_cli_logs_project_version_and_all_effective_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """起動直後にproject情報と明示した選択枚数を含む全設定を確認できること."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    (input_dir / "Sample Game Part1.mp4").write_bytes(b"video")
    output_dir = tmp_path / "selected"
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.example:11434")
    monkeypatch.setattr("src.main._project_version", lambda: "1.8.0-test")
    monkeypatch.setattr("src.main.run_video_application", lambda _request: None)
    caplog.set_level(logging.INFO)

    run(
        [
            "-n",
            "30",
            "--game-context",
            "探索を含む",
            str(input_dir),
            str(output_dir),
        ]
    )

    messages = [record.getMessage() for record in caplog.records]
    assert "game-screen-pick 1.8.0-test の画像選定処理を開始します。" in messages
    assert "実効設定:" in messages
    assert [message for message in messages if message.startswith("  ")] == [
        '  --config: "config/config.toml"',
        "  --num: 30",
        '  --game-title: ""',
        '  --game-context: "探索を含む"',
        '  [run].game_context_provider: "ollama"',
        '  [run].game_context_model: "qwen3.8:27b"',
        '  [run].primary_model: "qwen3.8:27b"',
        '  [run].secondary_model: "muse-glimmer:30b"',
        '  [run].ollama_host: "http://ollama.example:11434（自動決定: OLLAMA_HOST）"',
        "  [run].ollama_timeout: 900.0",
        "  [run].allow_cpu: false",
        "  [run].ffmpeg_workers: 2",
        '  [run].sample_interval_seconds: "<自動決定: 動画時間と選択枚数>"',
        "  [run].debug: false",
        f"  INPUT_VIDEO_DIR: {json.dumps(str(input_dir), ensure_ascii=False)}",
        f"  OUTPUT_DIR: {json.dumps(str(output_dir), ensure_ascii=False)}",
    ]


@pytest.mark.parametrize(
    "args,error_pattern",
    [
        (["-n", "0"], "正の整数"),
        (["-n", "-1"], "正の整数"),
        (["-n", "1000"], "999以下"),
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


def test_cli_requires_output_count_before_application(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """選択枚数を省略するとapplication開始前のusage errorになること."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    (input_dir / "game.mp4").write_bytes(b"video")
    monkeypatch.setattr(
        "src.main.run_video_application",
        lambda _request: pytest.fail("applicationは呼ばれないこと"),
    )

    result = CliRunner().invoke(
        execute,
        ["--game-context", "探索を含む", str(input_dir), str(tmp_path / "selected")],
    )

    assert result.exit_code == 2
    assert "Missing option '-n' / '--num'" in result.output


@pytest.mark.parametrize("output_count", [1, 999])
def test_cli_accepts_boundary_output_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    output_count: int,
) -> None:
    """選択枚数の下限と上限をapplicationへ渡すこと."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    (input_dir / "game.mp4").write_bytes(b"video")
    captured_requests: list[VideoSelectionRequest] = []
    monkeypatch.setattr("src.main.run_video_application", captured_requests.append)

    run(
        [
            "-n",
            str(output_count),
            "--game-context",
            "探索を含む",
            str(input_dir),
            str(tmp_path / "selected"),
        ]
    )

    assert captured_requests[0].output_count == output_count


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
        run(["-n", "1", str(input_video), str(tmp_path / "selected")])

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
        run(["-n", "1", str(input_dir), str(tmp_path / "selected")])

    assert "処理対象の動画が見つかりません" in capsys.readouterr().err


def test_cli_help_describes_directory_input() -> None:
    """helpが実行時入力だけをCLI optionとして案内すること."""
    result = CliRunner().invoke(execute, ["--help"])

    assert result.exit_code == 0
    assert "INPUT_VIDEO_DIR OUTPUT_DIR" in result.output
    assert "INPUT_VIDEO..." not in result.output
    assert "-c, --config FILE" in result.output
    assert "-n, --num INTEGER" in result.output
    assert "選択枚数（1から999）" in result.output
    assert "[required]" in result.output
    assert "--game-title TEXT" in result.output
    assert "--game-context TEXT" in result.output
    for removed_option in (
        "--game-context-provider",
        "--game-context-model",
        "--primary-model",
        "--secondary-model",
        "--ollama-host",
        "--ollama-timeout",
        "--allow-cpu",
        "--ffmpeg-workers",
        "--sample-interval-seconds",
        "--auto-sample-interval",
        "--debug",
    ):
        assert removed_option not in result.output


@pytest.mark.parametrize(
    "context_args",
    [
        [],
        ["--game-title", "Game", "--game-context", "直接指定"],
        ["--game-title", "  "],
        ["--game-context", "  "],
    ],
)
def test_cli_rejects_game_title_and_game_context_without_exactly_one_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    context_args: list[str],
) -> None:
    """新規実行ではtitleとcontextの両方指定・両方未指定を拒否すること."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    (input_dir / "game.mp4").write_bytes(b"video")
    monkeypatch.setattr(
        "src.main.run_video_application",
        lambda _request: pytest.fail("applicationは呼ばれないこと"),
    )

    with pytest.raises(SystemExit):
        run(["-n", "1", *context_args, str(input_dir), str(tmp_path / "selected")])

    assert "--game-titleと--game-contextのどちらか一方" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("config_text", "missing_key"),
    [
        ('[run]\ngame_context_model = "context-model"\n', "game_context_provider"),
        (
            '[run]\ngame_context_provider = ""\ngame_context_model = "context-model"\n',
            "game_context_provider",
        ),
        ('[run]\ngame_context_provider = "openai"\n', "game_context_model"),
        (
            '[run]\ngame_context_provider = "openai"\ngame_context_model = "  "\n',
            "game_context_model",
        ),
    ],
)
def test_cli_requires_explicit_context_provider_and_model_for_game_title(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    config_text: str,
    missing_key: str,
) -> None:
    """titleから生成する場合はproviderとmodelの明示指定を必須にすること."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    (input_dir / "game.mp4").write_bytes(b"video")
    config_path = tmp_path / "picker.toml"
    config_path.write_text(config_text, encoding="utf-8")
    monkeypatch.setattr(
        "src.main.run_video_application",
        lambda _request: pytest.fail("applicationは呼ばれないこと"),
    )

    with pytest.raises(SystemExit):
        run(
            [
                "-n",
                "1",
                "--game-title",
                "Game",
                "-c",
                str(config_path),
                str(input_dir),
                str(tmp_path / "selected"),
            ]
        )

    assert f"[run].{missing_key}" in capsys.readouterr().err


def test_cli_allows_direct_context_without_context_provider_or_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """直接指定では未設定のproviderとmodelを要求しないこと."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    input_video = input_dir / "game.mp4"
    input_video.write_bytes(b"video")
    config_path = tmp_path / "picker.toml"
    config_path.write_text("[run]\n", encoding="utf-8")
    captured_requests: list[VideoSelectionRequest] = []
    monkeypatch.setattr("src.main.run_video_application", captured_requests.append)

    run(
        [
            "-n",
            "1",
            "--game-context",
            "直接指定",
            "-c",
            str(config_path),
            str(input_dir),
            str(tmp_path / "selected"),
        ]
    )

    assert captured_requests[0].game_context_provider is None
    assert captured_requests[0].game_context_model is None


def test_cli_loads_all_run_options_from_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """反復するoptionを設定ファイルだけからrequestへ渡せること."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    input_video = input_dir / "game.mp4"
    input_video.write_bytes(b"video")
    output_dir = tmp_path / "selected"
    config_path = tmp_path / "picker.toml"
    config_path.write_text(
        """[run]
game_context_provider = "openai"
game_context_model = "gpt-context"
openai_api_key = "configured-secret"
primary_model = "primary:latest"
secondary_model = "secondary:latest"
ollama_host = "192.168.1.31:11434"
ollama_timeout = 120
allow_cpu = true
ffmpeg_workers = 4
sample_interval_seconds = 2.5
debug = true
""",
        encoding="utf-8",
    )
    captured_requests: list[VideoSelectionRequest] = []
    monkeypatch.setattr("src.main.run_video_application", captured_requests.append)

    run(
        [
            "-c",
            str(config_path),
            "-n",
            "12",
            "--game-title",
            "ゲーム名",
            str(input_dir),
            str(output_dir),
        ]
    )

    assert captured_requests == [
        VideoSelectionRequest(
            input_videos=(str(input_video),),
            output_dir=str(output_dir),
            output_count=12,
            game_title="ゲーム名",
            game_context="",
            game_context_provider="openai",
            game_context_model="gpt-context",
            game_context_api_key="configured-secret",
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


def test_cli_uses_config_directory_toml_from_current_directory_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """config未指定時はcurrent directoryのconfig/config.tomlを使うこと."""
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    (input_dir / "game.mp4").write_bytes(b"video")
    config_path = tmp_path / "config" / "config.toml"
    config_path.write_text(
        """[run]
primary_model = "configured-primary"
allow_cpu = true
debug = true
""",
        encoding="utf-8",
    )
    captured_requests: list[VideoSelectionRequest] = []
    monkeypatch.setattr("src.main.run_video_application", captured_requests.append)

    run(
        [
            "--num",
            "20",
            "--game-context",
            "CLIの文脈",
            str(input_dir),
            str(tmp_path / "selected"),
        ]
    )

    request = captured_requests[0]
    assert request.output_count == 20
    assert request.game_context == "CLIの文脈"
    assert request.primary_model == "configured-primary"
    assert request.allow_cpu is True
    assert request.debug is True


def test_cli_prefers_non_empty_config_api_key_over_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """選択providerの設定値を環境変数より優先し、秘密値をlogへ出さないこと."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    (input_dir / "game.mp4").write_bytes(b"video")
    config_path = tmp_path / "picker.toml"
    config_path.write_text(
        """[run]
game_context_provider = "openai"
game_context_model = "gpt-context"
openai_api_key = "configured-secret"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    captured_requests: list[VideoSelectionRequest] = []
    monkeypatch.setattr("src.main.run_video_application", captured_requests.append)
    caplog.set_level(logging.INFO)

    run(
        [
            "-n",
            "1",
            "--game-title",
            "Game",
            "-c",
            str(config_path),
            str(input_dir),
            str(tmp_path / "selected"),
        ]
    )

    assert captured_requests[0].game_context_api_key == "configured-secret"
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "configured-secret" not in log_text
    assert "environment-secret" not in log_text
    assert "configured-secret" not in repr(captured_requests[0])


def test_cli_falls_back_to_selected_provider_environment_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """設定値が空なら選択providerの環境変数だけを使用すること."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    (input_dir / "game.mp4").write_bytes(b"video")
    config_path = tmp_path / "picker.toml"
    config_path.write_text(
        """[run]
game_context_provider = "gemini"
game_context_model = "gemini-test"
gemini_api_key = "  "
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "environment-secret")
    captured_requests: list[VideoSelectionRequest] = []
    monkeypatch.setattr("src.main.run_video_application", captured_requests.append)

    run(
        [
            "-n",
            "1",
            "--game-title",
            "Game",
            "-c",
            str(config_path),
            str(input_dir),
            str(tmp_path / "selected"),
        ]
    )

    assert captured_requests[0].game_context_api_key == "environment-secret"


def test_cli_rejects_missing_selected_provider_api_key_before_application(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """選択providerの認証値が両方なければapplication開始前に拒否すること."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    (input_dir / "game.mp4").write_bytes(b"video")
    config_path = tmp_path / "picker.toml"
    config_path.write_text(
        """[run]
game_context_provider = "xai"
game_context_model = "grok-test"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.main.run_video_application",
        lambda _request: pytest.fail("applicationは呼ばれないこと"),
    )

    with pytest.raises(SystemExit):
        run(
            [
                "-n",
                "1",
                "--game-title",
                "Game",
                "-c",
                str(config_path),
                str(input_dir),
                str(tmp_path / "selected"),
            ]
        )

    error = capsys.readouterr().err
    assert "[run].xai_api_key" in error
    assert "XAI_API_KEY" in error


def test_local_config_directory_is_ignored_except_gitkeep() -> None:
    """任意名のlocal configを通常のgit add対象から除外すること."""
    ignored_results = [
        subprocess.run(
            ["git", "check-ignore", "--quiet", path],
            cwd=PROJECT_ROOT,
            check=False,
        )
        for path in ("config/config.toml", "config/openai.toml")
    ]
    tracked_placeholder = subprocess.run(
        ["git", "check-ignore", "--quiet", "config/.gitkeep"],
        cwd=PROJECT_ROOT,
        check=False,
    )

    assert all(result.returncode == 0 for result in ignored_results)
    assert tracked_placeholder.returncode == 1


@pytest.mark.parametrize(
    "removed_args",
    [
        ["--game-context-provider", "openai"],
        ["--game-context-model", "model"],
        ["--primary-model", "model"],
        ["--secondary-model", "model"],
        ["--ollama-host", "localhost:11434"],
        ["--ollama-timeout", "120"],
        ["--allow-cpu"],
        ["--ffmpeg-workers", "4"],
        ["--sample-interval-seconds", "2.5"],
        ["--auto-sample-interval"],
        ["--debug"],
    ],
)
def test_cli_rejects_config_only_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    removed_args: list[str],
) -> None:
    """設定ファイル専用項目をCLI optionとして受け付けないこと."""
    monkeypatch.setattr(
        "src.main.run_video_application",
        lambda _request: pytest.fail("applicationは呼ばれないこと"),
    )

    with pytest.raises(SystemExit):
        run([*removed_args, "input", "output"])

    assert "No such option" in capsys.readouterr().err


@pytest.mark.parametrize(
    "config_text,error_pattern",
    [
        ("[unknown]\nvalue = 1\n", "未知の設定セクション"),
        ("[run]\nunknown = 1\n", "未知の設定キー"),
        ("[run]\nnum = 30\n", "未知の設定キー"),
        ('[run]\ngame_title = "Game"\n', "未知の設定キー"),
        ('[run]\ngame_context = "Context"\n', "未知の設定キー"),
        ('[run]\nollama_timeout = "30"\n', "number"),
        ("[run]\nopenai_api_key = 123\n", "string"),
        ("[run]\nollama_timeout = 0\n", "正の数"),
        ("[run]\nffmpeg_workers = 5\n", "1から4"),
        ("[run]\nsample_interval_seconds = 0.1\n", "0.25以上"),
        ('[run]\ngame_context_provider = "invalid"\n', "ollama, openai"),
        ("[run\n", "設定ファイルを読み込めません"),
    ],
)
def test_cli_rejects_invalid_config_before_application(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    config_text: str,
    error_pattern: str,
) -> None:
    """設定契約違反をapplication開始前に拒否すること."""
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(config_text, encoding="utf-8")
    monkeypatch.setattr(
        "src.main.run_video_application",
        lambda _request: pytest.fail("applicationは呼ばれないこと"),
    )

    with pytest.raises(SystemExit):
        run(["-n", "1", "-c", str(config_path), "missing-input", "output"])

    assert error_pattern in capsys.readouterr().err
