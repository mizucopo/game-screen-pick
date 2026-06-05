"""main.py CLI adapterの単体テスト."""

from pathlib import Path

import pytest

from src.main import run, validate_similarity_range
from src.models.application_run_request import ApplicationRunRequest


def test_cli_translates_options_to_application_run_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLIオプションがapplication実行リクエストへ変換されること.

    Arrange:
        - 有効な入力ディレクトリと設定ファイルがある
        - CLIオプションが一通り指定されている
    Act:
        - CLIが実行される
    Assert:
        - application実行層へ変換済みリクエストが渡されること
    """
    # Arrange
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    config_path = tmp_path / "picker.toml"
    report_path = tmp_path / "report.json"
    input_dir.mkdir()
    config_path.write_text("[thresholds]\nsimilarity = 0.7\n", encoding="utf-8")
    captured_requests: list[ApplicationRunRequest] = []

    def capture_request(request: ApplicationRunRequest) -> None:
        captured_requests.append(request)

    monkeypatch.setattr("src.main.run_application", capture_request)

    # Act
    run(
        [
            "-n",
            "3",
            "--similarity",
            "0.8",
            "--recursive",
            "--config",
            str(config_path),
            "--ollama-model",
            "gemma4",
            "--ollama-host",
            "http://localhost:11435",
            "--ollama-timeout",
            "30",
            "--ollama-max-workers",
            "2",
            "--no-ollama-cache",
            "--no-resume-cache",
            "--scene-hint",
            "アドベンチャーゲーム。会話差分が多い",
            "--report-json",
            str(report_path),
            "--rename",
            "--batch-size",
            "64",
            "--result-max-workers",
            "0",
            "--max-dim",
            "1080",
            "--max-memory-gb",
            "4",
            "--debug",
            str(input_dir),
            str(output_dir),
        ]
    )

    # Assert
    request = captured_requests[0]
    assert request.num == 3
    assert request.similarity == 0.8
    assert request.recursive is True
    assert request.config_path == str(config_path)
    assert request.ollama_model == "gemma4"
    assert request.ollama_host == "http://localhost:11435"
    assert request.ollama_timeout == 30.0
    assert request.ollama_max_workers == 2
    assert request.ollama_cache_enabled is False
    assert request.resume_cache_enabled is False
    assert request.scene_hint == "アドベンチャーゲーム。会話差分が多い"
    assert request.report_json == str(report_path)
    assert request.rename is True
    assert request.batch_size == 64
    assert request.result_max_workers == 1
    assert request.max_dim == 1080
    assert request.max_memory_gb == 4
    assert request.debug is True
    assert request.input_dir == str(input_dir)
    assert request.output_dir == str(output_dir)


@pytest.mark.parametrize(
    "args,error_pattern",
    [
        (["-n", "-1"], "正の整数"),
        (["-n", "0"], "正の整数"),
        (["--similarity", "-0.1"], "0.0~1.0"),
        (["--similarity", "1.1"], "0.0~1.0"),
        (["--ollama-timeout", "0"], "正の数"),
        (["--ollama-max-workers", "0"], "正の整数"),
        (["--profile", "active"], "No such option"),
        (["--scene-mix", "play=0.7,event=0.3"], "No such option"),
    ],
)
def test_cli_validates_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    error_pattern: str,
) -> None:
    """CLIの不正な入力が実行層へ渡される前に拒否されること.

    Arrange:
        - 無効な入力値が指定されている
    Act:
        - CLIが実行される
    Assert:
        - 適切なエラーメッセージが表示され、application実行層が呼ばれないこと
    """
    # Arrange
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()
    output_path.mkdir()
    monkeypatch.setattr(
        "src.main.run_application",
        lambda _request: pytest.fail("application実行層は呼ばれないこと"),
    )

    # Act / Assert
    with pytest.raises(SystemExit):
        run([*args, str(input_path), str(output_path)])
    captured = capsys.readouterr()
    assert error_pattern in captured.err


@pytest.mark.parametrize("value", ["0.0", "1.0", 0.0, 1.0])
def test_validate_similarity_range_accepts_inclusive_boundaries(
    value: float | str,
) -> None:
    """類似度しきい値の境界値が受け入れられること.

    Arrange:
        - 0.0または1.0の境界値がある
    Act:
        - 類似度しきい値として検証される
    Assert:
        - 浮動小数点値として返されること
    """
    # Arrange / Act
    result = validate_similarity_range(value)

    # Assert
    assert result == float(value)
