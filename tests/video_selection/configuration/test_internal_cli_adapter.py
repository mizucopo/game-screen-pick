"""動画入力internal CLI adapterのcontract test。"""

from pathlib import Path

import pytest

from src.video_selection.configuration.configuration_error import ConfigurationError
from src.video_selection.configuration.configuration_source import ConfigurationSource
from src.video_selection.configuration.internal_cli_adapter import (
    run_internal_cli_adapter,
)
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.run_outcome import RunOutcome
from src.video_selection.models.run_status import RunStatus


def test_adapter_passes_effective_configuration_to_application_boundary(
    tmp_path: Path,
) -> None:
    """解決済みEffective Configurationがapplication境界へ渡されること。

    Arrange:
        - TOML、環境変数、CLI値と記録用application callableが用意される
    Act:
        - internal CLI adapterが実行される
    Assert:
        - 優先順位解決済み設定が一度だけ渡されRunOutcomeが返されること
    """
    # Arrange
    config_path = tmp_path / "video-selection.toml"
    config_path.write_text(
        'config_version = "1.0.0"\n'
        "[input]\nrecursive = true\n"
        "[selection]\nimage_count = 80\n"
        '[ollama]\nhost = "http://toml.example:11434"\n',
        encoding="utf-8",
    )
    captured: list[EffectiveConfiguration] = []
    expected = RunOutcome(
        output_folder=tmp_path / "output",
        status=RunStatus.COMPLETED,
        requested_count=60,
        selected_count=60,
        completed_stages=(),
    )

    def application_run(configuration: EffectiveConfiguration) -> RunOutcome:
        captured.append(configuration)
        return expected

    # Act
    outcome = run_internal_cli_adapter(
        application_run,
        video_input_folder=tmp_path / "videos",
        output_folder=tmp_path / "output",
        config_path=config_path,
        image_count=60,
        recursive=False,
        video_scan_workers=5,
        video_scan_auto_max_workers=7,
        reset_cache=True,
        debug=True,
        environ={"OLLAMA_HOST": "http://env.example:11434"},
    )

    # Assert
    assert outcome is expected
    assert len(captured) == 1
    configuration = captured[0]
    assert configuration.image_count == 60
    assert configuration.recursive is False
    assert configuration.ollama_host == "http://toml.example:11434"
    assert configuration.video_scan_workers == 5
    assert configuration.video_scan_auto_max_workers == 7
    assert configuration.reset_cache is True
    assert configuration.debug is True
    assert configuration.source_for("selection.image_count") is ConfigurationSource.CLI
    assert configuration.source_for("input.recursive") is ConfigurationSource.CLI
    assert configuration.source_for("ollama.host") is ConfigurationSource.TOML
    assert configuration.source_for("video_scan.workers") is ConfigurationSource.CLI
    assert configuration.source_for("reset_cache") is ConfigurationSource.CLI
    assert configuration.source_for("debug") is ConfigurationSource.CLI


def test_adapter_rejects_config_before_calling_application(
    tmp_path: Path,
) -> None:
    """config error時にapplication境界とfilesystem副作用が実行されないこと。

    Arrange:
        - unknown keyを含むTOMLと呼出禁止application callableが用意される
    Act:
        - internal CLI adapterが実行される
    Assert:
        - exit 2相当で停止しapplication、cache、outputが未実行であること
    """
    # Arrange
    config_path = tmp_path / "video-selection.toml"
    config_path.write_text(
        'config_version = "1.0.0"\n[selection]\nimage_cout = 10\n',
        encoding="utf-8",
    )
    video_input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"

    def forbidden_application_run(
        _configuration: EffectiveConfiguration,
    ) -> RunOutcome:
        pytest.fail("application境界は呼ばれないこと")

    # Act / Assert
    with pytest.raises(ConfigurationError) as error:
        run_internal_cli_adapter(
            forbidden_application_run,
            video_input_folder=video_input_folder,
            output_folder=output_folder,
            config_path=config_path,
            environ={},
        )
    assert error.value.exit_code == 2
    assert not output_folder.exists()
    assert not (video_input_folder / ".game-screen-pick").exists()
