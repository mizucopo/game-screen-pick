"""動画入力Effective Configuration解決のcontract test。"""

from pathlib import Path

import pytest

from src.video_selection.configuration.configuration_error import ConfigurationError
from src.video_selection.configuration.configuration_source import ConfigurationSource
from src.video_selection.configuration.resolve_effective_configuration import (
    resolve_effective_configuration,
)
from src.video_selection.models.effective_configuration import EffectiveConfiguration


def _resolve_with_cli_override(
    *,
    video_input_folder: Path,
    output_folder: Path,
    config_path: Path,
    environ: dict[str, str],
    cli_name: str,
    cli_value: object,
) -> EffectiveConfiguration:
    """table testのCLI keyを型付き引数へ変換する。"""
    if cli_name == "image_count":
        assert type(cli_value) is int
        return resolve_effective_configuration(
            video_input_folder=video_input_folder,
            output_folder=output_folder,
            config_path=config_path,
            environ=environ,
            image_count=cli_value,
        )
    if cli_name == "recursive":
        assert type(cli_value) is bool
        return resolve_effective_configuration(
            video_input_folder=video_input_folder,
            output_folder=output_folder,
            config_path=config_path,
            environ=environ,
            recursive=cli_value,
        )
    if cli_name in {"scene_hint", "spoiler_sensitivity", "ollama_host"}:
        assert type(cli_value) is str
        if cli_name == "scene_hint":
            return resolve_effective_configuration(
                video_input_folder=video_input_folder,
                output_folder=output_folder,
                config_path=config_path,
                environ=environ,
                scene_hint=cli_value,
            )
        if cli_name == "spoiler_sensitivity":
            return resolve_effective_configuration(
                video_input_folder=video_input_folder,
                output_folder=output_folder,
                config_path=config_path,
                environ=environ,
                spoiler_sensitivity=cli_value,
            )
        return resolve_effective_configuration(
            video_input_folder=video_input_folder,
            output_folder=output_folder,
            config_path=config_path,
            environ=environ,
            ollama_host=cli_value,
        )
    if cli_name == "similarity_threshold":
        assert type(cli_value) is float
        return resolve_effective_configuration(
            video_input_folder=video_input_folder,
            output_folder=output_folder,
            config_path=config_path,
            environ=environ,
            similarity_threshold=cli_value,
        )
    raise AssertionError(f"未対応のCLI test keyです: {cli_name}")


@pytest.mark.parametrize(
    (
        "key",
        "attribute",
        "default_value",
        "toml_body",
        "toml_value",
        "cli_name",
        "cli_value",
        "environment_value",
    ),
    [
        pytest.param(
            "config_version",
            "config_version",
            "1.0.0",
            "",
            "1.0.0",
            None,
            None,
            None,
            id="config-version",
        ),
        pytest.param(
            "input.recursive",
            "recursive",
            False,
            "[input]\nrecursive = true\n",
            True,
            "recursive",
            False,
            None,
            id="recursive",
        ),
        pytest.param(
            "selection.image_count",
            "image_count",
            100,
            "[selection]\nimage_count = 80\n",
            80,
            "image_count",
            60,
            None,
            id="image-count",
        ),
        pytest.param(
            "selection.scene_hint",
            "scene_hint",
            None,
            '[selection]\nscene_hint = "TOML hint"\n',
            "TOML hint",
            "scene_hint",
            "CLI hint",
            None,
            id="scene-hint",
        ),
        pytest.param(
            "selection.spoiler_sensitivity",
            "spoiler_sensitivity",
            "medium",
            '[selection]\nspoiler_sensitivity = "high"\n',
            "high",
            "spoiler_sensitivity",
            "low",
            None,
            id="spoiler-sensitivity",
        ),
        pytest.param(
            "selection.similarity_threshold",
            "similarity_threshold",
            0.72,
            "[selection]\nsimilarity_threshold = 0.8\n",
            0.8,
            "similarity_threshold",
            0.7,
            None,
            id="similarity-threshold",
        ),
        pytest.param(
            "frame_extraction.heartbeat_interval_seconds",
            "heartbeat_interval_seconds",
            1.0,
            "[frame_extraction]\nheartbeat_interval_seconds = 2.0\n",
            2.0,
            None,
            None,
            None,
            id="heartbeat-interval",
        ),
        pytest.param(
            "frame_extraction.scene_change_threshold",
            "scene_change_threshold",
            0.25,
            "[frame_extraction]\nscene_change_threshold = 0.4\n",
            0.4,
            None,
            None,
            None,
            id="scene-change-threshold",
        ),
        pytest.param(
            "frame_extraction.scene_min_interval_seconds",
            "scene_min_interval_seconds",
            0.5,
            "[frame_extraction]\nscene_min_interval_seconds = 0.75\n",
            0.75,
            None,
            None,
            None,
            id="scene-min-interval",
        ),
        pytest.param(
            "frame_extraction.decode_backend",
            "decode_backend",
            "cpu",
            '[frame_extraction]\ndecode_backend = "nvdec"\n',
            "nvdec",
            None,
            None,
            None,
            id="decode-backend",
        ),
        pytest.param(
            "frame_extraction.refinement_radius_seconds",
            "refinement_radius_seconds",
            1.0,
            "[frame_extraction]\nrefinement_radius_seconds = 2.0\n",
            2.0,
            None,
            None,
            None,
            id="refinement-radius",
        ),
        pytest.param(
            "frame_extraction.max_frame_candidates",
            "max_frame_candidates",
            3,
            "[frame_extraction]\nmax_frame_candidates = 2\n",
            2,
            None,
            None,
            None,
            id="max-frame-candidates",
        ),
        pytest.param(
            "candidate_moments.density_per_minute",
            "candidate_density_per_minute",
            2.0,
            "[candidate_moments]\ndensity_per_minute = 3.5\n",
            3.5,
            None,
            None,
            None,
            id="candidate-density",
        ),
        pytest.param(
            "context.language",
            "language",
            "ja",
            '[context]\nlanguage = "en-US"\n',
            "en-US",
            None,
            None,
            None,
            id="language",
        ),
        pytest.param(
            "context.subtitle_stream_index",
            "subtitle_stream_index",
            None,
            "[context]\nsubtitle_stream_index = 2\n",
            2,
            None,
            None,
            None,
            id="subtitle-stream-index",
        ),
        pytest.param(
            "context.audio_stream_index",
            "audio_stream_index",
            None,
            "[context]\naudio_stream_index = 1\n",
            1,
            None,
            None,
            None,
            id="audio-stream-index",
        ),
        pytest.param(
            "ollama.host",
            "ollama_host",
            "http://localhost:11434",
            '[ollama]\nhost = "http://toml.example:11434"\n',
            "http://toml.example:11434",
            "ollama_host",
            "http://cli.example:11434",
            "http://env.example:11434",
            id="ollama-host",
        ),
        pytest.param(
            "ollama.timeout_seconds",
            "ollama_timeout_seconds",
            60.0,
            "[ollama]\ntimeout_seconds = 30\n",
            30.0,
            None,
            None,
            None,
            id="ollama-timeout",
        ),
        pytest.param(
            "ollama.max_parallel_requests",
            "ollama_max_parallel_requests",
            1,
            "[ollama]\nmax_parallel_requests = 2\n",
            2,
            None,
            None,
            None,
            id="ollama-parallelism",
        ),
        pytest.param(
            "models.auto_upgrade",
            "models_auto_upgrade",
            True,
            "[models]\nauto_upgrade = false\n",
            False,
            None,
            None,
            None,
            id="model-auto-upgrade",
        ),
        pytest.param(
            "models.scene_catalog.name",
            "scene_catalog_model",
            "qwen3-vl:8b-instruct",
            '[models.scene_catalog]\nname = "scene-model:latest"\n',
            "scene-model:latest",
            None,
            None,
            None,
            id="scene-catalog-model",
        ),
        pytest.param(
            "models.scene_catalog.num_ctx",
            "scene_catalog_num_ctx",
            32768,
            "[models.scene_catalog]\nnum_ctx = 65536\n",
            65536,
            None,
            None,
            None,
            id="scene-catalog-context",
        ),
        pytest.param(
            "models.candidate_annotation.name",
            "candidate_annotation_model",
            "qwen3-vl:8b-instruct",
            '[models.candidate_annotation]\nname = "annotation-model:latest"\n',
            "annotation-model:latest",
            None,
            None,
            None,
            id="candidate-annotation-model",
        ),
        pytest.param(
            "models.candidate_annotation.num_ctx",
            "candidate_annotation_num_ctx",
            32768,
            "[models.candidate_annotation]\nnum_ctx = 65536\n",
            65536,
            None,
            None,
            None,
            id="candidate-annotation-context",
        ),
        pytest.param(
            "models.speech_to_text.name",
            "speech_to_text_model",
            "dropbox-dash/faster-whisper-large-v3-turbo",
            '[models.speech_to_text]\nname = "openai/whisper-large-v3"\n',
            "openai/whisper-large-v3",
            None,
            None,
            None,
            id="speech-model",
        ),
        pytest.param(
            "models.speech_to_text.device",
            "speech_to_text_device",
            "cuda",
            '[models.speech_to_text]\ndevice = "cpu"\n',
            "cpu",
            None,
            None,
            None,
            id="speech-device",
        ),
        pytest.param(
            "models.speech_to_text.compute_type",
            "speech_to_text_compute_type",
            "float16",
            '[models.speech_to_text]\ncompute_type = "int8"\n',
            "int8",
            None,
            None,
            None,
            id="speech-compute-type",
        ),
        pytest.param(
            "models.speech_to_text.beam_size",
            "speech_to_text_beam_size",
            5,
            "[models.speech_to_text]\nbeam_size = 3\n",
            3,
            None,
            None,
            None,
            id="speech-beam-size",
        ),
        pytest.param(
            "speech_to_text.vad_filter",
            "speech_vad_filter",
            True,
            "[speech_to_text]\nvad_filter = false\n",
            False,
            None,
            None,
            None,
            id="speech-vad-filter",
        ),
        pytest.param(
            "speech_to_text.chunk_seconds",
            "speech_chunk_seconds",
            600.0,
            "[speech_to_text]\nchunk_seconds = 300\n",
            300.0,
            None,
            None,
            None,
            id="speech-chunk-seconds",
        ),
        pytest.param(
            "speech_to_text.overlap_seconds",
            "speech_overlap_seconds",
            5.0,
            "[speech_to_text]\noverlap_seconds = 10\n",
            10.0,
            None,
            None,
            None,
            id="speech-overlap-seconds",
        ),
    ],
)
def test_every_config_key_is_resolved_by_its_supported_precedence(
    tmp_path: Path,
    key: str,
    attribute: str,
    default_value: object,
    toml_body: str,
    toml_value: object,
    cli_name: str | None,
    cli_value: object,
    environment_value: str | None,
) -> None:
    """全設定keyがCLI、TOML、環境変数、既定値の順で解決されること。

    Arrange:
        - 各keyについて組み込み既定値と明示TOML値が用意される
        - 公開対象keyにはCLI値またはOLLAMA_HOSTも用意される
    Act:
        - 各sourceの組み合わせからEffective Configurationが解決される
    Assert:
        - 利用可能な最上位sourceの値とsource名が保持されること
    """
    # Arrange
    config_path = tmp_path / "video-selection.toml"
    config_path.write_text(
        f'config_version = "1.0.0"\n{toml_body}',
        encoding="utf-8",
    )
    environment: dict[str, str] = (
        {"OLLAMA_HOST": environment_value} if environment_value is not None else {}
    )
    video_input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"

    # Act
    built_in = resolve_effective_configuration(
        video_input_folder=video_input_folder,
        output_folder=output_folder,
        environ={},
    )
    from_toml = resolve_effective_configuration(
        video_input_folder=video_input_folder,
        output_folder=output_folder,
        config_path=config_path,
        environ=environment,
    )

    # Assert
    assert getattr(built_in, attribute) == default_value
    assert built_in.source_for(key) is ConfigurationSource.BUILT_IN
    assert getattr(from_toml, attribute) == toml_value
    assert from_toml.source_for(key) is ConfigurationSource.TOML

    if environment_value is not None:
        from_environment = resolve_effective_configuration(
            video_input_folder=video_input_folder,
            output_folder=output_folder,
            environ=environment,
        )
        assert getattr(from_environment, attribute) == environment_value
        assert from_environment.source_for(key) is ConfigurationSource.ENVIRONMENT

    if cli_name is not None:
        from_cli = _resolve_with_cli_override(
            video_input_folder=video_input_folder,
            output_folder=output_folder,
            config_path=config_path,
            environ=environment,
            cli_name=cli_name,
            cli_value=cli_value,
        )
        assert getattr(from_cli, attribute) == cli_value
        assert from_cli.source_for(key) is ConfigurationSource.CLI


def test_recursive_absence_and_explicit_false_are_distinguished(
    tmp_path: Path,
) -> None:
    """recursiveの未指定と明示falseが異なるsourceとして解決されること。

    Arrange:
        - TOMLでrecursiveがtrueに設定される
    Act:
        - CLI未指定とCLI明示falseでそれぞれ解決される
    Assert:
        - 未指定ではTOML値、明示falseではCLI値が採用されること
    """
    # Arrange
    config_path = tmp_path / "video-selection.toml"
    config_path.write_text(
        'config_version = "1.0.0"\n[input]\nrecursive = true\n',
        encoding="utf-8",
    )
    # Act
    unspecified = resolve_effective_configuration(
        video_input_folder=tmp_path / "videos",
        output_folder=tmp_path / "output",
        config_path=config_path,
        recursive=None,
        environ={},
    )
    explicitly_disabled = resolve_effective_configuration(
        video_input_folder=tmp_path / "videos",
        output_folder=tmp_path / "output",
        config_path=config_path,
        recursive=False,
        environ={},
    )

    # Assert
    assert unspecified.recursive is True
    assert unspecified.source_for("input.recursive") is ConfigurationSource.TOML
    assert explicitly_disabled.recursive is False
    assert explicitly_disabled.source_for("input.recursive") is ConfigurationSource.CLI


@pytest.mark.parametrize(
    "document",
    [
        pytest.param('[unknown]\nvalue = "x"\n', id="unknown-section"),
        pytest.param(
            "[selection]\nimage_cout = 10\n",
            id="unknown-key",
        ),
        pytest.param(
            '[models.scene_catalog]\nrevision = "secret"\n',
            id="unknown-model-key",
        ),
        pytest.param(
            '[models.scene_catalog]\nexpected_digest = "sha256:manual"\n',
            id="manual-model-hash",
        ),
        pytest.param('[input]\nrecursive = "yes"\n', id="invalid-type"),
        pytest.param(
            '[selection]\nspoiler_sensitivity = "extreme"\n',
            id="invalid-enum",
        ),
        pytest.param(
            "[selection]\nsimilarity_threshold = 0.99\n",
            id="invalid-range",
        ),
        pytest.param(
            "[video_scan]\nworkers = 0\n",
            id="invalid-video-scan-workers-range",
        ),
        pytest.param(
            '[video_scan]\nworkers = "dynamic"\n',
            id="invalid-video-scan-workers-enum",
        ),
        pytest.param(
            "[video_scan]\nauto_max_workers = 33\n",
            id="invalid-video-scan-auto-max",
        ),
        pytest.param(
            "[speech_to_text]\nchunk_seconds = 5\noverlap_seconds = 5\n",
            id="invalid-cross-constraint",
        ),
        pytest.param('config_version = "2.0.0"\n', id="unsupported-version"),
    ],
)
def test_invalid_toml_is_an_exit_two_error_before_side_effects(
    tmp_path: Path,
    document: str,
) -> None:
    """不正なTOMLが副作用前にexit 2相当として拒否されること。

    Arrange:
        - schemaに違反する明示TOMLが用意される
    Act:
        - Effective Configurationの解決が試行される
    Assert:
        - exit 2相当のerrorとなりcacheとoutputが作成されないこと
    """
    # Arrange
    config_path = tmp_path / "video-selection.toml"
    prefix = (
        "" if document.startswith("config_version") else ('config_version = "1.0.0"\n')
    )
    config_path.write_text(f"{prefix}{document}", encoding="utf-8")
    video_input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"

    # Act / Assert
    with pytest.raises(ConfigurationError) as error:
        resolve_effective_configuration(
            video_input_folder=video_input_folder,
            output_folder=output_folder,
            config_path=config_path,
            environ={},
        )
    assert error.value.exit_code == 2
    assert not output_folder.exists()
    assert not (video_input_folder / ".game-screen-pick").exists()


@pytest.mark.parametrize(
    "document",
    [
        pytest.param("[input]\nrecursive = false\n", id="missing-version"),
        pytest.param("config_version = 1\n", id="non-string-version"),
        pytest.param('config_version = "1.0.0"\ninvalid =', id="malformed-toml"),
    ],
)
def test_invalid_document_shape_is_an_exit_two_error(
    tmp_path: Path,
    document: str,
) -> None:
    """version欠落・型違反・構文違反がexit 2相当で拒否されること。

    Arrange:
        - 文書全体の契約に違反するTOMLが用意される
    Act:
        - Effective Configurationの解決が試行される
    Assert:
        - exit 2相当のConfigurationErrorが返されること
    """
    # Arrange
    config_path = tmp_path / "video-selection.toml"
    config_path.write_text(document, encoding="utf-8")

    # Act / Assert
    with pytest.raises(ConfigurationError) as error:
        resolve_effective_configuration(
            video_input_folder=tmp_path / "videos",
            output_folder=tmp_path / "output",
            config_path=config_path,
            environ={},
        )
    assert error.value.exit_code == 2


def test_only_explicit_config_path_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """config path未指定時にcurrent directoryのTOMLが探索されないこと。

    Arrange:
        - current directoryに無効なvideo-selection.tomlが置かれる
    Act:
        - config pathを指定せずEffective Configurationが解決される
    Assert:
        - TOMLが読まれず組み込み既定値が返されること
    """
    # Arrange
    (tmp_path / "video-selection.toml").write_text(
        'config_version = "unsupported"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    # Act
    configuration = resolve_effective_configuration(
        video_input_folder=tmp_path / "videos",
        output_folder=tmp_path / "output",
        environ={},
    )

    # Assert
    assert configuration.image_count == 100
    assert (
        configuration.source_for("selection.image_count")
        is ConfigurationSource.BUILT_IN
    )


def test_documented_complete_example_is_accepted(tmp_path: Path) -> None:
    """公開されたv1 TOML完全例がstrict schemaで受理されること。

    Arrange:
        - repositoryが所有する動画入力TOML完全例が指定される
    Act:
        - Effective Configurationが解決される
    Assert:
        - 全体が受理され代表値のsourceがTOMLとして保持されること
    """
    # Arrange
    config_path = Path("docs/examples/video-selection.toml")

    # Act
    configuration = resolve_effective_configuration(
        video_input_folder=tmp_path / "videos",
        output_folder=tmp_path / "output",
        config_path=config_path,
        environ={},
    )

    # Assert
    assert configuration.config_version == "1.0.0"
    assert configuration.models_auto_upgrade is True
    assert configuration.speech_to_text_device == "cuda"
    assert (
        configuration.source_for("models.speech_to_text.name")
        is ConfigurationSource.TOML
    )


def test_credentials_are_excluded_from_errors_and_provenance(tmp_path: Path) -> None:
    """credentialと無関係な環境変数がerrorとprovenanceに含まれないこと。

    Arrange:
        - credentialを含む無効なOLLAMA_HOSTと非公開環境変数が用意される
    Act:
        - environment値からEffective Configurationの解決が試行される
    Assert:
        - errorとsource provenanceにcredentialが出力されないこと
    """
    # Arrange
    secret = "do-not-print-this-secret"
    environment = {
        "OLLAMA_HOST": f"ftp://user:{secret}@ollama.example",
        "HF_TOKEN": secret,
    }

    # Act / Assert
    with pytest.raises(ConfigurationError) as error:
        resolve_effective_configuration(
            video_input_folder=tmp_path / "videos",
            output_folder=tmp_path / "output",
            environ=environment,
        )
    assert secret not in str(error.value)
    assert "HF_TOKEN" not in str(error.value)
    assert "ftp://" not in str(error.value)

    configuration = resolve_effective_configuration(
        video_input_folder=tmp_path / "videos",
        output_folder=tmp_path / "output",
        environ={"OLLAMA_HOST": f"http://user:{secret}@ollama.example"},
    )
    assert secret not in repr(configuration.provenance)
    assert configuration.source_for("ollama.host") is ConfigurationSource.ENVIRONMENT


def test_video_scan_worker_configuration_uses_documented_precedence(
    tmp_path: Path,
) -> None:
    """Video Scan worker設定がCLI、TOML、環境変数、既定値の順で解決されること。

    Arrange:
        - worker数とauto上限についてTOML値、環境変数値、CLI値が用意される
    Act:
        - sourceの異なるEffective Configurationが解決される
    Assert:
        - 各keyで利用可能な最上位sourceの値とsource名が保持されること
    """
    # Arrange
    config_path = tmp_path / "video-selection.toml"
    config_path.write_text(
        'config_version = "1.0.0"\n[video_scan]\nworkers = 4\nauto_max_workers = 8\n',
        encoding="utf-8",
    )
    environment = {
        "GAME_SCREEN_PICK_VIDEO_SCAN_WORKERS": "5",
        "GAME_SCREEN_PICK_VIDEO_SCAN_AUTO_MAX_WORKERS": "7",
    }
    video_input_folder = tmp_path / "videos"
    output_folder = tmp_path / "output"

    # Act
    built_in = resolve_effective_configuration(
        video_input_folder=video_input_folder,
        output_folder=output_folder,
        environ={},
    )
    from_environment = resolve_effective_configuration(
        video_input_folder=video_input_folder,
        output_folder=output_folder,
        environ=environment,
    )
    from_toml = resolve_effective_configuration(
        video_input_folder=video_input_folder,
        output_folder=output_folder,
        config_path=config_path,
        environ=environment,
    )
    from_cli = resolve_effective_configuration(
        video_input_folder=video_input_folder,
        output_folder=output_folder,
        config_path=config_path,
        video_scan_workers="auto",
        video_scan_auto_max_workers=6,
        environ=environment,
    )

    # Assert
    assert (built_in.video_scan_workers, built_in.video_scan_auto_max_workers) == (
        "auto",
        6,
    )
    assert (
        from_environment.video_scan_workers,
        from_environment.video_scan_auto_max_workers,
    ) == (5, 7)
    assert from_environment.source_for("video_scan.workers") is (
        ConfigurationSource.ENVIRONMENT
    )
    assert from_toml.video_scan_workers == 4
    assert from_toml.video_scan_auto_max_workers == 8
    assert from_toml.source_for("video_scan.workers") is ConfigurationSource.TOML
    assert from_cli.video_scan_workers == "auto"
    assert from_cli.video_scan_auto_max_workers == 6
    assert from_cli.source_for("video_scan.workers") is ConfigurationSource.CLI


@pytest.mark.parametrize(
    ("environment_key", "environment_value"),
    [
        pytest.param(
            "GAME_SCREEN_PICK_VIDEO_SCAN_WORKERS",
            "dynamic",
            id="workers-enum",
        ),
        pytest.param(
            "GAME_SCREEN_PICK_VIDEO_SCAN_AUTO_MAX_WORKERS",
            "0",
            id="auto-max-range",
        ),
    ],
)
def test_invalid_video_scan_environment_value_is_a_config_error(
    tmp_path: Path,
    environment_key: str,
    environment_value: str,
) -> None:
    """不正なVideo Scan環境変数が副作用前のconfig errorにされること。

    Arrange:
        - 型または範囲が不正なVideo Scan環境変数が用意される
    Act:
        - Effective Configurationの解決が試行される
    Assert:
        - exit 2相当のConfigurationErrorが返されること
    """
    # Arrange
    environment = {environment_key: environment_value}

    # Act
    def resolve() -> EffectiveConfiguration:
        return resolve_effective_configuration(
            video_input_folder=tmp_path / "videos",
            output_folder=tmp_path / "output",
            environ=environment,
        )

    # Assert
    with pytest.raises(ConfigurationError) as error:
        resolve()
    assert error.value.exit_code == 2
