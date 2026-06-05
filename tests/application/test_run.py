"""application run moduleの単体テスト."""

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import click
import pytest

from src.application.run import run_application
from src.models.analyzer_config import AnalyzerConfig
from src.models.application_run_request import ApplicationRunRequest
from src.models.ollama_config import OllamaConfig
from src.models.picker_statistics import PickerStatistics
from src.models.scored_candidate import ScoredCandidate
from src.models.selection_annotation import SelectionAnnotation
from src.models.selection_config import SelectionConfig
from tests.conftest import create_scored_candidate


def _build_request(
    input_dir: Path,
    output_dir: Path,
    *,
    num: int = 100,
    report_json: str | None = None,
    rename: bool = False,
) -> ApplicationRunRequest:
    return ApplicationRunRequest(
        num=num,
        similarity=None,
        recursive=False,
        config_path=None,
        ollama_model="gemma4",
        ollama_host=None,
        ollama_timeout=None,
        ollama_max_workers=None,
        reset_cache=False,
        scene_hint=None,
        report_json=report_json,
        rename=rename,
        batch_size=None,
        result_max_workers=None,
        max_dim=720,
        max_memory_gb=1,
        debug=False,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
    )


def _build_stats(
    *,
    total_files: int,
    selected_count: int,
    selection_annotations_by_path: dict[str, SelectionAnnotation] | None = None,
) -> PickerStatistics:
    return PickerStatistics(
        total_files=total_files,
        analyzed_ok=total_files,
        analyzed_fail=0,
        rejected_by_similarity=0,
        rejected_by_content_filter=0,
        selected_count=selected_count,
        scene_distribution={"battle": selected_count},
        scene_mix_target={"battle": selected_count},
        scene_mix_actual={"battle": selected_count},
        threshold_relaxation_steps=[0.72],
        content_filter_breakdown={
            "blackout": 0,
            "whiteout": 0,
            "single_tone": 0,
            "fade_transition": 0,
            "temporal_transition": 0,
        },
        selection_annotations_by_path=selection_annotations_by_path or {},
    )


def _arrange_picker(
    monkeypatch: pytest.MonkeyPatch,
    selected: list[ScoredCandidate],
    stats: PickerStatistics,
) -> None:
    picker = MagicMock()
    picker.select.return_value = (selected, [], stats)
    analyzer = MagicMock()

    monkeypatch.setattr(
        "src.application.run.ImageQualityAnalyzer",
        lambda *_args, **_kwargs: nullcontext(analyzer),
    )
    monkeypatch.setattr(
        "src.application.run.GameScreenPicker",
        lambda *_args, **_kwargs: picker,
    )
    monkeypatch.setattr(
        "src.application.run.OllamaSceneAnalyzer",
        lambda *_args, **_kwargs: MagicMock(),
    )


def _arrange_keyboard_interrupt_picker(monkeypatch: pytest.MonkeyPatch) -> None:
    """画像選定中にKeyboardInterruptを送出するpickerを設定する."""
    picker = MagicMock()
    picker.select.side_effect = KeyboardInterrupt
    analyzer = MagicMock()
    monkeypatch.setattr(
        "src.application.run.ImageQualityAnalyzer",
        lambda *_args, **_kwargs: nullcontext(analyzer),
    )
    monkeypatch.setattr(
        "src.application.run.GameScreenPicker",
        lambda *_args, **_kwargs: picker,
    )
    monkeypatch.setattr(
        "src.application.run.OllamaSceneAnalyzer",
        lambda *_args, **_kwargs: MagicMock(),
    )


def test_run_application_selects_and_copies_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """application実行で選択画像がコピーされること.

    Arrange:
        - 入力ディレクトリに5件の画像がある
        - pickerが3件を選択して返す
    Act:
        - applicationが実行される
    Assert:
        - 出力ディレクトリに選択画像がコピーされること
    """
    # Arrange
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for index in range(5):
        (input_dir / f"image{index}.jpg").write_bytes(b"fake_image_data")
    selected = [
        create_scored_candidate(path=str(input_dir / f"image{index}.jpg"))
        for index in range(3)
    ]
    _arrange_picker(
        monkeypatch,
        selected,
        _build_stats(total_files=5, selected_count=3),
    )

    # Act
    run_application(_build_request(input_dir, output_dir, num=3))

    # Assert
    assert (output_dir / "image0.jpg").exists()
    assert (output_dir / "image1.jpg").exists()
    assert (output_dir / "image2.jpg").exists()


def test_run_application_writes_report_json_with_output_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """application実行でJSONレポートが出力されること.

    Arrange:
        - 選択結果にscene slug/display nameが含まれる
        - 統計情報にscore_bandの選定注釈が含まれる
        - report_jsonが指定されている
    Act:
        - applicationが実行される
    Assert:
        - JSONレポートに各スコアとoutput_pathが出力されること
    """
    # Arrange
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    report_path = tmp_path / "report.json"
    input_dir.mkdir()
    source = input_dir / "image0.jpg"
    source.write_bytes(b"fake_image_data")
    selected = [
        create_scored_candidate(
            path=str(source),
            scene_slug="battle",
            scene_display_name="戦闘",
            scene_description="敵との戦闘場面",
            selection_score=0.8,
        )
    ]
    _arrange_picker(
        monkeypatch,
        selected,
        _build_stats(
            total_files=1,
            selected_count=1,
            selection_annotations_by_path={
                str(source): SelectionAnnotation(score_band="high")
            },
        ),
    )

    # Act
    run_application(
        _build_request(
            input_dir,
            output_dir,
            report_json=str(report_path),
        )
    )

    # Assert
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["selected"][0]["path"] == str(source)
    assert payload["selected"][0]["output_path"] == str(
        (output_dir / "image0.jpg").resolve()
    )
    assert payload["selected"][0]["scene_slug"] == "battle"
    assert payload["selected"][0]["scene_display_name"] == "戦闘"
    assert payload["selected"][0]["score_band"] == "high"


def test_run_application_renames_outputs_by_scene(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """application実行でscene別ファイル名へ変更されること.

    Arrange:
        - play画像2件、event画像1件が選択される
        - renameが指定されている
    Act:
        - applicationが実行される
    Assert:
        - scene別連番のファイル名でコピーされること
    """
    # Arrange
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    sources = {
        "play_png": input_dir / "source_play.png",
        "event_jpg": input_dir / "source_event.jpg",
        "play_jpg": input_dir / "source_play2.jpg",
    }
    for path in sources.values():
        path.write_bytes(b"fake_image_data")
    selected = [
        create_scored_candidate(
            path=str(sources["play_png"]),
            scene_slug="battle",
            scene_display_name="戦闘",
            scene_description="敵との戦闘場面",
        ),
        create_scored_candidate(
            path=str(sources["event_jpg"]),
            scene_slug="conversation",
            scene_display_name="会話",
            scene_description="人物同士の会話場面",
        ),
        create_scored_candidate(
            path=str(sources["play_jpg"]),
            scene_slug="battle",
            scene_display_name="戦闘",
            scene_description="敵との戦闘場面",
        ),
    ]
    _arrange_picker(
        monkeypatch,
        selected,
        _build_stats(total_files=3, selected_count=3),
    )

    # Act
    run_application(_build_request(input_dir, output_dir, num=3, rename=True))

    # Assert
    assert (output_dir / "battle0001.png").exists()
    assert (output_dir / "conversation0001.jpg").exists()
    assert (output_dir / "battle0002.jpg").exists()


def test_run_application_resolves_configs_and_constructs_picker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """解決済み設定でanalyzerとpickerが構築されること.

    Arrange:
        - 設定ファイルとCLI上書き値が指定されている
        - pickerが空の選択結果を返す
    Act:
        - applicationが実行される
    Assert:
        - analyzerとpickerへ解決済み設定が渡されること
    """
    # Arrange
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    config_path = tmp_path / "picker.toml"
    input_dir.mkdir()
    config_path.write_text(
        "[thresholds]\nsimilarity = 0.66\n",
        encoding="utf-8",
    )
    analyzer_configs: list[AnalyzerConfig] = []
    selection_configs: list[SelectionConfig] = []
    ollama_configs: list[OllamaConfig] = []
    picker = MagicMock()
    picker.select.return_value = (
        [],
        [],
        _build_stats(total_files=0, selected_count=0),
    )
    request = replace(
        _build_request(input_dir, output_dir),
        config_path=str(config_path),
        similarity=0.8,
        batch_size=64,
        result_max_workers=2,
        max_dim=1080,
        max_memory_gb=4,
        ollama_model="cli-model",
        ollama_host="http://cli:11434",
        ollama_timeout=30.0,
        ollama_max_workers=2,
    )

    def capture_analyzer_config(
        *_args: object,
        config: AnalyzerConfig,
    ) -> object:
        analyzer_configs.append(config)
        return nullcontext(MagicMock())

    def capture_selection_config(
        *_args: object,
        config: SelectionConfig,
        scene_analyzer: object,
    ) -> MagicMock:
        del scene_analyzer
        selection_configs.append(config)
        return picker

    def capture_ollama_config(config: OllamaConfig) -> MagicMock:
        ollama_configs.append(config)
        return MagicMock()

    monkeypatch.setattr(
        "src.application.run.ImageQualityAnalyzer",
        capture_analyzer_config,
    )
    monkeypatch.setattr(
        "src.application.run.GameScreenPicker",
        capture_selection_config,
    )
    monkeypatch.setattr(
        "src.application.run.OllamaSceneAnalyzer",
        capture_ollama_config,
    )

    # Act
    run_application(request)

    # Assert
    assert analyzer_configs[0].result_max_workers == 2
    assert analyzer_configs[0].max_dim == 1080
    assert analyzer_configs[0].max_memory_gb == 4
    assert selection_configs[0].similarity_threshold == 0.8
    assert selection_configs[0].batch_size == 64
    assert selection_configs[0].ollama is not None
    assert selection_configs[0].ollama.model == "cli-model"
    assert ollama_configs[0].host == "http://cli:11434"
    assert ollama_configs[0].timeout == 30.0


def test_run_application_converts_unexpected_errors_to_system_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """想定外の実行時エラーが終了コード1へ変換されること.

    Arrange:
        - 入力ディレクトリが存在している
        - 設定解決で想定外エラーが発生する
    Act:
        - applicationが実行される
    Assert:
        - SystemExitの終了コード1へ変換されること
    """
    # Arrange
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    def raise_unexpected_error(**_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.application.run.ConfigResolver.resolve_configs",
        raise_unexpected_error,
    )

    # Act / Assert
    with pytest.raises(SystemExit) as exc_info:
        run_application(_build_request(input_dir, output_dir))
    assert exc_info.value.code == 1


def test_run_application_reports_keyboard_interrupt_as_resumable_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ctrl+C中断時に再実行で再開できることが案内されること.

    Arrange:
        - 画像選定中にKeyboardInterruptが発生する
    Act:
        - applicationが実行される
    Assert:
        - 終了コード130で終了し、再開案内が出力されること
    """
    # Arrange
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _arrange_keyboard_interrupt_picker(monkeypatch)
    caplog.set_level("INFO")

    # Act / Assert
    with pytest.raises(SystemExit) as exc_info:
        run_application(_build_request(input_dir, output_dir))
    assert exc_info.value.code == 130
    assert "中断されました" in caplog.text
    assert "再実行するとcacheから再開します" in caplog.text


def test_run_application_resets_cache_before_selecting_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """reset cache指定時は画像選定前にcache directoryが削除されること.

    Arrange:
        - 入力ディレクトリに既存cache fileがある
        - reset cacheが指定されている
    Act:
        - applicationが実行される
    Assert:
        - picker実行時点で既存cache fileが削除されていること
    """
    # Arrange
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    root_cache_file = input_dir / ".game-screen-pick" / "cache" / "ollama-scenes.json"
    nested_cache_file = (
        input_dir / "chapter1" / ".game-screen-pick" / "cache" / "ollama-scenes.json"
    )
    for cache_file in (root_cache_file, nested_cache_file):
        cache_file.parent.mkdir(parents=True)
        cache_file.write_text("cached", encoding="utf-8")
    observed_cache_exists: list[tuple[bool, bool]] = []
    picker = MagicMock()
    picker.select.return_value = ([], [], _build_stats(total_files=0, selected_count=0))

    def capture_picker(
        *_args: object,
        **_kwargs: object,
    ) -> MagicMock:
        observed_cache_exists.append(
            (root_cache_file.exists(), nested_cache_file.exists())
        )
        return picker

    monkeypatch.setattr(
        "src.application.run.ImageQualityAnalyzer",
        lambda *_args, **_kwargs: nullcontext(MagicMock()),
    )
    monkeypatch.setattr("src.application.run.GameScreenPicker", capture_picker)
    monkeypatch.setattr(
        "src.application.run.OllamaSceneAnalyzer",
        lambda *_args, **_kwargs: MagicMock(),
    )

    # Act
    run_application(replace(_build_request(input_dir, output_dir), reset_cache=True))

    # Assert
    assert observed_cache_exists == [(False, False)]
    assert not root_cache_file.exists()
    assert not nested_cache_file.exists()


def test_run_application_keeps_click_exceptions(
    tmp_path: Path,
) -> None:
    """Click例外がSystemExitへ変換されず維持されること.

    Arrange:
        - 入力ディレクトリが存在しない
    Act:
        - applicationが実行される
    Assert:
        - click.BadParameterとして送出されること
    """
    # Arrange
    input_dir = tmp_path / "missing"
    output_dir = tmp_path / "output"

    # Act / Assert
    with pytest.raises(click.BadParameter):
        run_application(_build_request(input_dir, output_dir))
