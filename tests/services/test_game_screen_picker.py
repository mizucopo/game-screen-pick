"""GameScreenPickerの単体テスト."""

import logging
from pathlib import Path

import pytest

from src.models.analyzer_config import AnalyzerConfig
from src.models.selection_config import SelectionConfig
from src.services.game_screen_picker import GameScreenPicker
from tests.conftest import _feature, _near_duplicate, create_analyzed_image
from tests.fake_analyzer import FakeAnalyzer
from tests.fake_counting_analyzer import FakeCountingAnalyzer
from tests.fake_mutating_analyzer import FakeMutatingAnalyzer
from tests.fake_scene_analyzer import FakeSceneAnalyzer


def test_select_from_analyzed_excludes_content_filtered_images() -> None:
    """content filter で落ちた画像は選定対象に入らないこと.

    Arrange:
        - 低情報量の暗いフレームと正常なフレームを含む画像群がある
    Act:
        - GameScreenPickerで選定される
    Assert:
        - content filterで除外された画像は選定対象に入らないこと
    """
    # Arrange
    dark = create_analyzed_image(
        path="/tmp/dark.jpg",
        raw_metrics_dict={
            "near_black_ratio": 0.98,
            "luminance_entropy": 0.2,
            "luminance_range": 10.0,
        },
        combined_features=_feature(0),
    )
    play = create_analyzed_image(
        path="/tmp/play.jpg",
        combined_features=_feature(1),
    )
    event = create_analyzed_image(
        path="/tmp/event.jpg",
        combined_features=_feature(100),
    )
    picker = GameScreenPicker(
        analyzer=FakeAnalyzer([dark, play, event]),
        config=SelectionConfig(),
        scene_analyzer=FakeSceneAnalyzer(),
    )

    # Act
    selected, rejected, stats = picker.select_from_analyzed(
        [dark, play, event],
        num=2,
    )

    # Assert
    assert {candidate.path for candidate in selected} == {
        "/tmp/play.jpg",
        "/tmp/event.jpg",
    }
    assert rejected == []
    assert stats.selected_count == 2
    assert stats.rejected_by_content_filter == 1
    assert stats.content_filter_breakdown["blackout"] == 1


def test_select_tracks_total_files_and_analysis_failures(tmp_path: Path) -> None:
    """`select` が入力総数と解析失敗数を統計へ反映すること.

    Arrange:
        - 複数の画像ファイルを含むディレクトリがある
        - 解析に成功する画像と失敗する画像が混在している
    Act:
        - GameScreenPickerで選定される
    Assert:
        - 入力総数と解析成功数・失敗数が統計に反映されること
    """
    # Arrange
    for name in ["frame10.jpg", "frame1.jpg", "frame2.jpg"]:
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")

    analyzed_images = [
        create_analyzed_image(path="/tmp/frame1.jpg", combined_features=_feature(0)),
        None,
        create_analyzed_image(path="/tmp/frame2.jpg", combined_features=_feature(10)),
    ]
    picker = GameScreenPicker(
        analyzer=FakeAnalyzer(analyzed_images),
        config=SelectionConfig(),
        scene_analyzer=FakeSceneAnalyzer(),
    )

    # Act
    selected, rejected, stats = picker.select(
        str(tmp_path),
        num=2,
        recursive=False,
        show_progress=False,
    )

    # Assert
    assert len(selected) == 2
    assert rejected == []
    assert stats.total_files == 3
    assert stats.analyzed_ok == 2
    assert stats.analyzed_fail == 1


def test_select_logs_file_count_and_neutral_cache_summary(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """入力件数と中立解析cache状況が進捗ログに出力されること.

    Arrange:
        - 複数の画像ファイルを含むディレクトリがある
        - 進捗表示が有効である
    Act:
        - GameScreenPickerで選定される
    Assert:
        - 入力画像件数と中立解析cacheのhit/miss件数がログ出力されること
    """
    # Arrange
    for name in ["frame1.jpg", "frame2.jpg"]:
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")
    analyzed_images = [
        create_analyzed_image(path=str(tmp_path / "frame1.jpg")),
        create_analyzed_image(path=str(tmp_path / "frame2.jpg")),
    ]
    picker = GameScreenPicker(
        analyzer=FakeAnalyzer(analyzed_images),
        config=SelectionConfig(),
        scene_analyzer=FakeSceneAnalyzer(),
    )
    caplog.set_level(logging.INFO)

    # Act
    picker.select(str(tmp_path), num=2, recursive=False, show_progress=True)

    # Assert
    assert "入力画像: 2件" in caplog.text
    assert "中立解析cache: hit=0件, miss=2件" in caplog.text
    assert "未cache画像の中立解析を開始します: 2件" in caplog.text


def test_select_logs_neutral_cache_check_progress_during_scan(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """中立解析cache確認中の進捗が定期的に出力されること.

    Arrange:
        - 複数の画像ファイルを含むディレクトリがある
        - cache確認進捗の出力間隔が2件に設定されている
    Act:
        - GameScreenPickerで選定される
    Assert:
        - cache確認ループ中の処理件数がログ出力されること
    """
    # Arrange
    for name in ["frame1.jpg", "frame2.jpg", "frame3.jpg"]:
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")
    monkeypatch.setattr(
        GameScreenPicker,
        "CACHE_CHECK_PROGRESS_INTERVAL",
        2,
        raising=False,
    )
    picker = GameScreenPicker(
        analyzer=FakeCountingAnalyzer(),
        config=SelectionConfig(),
        scene_analyzer=FakeSceneAnalyzer(),
    )
    caplog.set_level(logging.INFO)

    # Act
    picker.select(str(tmp_path), num=2, recursive=False, show_progress=True)

    # Assert
    assert "中立解析cache確認中: 2/3件" in caplog.text


def test_select_reuses_neutral_analysis_cache_on_later_run(tmp_path: Path) -> None:
    """同じ入力画像の中立解析結果が後続実行で再利用されること.

    Arrange:
        - 初回実行で複数の入力画像が解析されている
        - 後続実行ではscene hintだけが変わっている
    Act:
        - 同じ入力フォルダが再び選定される
    Assert:
        - 後続実行では中立解析が呼び出されず、選定結果が返されること
    """
    # Arrange
    for name in ["frame1.jpg", "frame2.jpg"]:
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")

    first_analyzer = FakeCountingAnalyzer()
    first_picker = GameScreenPicker(
        analyzer=first_analyzer,
        config=SelectionConfig(scene_hint="RPG"),
        scene_analyzer=FakeSceneAnalyzer(),
    )
    second_analyzer = FakeCountingAnalyzer()
    second_picker = GameScreenPicker(
        analyzer=second_analyzer,
        config=SelectionConfig(scene_hint="ADV"),
        scene_analyzer=FakeSceneAnalyzer(),
    )

    # Act
    first_picker.select(str(tmp_path), num=2, recursive=False, show_progress=False)
    selected, rejected, stats = second_picker.select(
        str(tmp_path),
        num=2,
        recursive=False,
        show_progress=False,
    )

    # Assert
    assert len(first_analyzer.requested_paths) == 1
    assert second_analyzer.requested_paths == []
    assert len(selected) == 2
    assert rejected == []
    assert stats.analyzed_ok == 2


def test_select_reanalyzes_when_clip_model_changes(tmp_path: Path) -> None:
    """CLIP modelが変わった場合は中立解析cacheが再利用されないこと.

    Arrange:
        - 初回実行でCLIP model Aの中立解析cacheが作成されている
        - 後続実行ではCLIP model Bが使われている
    Act:
        - 同じ入力フォルダが再び選定される
    Assert:
        - 後続実行でも中立解析が呼び出されること
    """
    # Arrange
    for name in ["frame1.jpg", "frame2.jpg"]:
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")

    first_picker = GameScreenPicker(
        analyzer=FakeCountingAnalyzer(model_name="clip-a"),
        config=SelectionConfig(),
        scene_analyzer=FakeSceneAnalyzer(),
    )
    second_analyzer = FakeCountingAnalyzer(model_name="clip-b")
    second_picker = GameScreenPicker(
        analyzer=second_analyzer,
        config=SelectionConfig(),
        scene_analyzer=FakeSceneAnalyzer(),
    )

    # Act
    first_picker.select(str(tmp_path), num=2, recursive=False, show_progress=False)
    second_picker.select(str(tmp_path), num=2, recursive=False, show_progress=False)

    # Assert
    assert len(second_analyzer.requested_paths) == 1


def test_select_reuses_cache_when_runtime_only_analyzer_settings_change(
    tmp_path: Path,
) -> None:
    """実行時リソース設定だけが変わった場合は中立解析cacheが再利用されること.

    Arrange:
        - 初回実行で中立解析cacheが作成されている
        - 後続実行ではメモリ予算と結果構築worker数だけが変わっている
    Act:
        - 同じ入力フォルダが再び選定される
    Assert:
        - 後続実行では中立解析が呼び出されないこと
    """
    # Arrange
    for name in ["frame1.jpg", "frame2.jpg"]:
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")

    first_analyzer = FakeCountingAnalyzer(
        config=AnalyzerConfig(max_memory_gb=1, result_max_workers=1),
    )
    first_picker = GameScreenPicker(
        analyzer=first_analyzer,
        config=SelectionConfig(),
        scene_analyzer=FakeSceneAnalyzer(),
    )
    second_analyzer = FakeCountingAnalyzer(
        config=AnalyzerConfig(max_memory_gb=2, result_max_workers=2),
    )
    second_picker = GameScreenPicker(
        analyzer=second_analyzer,
        config=SelectionConfig(),
        scene_analyzer=FakeSceneAnalyzer(),
    )

    # Act
    first_picker.select(str(tmp_path), num=2, recursive=False, show_progress=False)
    second_picker.select(str(tmp_path), num=2, recursive=False, show_progress=False)

    # Assert
    assert len(first_analyzer.requested_paths) == 1
    assert second_analyzer.requested_paths == []


def test_select_skips_cache_write_when_image_changes_during_analysis(
    tmp_path: Path,
) -> None:
    """解析中に画像が差し替わった場合は中立解析cacheへ保存されないこと.

    Arrange:
        - 解析完了直後に入力画像を書き換えるAnalyzerがある
    Act:
        - 同じ入力フォルダが再び選定される
    Assert:
        - 後続実行では中立解析が呼び出されること
    """
    # Arrange
    for name in ["frame1.jpg", "frame2.jpg"]:
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")

    first_picker = GameScreenPicker(
        analyzer=FakeMutatingAnalyzer(),
        config=SelectionConfig(),
        scene_analyzer=FakeSceneAnalyzer(),
    )
    second_analyzer = FakeCountingAnalyzer()
    second_picker = GameScreenPicker(
        analyzer=second_analyzer,
        config=SelectionConfig(),
        scene_analyzer=FakeSceneAnalyzer(),
    )

    # Act
    first_picker.select(str(tmp_path), num=2, recursive=False, show_progress=False)
    second_picker.select(str(tmp_path), num=2, recursive=False, show_progress=False)

    # Assert
    assert len(second_analyzer.requested_paths) == 1


def test_select_from_analyzed_sorts_remaining_candidates_by_selection_score() -> None:
    """非選択候補は selection_score の降順で返ること.

    Arrange:
        - 複数の候補画像がある
        - 一部は重複に近い画像である
    Act:
        - GameScreenPickerで選定される
    Assert:
        - 非選択候補はselection_scoreの降順で返されること
    """
    # Arrange
    base = _feature(0)
    analyzed_images = [
        create_analyzed_image(path="/tmp/play_0.jpg", combined_features=base),
        create_analyzed_image(
            path="/tmp/play_1.jpg",
            combined_features=_near_duplicate(base, 1),
        ),
        create_analyzed_image(
            path="/tmp/play_2.jpg",
            combined_features=_near_duplicate(base, 2),
        ),
        create_analyzed_image(
            path="/tmp/play_3.jpg",
            combined_features=_feature(100),
        ),
        create_analyzed_image(
            path="/tmp/play_4.jpg",
            combined_features=_feature(200),
        ),
    ]
    picker = GameScreenPicker(
        analyzer=FakeAnalyzer(analyzed_images),
        config=SelectionConfig(),
        scene_analyzer=FakeSceneAnalyzer(),
    )

    # Act
    selected, rejected, stats = picker.select_from_analyzed(analyzed_images, num=3)

    # Assert
    assert len(selected) == 3
    assert stats.selected_count == 3
    assert not (
        {candidate.path for candidate in selected}
        & {candidate.path for candidate in rejected}
    )
    rejected_scores = [candidate.selection_score for candidate in rejected]
    assert rejected_scores == sorted(rejected_scores, reverse=True)


def test_load_image_files_returns_natural_order(tmp_path: Path) -> None:
    """自然順で返されること.

    Arrange:
        - 数字を含むファイル名の画像ファイルがディレクトリにある
        - ファイル名は辞書順ではない順序で格納されている
    Act:
        - load_image_filesが呼び出される
    Assert:
        - ファイルが自然順（file1, file2, file10）で返されること
    """
    # Arrange
    for name in ["file10.jpg", "file1.jpg", "file2.jpg"]:
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")

    # Act
    result = GameScreenPicker.load_image_files(str(tmp_path), recursive=False)

    # Assert
    assert [path.name for path in result] == ["file1.jpg", "file2.jpg", "file10.jpg"]
