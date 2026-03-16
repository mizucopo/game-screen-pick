"""GameScreenPickerの単体テスト."""

import random
from pathlib import Path

import numpy as np

from src.models.scene_mix import SceneMix
from src.models.selection_config import SelectionConfig
from src.services.game_screen_picker import GameScreenPicker
from tests.conftest import create_analyzed_image
from tests.fake_analyzer import FakeAnalyzer


def _feature(index: int) -> np.ndarray:
    feature = np.zeros(576, dtype=np.float32)
    feature[index] = 1.0
    return feature


def _near_duplicate(base: np.ndarray, index: int) -> np.ndarray:
    feature = base.copy()
    feature[index] = 0.01
    return feature


def test_select_from_analyzed_filters_content_before_play_event_assignment() -> None:
    """content filter の除外が先に適用されること.

    Given:
        - blackout判定される暗い画像がある
        - 正常なplay画像とevent画像がある
        - scene_mix比率が50/50に設定されている
    When:
        - 2件を選択する
    Then:
        - play画像とevent画像が選ばれること
        - blackout画像がcontent_filterで除外されること
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
    analyzed_images = [dark, play, event]
    picker = GameScreenPicker(
        analyzer=FakeAnalyzer(analyzed_images),
        config=SelectionConfig(scene_mix=SceneMix(play=0.5, event=0.5)),
    )

    # Act
    selected, rejected, stats = picker.select_from_analyzed(analyzed_images, num=2)

    # Assert
    assert {candidate.path for candidate in selected} == {
        "/tmp/play.jpg",
        "/tmp/event.jpg",
    }
    assert rejected == []
    assert stats.rejected_by_content_filter == 1
    assert stats.content_filter_breakdown["blackout"] == 1
    assert stats.scene_mix_actual == {"play": 1, "event": 1}


def test_select_from_analyzed_assigns_dense_candidates_to_play() -> None:
    """密度の高いクラスタが play へ寄ること.

    Given:
        - 互いに似た3件の画像（高密度クラスタ）がある
        - 孤立した2件の画像（低密度）がある
        - scene_mix比率が70/30に設定されている
    When:
        - 5件を選択する
    Then:
        - 高密度クラスタがplayに割り当てられること
        - 低密度画像の一部がeventに割り当てられること
    """
    # Arrange
    base = _feature(0)
    analyzed_images = [
        create_analyzed_image(
            path="/tmp/play_0.jpg",
            combined_features=base,
        ),
        create_analyzed_image(
            path="/tmp/play_1.jpg",
            combined_features=_near_duplicate(base, 1),
        ),
        create_analyzed_image(
            path="/tmp/play_2.jpg",
            combined_features=_near_duplicate(base, 2),
        ),
        create_analyzed_image(
            path="/tmp/event_0.jpg",
            combined_features=_feature(100),
        ),
        create_analyzed_image(
            path="/tmp/event_1.jpg",
            combined_features=_feature(200),
        ),
    ]
    picker = GameScreenPicker(
        analyzer=FakeAnalyzer(analyzed_images),
        config=SelectionConfig(scene_mix=SceneMix(play=0.7, event=0.3)),
    )

    # Act
    selected, rejected, stats = picker.select_from_analyzed(analyzed_images, num=5)

    # Assert
    assert len(selected) >= 2
    assert all(
        candidate.scene_assessment.scene_label.value in {"play", "event"}
        for candidate in selected
    )
    assert len(rejected) >= 1
    assert stats.scene_distribution == {"play": 4, "event": 1}
    assert stats.scene_mix_target == {"play": 4, "event": 1}
    assert stats.scene_mix_actual["play"] >= 1
    assert stats.scene_mix_actual["event"] >= 1


def test_select_from_analyzed_spreads_score_bands_and_rejects_duplicates() -> None:
    """band 分散と global 類似度除外が同時に働くこと.

    Given:
        - 異なるスコア帯のplay画像がある
        - play画像と類似するevent画像がある
        - 類似しないevent画像がある
        - scene_mix比率が50/50に設定されている
    When:
        - 4件を選択する
    Then:
        - 類似画像が類似度除外されること
        - 選択候補に複数のscore_bandが含まれること
    """
    # Arrange
    base = _feature(0)
    analyzed_images = [
        create_analyzed_image(path="/tmp/play_low.jpg", combined_features=_feature(10)),
        create_analyzed_image(path="/tmp/play_mid.jpg", combined_features=_feature(11)),
        create_analyzed_image(path="/tmp/play_high.jpg", combined_features=base),
        create_analyzed_image(
            path="/tmp/event_dup.jpg",
            combined_features=_near_duplicate(base, 1),
        ),
        create_analyzed_image(
            path="/tmp/event_far.jpg", combined_features=_feature(100)
        ),
        create_analyzed_image(
            path="/tmp/event_far2.jpg", combined_features=_feature(200)
        ),
    ]
    picker = GameScreenPicker(
        analyzer=FakeAnalyzer(analyzed_images),
        config=SelectionConfig(scene_mix=SceneMix(play=0.5, event=0.5)),
    )

    # Act
    selected, _rejected, stats = picker.select_from_analyzed(analyzed_images, num=4)

    # Assert
    assert stats.rejected_by_similarity >= 1
    assert len({candidate.score_band for candidate in selected}) >= 2
    assert all(candidate.score_band is not None for candidate in selected)


def test_load_image_files_with_same_seed_returns_same_order(
    tmp_path: Path,
) -> None:
    """同じシードで同じ順序が返されること.

    Given:
        - 名前順で並べると異なる順序になる複数の画像ファイルがある
        - 同じシード値で2つの乱数生成器を作成する
    When:
        - 両方の乱数生成器を使ってload_image_filesを実行する
    Then:
        - 両方の結果が同じ順序で返されること
    """
    # Arrange
    for name in ["z.jpg", "a.jpg", "m.jpg"]:
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")

    rng1 = random.Random(42)
    rng2 = random.Random(42)

    # Act
    result1 = GameScreenPicker.load_image_files(
        str(tmp_path), recursive=False, rng=rng1
    )
    result2 = GameScreenPicker.load_image_files(
        str(tmp_path), recursive=False, rng=rng2
    )

    # Assert
    names1 = [p.name for p in result1]
    names2 = [p.name for p in result2]
    assert names1 == names2


def test_load_image_files_with_different_seeds_returns_different_order(
    tmp_path: Path,
) -> None:
    """異なるシードで異なる順序が返されること.

    Given:
        - 名前順で並べると異なる順序になる複数の画像ファイルがある
        - 異なるシード値で2つの乱数生成器を作成する
    When:
        - 両方の乱数生成器を使ってload_image_filesを実行する
    Then:
        - 両方の結果が異なる順序で返されること
    """
    # Arrange
    for name in ["z.jpg", "a.jpg", "m.jpg", "p.jpg", "q.jpg"]:
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")

    rng1 = random.Random(42)
    rng2 = random.Random(123)

    # Act
    result1 = GameScreenPicker.load_image_files(
        str(tmp_path), recursive=False, rng=rng1
    )
    result2 = GameScreenPicker.load_image_files(
        str(tmp_path), recursive=False, rng=rng2
    )

    # Assert
    names1 = [p.name for p in result1]
    names2 = [p.name for p in result2]
    assert names1 != names2


def test_load_image_files_without_rng_returns_natural_order(tmp_path: Path) -> None:
    """rng未指定時に自然順で返されること.

    Given:
        - 辞書順とは異なる自然順を持つ複数の画像ファイルがある
    When:
        - rngを指定せずにload_image_filesを実行する
    Then:
        - 自然順ソートされた結果が返されること
    """
    # Arrange
    for name in ["file10.jpg", "file1.jpg", "file2.jpg"]:
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")

    # Act
    result = GameScreenPicker.load_image_files(str(tmp_path), recursive=False, rng=None)

    # Assert
    names = [p.name for p in result]
    assert names == ["file1.jpg", "file2.jpg", "file10.jpg"]
