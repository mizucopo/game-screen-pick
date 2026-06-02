"""SceneMixSelectorの単体テスト."""

from types import SimpleNamespace

from src.constants.scene_label import SceneLabel
from src.models.scene_mix import SceneMix
from src.models.selection_config import SelectionConfig
from src.services.scene_mix_selector import SceneMixSelector
from tests.conftest import _feature, _near_duplicate, create_scored_candidate


def test_scene_mix_selector_respects_play_event_ratio() -> None:
    """既定の 70/30 比率で選ばれること.

    Arrange:
        - play候補7件、event候補3件がある
        - scene_mix比率が70/30に設定されている
    Act:
        - 10件を選択する
    Assert:
        - playが7件、eventが3件選ばれること
        - targetsとactualsが一致すること
    """
    # Arrange
    selector = SceneMixSelector(
        SelectionConfig(scene_mix=SceneMix(play=0.7, event=0.3))
    )
    candidates = [
        *[
            create_scored_candidate(
                path=f"/tmp/play_{index}.jpg",
                scene_label=SceneLabel.PLAY,
                play_score=0.30 + index * 0.05,
                event_score=0.70 - index * 0.05,
                density_score=0.30 + index * 0.05,
                selection_score=0.30 + index * 0.05,
                combined_features=_feature(index),
            )
            for index in range(7)
        ],
        *[
            create_scored_candidate(
                path=f"/tmp/event_{index}.jpg",
                scene_label=SceneLabel.EVENT,
                play_score=0.10,
                event_score=0.90 - index * 0.05,
                density_score=0.10,
                selection_score=0.90 - index * 0.05,
                combined_features=_feature(100 + index),
            )
            for index in range(3)
        ],
    ]

    # Act
    result = selector.select(candidates, 10)

    # Assert
    assert len(result.selected) == 10
    assert result.target_counts == {"play": 7, "event": 3}
    assert result.actual_counts == {"play": 7, "event": 3}


def test_scene_mix_selector_accepts_scene_mix_candidate_seam() -> None:
    """scene mix選定に必要な候補情報だけで選定されること.

    Arrange:
        - full domain graphを持たないplay候補とevent候補がある
        - scene_mix比率が50/50に設定されている
    Act:
        - 2件を選択する
    Assert:
        - playとeventが1件ずつ選ばれること
        - score_band注釈が返されること
    """
    # Arrange
    selector = SceneMixSelector(
        SelectionConfig(scene_mix=SceneMix(play=0.5, event=0.5))
    )
    candidates = [
        SimpleNamespace(
            path="/tmp/play_seam.jpg",
            scene_label=SceneLabel.PLAY,
            quality_score=0.9,
            selection_score=0.2,
            combined_features=_feature(0),
        ),
        SimpleNamespace(
            path="/tmp/event_seam.jpg",
            scene_label=SceneLabel.EVENT,
            quality_score=0.8,
            selection_score=0.7,
            combined_features=_feature(1),
        ),
    ]

    # Act
    result = selector.select(candidates, 2)

    # Assert
    assert [candidate.path for candidate in result.selected] == [
        "/tmp/play_seam.jpg",
        "/tmp/event_seam.jpg",
    ]
    assert result.target_counts == {"play": 1, "event": 1}
    assert result.actual_counts == {"play": 1, "event": 1}
    assert {
        result.annotation_for(candidate).score_band for candidate in result.selected
    } == {"mid"}


def test_scene_mix_selector_keeps_similar_candidates_out_globally() -> None:
    """カテゴリをまたいでも類似画像を戻さないこと.

    Arrange:
        - play候補と、それに類似するevent候補がある
        - 類似しない別のevent候補がある
        - scene_mix比率が50/50に設定されている
    Act:
        - 2件を選択する
    Assert:
        - play候補と、類似しないevent候補が選ばれること
        - 類似候補が除外されること
    """
    # Arrange
    selector = SceneMixSelector(
        SelectionConfig(scene_mix=SceneMix(play=0.5, event=0.5))
    )
    base = _feature(0)
    candidates = [
        create_scored_candidate(
            path="/tmp/play_a.jpg",
            scene_label=SceneLabel.PLAY,
            selection_score=0.20,
            combined_features=base,
        ),
        create_scored_candidate(
            path="/tmp/event_near.jpg",
            scene_label=SceneLabel.EVENT,
            selection_score=0.20,
            combined_features=_near_duplicate(base, 1),
        ),
        create_scored_candidate(
            path="/tmp/event_far.jpg",
            scene_label=SceneLabel.EVENT,
            selection_score=0.80,
            combined_features=_feature(10),
        ),
    ]

    # Act
    result = selector.select(candidates, 2)

    # Assert
    assert [candidate.path for candidate in result.selected] == [
        "/tmp/play_a.jpg",
        "/tmp/event_far.jpg",
    ]
    assert result.rejected_by_similarity == 1
    assert result.actual_counts == {"play": 1, "event": 1}


def test_scene_mix_selector_returns_score_bands_as_annotations() -> None:
    """選択候補のscore_bandが注釈として返されること.

    Arrange:
        - 異なるselection_scoreを持つ5件のplay候補がある
        - scene_mix比率が100/0に設定されている
    Act:
        - 5件を選択する
    Assert:
        - 各候補の注釈にlow/mid_low/mid/mid_high/highのscore_bandが返されること
    """
    # Arrange
    selector = SceneMixSelector(
        SelectionConfig(scene_mix=SceneMix(play=1.0, event=0.0))
    )
    candidates = [
        create_scored_candidate(
            path=f"/tmp/play_{index}.jpg",
            scene_label=SceneLabel.PLAY,
            selection_score=score,
            play_score=score,
            event_score=1.0 - score,
            density_score=score,
            combined_features=_feature(index),
        )
        for index, score in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])
    ]

    # Act
    result = selector.select(candidates, 5)

    # Assert
    assert {
        result.annotation_for(candidate).score_band for candidate in result.selected
    } == {
        "low",
        "mid_low",
        "mid",
        "mid_high",
        "high",
    }


def test_scene_mix_selector_returns_annotations_without_mutating_candidates() -> None:
    """選定注釈が結果として返され候補が変更されないこと.

    Arrange:
        - 異なるselection_scoreを持つ5件のplay候補がある
        - scene_mix比率が100/0に設定されている
    Act:
        - 5件を選択する
    Assert:
        - 選定結果にscore_band注釈が含まれること
        - 元候補にscore_band属性が追加されないこと
    """
    # Arrange
    selector = SceneMixSelector(
        SelectionConfig(scene_mix=SceneMix(play=1.0, event=0.0))
    )
    candidates = [
        create_scored_candidate(
            path=f"/tmp/play_result_{index}.jpg",
            scene_label=SceneLabel.PLAY,
            selection_score=score,
            play_score=score,
            event_score=1.0 - score,
            density_score=score,
            combined_features=_feature(index),
        )
        for index, score in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])
    ]

    # Act
    result = selector.select(candidates, 5)

    # Assert
    assert [candidate.path for candidate in result.selected] == [
        "/tmp/play_result_0.jpg",
        "/tmp/play_result_1.jpg",
        "/tmp/play_result_2.jpg",
        "/tmp/play_result_3.jpg",
        "/tmp/play_result_4.jpg",
    ]
    assert result.target_counts == {"play": 5, "event": 0}
    assert result.actual_counts == {"play": 5, "event": 0}
    assert result.rejected_by_similarity == 0
    assert {
        result.annotation_for(candidate).score_band for candidate in result.selected
    } == {
        "low",
        "mid_low",
        "mid",
        "mid_high",
        "high",
    }
    assert all(
        result.annotation_for(candidate).score_band is not None
        for candidate in candidates
    )
    assert all(not hasattr(candidate, "score_band") for candidate in candidates)
    assert all(not hasattr(candidate, "outlier_rejected") for candidate in candidates)


def test_scene_mix_selector_fallback_includes_outliers() -> None:
    """外れ値除外された候補が fallback で選ばれること.

    Arrange:
        - 4件のplay候補がある
        - selection_score=[0.1, 0.2, 0.3, 100.0] で100.0が外れ値
        - scene_mix比率が100/0に設定されている
    Act:
        - 4件を選択する
    Assert:
        - 4件全てが選ばれること
        - targetsとactualsが一致すること
        - 外れ値が最後に選ばれること
    """
    # Arrange
    selector = SceneMixSelector(
        SelectionConfig(scene_mix=SceneMix(play=1.0, event=0.0))
    )
    candidates = [
        create_scored_candidate(
            path="/tmp/play_normal_1.jpg",
            scene_label=SceneLabel.PLAY,
            selection_score=0.1,
            combined_features=_feature(0),
        ),
        create_scored_candidate(
            path="/tmp/play_normal_2.jpg",
            scene_label=SceneLabel.PLAY,
            selection_score=0.2,
            combined_features=_feature(1),
        ),
        create_scored_candidate(
            path="/tmp/play_normal_3.jpg",
            scene_label=SceneLabel.PLAY,
            selection_score=0.3,
            combined_features=_feature(2),
        ),
        create_scored_candidate(
            path="/tmp/play_outlier.jpg",
            scene_label=SceneLabel.PLAY,
            selection_score=100.0,
            combined_features=_feature(3),
        ),
    ]

    # Act
    result = selector.select(candidates, 4)

    # Assert
    assert len(result.selected) == 4, "4件全てが選ばれること"
    assert result.target_counts == {"play": 4, "event": 0}
    assert result.actual_counts == {"play": 4, "event": 0}, (
        "targetsとactualsが一致すること"
    )
    assert result.selected[-1].path == "/tmp/play_outlier.jpg", (
        "外れ値が最後に選ばれること"
    )
    assert result.annotation_for(result.selected[-1]).score_band == "outlier", (
        "外れ値のscore_bandが保持されること"
    )
