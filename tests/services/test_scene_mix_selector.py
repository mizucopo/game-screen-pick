"""SceneMixSelectorの単体テスト."""

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
    selected, _, targets, actuals = selector.select(candidates, 10)

    # Assert
    assert len(selected) == 10
    assert targets == {"play": 7, "event": 3}
    assert actuals == {"play": 7, "event": 3}


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
    selected, rejected, _, actuals = selector.select(candidates, 2)

    # Assert
    assert [candidate.path for candidate in selected] == [
        "/tmp/play_a.jpg",
        "/tmp/event_far.jpg",
    ]
    assert rejected == 1
    assert actuals == {"play": 1, "event": 1}


def test_scene_mix_selector_assigns_score_bands() -> None:
    """選択候補へ score_band が設定されること.

    Arrange:
        - 異なるselection_scoreを持つ5件のplay候補がある
        - scene_mix比率が100/0に設定されている
    Act:
        - 5件を選択する
    Assert:
        - 各候補にlow/mid_low/mid/mid_high/highのscore_bandが設定されること
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
    selected, _, _, _ = selector.select(candidates, 5)

    # Assert
    assert {candidate.score_band for candidate in selected} == {
        "low",
        "mid_low",
        "mid",
        "mid_high",
        "high",
    }


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
    selected, _, targets, actuals = selector.select(candidates, 4)

    # Assert
    assert len(selected) == 4, "4件全てが選ばれること"
    assert targets == {"play": 4, "event": 0}
    assert actuals == {"play": 4, "event": 0}, "targetsとactualsが一致すること"
    assert selected[-1].path == "/tmp/play_outlier.jpg", "外れ値が最後に選ばれること"
    assert selected[-1].score_band == "outlier", "外れ値のscore_bandが保持されること"
