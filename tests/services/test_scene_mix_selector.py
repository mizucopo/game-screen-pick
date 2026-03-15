"""SceneMixSelectorの単体テスト.

scene mix 比率の目標計算と不足時の再配分が、
既定値および設定上書きで期待通りに動くかを検証する。
"""

import numpy as np

from src.constants.scene_label import SceneLabel
from src.constants.selection_profiles import ACTIVE_PROFILE
from src.models.scene_mix import SceneMix
from src.models.scored_candidate import ScoredCandidate
from src.models.selection_config import SelectionConfig
from src.services.scene_mix_selector import SceneMixSelector
from tests.conftest import create_scored_candidate


def _make_feature(index: int) -> np.ndarray:
    """one-hot特徴ベクトルを作る."""
    feature = np.zeros(576, dtype=np.float32)
    feature[index] = 1.0
    return feature


def _make_near_duplicate(
    base: np.ndarray,
    index: int,
    delta: float = 0.01,
) -> np.ndarray:
    """base に非常に近い特徴ベクトルを作る."""
    feature = base.copy()
    feature[index] = delta
    return feature


def _make_candidate(index: int, label: SceneLabel) -> ScoredCandidate:
    """指定scene labelを持つ候補を1件作成する.

    Args:
        index: 結合特徴とスコア順を変えるための連番。
        label: 候補に与える scene label 。

    Returns:
        類似度判定用特徴を持つ `ScoredCandidate` 。
    """
    base = _make_feature(index)
    gameplay_score = 0.8 if label == SceneLabel.GAMEPLAY else 0.2
    event_score = 0.8 if label == SceneLabel.EVENT else 0.2
    other_score = 0.8 if label == SceneLabel.OTHER else 0.2
    return create_scored_candidate(
        path=f"/tmp/{label.value}_{index}.jpg",
        scene_label=label,
        gameplay_score=gameplay_score,
        event_score=event_score,
        other_score=other_score,
        selection_score=100.0 - index,
        activity_score=0.2 + (index % 3) * 0.2,
        combined_features=base,
    )


def test_scene_mix_selector_respects_default_ratio() -> None:
    """候補が十分あるとき、既定の50/40/10に沿うこと.

    Given:
        - gameplay / event / other の候補が十分にある
    When:
        - 既定設定で10件を選択する
    Then:
        - 目標値と実績が 50 / 40 / 10 に一致すること
    """
    # Arrange
    config = SelectionConfig()
    selector = SceneMixSelector(config)
    candidates = [
        *[_make_candidate(i, SceneLabel.GAMEPLAY) for i in range(10)],
        *[_make_candidate(100 + i, SceneLabel.EVENT) for i in range(10)],
        *[_make_candidate(200 + i, SceneLabel.OTHER) for i in range(10)],
    ]

    # Act
    selected, _, targets, actuals = selector.select(candidates, 10, ACTIVE_PROFILE)

    # Assert
    assert len(selected) == 10
    assert targets == {"gameplay": 5, "event": 4, "other": 1}
    assert actuals == {"gameplay": 5, "event": 4, "other": 1}


def test_scene_mix_selector_accepts_override_ratio() -> None:
    """設定でscene mix比率を変更できること.

    Given:
        - 設定で 60 / 30 / 10 の比率を指定している
        - 各sceneの候補が十分にある
    When:
        - 10件を選択する
    Then:
        - 目標値と実績が設定比率に従うこと
    """
    # Arrange
    config = SelectionConfig(scene_mix=SceneMix(gameplay=0.6, event=0.3, other=0.1))
    selector = SceneMixSelector(config)
    candidates = [
        *[_make_candidate(i, SceneLabel.GAMEPLAY) for i in range(10)],
        *[_make_candidate(100 + i, SceneLabel.EVENT) for i in range(10)],
        *[_make_candidate(200 + i, SceneLabel.OTHER) for i in range(10)],
    ]

    # Act
    _, _, targets, actuals = selector.select(candidates, 10, ACTIVE_PROFILE)

    # Assert
    assert targets == {"gameplay": 6, "event": 3, "other": 1}
    assert actuals == {"gameplay": 6, "event": 3, "other": 1}


def test_scene_mix_selector_redistributes_when_other_is_missing() -> None:
    """otherが不足した場合はgameplay/eventに再配分されること.

    Given:
        - gameplay と event の候補だけが十分にある
        - other 候補は存在しない
    When:
        - 既定設定で10件を選択する
    Then:
        - other の不足分が gameplay / event へ再配分されること
    """
    # Arrange
    config = SelectionConfig()
    selector = SceneMixSelector(config)
    candidates = [
        *[_make_candidate(i, SceneLabel.GAMEPLAY) for i in range(10)],
        *[_make_candidate(100 + i, SceneLabel.EVENT) for i in range(10)],
    ]

    # Act
    _, _, targets, actuals = selector.select(candidates, 10, ACTIVE_PROFILE)

    # Assert
    assert targets == {"gameplay": 5, "event": 4, "other": 1}
    assert actuals["other"] == 0
    assert actuals["gameplay"] + actuals["event"] == 10


def test_scene_mix_selector_keeps_similar_candidates_out() -> None:
    """scene target を満たすために類似画像を戻さないこと.

    Given:
        - gameplay 100%の設定で選択が行われる
        - 類似した特徴を持つ候補と異なる特徴を持つ候補が混在している
    When:
        - 3件の選択が実行される
    Then:
        - 類似画像は除外され、多様な候補のみが選択されること
    """
    # Arrange
    config = SelectionConfig(scene_mix=SceneMix(gameplay=1.0, event=0.0, other=0.0))
    selector = SceneMixSelector(config)
    base = _make_feature(0)
    candidates = [
        create_scored_candidate(
            path="/tmp/gameplay_a.jpg",
            scene_label=SceneLabel.GAMEPLAY,
            selection_score=100.0,
            combined_features=base,
        ),
        create_scored_candidate(
            path="/tmp/gameplay_b.jpg",
            scene_label=SceneLabel.GAMEPLAY,
            selection_score=99.0,
            combined_features=_make_near_duplicate(base, 1),
        ),
        create_scored_candidate(
            path="/tmp/gameplay_c.jpg",
            scene_label=SceneLabel.GAMEPLAY,
            selection_score=80.0,
            combined_features=_make_feature(10),
        ),
    ]

    # Act
    selected, rejected, _, actuals = selector.select(candidates, 3, ACTIVE_PROFILE)

    # Assert
    assert [candidate.path for candidate in selected] == [
        "/tmp/gameplay_a.jpg",
        "/tmp/gameplay_c.jpg",
    ]
    assert rejected == 1
    assert actuals == {"gameplay": 2, "event": 0, "other": 0}


def test_scene_mix_selector_rejects_similar_candidate_across_scenes() -> None:
    """別sceneでも既選択画像に近い候補は除外されること.

    Given:
        - gameplay/event 50%ずつの設定で選択が行われる
        - event候補の1件が既選択gameplay画像に非常に近い特徴を持つ
    When:
        - 2件の選択が実行される
    Then:
        - gameplayに近いevent候補は除外され、遠い候補が選択されること
    """
    # Arrange
    config = SelectionConfig(scene_mix=SceneMix(gameplay=0.5, event=0.5, other=0.0))
    selector = SceneMixSelector(config)
    gameplay_feature = _make_feature(0)
    candidates = [
        create_scored_candidate(
            path="/tmp/gameplay_a.jpg",
            scene_label=SceneLabel.GAMEPLAY,
            selection_score=100.0,
            combined_features=gameplay_feature,
        ),
        create_scored_candidate(
            path="/tmp/event_near.jpg",
            scene_label=SceneLabel.EVENT,
            selection_score=99.0,
            combined_features=_make_near_duplicate(gameplay_feature, 1),
        ),
        create_scored_candidate(
            path="/tmp/event_far.jpg",
            scene_label=SceneLabel.EVENT,
            selection_score=80.0,
            combined_features=_make_feature(10),
        ),
    ]

    # Act
    selected, rejected, _, actuals = selector.select(candidates, 2, ACTIVE_PROFILE)

    # Assert
    assert [candidate.path for candidate in selected] == [
        "/tmp/gameplay_a.jpg",
        "/tmp/event_far.jpg",
    ]
    assert rejected == 1
    assert actuals == {"gameplay": 1, "event": 1, "other": 0}


def test_scene_mix_selector_redistributes_shortage_to_other_scene() -> None:
    """不足したscene分を他sceneの多様な候補で埋めること.

    Given:
        - gameplay/event 50%ずつの設定で選択が行われる
        - gameplay候補の一部が類似しており除外される
        - event候補は十分に多様である
    When:
        - 4件の選択が実行される
    Then:
        - gameplayの不足分がeventの多様な候補で補完されること
    """
    # Arrange
    config = SelectionConfig(scene_mix=SceneMix(gameplay=0.5, event=0.5, other=0.0))
    selector = SceneMixSelector(config)
    gameplay_feature = _make_feature(0)
    candidates = [
        create_scored_candidate(
            path="/tmp/gameplay_a.jpg",
            scene_label=SceneLabel.GAMEPLAY,
            selection_score=100.0,
            combined_features=gameplay_feature,
        ),
        create_scored_candidate(
            path="/tmp/gameplay_b.jpg",
            scene_label=SceneLabel.GAMEPLAY,
            selection_score=99.0,
            combined_features=_make_near_duplicate(gameplay_feature, 1),
        ),
        create_scored_candidate(
            path="/tmp/event_a.jpg",
            scene_label=SceneLabel.EVENT,
            selection_score=98.0,
            combined_features=_make_feature(10),
        ),
        create_scored_candidate(
            path="/tmp/event_b.jpg",
            scene_label=SceneLabel.EVENT,
            selection_score=97.0,
            combined_features=_make_feature(20),
        ),
        create_scored_candidate(
            path="/tmp/event_c.jpg",
            scene_label=SceneLabel.EVENT,
            selection_score=96.0,
            combined_features=_make_feature(30),
        ),
    ]

    # Act
    selected, rejected, _, actuals = selector.select(candidates, 4, ACTIVE_PROFILE)

    # Assert
    assert [candidate.path for candidate in selected] == [
        "/tmp/gameplay_a.jpg",
        "/tmp/event_a.jpg",
        "/tmp/event_b.jpg",
        "/tmp/event_c.jpg",
    ]
    assert rejected == 1
    assert actuals == {"gameplay": 1, "event": 3, "other": 0}


def test_scene_mix_selector_returns_fewer_when_all_candidates_are_homogeneous() -> None:
    """入力全体が同質なら要求枚数未満で返すこと.

    Given:
        - gameplay 100%の設定で選択が行われる
        - すべての候補が同一の特徴を持つ
    When:
        - 3件の選択が要求される
    Then:
        - 類似除外により1件のみ選択され、要求枚数未満で返されること
    """
    # Arrange
    config = SelectionConfig(scene_mix=SceneMix(gameplay=1.0, event=0.0, other=0.0))
    selector = SceneMixSelector(config)
    feature = _make_feature(0)
    candidates = [
        create_scored_candidate(
            path=f"/tmp/gameplay_{index}.jpg",
            scene_label=SceneLabel.GAMEPLAY,
            selection_score=100.0 - index,
            combined_features=feature.copy(),
        )
        for index in range(5)
    ]

    # Act
    selected, rejected, _, actuals = selector.select(candidates, 3, ACTIVE_PROFILE)

    # Assert
    assert len(selected) == 1
    assert selected[0].path == "/tmp/gameplay_0.jpg"
    assert rejected == 4
    assert actuals == {"gameplay": 1, "event": 0, "other": 0}
