"""SceneMixSelectorの単体テスト.

scene mix 比率の目標計算と不足時の再配分が、
既定値および設定上書きで期待通りに動くかを検証する。
"""

import numpy as np

from src.constants.selection_profiles import ACTIVE_PROFILE
from src.models.scene_label import SceneLabel
from src.models.scene_mix import SceneMix
from src.models.scored_candidate import ScoredCandidate
from src.models.selection_config import SelectionConfig
from src.services.scene_mix_selector import SceneMixSelector
from tests.conftest import create_scored_candidate


def _make_candidate(index: int, label: SceneLabel) -> ScoredCandidate:
    """指定scene labelを持つ候補を1件作成する.

    Args:
        index: 結合特徴とスコア順を変えるための連番。
        label: 候補に与える scene label 。

    Returns:
        類似度判定用特徴を持つ `ScoredCandidate` 。
    """
    base = np.zeros(576, dtype=np.float32)
    base[index] = 1.0
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
