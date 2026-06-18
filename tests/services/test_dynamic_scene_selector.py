"""DynamicSceneSelectorの単体テスト."""

from typing import Any

import numpy as np

from src.models.scene_assessment import SceneAssessment
from src.models.scene_selection_role import SceneSelectionRole
from src.models.scored_candidate import ScoredCandidate
from src.services.dynamic_scene_selector import DynamicSceneSelector
from tests.conftest import _feature, _near_duplicate, create_analyzed_image


def test_select_prefers_one_candidate_per_variant_group() -> None:
    """必要枚数が埋まる場合は各variant groupから1枚ずつ選ばれること.

    Arrange:
        - 同一sceneに近い差分画像2枚と別groupの画像がある
        - 別sceneにも候補がある
    Act:
        - 動的scene選定が実行される
    Assert:
        - 近い差分画像の2枚目より別groupの画像が優先されること
    """
    # Arrange
    base = _feature(1)
    duplicate = _near_duplicate(base, 2)
    candidates = [
        build_dynamic_candidate("/tmp/talk_a.jpg", "conversation", base, 0.9),
        build_dynamic_candidate("/tmp/talk_b.jpg", "conversation", duplicate, 0.8),
        build_dynamic_candidate("/tmp/talk_c.jpg", "conversation", _feature(40), 0.7),
        build_dynamic_candidate("/tmp/menu.jpg", "menu", _feature(80), 0.6),
    ]
    selector = DynamicSceneSelector(
        similarity_threshold=0.98,
        threshold_steps=[0.98],
        variant_similarity_threshold=0.95,
    )

    # Act
    result = selector.select(candidates, num=3)

    # Assert
    assert [candidate.path for candidate in result.selected] == [
        "/tmp/talk_a.jpg",
        "/tmp/menu.jpg",
        "/tmp/talk_c.jpg",
    ]
    assert result.annotations_by_path["/tmp/talk_a.jpg"].variant_group == (
        "conversation_001"
    )
    assert result.annotations_by_path["/tmp/talk_b.jpg"].variant_group == (
        "conversation_001"
    )


def test_select_uses_second_variant_when_slots_cannot_be_filled_otherwise() -> None:
    """必要枚数が埋まらない場合は同じvariant groupの2枚目が使われること.

    Arrange:
        - 同じvariant groupに属する候補だけがある
    Act:
        - 候補数と同じ枚数が要求される
    Assert:
        - 2枚目のvariantも選択されること
    """
    # Arrange
    base = _feature(1)
    duplicate = _near_duplicate(base, 2)
    candidates = [
        build_dynamic_candidate("/tmp/talk_a.jpg", "conversation", base, 0.9),
        build_dynamic_candidate("/tmp/talk_b.jpg", "conversation", duplicate, 0.8),
    ]
    selector = DynamicSceneSelector(
        similarity_threshold=1.0,
        threshold_steps=[1.0],
        variant_similarity_threshold=0.95,
    )

    # Act
    result = selector.select(candidates, num=2)

    # Assert
    assert [candidate.path for candidate in result.selected] == [
        "/tmp/talk_a.jpg",
        "/tmp/talk_b.jpg",
    ]


def test_select_allocates_scarce_scene_slots_by_best_score() -> None:
    """scene数が要求枚数を超える場合は高score sceneに枠が割り当てられること.

    Arrange:
        - 3つのsceneに候補があり、要求枚数は2枚である
        - 入力順の最後のsceneが最も高いscoreを持つ
    Act:
        - 動的scene選定が実行される
    Assert:
        - 入力順ではなくscore上位のsceneから選ばれること
    """
    # Arrange
    candidates = [
        build_dynamic_candidate("/tmp/battle.jpg", "battle", _feature(1), 0.2),
        build_dynamic_candidate("/tmp/menu.jpg", "menu", _feature(40), 0.6),
        build_dynamic_candidate("/tmp/climax.jpg", "climax", _feature(80), 0.9),
    ]
    selector = DynamicSceneSelector(
        similarity_threshold=1.0,
        threshold_steps=[1.0],
        variant_similarity_threshold=0.95,
    )

    # Act
    result = selector.select(candidates, num=2)

    # Assert
    assert {candidate.path for candidate in result.selected} == {
        "/tmp/menu.jpg",
        "/tmp/climax.jpg",
    }
    assert result.target_counts == {
        "battle": 0,
        "menu": 1,
        "climax": 1,
    }


def test_select_caps_cinematic_targets_and_redistributes_slots() -> None:
    """cinematic sceneの目標枚数が合計soft cap内に抑えられること.

    Arrange:
        - cinematic、ordinary、recurring gameplayのsceneに十分な候補がある
        - 要求枚数が10枚である
    Act:
        - 動的scene選定が実行される
    Assert:
        - cinematic sceneの目標枚数が1枚に抑えられること
        - 空いた枠がnon-cinematic sceneへ再配分されること
    """
    # Arrange
    candidates = [
        build_dynamic_candidate(
            f"/tmp/event_{index}.jpg",
            "event",
            _feature(index),
            0.9,
            SceneSelectionRole.CINEMATIC,
        )
        for index in range(10)
    ]
    candidates.extend(
        build_dynamic_candidate(
            f"/tmp/battle_{index}.jpg",
            "battle",
            _feature(20 + index),
            0.8,
            SceneSelectionRole.ORDINARY,
        )
        for index in range(10)
    )
    candidates.extend(
        build_dynamic_candidate(
            f"/tmp/puzzle_{index}.jpg",
            "puzzle",
            _feature(40 + index),
            0.7,
            SceneSelectionRole.RECURRING_GAMEPLAY,
        )
        for index in range(10)
    )
    selector = DynamicSceneSelector(
        similarity_threshold=1.0,
        threshold_steps=[1.0],
        variant_similarity_threshold=0.95,
    )

    # Act
    result = selector.select(candidates, num=10)

    # Assert
    assert result.target_counts == {
        "event": 1,
        "battle": 5,
        "puzzle": 4,
    }
    assert result.actual_counts["event"] == 1


def test_select_relaxes_similarity_for_recurring_gameplay_variants() -> None:
    """recurring gameplayでは類似する状態差画像も選ばれること.

    Arrange:
        - recurring gameplay sceneに類似度0.9程度の候補が2枚ある
        - 通常の類似度しきい値は0.72である
    Act:
        - 2枚の選定が要求される
    Assert:
        - 類似する2枚がどちらも選ばれること
    """
    # Arrange
    first_feature = np.array([np.sqrt(0.9), np.sqrt(0.1), 0.0], dtype=np.float32)
    second_feature = np.array([np.sqrt(0.9), 0.0, np.sqrt(0.1)], dtype=np.float32)
    candidates = [
        build_dynamic_candidate(
            "/tmp/battle_state_a.jpg",
            "battle",
            first_feature,
            0.9,
            SceneSelectionRole.RECURRING_GAMEPLAY,
        ),
        build_dynamic_candidate(
            "/tmp/battle_state_b.jpg",
            "battle",
            second_feature,
            0.8,
            SceneSelectionRole.RECURRING_GAMEPLAY,
        ),
    ]
    selector = DynamicSceneSelector(
        similarity_threshold=0.72,
        threshold_steps=[0.72],
        variant_similarity_threshold=0.85,
    )

    # Act
    result = selector.select(candidates, num=2)

    # Assert
    assert [candidate.path for candidate in result.selected] == [
        "/tmp/battle_state_a.jpg",
        "/tmp/battle_state_b.jpg",
    ]


def build_dynamic_candidate(
    path: str,
    scene_slug: str,
    combined_features: np.ndarray[Any, Any],
    selection_score: float = 0.8,
    scene_selection_role: SceneSelectionRole = SceneSelectionRole.ORDINARY,
) -> ScoredCandidate:
    """動的scene候補を作る."""
    analyzed = create_analyzed_image(path=path, combined_features=combined_features)
    return ScoredCandidate(
        analyzed_image=analyzed,
        scene_assessment=SceneAssessment(
            scene_slug=scene_slug,
            scene_display_name=scene_slug,
            scene_description=scene_slug,
            scene_selection_role=scene_selection_role,
            scene_confidence=selection_score,
        ),
        quality_score=selection_score,
        selection_score=selection_score,
    )
