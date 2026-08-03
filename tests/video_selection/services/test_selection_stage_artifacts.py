"""Video Set Selection cache artifact変換のtest。"""

from dataclasses import replace
from fractions import Fraction

import pytest

from src.video_selection.services.select_video_set_images import (
    select_video_set_images,
)
from src.video_selection.services.selection_stage_artifacts import (
    restore_video_set_selection_result,
    serialize_video_set_selection_result,
)
from tests.video_selection.fakes.selection_model_factory import (
    build_blog_candidate,
)


def test_selection_result_round_trips_without_rerunning_selector() -> None:
    """全選定判断が現在のcandidateへ結び直して復元されること。

    Arrange:
        - selectedとrejectedを持つ実selector結果が用意される
    Act:
        - selection resultがJSON artifact化され現在candidateから復元される
    Assert:
        - score、reason、coverageを含む完全な結果が一致すること
    """
    # Arrange
    first = build_blog_candidate("a")
    second = replace(
        build_blog_candidate("b"),
        video_set_progress=Fraction(2, 10),
        shortlist_rank=2,
    )
    selection = select_video_set_images(
        (first, second),
        requested_count=1,
        spoiler_sensitivity="medium",
        similarity_threshold=0.85,
    )

    # Act
    restored = restore_video_set_selection_result(
        serialize_video_set_selection_result(selection),
        (first, second),
    )

    # Assert
    assert restored == selection


def test_selection_artifact_rejects_changed_candidate_set() -> None:
    """artifact候補と現在候補の不一致がcache reuseとして受理されないこと。

    Arrange:
        - 2候補から確定したselection artifactが用意される
    Act:
        - 1候補だけを渡してartifact復元が試行される
    Assert:
        - candidate集合不一致として拒否されること
    """
    # Arrange
    first = build_blog_candidate("a")
    second = replace(build_blog_candidate("b"), shortlist_rank=2)
    selection = select_video_set_images(
        (first, second),
        requested_count=1,
        spoiler_sensitivity="medium",
        similarity_threshold=0.85,
    )
    artifact = serialize_video_set_selection_result(selection)

    # Act
    # Assert
    with pytest.raises(ValueError, match="candidate集合"):
        restore_video_set_selection_result(artifact, (first,))


def test_semantic_group_decision_round_trips_with_privacy_safe_basis() -> None:
    """Semantic Duplicate GroupのIDと根拠がcache artifactで復元されること。

    Arrange:
        - 同じ動画とScene Slugに属する主要戦闘候補2件が用意される
    Act:
        - semantic duplicateを含む選定結果がartifact化され復元される
    Assert:
        - schemaが更新されGroup ID、根拠、blocking IDを含む結果が一致すること
    """
    # Arrange
    first = build_blog_candidate("a")
    first = replace(
        first,
        annotation=replace(
            first.annotation,
            combat_encounter_kind="major",
            combat_encounter_basis="major_opponent_presentation",
        ),
    )
    second = build_blog_candidate("b")
    second_frame = replace(
        second.annotation.candidate,
        video_fingerprint=first.annotation.candidate.video_fingerprint,
        source_pts=200,
        video_time=Fraction(2, 10),
    )
    second = replace(
        second,
        annotation=replace(
            second.annotation,
            candidate=second_frame,
            combat_encounter_kind="major",
            combat_encounter_basis="major_opponent_presentation",
        ),
        video_set_progress=Fraction(2, 10),
        shortlist_rank=2,
    )
    selection = select_video_set_images(
        (first, second),
        requested_count=2,
        spoiler_sensitivity="medium",
        similarity_threshold=0.85,
    )

    # Act
    artifact = serialize_video_set_selection_result(selection)
    restored = restore_video_set_selection_result(artifact, (first, second))

    # Assert
    assert artifact["schema"] == "game-screen-pick/video-set-selection@2.0.0"
    selected = artifact["selected"]
    rejected = artifact["rejected"]
    assert isinstance(selected, list)
    assert isinstance(rejected, list)
    assert selected[0]["semantic_group_basis"] == "combat_encounter_sequence"
    assert rejected[0]["semantic_group_basis"] == "combat_encounter_sequence"
    assert rejected[0]["blocked_by_image_id"] == first.identifier
    assert restored == selection
