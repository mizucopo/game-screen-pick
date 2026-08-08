"""公開前のCandidate Annotation安全化test。"""

from dataclasses import replace
from pathlib import Path

import pytest

from src.video_selection.models.candidate_annotation import (
    candidate_annotation_free_text_is_safe,
)
from src.video_selection.services import (
    sanitize_selection_annotations_for_publication as sanitize_module,
)
from tests.video_selection.fakes.canonical_publication_factory import (
    build_canonical_publication_request,
)


def test_annotation_matching_an_unseen_video_set_cue_is_sanitized(
    tmp_path: Path,
) -> None:
    """Candidateに未提示のVideo Set Cueと一致する生成文が公開前に安全化されること。

    Arrange:
        - 未提示のContext Cue本文と偶然一致する未採用Annotationが用意される
    Act:
        - Video Set全Cueを境界とする公開前安全化が実行される
    Assert:
        - 公開requestが受理されCue逐語一致が残らないこと
        - 選定結果の件数と判定値が保持されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    cue_text = request.video_stage_results[0].context.cues[0].text
    rejected = request.selection_result.rejected[0]
    unsafe_annotation = replace(rejected.candidate.annotation, summary=cue_text)
    unsafe_candidate = replace(rejected.candidate, annotation=unsafe_annotation)
    unsafe_selection = replace(
        request.selection_result,
        rejected=(replace(rejected, candidate=unsafe_candidate),),
    )
    with pytest.raises(ValueError):
        replace(request, selection_result=unsafe_selection)

    # Act
    sanitized = sanitize_module.sanitize_selection_annotations_for_publication(
        unsafe_selection,
        request.scene_catalog,
        (cue_text,),
    )
    safe_request = replace(request, selection_result=sanitized)

    # Assert
    sanitized_annotation = sanitized.rejected[0].candidate.annotation
    assert safe_request.selection_result is sanitized
    assert candidate_annotation_free_text_is_safe(
        (
            sanitized_annotation.summary,
            sanitized_annotation.frame_choice_reason or "",
            sanitized_annotation.spoiler_evidence,
        ),
        (cue_text,),
    )
    assert sanitized_annotation.summary == "通常プレイ画面"
    assert sanitized.requested_count == unsafe_selection.requested_count
    assert sanitized.annotated_candidate_count == (
        unsafe_selection.annotated_candidate_count
    )
    assert sanitized.rejected[0].reason_code is rejected.reason_code


def test_publication_replaces_internal_event_discriminator(tmp_path: Path) -> None:
    """内部のScene識別子が公開用の有限event説明へ置換されること。

    Arrange:
        - Scene識別子を含む会話eventの選定結果が用意される
    Act:
        - 公開前Annotation安全化が実行される
    Assert:
        - 元の選定値を変えず、公開copyだけが画像由来の汎用説明になること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    selected = request.selection_result.selected[0]
    internal_annotation = replace(
        selected.candidate.annotation,
        summary="キャラクターとの会話の画面内テキストのあるイベント",
        blog_image_type="event",
        screen_text_kind="dialogue",
    )
    internal_candidate = replace(selected.candidate, annotation=internal_annotation)
    internal_selection = replace(
        request.selection_result,
        selected=(replace(selected, candidate=internal_candidate),),
    )

    # Act
    sanitized = sanitize_module.sanitize_selection_annotations_for_publication(
        internal_selection,
        request.scene_catalog,
        (),
    )

    # Assert
    assert internal_selection.selected[0].candidate.annotation.summary.startswith(
        "キャラクターとの会話"
    )
    assert sanitized.selected[0].candidate.annotation.summary == (
        "画面内テキストのあるイベント"
    )


def test_publication_uses_verified_combat_kind_in_summary(tmp_path: Path) -> None:
    """検証済みの通常戦闘種別だけから公開説明が生成されること。

    Arrange:
        - 誤った固有Scene名と検証済みordinary戦闘を持つ選定結果が用意される
    Act:
        - 公開前Annotation安全化が実行される
    Assert:
        - 固有Scene名を出さず通常戦闘の有限説明が返されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    selected = request.selection_result.selected[0]
    internal_annotation = replace(
        selected.candidate.annotation,
        summary="誤った固有ボス名との戦闘",
        combat_encounter_kind="ordinary",
        combat_encounter_basis="ordinary_opponent_presentation",
    )
    internal_candidate = replace(selected.candidate, annotation=internal_annotation)
    internal_selection = replace(
        request.selection_result,
        selected=(replace(selected, candidate=internal_candidate),),
    )

    # Act
    sanitized = sanitize_module.sanitize_selection_annotations_for_publication(
        internal_selection,
        request.scene_catalog,
        (),
    )

    # Assert
    assert sanitized.selected[0].candidate.annotation.summary == (
        "通常戦闘の具体的なプレイ"
    )
