"""Candidate AnnotationからBlog Candidateを構築するtest。"""

from pathlib import Path

from src.video_selection.models.scene_catalog import SceneCatalog
from src.video_selection.models.scene_catalog_entry import SceneCatalogEntry
from src.video_selection.services.build_blog_candidates import build_blog_candidates
from src.video_selection.services.build_candidate_annotation_requests import (
    build_candidate_annotation_requests,
)
from tests.video_selection.fakes.canonical_publication_factory import (
    build_canonical_publication_request,
)


def test_annotations_are_mapped_to_source_role_progress_and_global_rank(
    tmp_path: Path,
) -> None:
    """Annotationがsource、Scene role、進行率、global rankへ対応付けられること。

    Arrange:
        - 2件のMoment request、Annotation、Scene Catalogが用意される
    Act:
        - 2番目のbatchとしてBlog Candidateへ変換される
    Assert:
        - source orderとrequest進行率が保持されること
        - Scene Catalogのselection roleが解決されること
        - shortlist rankがbatchをまたぐglobal値になること
    """
    # Arrange
    publication = build_canonical_publication_request(tmp_path)
    requests = build_candidate_annotation_requests(
        publication.video_stage_results,
        selection_intent="ブログ本文を説明できる画像を選ぶ",
    )
    annotations_by_moment = {
        item.candidate.annotation.candidate_moment_id: item.candidate.annotation
        for item in publication.selection_result.selected
    }
    annotations_by_moment.update(
        {
            item.candidate.annotation.candidate_moment_id: item.candidate.annotation
            for item in publication.selection_result.rejected
        }
    )
    annotations = tuple(
        annotations_by_moment[request.moment.identifier] for request in requests
    )
    catalog = SceneCatalog(
        (
            SceneCatalogEntry(
                "test-scene",
                "テスト場面",
                "テスト用場面",
                "exploration",
                "recurring_gameplay",
            ),
            SceneCatalogEntry(
                "event", "イベント", "イベント場面", "event", "cinematic"
            ),
            SceneCatalogEntry("other", "その他", "分類不能", "other", "ordinary"),
        )
    )

    # Act
    candidates = build_blog_candidates(
        requests,
        annotations,
        catalog,
        publication.video_stage_results,
        shortlist_rank_offset=24,
    )

    # Assert
    assert [candidate.video_order for candidate in candidates] == [0, 0]
    assert [candidate.video_set_progress for candidate in candidates] == [
        request.video_set_progress for request in requests
    ]
    assert [candidate.scene_selection_role for candidate in candidates] == [
        "recurring_gameplay",
        "recurring_gameplay",
    ]
    assert [candidate.shortlist_rank for candidate in candidates] == [24, 25]


def test_annotation_must_match_request_moment(tmp_path: Path) -> None:
    """別MomentのAnnotationが対応requestへ混入すると拒否されること。

    Arrange:
        - 順序を逆転した2件のAnnotationが用意される
    Act:
        - Blog Candidateへの変換が試行される
    Assert:
        - requestとAnnotationの不一致がValueErrorになること
    """
    # Arrange
    publication = build_canonical_publication_request(tmp_path)
    requests = build_candidate_annotation_requests(
        publication.video_stage_results,
        selection_intent="ブログ本文を説明できる画像を選ぶ",
    )
    annotations = (
        *(item.candidate.annotation for item in publication.selection_result.selected),
        *(item.candidate.annotation for item in publication.selection_result.rejected),
    )[::-1]
    scene_catalog = publication.scene_catalog
    assert scene_catalog is not None

    # Act
    # Assert
    try:
        build_blog_candidates(
            requests,
            annotations,
            scene_catalog,
            publication.video_stage_results,
        )
    except ValueError as error:
        assert str(error) == "AnnotationとCandidate Momentが一致しません"
    else:  # pragma: no cover - assertion failureを読みやすくする
        raise AssertionError("ValueErrorが送出されませんでした")
