"""Video Set全Context Cueに対する公開前Annotation安全化。"""

from dataclasses import replace

from ..models.blog_candidate import BlogCandidate
from ..models.candidate_annotation import (
    BlogImageType,
    CandidateAnnotation,
    privacy_safe_candidate_text,
)
from ..models.candidate_frame_observation import CandidateFrameContentKind
from ..models.scene_catalog import SceneCatalog
from ..models.video_set_selection_result import VideoSetSelectionResult

_PUBLICATION_CONTENT_SUMMARIES: dict[CandidateFrameContentKind, str] = {
    "gameplay_action": "具体的なプレイ",
    "gameplay_idle": "通常プレイ画面",
    "event_dialogue": "画面内テキストのあるイベント",
    "event_action": "動きのあるイベント",
    "event_setup": "イベント場面",
    "document": "文書画面",
    "shop": "ショップ画面",
    "map": "マップ画面",
    "save": "セーブ画面",
    "tutorial_help": "チュートリアル画面",
    "other_interface": "操作画面",
    "title": "タイトル画面",
    "other": "その他の場面",
}
_PUBLICATION_BLOG_TYPE_SUMMARIES: dict[BlogImageType, str] = {
    "normal_gameplay": "通常プレイ画面",
    "event": "イベント場面",
    "menu": "操作画面",
    "title": "タイトル画面",
    "other": "その他の場面",
}


def sanitize_selection_annotations_for_publication(
    selection: VideoSetSelectionResult,
    scene_catalog: SceneCatalog | None,
    raw_context_texts: tuple[str, ...],
) -> VideoSetSelectionResult:
    """選定値を保ち、公開自由文だけをVideo Set全Cueから安全化する。"""
    if not selection.selected and not selection.rejected:
        return selection
    if scene_catalog is None:
        raise ValueError("Annotationがある選定にはScene Catalogが必要です")
    return replace(
        selection,
        selected=tuple(
            replace(
                item,
                candidate=_sanitize_candidate(
                    item.candidate,
                    scene_catalog,
                    raw_context_texts,
                ),
            )
            for item in selection.selected
        ),
        rejected=tuple(
            replace(
                item,
                candidate=_sanitize_candidate(
                    item.candidate,
                    scene_catalog,
                    raw_context_texts,
                ),
            )
            for item in selection.rejected
        ),
    )


def _sanitize_candidate(
    candidate: BlogCandidate,
    scene_catalog: SceneCatalog,
    raw_context_texts: tuple[str, ...],
) -> BlogCandidate:
    annotation = candidate.annotation
    scene_catalog.for_slug(annotation.scene_slug)
    summary, _ = privacy_safe_candidate_text(
        _publication_annotation_summary(annotation),
        "画像内容を示す場面",
        raw_context_texts,
    )
    frame_choice_reason, _ = privacy_safe_candidate_text(
        annotation.frame_choice_reason or annotation.summary,
        "画像内容が候補内で最も明瞭なフレーム",
        raw_context_texts,
    )
    spoiler_evidence, _ = privacy_safe_candidate_text(
        annotation.spoiler_evidence,
        (
            ""
            if annotation.spoiler_risk == "none"
            else f"{annotation.spoiler_risk}相当の進行情報を映像から判定"
        ),
        raw_context_texts,
    )
    sanitized = replace(
        annotation,
        summary=summary,
        frame_choice_reason=frame_choice_reason,
        spoiler_evidence=spoiler_evidence,
    )
    return replace(candidate, annotation=sanitized)


def _publication_annotation_summary(annotation: CandidateAnnotation) -> str:
    """内部の意味識別子を使わず検証済み観測だけから公開説明を返す。"""
    evidence = annotation.representative_frame_evidence
    has_gameplay_semantics = (
        annotation.blog_image_type == "normal_gameplay"
        if evidence is None
        else evidence.content_kind in {"gameplay_action", "gameplay_idle"}
    )
    if has_gameplay_semantics:
        if annotation.combat_encounter_kind == "ordinary":
            return "通常戦闘の具体的なプレイ"
        if annotation.combat_encounter_kind == "major":
            return "主要戦闘の具体的なプレイ"
        if annotation.combat_encounter_kind == "uncertain":
            return "戦闘の具体的なプレイ"
    if evidence is not None:
        return _PUBLICATION_CONTENT_SUMMARIES[evidence.content_kind]
    if annotation.has_title_semantics:
        return "タイトル画面"
    if (
        annotation.blog_image_type == "event"
        and annotation.screen_text_kind == "dialogue"
    ):
        return "画面内テキストのあるイベント"
    return _PUBLICATION_BLOG_TYPE_SUMMARIES[annotation.blog_image_type]
