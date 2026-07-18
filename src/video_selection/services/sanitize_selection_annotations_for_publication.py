"""Video Set全Context Cueに対する公開前Annotation安全化。"""

from dataclasses import replace

from ..models.blog_candidate import BlogCandidate
from ..models.candidate_annotation import (
    privacy_safe_candidate_text,
)
from ..models.scene_catalog import SceneCatalog
from ..models.video_set_selection_result import VideoSetSelectionResult


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
    scene = scene_catalog.for_slug(annotation.scene_slug)
    summary, _ = privacy_safe_candidate_text(
        annotation.summary,
        f"{scene.display_name}に分類される{annotation.blog_image_type}の場面",
        raw_context_texts,
    )
    frame_choice_reason, _ = privacy_safe_candidate_text(
        annotation.frame_choice_reason or annotation.summary,
        f"{scene.description}を視覚的に表すフレーム",
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
