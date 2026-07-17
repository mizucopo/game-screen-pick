"""Candidate Annotationを決定的selectorのBlog Candidateへ変換する。"""

from ..models.blog_candidate import BlogCandidate
from ..models.candidate_annotation import CandidateAnnotation
from ..models.candidate_annotation_request import CandidateAnnotationRequest
from ..models.scene_catalog import SceneCatalog
from ..models.video_stage_result import VideoStageResult


def build_blog_candidates(
    requests: tuple[CandidateAnnotationRequest, ...],
    annotations: tuple[CandidateAnnotation, ...],
    scene_catalog: SceneCatalog,
    video_stage_results: tuple[VideoStageResult, ...],
    *,
    shortlist_rank_offset: int = 0,
) -> tuple[BlogCandidate, ...]:
    """対応済みAnnotationへsource order、Scene role、global rankを付与する。"""
    if len(requests) != len(annotations) or shortlist_rank_offset < 0:
        msg = "Annotation requestと結果の件数またはrankが不正です"
        raise ValueError(msg)
    source_orders = {
        result.source.fingerprint: order
        for order, result in enumerate(video_stage_results)
    }
    result: list[BlogCandidate] = []
    for index, (request, annotation) in enumerate(
        zip(requests, annotations, strict=True)
    ):
        if (
            annotation.candidate_moment_id != request.moment.identifier
            or annotation.candidate not in request.frame_candidates
        ):
            msg = "AnnotationとCandidate Momentが一致しません"
            raise ValueError(msg)
        fingerprint = annotation.candidate.video_fingerprint
        if fingerprint is None or fingerprint not in source_orders:
            msg = "AnnotationのVideo Sourceが見つかりません"
            raise ValueError(msg)
        try:
            scene = scene_catalog.for_slug(annotation.scene_slug)
        except KeyError:
            raise ValueError("AnnotationのSceneがCatalogにありません") from None
        result.append(
            BlogCandidate(
                annotation=annotation,
                scene_selection_role=scene.selection_role,
                video_order=source_orders[fingerprint],
                video_set_progress=request.video_set_progress,
                shortlist_rank=shortlist_rank_offset + index,
            )
        )
    return tuple(result)
