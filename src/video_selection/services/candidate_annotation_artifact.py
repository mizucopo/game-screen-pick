"""Candidate AnnotationのCompleted Stage artifact変換。"""

from ..models.candidate_annotation import CandidateAnnotation
from ..models.frame_candidate import FrameCandidate


def build_candidate_annotation_artifact(
    annotations: tuple[CandidateAnnotation, ...],
    candidates: tuple[FrameCandidate, ...],
) -> dict[str, object]:
    """抽出候補に属するCandidate Annotationをartifactへ変換する。"""
    candidates_by_id = {candidate.identifier: candidate for candidate in candidates}
    annotation_ids: list[str] = []
    for annotation in annotations:
        candidate_id = annotation.candidate.identifier
        if candidates_by_id.get(candidate_id) != annotation.candidate:
            msg = (
                f"Candidate Annotationに未知のFrame Candidateがあります: {candidate_id}"
            )
            raise ValueError(msg)
        annotation_ids.append(candidate_id)
    if len(set(annotation_ids)) != len(annotation_ids):
        msg = "Candidate AnnotationのFrame Candidate IDが重複しています"
        raise ValueError(msg)
    return {
        "annotations": [
            {
                "candidate_id": annotation.candidate.identifier,
                "summary": annotation.summary,
            }
            for annotation in annotations
        ]
    }


def restore_candidate_annotations(
    artifact: dict[str, object],
    candidates: tuple[FrameCandidate, ...],
) -> tuple[CandidateAnnotation, ...]:
    """Completed Stage artifactを現在のFrame Candidateへ復元する。"""
    records = artifact.get("annotations")
    if not isinstance(records, list):
        msg = "Candidate Annotation artifactのannotationsが不正です"
        raise ValueError(msg)
    candidates_by_id = {candidate.identifier: candidate for candidate in candidates}
    restored: list[CandidateAnnotation] = []
    for record in records:
        if not isinstance(record, dict):
            msg = "Candidate Annotation artifactのrecordが不正です"
            raise ValueError(msg)
        candidate_id = record.get("candidate_id")
        summary = record.get("summary")
        if not isinstance(candidate_id, str) or not isinstance(summary, str):
            msg = "Candidate Annotation artifactのfieldが不正です"
            raise ValueError(msg)
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None:
            msg = (
                "Candidate Annotation artifactに未知のCandidate IDがあります: "
                f"{candidate_id}"
            )
            raise ValueError(msg)
        restored.append(CandidateAnnotation(candidate=candidate, summary=summary))
    return tuple(restored)
