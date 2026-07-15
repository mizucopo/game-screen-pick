"""Candidate AnnotationのCompleted Stage artifact変換。"""

from ..models.candidate_annotation import CandidateAnnotation
from ..models.frame_candidate import FrameCandidate


def build_candidate_annotation_artifact(
    annotations: tuple[CandidateAnnotation, ...],
    candidates: tuple[FrameCandidate, ...],
) -> dict[str, object]:
    """抽出候補に属するCandidate Annotationをartifactへ変換する。"""
    normalized_annotations = normalize_candidate_annotations(annotations, candidates)
    return {
        "annotations": [
            {
                "candidate_id": annotation.candidate.identifier,
                "summary": annotation.summary,
            }
            for annotation in normalized_annotations
        ]
    }


def normalize_candidate_annotations(
    annotations: tuple[CandidateAnnotation, ...],
    candidates: tuple[FrameCandidate, ...],
) -> tuple[CandidateAnnotation, ...]:
    """Annotationを検証してFrame Candidate順へ正規化する。"""
    candidates_by_id = {candidate.identifier: candidate for candidate in candidates}
    annotations_by_id: dict[str, CandidateAnnotation] = {}
    for annotation in annotations:
        candidate_id = annotation.candidate.identifier
        if candidates_by_id.get(candidate_id) != annotation.candidate:
            msg = (
                f"Candidate Annotationに未知のFrame Candidateがあります: {candidate_id}"
            )
            raise ValueError(msg)
        if candidate_id in annotations_by_id:
            msg = "Candidate AnnotationのFrame Candidate IDが重複しています"
            raise ValueError(msg)
        annotations_by_id[candidate_id] = annotation
    missing_candidate_ids = set(candidates_by_id) - set(annotations_by_id)
    if missing_candidate_ids:
        msg = f"Candidate Annotationが不足しています: {sorted(missing_candidate_ids)}"
        raise ValueError(msg)
    return tuple(annotations_by_id[candidate.identifier] for candidate in candidates)


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
    return normalize_candidate_annotations(tuple(restored), candidates)
