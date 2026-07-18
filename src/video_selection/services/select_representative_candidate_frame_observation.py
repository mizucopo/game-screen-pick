"""frame別意味観測からRepresentative Frameを決定する。"""

from ..models.candidate_frame_observation import CandidateFrameObservation

_EXPLANATION_PRIORITY = {"none": 0, "low": 1, "medium": 2, "high": 3}
_CONTENT_PRIORITY = {
    "document": 0,
    "tutorial_help": 0,
    "event_setup": 0,
    "gameplay_idle": 1,
    "save": 2,
    "map": 3,
    "other_interface": 3,
    "other": 4,
    "shop": 4,
    "title": 4,
    "event_action": 5,
    "gameplay_action": 5,
    "event_dialogue": 6,
}
_SUBJECT_PRIORITY = {"absent": 0, "partial": 1, "clear": 2}
_OBSTRUCTION_PRIORITY = {"severe": 0, "partial": 1, "none": 2}


def select_representative_candidate_frame_observation(
    observations: tuple[CandidateFrameObservation, ...],
) -> CandidateFrameObservation:
    """意味、視認性、Neutral画質の順で一つのframe観測を返す。"""
    identifiers = tuple(item.candidate.identifier for item in observations)
    if not observations or len(identifiers) != len(set(identifiers)):
        msg = "Representative Frame候補の観測集合が不正です"
        raise ValueError(msg)
    eligible = tuple(
        item for item in observations if not _is_grossly_degraded(item, observations)
    )
    return min(eligible or observations, key=_selection_key)


def _is_grossly_degraded(
    observation: CandidateFrameObservation,
    peers: tuple[CandidateFrameObservation, ...],
) -> bool:
    analysis = observation.candidate.analysis
    if analysis is None or analysis.quality_score >= 0.35:
        return False
    metrics = analysis.metrics
    return any(
        peer_analysis is not None
        and peer_analysis.quality_score - analysis.quality_score >= 0.35
        and peer_analysis.metrics.visibility_score - metrics.visibility_score >= 0.10
        and peer_analysis.metrics.information_score - metrics.information_score >= 0.20
        for peer in peers
        if peer is not observation
        for peer_analysis in (peer.candidate.analysis,)
    )


def _selection_key(
    observation: CandidateFrameObservation,
) -> tuple[int, int, int, int, float, str]:
    analysis = observation.candidate.analysis
    quality_score = 0.0 if analysis is None else analysis.quality_score
    return (
        -_EXPLANATION_PRIORITY[observation.effective_explanation_value],
        -_CONTENT_PRIORITY[observation.effective_content_kind],
        -_SUBJECT_PRIORITY[observation.primary_subject_visibility],
        -_OBSTRUCTION_PRIORITY[observation.transient_obstruction],
        -quality_score,
        observation.candidate.identifier,
    )
