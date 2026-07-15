"""Frame Candidateのordered content snapshotを構築する。"""

import hashlib

from ..models.frame_candidate import FrameCandidate


def snapshot_frame_candidates(
    candidates: tuple[FrameCandidate, ...],
) -> tuple[dict[str, str], ...]:
    """Candidate IDとimage SHA-256を返し、重複IDを拒否する。"""
    identifiers = tuple(candidate.identifier for candidate in candidates)
    if len(set(identifiers)) != len(identifiers):
        msg = "Frame Candidate IDが重複しています"
        raise ValueError(msg)
    return tuple(
        {
            "id": candidate.identifier,
            "image_sha256": hashlib.sha256(candidate.image_bytes).hexdigest(),
        }
        for candidate in candidates
    )
