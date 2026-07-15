"""Frame Candidateのordered content snapshotを構築する。"""

import hashlib
import re

from ..models.frame_candidate import FrameCandidate

_SAFE_CANDIDATE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def snapshot_frame_candidates(
    candidates: tuple[FrameCandidate, ...],
) -> tuple[dict[str, str], ...]:
    """Candidate IDとimage SHA-256を返し、重複IDを拒否する。"""
    identifiers = tuple(candidate.identifier for candidate in candidates)
    unsafe_identifier = next(
        (
            identifier
            for identifier in identifiers
            if _SAFE_CANDIDATE_ID_PATTERN.fullmatch(identifier) is None
        ),
        None,
    )
    if unsafe_identifier is not None:
        msg = f"Frame Candidate IDが安全ではありません: {unsafe_identifier}"
        raise ValueError(msg)
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
