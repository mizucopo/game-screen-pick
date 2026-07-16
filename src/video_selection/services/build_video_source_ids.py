"""Video Set内で一意な公開Video Source IDを構築する。"""

from collections import Counter

from ..models.video_source import VideoSource

_SHORT_DIGEST_LENGTH = 12


def build_video_source_ids(
    sources: tuple[VideoSource, ...],
) -> dict[str, str]:
    """衝突した短縮digestだけを完全digestへ拡張して返す。"""
    prefix_counts = Counter(
        source.fingerprint[:_SHORT_DIGEST_LENGTH] for source in sources
    )
    identifiers: dict[str, str] = {}
    for source in sources:
        fingerprint = source.fingerprint
        prefix = fingerprint[:_SHORT_DIGEST_LENGTH]
        digest = fingerprint if prefix_counts[prefix] > 1 else prefix
        identifiers[fingerprint] = f"vid_{digest}"
    return identifiers
