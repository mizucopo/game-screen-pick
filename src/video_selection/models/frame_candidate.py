"""Video Source内のFrame Candidate。"""

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from .decoded_video_frame import DecodedVideoFrame
from .neutral_image_analysis import NeutralImageAnalysis


@dataclass(frozen=True)
class FrameCandidate:
    """安定ID、exact時刻、proxy、Neutral Image Analysisを持つframe。"""

    identifier: str
    image_bytes: bytes
    video_fingerprint: str | None = None
    stream_index: int | None = None
    source_pts: int | None = None
    origin_pts: int | None = None
    time_base: Fraction | None = None
    video_time: Fraction | None = None
    analysis: NeutralImageAnalysis | None = None
    proxy_path: Path | None = None
    decoded_frame: DecodedVideoFrame | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Video Stage由来candidateのidentityとtimingを検証する。"""
        if self.video_fingerprint is None:
            return
        if (
            len(self.video_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.video_fingerprint
            )
            or not self.identifier.startswith("frm_")
            or len(self.identifier) != 68
        ):
            msg = "Frame CandidateにはVideo Fingerprintとfrm_ IDが必要です"
            raise ValueError(msg)
        if any(
            item is None
            for item in (
                self.stream_index,
                self.source_pts,
                self.origin_pts,
                self.time_base,
                self.video_time,
                self.analysis,
            )
        ):
            msg = "Video Stage由来Frame Candidateにはexact timingとanalysisが必要です"
            raise ValueError(msg)
