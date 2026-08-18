"""単一動画の画像選定で共有する値オブジェクト."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoMetadata:
    """ffprobeから得た動画メタデータ."""

    duration_seconds: float
    width: int
    height: int
    codec_name: str
    average_frame_rate: str


@dataclass(frozen=True)
class FrameCandidate:
    """動画内の選定候補フレーム."""

    frame_id: str
    timestamp_seconds: float
    path: str
    quality_score: float = 0.0
    difference_hash: int = 0


@dataclass(frozen=True)
class FrameAssessment:
    """Ollamaによるブログ掲載候補の評価."""

    frame_id: str
    blog_score: float
    is_transition: bool
    scene: str
    reason: str


@dataclass(frozen=True)
class SelectedFrame:
    """最終選定されたフレームと評価."""

    candidate: FrameCandidate
    aggregate_score: float
    primary_assessment: FrameAssessment
    secondary_assessment: FrameAssessment
