"""一つのVideo SourceのVideo Stage結果。"""

from dataclasses import dataclass

from .completed_stage import CompletedStage
from .context_stage_result import ContextStageResult
from .frame_candidate_extraction import FrameCandidateExtraction
from .frame_candidate_extraction_metrics import FrameCandidateExtractionMetrics
from .video_scan_result import VideoScanResult
from .video_source import VideoSource


@dataclass(frozen=True)
class VideoStageResult:
    """scanとcandidate抽出のCompleted Stageをまとめる。"""

    source: VideoSource
    scan: VideoScanResult
    extraction: FrameCandidateExtraction
    extraction_metrics: FrameCandidateExtractionMetrics
    context: ContextStageResult
    completed_stages: tuple[CompletedStage, ...]
