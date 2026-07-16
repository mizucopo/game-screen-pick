"""Processing Stageごとのalgorithm version。"""

from ..models.processing_stage import ProcessingStage


def stage_version(stage: ProcessingStage) -> str:
    """Stage Fingerprintとmanifestへ使うversionを返す。"""
    if stage is ProcessingStage.SCAN_VIDEO:
        return "video-scan-v1"
    if stage is ProcessingStage.EXTRACT_FRAME_CANDIDATES:
        return "frame-candidate-extraction-v1"
    if stage is ProcessingStage.COLLECT_CONTEXT:
        return "context-collection-v1"
    return "walking-skeleton-0"
