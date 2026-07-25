"""Processing Stageごとのalgorithm version。"""

from ..models.processing_stage import ProcessingStage


def stage_version(stage: ProcessingStage) -> str:
    """Stage Fingerprintとmanifestへ使うversionを返す。"""
    if stage is ProcessingStage.SCAN_VIDEO:
        return "video-scan-v1"
    if stage is ProcessingStage.EXTRACT_FRAME_CANDIDATES:
        return "frame-candidate-extraction-v3"
    if stage is ProcessingStage.COLLECT_CONTEXT:
        return "context-collection-v3"
    if stage is ProcessingStage.RESOLVE_MODELS:
        return "model-resolution-v1"
    if stage is ProcessingStage.BUILD_SCENE_CATALOG:
        return "scene-catalog-v1"
    if stage is ProcessingStage.ANNOTATE_CANDIDATE:
        return "candidate-annotation-v1"
    if stage is ProcessingStage.SELECT_IMAGES:
        return "video-set-selection-v2"
    return "walking-skeleton-0"
