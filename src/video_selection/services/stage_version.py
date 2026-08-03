"""Processing Stageごとのalgorithm version registry。"""

from ..models.processing_stage import ProcessingStage

_STAGE_VERSIONS = {
    ProcessingStage.DISCOVER_VIDEO_SET: "video-set-discovery-v1",
    ProcessingStage.SCAN_VIDEO: "video-scan-v6",
    ProcessingStage.EXTRACT_FRAME_CANDIDATES: "frame-candidate-extraction-v4",
    ProcessingStage.COLLECT_CONTEXT: "context-collection-v4",
    ProcessingStage.RESOLVE_MODELS: "model-resolution-v1",
    ProcessingStage.BUILD_SCENE_CATALOG: "scene-catalog-v1",
    ProcessingStage.ANNOTATE_CANDIDATE: "candidate-annotation-v1",
    ProcessingStage.ANNOTATE_CANDIDATES: "candidate-annotations-v1",
    ProcessingStage.SELECT_IMAGES: "video-set-selection-v5",
}

if set(_STAGE_VERSIONS) != set(ProcessingStage):
    raise RuntimeError("全Processing Stageに明示的なversion登録が必要です")


def stage_version(stage: ProcessingStage) -> str:
    """Stage Fingerprintとmanifestへ使う明示versionを返す。"""
    return _STAGE_VERSIONS[stage]
