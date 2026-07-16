"""Video Set選定のProcessing Stage。"""

from enum import StrEnum


class ProcessingStage(StrEnum):
    """walking skeletonを構成する順序付きProcessing Stage。"""

    DISCOVER_VIDEO_SET = "discover-video-set"
    SCAN_VIDEO = "scan-video"
    EXTRACT_FRAME_CANDIDATES = "extract-frame-candidates"
    COLLECT_CONTEXT = "collect-context"
    RESOLVE_MODELS = "resolve-models"
    BUILD_SCENE_CATALOG = "build-scene-catalog"
    ANNOTATE_CANDIDATE = "annotate-candidate"
    ANNOTATE_CANDIDATES = "annotate-candidates"
    SELECT_IMAGES = "select-images"


VIDEO_SET_STAGE_ORDER = (
    ProcessingStage.DISCOVER_VIDEO_SET,
    ProcessingStage.EXTRACT_FRAME_CANDIDATES,
    ProcessingStage.RESOLVE_MODELS,
    ProcessingStage.COLLECT_CONTEXT,
    ProcessingStage.ANNOTATE_CANDIDATES,
    ProcessingStage.SELECT_IMAGES,
)

VIDEO_STAGE_ORDER = (
    ProcessingStage.SCAN_VIDEO,
    ProcessingStage.EXTRACT_FRAME_CANDIDATES,
)
