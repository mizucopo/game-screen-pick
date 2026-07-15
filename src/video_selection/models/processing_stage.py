"""Video Set選定のProcessing Stage。"""

from enum import StrEnum


class ProcessingStage(StrEnum):
    """walking skeletonを構成する順序付きProcessing Stage。"""

    DISCOVER_VIDEO_SET = "discover-video-set"
    EXTRACT_FRAME_CANDIDATES = "extract-frame-candidates"
    COLLECT_CONTEXT = "collect-context"
    RESOLVE_MODELS = "resolve-models"
    ANNOTATE_CANDIDATES = "annotate-candidates"
    SELECT_IMAGES = "select-images"
