"""Migration acceptance evidence。"""

from enum import StrEnum


class Evidence(StrEnum):
    """public cutover前に必要なacceptance evidence。"""

    FAKE_E2E = "fake E2E"
    FFMPEG_INTEGRATION = "FFmpeg integration"
    INTERRUPTION_MATRIX = "interruption matrix"
    TRACEABILITY = "traceability matrix"
    TARGET_COLD = "30-minute cold <= 20m"
    TARGET_WARM = "30-minute warm <= 3m"
    FULL_COLD = "full-scale cold <= 24h"
    FULL_WARM = "full-scale warm <= 30m"
    CACHE_BUDGET = "cache <= 64/96 GiB"
    GPU_BUDGET = "GPU <= 18/8 GiB"
    HUMAN_QUALITY = "human quality gate"
