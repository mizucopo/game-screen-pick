"""model artifactへ要求する実行能力。"""

from enum import StrEnum


class ModelCapability(StrEnum):
    """store adapterが検証するrole別capability。"""

    VISION_STRUCTURED_OUTPUT = "vision_structured_output"
    SPEECH_TO_TEXT = "speech_to_text"
