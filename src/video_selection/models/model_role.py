"""Video Set pipeline内のmodel role。"""

from enum import StrEnum


class ModelRole(StrEnum):
    """設定とfingerprintを独立させるmodel用途。"""

    SCENE_CATALOG = "scene_catalog"
    CANDIDATE_ANNOTATION = "candidate_annotation"
    SPEECH_TO_TEXT = "speech_to_text"
