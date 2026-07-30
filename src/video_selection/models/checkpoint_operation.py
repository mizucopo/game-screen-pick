"""Durable checkpointとして登録されたWork Unit種別。"""

from enum import Enum


class CheckpointOperation(Enum):
    """engine versionを明示して再開できる最小処理単位。"""

    VIDEO_IDENTITY = "video-identity"
    VIDEO_SCAN_PARTITION = "video-scan-partition"
    FRAME_REFINEMENT_GROUP = "frame-refinement-group"
    PCM_AUDIO_CHUNK = "pcm-audio-chunk"
    SPEECH_RECOGNITION_CHUNK = "speech-recognition-chunk"
    EMBEDDED_SUBTITLE_STREAM = "embedded-subtitle-stream"
    SELECTED_IMAGE_WEBP = "selected-image-webp"
