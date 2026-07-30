"""Durable checkpointごとのengine version registry。"""

from ..models.checkpoint_operation import CheckpointOperation

_CHECKPOINT_VERSIONS = {
    CheckpointOperation.VIDEO_IDENTITY: "video-identity-engine-v2",
    CheckpointOperation.VIDEO_SCAN_PARTITION: "video-scan-partition-v3",
    CheckpointOperation.FRAME_REFINEMENT_GROUP: "frame-refinement-group-v1",
    CheckpointOperation.PCM_AUDIO_CHUNK: "pcm-audio-range-v2",
    CheckpointOperation.SPEECH_RECOGNITION_CHUNK: "speech-recognition-chunk-v1",
    CheckpointOperation.EMBEDDED_SUBTITLE_STREAM: "embedded-subtitle-stream-v1",
    CheckpointOperation.SELECTED_IMAGE_WEBP: "selected-image-webp-v1",
}

if set(_CHECKPOINT_VERSIONS) != set(CheckpointOperation):
    raise RuntimeError("全Durable checkpointに明示的なversion登録が必要です")


def checkpoint_version(operation: CheckpointOperation) -> str:
    """checkpoint fingerprintとmanifestへ使う明示versionを返す。"""
    return _CHECKPOINT_VERSIONS[operation]
