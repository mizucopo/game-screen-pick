"""target acceptance用の実runtime composition root。"""

from ..application.video_selection_application import VideoSelectionApplication
from ..media.ffmpeg_media_runtime import FfmpegMediaRuntime
from ..models.effective_configuration import EffectiveConfiguration
from ..models.model_role import ModelRole
from ..models.resolved_model import ResolvedModel
from ..models.resolved_models import ResolvedModels
from ..protocols.run_observer import RunObserver
from ..protocols.speech_runtime import SpeechRuntime
from ..services.gpu_work_coordinator import GpuWorkCoordinator
from ..services.run_progress_tracker import RunProgressTracker
from ..speech.faster_whisper_speech_runtime import FasterWhisperSpeechRuntime
from ..vision.ollama_vision_runtime import OllamaVisionRuntime
from .frozen_model_runtime import FrozenModelRuntime


def build_real_application(
    configuration: EffectiveConfiguration,
    resolved_models: ResolvedModels,
    observer: RunObserver,
    progress: RunProgressTracker,
) -> VideoSelectionApplication:
    """FFmpeg、faster-whisper、Ollamaを共有GPU coordinatorへ接続する。"""
    gpu_coordinator = GpuWorkCoordinator()
    media_runtime = FfmpegMediaRuntime()
    vision_runtime = OllamaVisionRuntime(
        configuration.ollama_host,
        timeout_seconds=configuration.ollama_timeout_seconds,
        gpu_coordinator=gpu_coordinator,
    )

    def speech_runtime_factory(
        model: ResolvedModel,
        effective: EffectiveConfiguration,
    ) -> SpeechRuntime:
        if model.role is not ModelRole.SPEECH_TO_TEXT:
            raise ValueError("Speech Runtimeにはspeech_to_text modelが必要です")
        artifact = model.artifact_location
        if artifact is None:
            raise ValueError("Speech-to-text model artifactがありません")
        return FasterWhisperSpeechRuntime.load_local(
            artifact,
            resolved_model_identity=model.execution_identity.identifier,
            device=effective.speech_to_text_device,
            compute_type=effective.speech_to_text_compute_type,
            gpu_coordinator=gpu_coordinator,
        )

    return VideoSelectionApplication(
        media_runtime=media_runtime,
        model_runtime=FrozenModelRuntime(resolved_models),
        speech_runtime_factory=speech_runtime_factory,
        vision_runtime=vision_runtime,
        observer=observer,
        progress=progress,
    )
