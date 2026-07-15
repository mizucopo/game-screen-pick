"""Video Set選定walking skeletonのapplication orchestration。"""

from ..models.effective_configuration import EffectiveConfiguration
from ..models.processing_stage import ProcessingStage
from ..models.run_outcome import RunOutcome
from ..protocols.media_runtime import MediaRuntime
from ..protocols.model_runtime import ModelRuntime
from ..protocols.run_observer import RunObserver
from ..protocols.speech_runtime import SpeechRuntime
from ..protocols.vision_runtime import VisionRuntime
from ..services.atomic_output_publisher import AtomicOutputPublisher
from ..services.discover_video_set import discover_video_set
from ..services.processing_stage_runner import ProcessingStageRunner
from ..services.select_images import select_images


class VideoSelectionApplication:
    """fake runtimeを通してVideo Setからoutputまでを接続する。"""

    def __init__(
        self,
        media_runtime: MediaRuntime,
        speech_runtime: SpeechRuntime,
        model_runtime: ModelRuntime,
        vision_runtime: VisionRuntime,
        observer: RunObserver,
    ) -> None:
        self._media_runtime = media_runtime
        self._speech_runtime = speech_runtime
        self._model_runtime = model_runtime
        self._vision_runtime = vision_runtime
        self._observer = observer

    def run(self, configuration: EffectiveConfiguration) -> RunOutcome:
        """内部Video Set選定を実行してRunOutcomeを返す。"""
        stage_runner = ProcessingStageRunner(
            configuration.processing_cache_folder,
            self._observer,
        )

        video_set = discover_video_set(configuration.video_input_folder)
        stage_runner.complete(
            ProcessingStage.DISCOVER_VIDEO_SET,
            {"videos": list(video_set.relative_paths)},
            {"videos": list(video_set.relative_paths)},
        )

        frame_candidates = self._media_runtime.extract_candidates(video_set)
        stage_runner.complete(
            ProcessingStage.EXTRACT_FRAME_CANDIDATES,
            {"video_count": len(video_set.videos)},
            {"candidate_ids": [item.identifier for item in frame_candidates]},
        )

        context_cues = self._speech_runtime.collect_context(video_set)
        stage_runner.complete(
            ProcessingStage.COLLECT_CONTEXT,
            {"video_count": len(video_set.videos)},
            {"context_cue_ids": [item.identifier for item in context_cues]},
        )

        model_identity = self._model_runtime.resolve_models()
        stage_runner.complete(
            ProcessingStage.RESOLVE_MODELS,
            {},
            {"model_identity": model_identity.identifier},
        )

        annotations = self._vision_runtime.annotate_candidates(
            frame_candidates,
            context_cues,
            model_identity,
        )
        stage_runner.complete(
            ProcessingStage.ANNOTATE_CANDIDATES,
            {"candidate_count": len(frame_candidates)},
            {"candidate_ids": [item.candidate.identifier for item in annotations]},
        )

        selected_images = select_images(annotations, configuration.image_count)
        stage_runner.complete(
            ProcessingStage.SELECT_IMAGES,
            {"image_count": configuration.image_count},
            {
                "selected_ids": [
                    item.annotation.candidate.identifier for item in selected_images
                ]
            },
        )

        publisher = AtomicOutputPublisher()
        report = publisher.publish(
            configuration.output_folder,
            video_set,
            selected_images,
        )
        stage_runner.complete(
            ProcessingStage.PUBLISH_OUTPUT,
            {"selected_count": len(selected_images)},
            {"report": report},
        )
        return RunOutcome(
            output_folder=configuration.output_folder,
            selected_count=len(selected_images),
            completed_stages=stage_runner.completed_stages,
        )
