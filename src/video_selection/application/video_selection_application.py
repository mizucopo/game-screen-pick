"""Video Set選定walking skeletonのapplication orchestration。"""

from contextlib import suppress

from ..models.effective_configuration import EffectiveConfiguration
from ..models.processing_stage import ProcessingStage
from ..models.run_outcome import RunOutcome
from ..models.run_status import RunStatus
from ..models.video_set import VideoSet
from ..protocols.media_runtime import MediaRuntime
from ..protocols.model_runtime import ModelRuntime
from ..protocols.run_observer import RunObserver
from ..protocols.speech_runtime import SpeechRuntime
from ..protocols.vision_runtime import VisionRuntime
from ..services.atomic_output_publisher import AtomicOutputPublisher
from ..services.candidate_annotation_artifact import (
    build_candidate_annotation_artifact,
    normalize_candidate_annotations,
    restore_candidate_annotations,
)
from ..services.discover_video_set import discover_video_set
from ..services.input_folder_lock import InputFolderLock
from ..services.prepare_processing_cache import prepare_processing_cache
from ..services.processing_stage_runner import ProcessingStageRunner
from ..services.select_images import select_images
from ..services.snapshot_frame_candidates import snapshot_frame_candidates
from ..services.snapshot_video_set import snapshot_video_set
from ..services.validate_output_folder import validate_output_folder
from ..services.validate_video_set_snapshot import validate_video_set_snapshot


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
        validate_output_folder(
            configuration.video_input_folder,
            configuration.output_folder,
        )

        video_set = discover_video_set(
            configuration.video_input_folder,
            recursive=configuration.recursive,
        )
        with InputFolderLock(configuration.video_input_folder) as input_lock:
            validate_video_set_snapshot(video_set)
            diagnostic = prepare_processing_cache(
                configuration.processing_cache_folder,
                input_lock=input_lock,
                reset_cache=configuration.reset_cache,
            )
            self._observer.legacy_cache_cleaned(diagnostic)
            return self._run_locked(configuration, video_set)

    def _run_locked(
        self,
        configuration: EffectiveConfiguration,
        video_set: VideoSet,
    ) -> RunOutcome:
        """Input Lockを保持したまま全Processing Stageを実行する。"""
        video_set_snapshot = snapshot_video_set(video_set)
        with suppress(FileNotFoundError):
            configuration.output_folder.rmdir()
        stage_runner = ProcessingStageRunner(
            configuration.processing_cache_folder,
            self._observer,
            subject_namespace="video-sets",
            subject_fingerprint=video_set.fingerprint,
            before_stage=lambda: validate_video_set_snapshot(video_set),
        )
        stage_runner.complete(
            ProcessingStage.DISCOVER_VIDEO_SET,
            {
                "video_set_fingerprint": video_set.fingerprint,
                "videos": list(video_set_snapshot),
            },
            {
                "video_set_fingerprint": video_set.fingerprint,
                "videos": list(video_set_snapshot),
            },
        )

        frame_candidates = self._media_runtime.extract_candidates(video_set)
        candidate_snapshot = snapshot_frame_candidates(frame_candidates)
        stage_runner.complete(
            ProcessingStage.EXTRACT_FRAME_CANDIDATES,
            {"candidates": list(candidate_snapshot)},
            {"candidates": list(candidate_snapshot)},
        )

        context_cues = self._speech_runtime.collect_context(video_set)
        context_cue_ids = [item.identifier for item in context_cues]
        stage_runner.complete(
            ProcessingStage.COLLECT_CONTEXT,
            {"context_cue_ids": context_cue_ids},
            {"context_cue_ids": context_cue_ids},
        )

        model_identity = self._model_runtime.resolve_models()
        stage_runner.complete(
            ProcessingStage.RESOLVE_MODELS,
            {"resolved_model_identity": model_identity.identifier},
            {"model_identity": model_identity.identifier},
        )

        annotation_semantic_input = {
            "candidates": list(candidate_snapshot),
            "context_cue_ids": context_cue_ids,
            "resolved_model_identity": model_identity.identifier,
        }
        annotations = stage_runner.reuse(
            ProcessingStage.ANNOTATE_CANDIDATES,
            annotation_semantic_input,
            lambda artifact: restore_candidate_annotations(
                artifact,
                frame_candidates,
            ),
        )
        if annotations is None:
            annotations = normalize_candidate_annotations(
                self._vision_runtime.annotate_candidates(
                    frame_candidates,
                    context_cues,
                    model_identity,
                ),
                frame_candidates,
            )
            stage_runner.complete(
                ProcessingStage.ANNOTATE_CANDIDATES,
                annotation_semantic_input,
                build_candidate_annotation_artifact(annotations, frame_candidates),
            )

        selected_images = select_images(annotations, configuration.image_count)
        selected_count = len(selected_images)
        run_status = RunStatus.from_selection_counts(
            configuration.image_count,
            selected_count,
        )
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
        prepared_output = publisher.prepare(
            configuration.output_folder,
            video_set,
            selected_images,
            configuration.image_count,
            run_status,
        )
        validate_video_set_snapshot(video_set)
        prepared_output.publish()
        return RunOutcome(
            output_folder=configuration.output_folder,
            status=run_status,
            requested_count=configuration.image_count,
            selected_count=selected_count,
            completed_stages=stage_runner.completed_stages,
        )
