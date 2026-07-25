"""実Video Stage、Vision、Selector、Canonical Publicationのcomposition。"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import suppress
from datetime import datetime, timezone
from typing import cast
from uuid import uuid4

from ..configuration.configuration_error import ConfigurationError
from ..models.blog_candidate import BlogCandidate
from ..models.canonical_publication_request import CanonicalPublicationRequest
from ..models.completed_stage import CompletedStage
from ..models.effective_configuration import EffectiveConfiguration
from ..models.model_role import ModelRole
from ..models.processing_stage import ProcessingStage
from ..models.resolved_models import ResolvedModels
from ..models.run_outcome import RunOutcome
from ..models.run_status import RunStatus
from ..models.scene_catalog import SceneCatalog
from ..models.stage_fingerprint import StageFingerprint
from ..models.video_set import VideoSet
from ..models.video_set_selection_result import VideoSetSelectionResult
from ..models.video_stage_result import VideoStageResult
from ..models.vision_inference_diagnostics import VisionInferenceDiagnostics
from ..protocols.model_runtime import ModelRuntime
from ..protocols.run_observer import RunObserver
from ..protocols.selection_media_runtime import SelectionMediaRuntime
from ..protocols.speech_runtime_factory import SpeechRuntimeFactory
from ..protocols.vision_runtime import VisionRuntime
from ..services.build_blog_candidates import build_blog_candidates
from ..services.build_candidate_annotation_requests import (
    build_candidate_annotation_requests,
    select_scene_catalog_representatives,
)
from ..services.build_report_provenance import build_report_provenance
from ..services.build_stage_fingerprint import build_stage_fingerprint
from ..services.canonical_output_publisher import CanonicalOutputPublisher
from ..services.completed_stage_writer import CompletedStageWriter
from ..services.discover_video_set import discover_video_set
from ..services.input_folder_lock import InputFolderLock
from ..services.prepare_processing_cache import prepare_processing_cache
from ..services.run_progress_tracker import RunProgressTracker
from ..services.sanitize_selection_annotations_for_publication import (
    sanitize_selection_annotations_for_publication,
)
from ..services.select_video_set_images import (
    SpoilerSensitivity,
    select_from_shortlist_batches,
    select_video_set_images,
)
from ..services.selection_stage_artifacts import (
    restore_video_set_selection_result,
    selection_artifact_candidate_count,
    serialize_video_set_selection_result,
)
from ..services.selection_stage_cache import SelectionStageCache
from ..services.validate_output_folder import validate_output_folder
from ..services.validate_video_set_snapshot import validate_video_set_snapshot_metadata
from ..services.video_identity_cache import VideoIdentityCache
from ..services.video_set_vision_processor import (
    VideoSetVisionProcessor,
    plan_vision_stage_fingerprints,
)
from ..services.video_stage_processor import VideoStageProcessor

_SELECTION_INTENT = "ブログ本文を説明できる画像を選ぶ"
_INITIAL_ANNOTATION_MINIMUM = 24
UtcClock = Callable[[], datetime]


class VideoSelectionApplication:
    """実runtimeをVideo Set入力からcanonical outputまで接続する深いmodule。"""

    def __init__(
        self,
        *,
        media_runtime: SelectionMediaRuntime,
        model_runtime: ModelRuntime,
        speech_runtime_factory: SpeechRuntimeFactory,
        vision_runtime: VisionRuntime,
        observer: RunObserver,
        progress: RunProgressTracker,
        clock: UtcClock | None = None,
    ) -> None:
        self._media_runtime = media_runtime
        self._model_runtime = model_runtime
        self._speech_runtime_factory = speech_runtime_factory
        self._vision_runtime = vision_runtime
        self._observer = observer
        self._progress = progress
        self._clock = clock or _utc_now

    def run(self, configuration: EffectiveConfiguration) -> RunOutcome:
        """実Video Selection pipelineを実行しatomic outputを公開する。"""
        started_at = self._clock()
        _validate_configuration_paths(configuration)
        identity_cache = VideoIdentityCache(configuration.processing_cache_folder)
        video_set = discover_video_set(
            configuration.video_input_folder,
            recursive=configuration.recursive,
            identity_cache=None if configuration.reset_cache else identity_cache,
        )
        with InputFolderLock(configuration.video_input_folder) as input_lock:
            validate_video_set_snapshot_metadata(video_set)
            resolved_models = self._model_runtime.resolve_models(configuration)
            speech_runtime = self._speech_runtime_factory(
                resolved_models.for_role(ModelRole.SPEECH_TO_TEXT),
                configuration,
            )
            try:
                media_runtime_identity = self._media_runtime.preflight()
                validate_video_set_snapshot_metadata(video_set)
                diagnostic = prepare_processing_cache(
                    configuration.processing_cache_folder,
                    input_lock=input_lock,
                    reset_cache=configuration.reset_cache,
                )
                self._observer.legacy_cache_cleaned(diagnostic)
                for source in video_set.sources:
                    identity_cache.store(source)
                video_stage_results = VideoStageProcessor(
                    self._media_runtime,
                    speech_runtime,
                    self._observer,
                    progress=self._progress,
                ).process(
                    video_set,
                    configuration,
                    runtime_identity=media_runtime_identity,
                )
                speech_runtime_identity = speech_runtime.runtime_identity
            finally:
                speech_runtime.close()
            return self._select_and_publish(
                configuration,
                video_set,
                video_stage_results,
                resolved_models,
                speech_runtime_identity,
                started_at,
            )

    def _select_and_publish(
        self,
        configuration: EffectiveConfiguration,
        video_set: VideoSet,
        video_stage_results: tuple[VideoStageResult, ...],
        resolved_models: ResolvedModels,
        speech_runtime_identity: str,
        started_at: datetime,
    ) -> RunOutcome:
        """shortlistを必要分だけ注釈し選定結果をcanonicalに公開する。"""
        requests = build_candidate_annotation_requests(
            video_stage_results,
            selection_intent=_SELECTION_INTENT,
            similarity_threshold=configuration.similarity_threshold,
        )
        vision_stages: list[CompletedStage] = []
        vision_diagnostics: dict[str, VisionInferenceDiagnostics] = {}
        scene_catalog: SceneCatalog | None = None
        base_completed_stages = _unique_completed_stages(
            tuple(
                stage
                for result in video_stage_results
                for stage in result.completed_stages
            )
        )
        spoiler_sensitivity = cast(
            SpoilerSensitivity,
            configuration.spoiler_sensitivity,
        )
        planned_vision_fingerprints: tuple[StageFingerprint, ...] = ()
        if requests:
            representatives = select_scene_catalog_representatives(requests)
            extraction_fingerprints = _extraction_fingerprints(video_stage_results)
            planned_vision_fingerprints = plan_vision_stage_fingerprints(
                video_set=video_set,
                representatives=representatives,
                representative_source_fingerprints=extraction_fingerprints,
                annotation_requests=requests,
                configuration=configuration,
                resolved_models=resolved_models,
            )
            vision_processor = VideoSetVisionProcessor(
                self._vision_runtime,
                self._observer,
                progress=self._progress,
            )

            def candidate_batches() -> Iterator[tuple[BlogCandidate, ...]]:
                nonlocal scene_catalog
                offset = 0
                for batch_size in _annotation_batch_sizes(
                    len(requests),
                    configuration.image_count,
                ):
                    batch = requests[offset : offset + batch_size]
                    vision = vision_processor.process(
                        video_set=video_set,
                        representatives=representatives,
                        representative_source_fingerprints=extraction_fingerprints,
                        annotation_requests=batch,
                        configuration=configuration,
                        resolved_models=resolved_models,
                    )
                    if scene_catalog is None:
                        scene_catalog = vision.catalog
                    elif scene_catalog != vision.catalog:
                        msg = "再利用されたScene Catalogがrun内で変化しました"
                        raise RuntimeError(msg)
                    vision_stages.extend(vision.completed_stages)
                    for completed, diagnostics in zip(
                        vision.completed_stages,
                        (
                            vision.catalog_diagnostics,
                            *vision.annotation_diagnostics,
                        ),
                        strict=True,
                    ):
                        vision_diagnostics.setdefault(
                            completed.fingerprint.value,
                            diagnostics,
                        )
                    yield build_blog_candidates(
                        batch,
                        vision.annotations,
                        vision.catalog,
                        video_stage_results,
                        shortlist_rank_offset=offset,
                    )
                    offset += batch_size

        selection_request_fingerprint = _selection_request_fingerprint(
            configuration,
            base_completed_stages,
            planned_vision_fingerprints,
            request_count=len(requests),
        )
        selection_cache = SelectionStageCache(
            configuration.processing_cache_folder,
            video_set_fingerprint=video_set.fingerprint,
        )
        cached_selection = selection_cache.read(selection_request_fingerprint)
        selection_reused = cached_selection is not None
        if cached_selection is not None:
            selection_artifact, selection_stage = cached_selection
            expected_candidate_count = selection_artifact_candidate_count(
                selection_artifact
            )
            if requests:
                annotated_candidates = _restore_annotated_candidates(
                    candidate_batches(),
                    expected_candidate_count,
                )
            elif expected_candidate_count == 0:
                annotated_candidates = ()
            else:
                raise ValueError("Video Set Selection cacheのcandidate件数が不正です")
            selection = restore_video_set_selection_result(
                selection_artifact,
                annotated_candidates,
            )
            if selection.requested_count != configuration.image_count:
                raise ValueError("Video Set Selection cacheのrequested countが不正です")
        elif requests:
            selection = select_from_shortlist_batches(
                candidate_batches(),
                requested_count=configuration.image_count,
                spoiler_sensitivity=spoiler_sensitivity,
                similarity_threshold=configuration.similarity_threshold,
            )
        else:
            selection = select_video_set_images(
                (),
                requested_count=configuration.image_count,
                spoiler_sensitivity=spoiler_sensitivity,
                similarity_threshold=configuration.similarity_threshold,
            )

        completed_stages = _unique_completed_stages(
            (
                *base_completed_stages,
                *vision_stages,
            )
        )
        if cached_selection is not None:
            expected_upstream = tuple(stage.fingerprint for stage in completed_stages)
            if selection_stage.upstream_fingerprints != expected_upstream:
                raise ValueError(
                    "Video Set Selection cacheのupstreamが現在のrunと一致しません"
                )
        else:
            selection_stage = self._write_selection_stage(
                configuration,
                video_set,
                selection,
                completed_stages,
                selection_request_fingerprint,
            )
            selection_cache.record(
                selection_request_fingerprint,
                selection_stage,
            )
        self._record_selection_progress(
            selection_stage,
            reused=selection_reused,
        )
        completed_stages = (*completed_stages, selection_stage)
        return self._publish(
            configuration,
            video_set,
            video_stage_results,
            resolved_models,
            scene_catalog,
            selection,
            completed_stages,
            vision_diagnostics,
            speech_runtime_identity,
            started_at,
        )

    def _write_selection_stage(
        self,
        configuration: EffectiveConfiguration,
        video_set: VideoSet,
        selection: VideoSetSelectionResult,
        completed_stages: tuple[CompletedStage, ...],
        selection_request_fingerprint: StageFingerprint,
    ) -> CompletedStage:
        """最終selection decisionをCompleted Stageとしてatomicに確定する。"""
        upstream = tuple(stage.fingerprint for stage in completed_stages)
        semantic_input = {
            "selection_request_fingerprint": selection_request_fingerprint.value,
            "requested_count": configuration.image_count,
            "spoiler_sensitivity": configuration.spoiler_sensitivity,
            "similarity_threshold": configuration.similarity_threshold,
            "annotated_candidate_ids": [
                *(item.candidate.identifier for item in selection.selected),
                *(item.candidate.identifier for item in selection.rejected),
            ],
        }
        fingerprint = build_stage_fingerprint(
            ProcessingStage.SELECT_IMAGES,
            upstream,
            semantic_input,
        )
        writer = CompletedStageWriter(
            configuration.processing_cache_folder,
            subject_namespace="video-sets",
            subject_fingerprint=video_set.fingerprint,
        )
        return writer.write(
            ProcessingStage.SELECT_IMAGES,
            fingerprint,
            upstream,
            semantic_input,
            serialize_video_set_selection_result(selection),
        )

    def _record_selection_progress(
        self,
        completed: CompletedStage,
        *,
        reused: bool,
    ) -> None:
        """実際のselector実行有無をSelect Images progressへ記録する。"""
        self._progress.start_stage(
            ProcessingStage.SELECT_IMAGES,
            work_unit_kind="selection",
        )
        self._progress.record_work_sample("reuse" if reused else "recompute")
        self._progress.cache_observed(
            cache_hit_count=1 if reused else 0,
            cache_miss_count=0 if reused else 1,
            reuse_count=1 if reused else 0,
            recompute_count=0 if reused else 1,
            reason_code="cache_reused" if reused else "stage_recomputed",
        )
        self._progress.complete_stage(stage_fingerprint=completed.fingerprint)
        self._observer.stage_completed(completed)

    def _publish(
        self,
        configuration: EffectiveConfiguration,
        video_set: VideoSet,
        video_stage_results: tuple[VideoStageResult, ...],
        resolved_models: ResolvedModels,
        scene_catalog: SceneCatalog | None,
        selection: VideoSetSelectionResult,
        completed_stages: tuple[CompletedStage, ...],
        vision_diagnostics: Mapping[str, VisionInferenceDiagnostics],
        speech_runtime_identity: str,
        started_at: datetime,
    ) -> RunOutcome:
        """Canonical Publication Requestを構築してOutput Folderへ公開する。"""
        with suppress(FileNotFoundError):
            configuration.output_folder.rmdir()
        publication_selection = sanitize_selection_annotations_for_publication(
            selection,
            scene_catalog,
            tuple(
                cue.text
                for stage in video_stage_results
                for cue in stage.context.cues
                if cue.text
            ),
        )
        status = RunStatus.from_selection_counts(
            configuration.image_count,
            len(selection.selected),
            has_other_warnings=bool(resolved_models.unavailable_roles()),
        )
        request = CanonicalPublicationRequest(
            video_set=video_set,
            video_stage_results=video_stage_results,
            scene_catalog=scene_catalog,
            selection_result=publication_selection,
            resolved_models=resolved_models,
            configuration=configuration,
            run_id=_run_id(started_at),
            started_at=started_at,
            completed_at=started_at,
            provenance=build_report_provenance(
                completed_stages,
                self._progress.completed_stage_events,
                configuration,
                vision_diagnostics,
                speech_runtime_identity,
            ),
        )
        CanonicalOutputPublisher(
            self._media_runtime,
            completion_clock=self._clock,
        ).publish(request)
        return RunOutcome(
            output_folder=configuration.output_folder,
            status=status,
            requested_count=configuration.image_count,
            selected_count=len(selection.selected),
            completed_stages=completed_stages,
        )


def _validate_configuration_paths(configuration: EffectiveConfiguration) -> None:
    """Configuration Errorへ変換可能なOutput Folder境界を検証する。"""
    try:
        validate_output_folder(
            configuration.video_input_folder,
            configuration.output_folder,
        )
    except ValueError as error:
        raise ConfigurationError("OUTPUT_FOLDER_INVALID", str(error)) from None


def _annotation_batch_sizes(total: int, requested_count: int) -> tuple[int, ...]:
    """initialと不足時追加の有限batch列を返す。"""
    first = min(total, max(_INITIAL_ANNOTATION_MINIMUM, requested_count * 2))
    sizes = [first]
    remaining = total - first
    while remaining > 0:
        size = min(remaining, requested_count)
        sizes.append(size)
        remaining -= size
    return tuple(sizes)


def _selection_request_fingerprint(
    configuration: EffectiveConfiguration,
    completed_stages: tuple[CompletedStage, ...],
    planned_vision_fingerprints: tuple[StageFingerprint, ...],
    *,
    request_count: int,
) -> StageFingerprint:
    """selector実行前に全依存入力を識別するfingerprintを返す。"""
    semantic_input = {
        "request_contract": "preselection-input-v1",
        "requested_count": configuration.image_count,
        "spoiler_sensitivity": configuration.spoiler_sensitivity,
        "similarity_threshold": configuration.similarity_threshold,
        "annotation_batch_sizes": list(
            _annotation_batch_sizes(request_count, configuration.image_count)
            if request_count
            else ()
        ),
    }
    return build_stage_fingerprint(
        ProcessingStage.SELECT_IMAGES,
        (
            *(stage.fingerprint for stage in completed_stages),
            *planned_vision_fingerprints,
        ),
        semantic_input,
    )


def _restore_annotated_candidates(
    batches: Iterator[tuple[BlogCandidate, ...]],
    expected_count: int,
) -> tuple[BlogCandidate, ...]:
    """cache artifactが使用したbatch境界まで注釈済みcandidateを復元する。"""
    if expected_count < 1:
        raise ValueError("Video Set Selection cacheのcandidate件数が不正です")
    candidates: list[BlogCandidate] = []
    for batch in batches:
        candidates.extend(batch)
        if len(candidates) >= expected_count:
            break
    if len(candidates) != expected_count:
        raise ValueError("Video Set Selection cacheのbatch境界が不正です")
    return tuple(candidates)


def _extraction_fingerprints(
    results: tuple[VideoStageResult, ...],
) -> tuple[StageFingerprint, ...]:
    """Scene Catalogが依存するsource-local extraction fingerprintを返す。"""
    values: dict[str, StageFingerprint] = {}
    for result in results:
        stage = next(
            item
            for item in result.completed_stages
            if item.stage is ProcessingStage.EXTRACT_FRAME_CANDIDATES
        )
        values.setdefault(stage.fingerprint.value, stage.fingerprint)
    return tuple(values.values())


def _unique_completed_stages(
    stages: tuple[CompletedStage, ...],
) -> tuple[CompletedStage, ...]:
    """同じCompleted Stage再利用通知を一つのrun resultへ畳む。"""
    unique: dict[str, CompletedStage] = {}
    for stage in stages:
        unique.setdefault(stage.fingerprint.value, stage)
    return tuple(unique.values())


def _run_id(started_at: datetime) -> str:
    """pathを含まない一意なrun IDを返す。"""
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return f"run_{timestamp}_{uuid4().hex}"


def _utc_now() -> datetime:
    """run lifecycle用のtimezone-aware UTC時刻を返す。"""
    return datetime.now(timezone.utc)
