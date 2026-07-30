"""一つのVideo SourceのContext Collection Stage。"""

from dataclasses import replace
from fractions import Fraction

from ..models.context_cue import ContextCue
from ..models.context_cue_provenance import ContextCueProvenance
from ..models.context_source_outcome import (
    ContextSourceOutcome,
    ContextSourceStatus,
)
from ..models.context_stage_error import ContextStageError
from ..models.context_stage_failure_reason import ContextStageFailureReason
from ..models.context_stage_result import ContextStageResult
from ..models.effective_configuration import EffectiveConfiguration
from ..models.embedded_subtitle import EmbeddedSubtitle
from ..models.media_probe import MediaProbe
from ..models.media_runtime_identity import MediaRuntimeIdentity
from ..models.media_stream import MediaStream
from ..models.processing_stage import ProcessingStage
from ..models.video_scan_result import VideoScanResult
from ..models.video_set import VideoSet
from ..models.video_source import VideoSource
from ..protocols.run_observer import RunObserver
from ..protocols.speech_runtime import SpeechRuntime
from ..protocols.video_stage_media_runtime import VideoStageMediaRuntime
from .build_context_cue_equivalence_groups import (
    build_context_cue_equivalence_groups,
)
from .build_context_cue_id import build_context_cue_id
from .build_context_stage_semantic_input import (
    build_context_stage_semantic_input,
)
from .collect_speech_context import collect_speech_context
from .context_stage_artifacts import restore_context_stage, serialize_context_stage
from .embedded_subtitle_checkpoint import EmbeddedSubtitleCheckpoint
from .external_work_monitor import ExternalWorkMonitor
from .pcm_audio_checkpoint import PcmAudioCheckpoint
from .processing_stage_runner import ProcessingStageRunner
from .run_progress_tracker import RunProgressTracker
from .select_context_audio_stream import select_context_audio_stream
from .select_context_subtitle_stream import select_context_subtitle_stream
from .speech_recognition_checkpoint import SpeechRecognitionCheckpoint
from .validate_video_set_snapshot import (
    validate_video_set_snapshot_metadata,
    validate_video_source_snapshot,
)


class ContextStageProcessor:
    """subtitleまたはaudioからsource-local Context Cueを確定する。"""

    def __init__(
        self,
        media_runtime: VideoStageMediaRuntime,
        speech_runtime: SpeechRuntime,
        observer: RunObserver,
        *,
        progress: RunProgressTracker | None = None,
    ) -> None:
        self._media_runtime = media_runtime
        self._speech_runtime = speech_runtime
        self._observer = observer
        self._progress = progress
        self._external_work_monitor = (
            ExternalWorkMonitor(progress) if progress is not None else None
        )

    def process(
        self,
        *,
        video_set: VideoSet,
        source: VideoSource,
        probe: MediaProbe,
        scan: VideoScanResult,
        configuration: EffectiveConfiguration,
        media_runtime_identity: MediaRuntimeIdentity,
    ) -> ContextStageResult:
        """一つのVideo SourceのContext Stageを確定または再利用する。"""
        subtitle_stream = select_context_subtitle_stream(
            probe,
            configuration.language,
            configuration.subtitle_stream_index,
        )
        stt_path_selected = subtitle_stream is None or subtitle_stream.is_forced
        audio_stream = (
            select_context_audio_stream(
                probe,
                configuration.language,
                configuration.audio_stream_index,
            )
            if stt_path_selected
            else None
        )
        use_stt = stt_path_selected and audio_stream is not None
        semantic_input = build_context_stage_semantic_input(
            source,
            scan,
            subtitle_stream,
            audio_stream,
            use_stt,
            configuration,
            media_runtime_identity,
            self._speech_runtime,
        )
        runner = ProcessingStageRunner(
            configuration.processing_cache_folder,
            self._observer,
            subject_namespace="videos",
            subject_fingerprint=source.fingerprint,
            before_stage=lambda: validate_video_set_snapshot_metadata(video_set),
            stage_order=(ProcessingStage.COLLECT_CONTEXT,),
            progress=self._progress,
            video_order=video_set.sources.index(source) + 1,
            video_count=len(video_set.sources),
            video_relative_path=source.relative_path,
            work_unit_kind="video",
        )
        cached = runner.reuse(
            ProcessingStage.COLLECT_CONTEXT,
            semantic_input,
            lambda artifact: restore_context_stage(
                artifact,
                expected_video_fingerprint=source.fingerprint,
                video_duration=scan.timeline.duration.seconds,
            ),
        )
        if cached is not None:
            return replace(cached, completed_stage=runner.completed_stages[0])
        try:
            result = self._collect(
                video_set,
                source,
                probe,
                scan,
                subtitle_stream,
                audio_stream,
                configuration,
                semantic_input,
            )
        except ValueError as error:
            if str(error) != "timestamp_drift":
                raise
            raise ContextStageError(
                ContextStageFailureReason.TIMESTAMP_DRIFT
            ) from error
        result = replace(
            result,
            equivalence_groups=build_context_cue_equivalence_groups(result.cues),
        )
        artifact = serialize_context_stage(result)
        restore_context_stage(
            artifact,
            expected_video_fingerprint=source.fingerprint,
            video_duration=scan.timeline.duration.seconds,
        )
        completed = runner.complete(
            ProcessingStage.COLLECT_CONTEXT,
            semantic_input,
            artifact,
        )
        return replace(result, completed_stage=completed)

    def _collect(
        self,
        video_set: VideoSet,
        source: VideoSource,
        probe: MediaProbe,
        scan: VideoScanResult,
        subtitle_stream: MediaStream | None,
        audio_stream: MediaStream | None,
        configuration: EffectiveConfiguration,
        context_semantic_input: dict[str, object],
    ) -> ContextStageResult:
        cues: list[ContextCue] = []
        outcomes: list[ContextSourceOutcome] = []
        if subtitle_stream is None:
            outcomes.append(
                ContextSourceOutcome(
                    source_kind="embedded_subtitle",
                    stream_index=None,
                    status="absent",
                    reason_code="no_subtitle_stream",
                )
            )
        else:
            subtitle_events = EmbeddedSubtitleCheckpoint(
                configuration.processing_cache_folder,
                source_fingerprint=source.fingerprint,
                stream_index=subtitle_stream.index,
                extraction_semantic_input=_subtitle_extraction_semantic_input(
                    context_semantic_input
                ),
                validate_source=lambda: validate_video_source_snapshot(
                    video_set,
                    source,
                ),
                observer=self._observer,
            ).resolve(
                lambda: self._media_runtime.read_embedded_subtitles(
                    source.path,
                    subtitle_stream.index,
                )
            )
            subtitle_cues = tuple(
                _subtitle_cue(source, scan, subtitle_stream, event)
                for event in subtitle_events
            )
            cues.extend(subtitle_cues)
            subtitle_status: ContextSourceStatus = (
                "available" if subtitle_cues else "no_context"
            )
            subtitle_reason = (
                "context_extracted" if subtitle_cues else "no_subtitle_events"
            )
            outcomes.append(
                ContextSourceOutcome(
                    source_kind="embedded_subtitle",
                    stream_index=subtitle_stream.index,
                    status=subtitle_status,
                    reason_code=subtitle_reason,
                    cue_count=len(subtitle_cues),
                )
            )
            if not subtitle_stream.is_forced:
                return ContextStageResult(cues=tuple(cues), outcomes=tuple(outcomes))
        if audio_stream is None:
            outcomes.append(
                ContextSourceOutcome(
                    source_kind="speech_to_text",
                    stream_index=None,
                    status="absent",
                    reason_code="no_audio_stream",
                )
            )
            return ContextStageResult(cues=tuple(cues), outcomes=tuple(outcomes))
        speech_result = collect_speech_context(
            media_runtime=self._media_runtime,
            speech_runtime=self._speech_runtime,
            source=source,
            scan=scan,
            stream=audio_stream,
            media_origin=_media_origin(probe),
            configuration=configuration,
            external_work_monitor=self._external_work_monitor,
            pcm_checkpoint=PcmAudioCheckpoint(
                configuration.processing_cache_folder,
                source_fingerprint=source.fingerprint,
                stream_index=audio_stream.index,
                sample_rate=16000,
                frame_sample_count=int(
                    (
                        configuration.speech_chunk_seconds
                        - configuration.speech_overlap_seconds
                    )
                    * 16000
                ),
                extraction_semantic_input=_pcm_extraction_semantic_input(
                    context_semantic_input
                ),
                validate_source=lambda: validate_video_source_snapshot(
                    video_set,
                    source,
                ),
                observer=self._observer,
            ),
            recognition_checkpoint=SpeechRecognitionCheckpoint(
                configuration.processing_cache_folder,
                source_fingerprint=source.fingerprint,
                recognition_semantic_input=_speech_recognition_semantic_input(
                    context_semantic_input
                ),
                validate_source=lambda: validate_video_source_snapshot(
                    video_set,
                    source,
                ),
                observer=self._observer,
            ),
        )
        cues.extend(speech_result.cues)
        outcomes.extend(speech_result.outcomes)
        return ContextStageResult(
            cues=tuple(cues),
            outcomes=tuple(outcomes),
            rejected_speech_diagnostics=(speech_result.rejected_speech_diagnostics),
        )


def _pcm_extraction_semantic_input(
    context_semantic_input: dict[str, object],
) -> dict[str, object]:
    """STT modelに依存しないPCM extraction入力だけを選ぶ。"""
    keys = (
        "video_fingerprint",
        "selected_audio_stream",
        "media_runtime_identity",
        "speech_chunk_seconds",
        "speech_overlap_seconds",
    )
    return {key: context_semantic_input[key] for key in keys}


def _subtitle_extraction_semantic_input(
    context_semantic_input: dict[str, object],
) -> dict[str, object]:
    """STTに依存しないsubtitle extraction入力だけを選ぶ。"""
    keys = (
        "subtitle_extraction_version",
        "video_fingerprint",
        "selected_subtitle_stream",
        "media_runtime_identity",
    )
    return {key: context_semantic_input[key] for key in keys}


def _speech_recognition_semantic_input(
    context_semantic_input: dict[str, object],
) -> dict[str, object]:
    """字幕・Cue groupingに依存しないSpeech Recognition入力だけを選ぶ。"""
    keys = (
        "video_fingerprint",
        "language",
        "selected_audio_stream",
        "speech_runtime_identity",
        "resolved_model_identity",
        "speech_device",
        "speech_compute_type",
        "speech_beam_size",
        "speech_vad_filter",
        "speech_chunk_seconds",
        "speech_overlap_seconds",
    )
    return {key: context_semantic_input[key] for key in keys}


def _media_origin(probe: MediaProbe) -> Fraction:
    """全streamのうち最も早いexact開始timestampを返す。"""
    origins = tuple(
        stream.start_pts * stream.time_base
        for stream in probe.streams
        if stream.start_pts is not None and stream.time_base is not None
    )
    if not origins:
        raise ValueError("timestamp_drift")
    return min(origins)


def _subtitle_cue(
    source: VideoSource,
    scan: VideoScanResult,
    stream: MediaStream,
    event: EmbeddedSubtitle,
) -> ContextCue:
    video_origin = scan.timeline.origin_pts * scan.timeline.time_base
    start = event.pts * event.time_base - video_origin
    end = start + event.duration_ts * event.time_base
    if start < 0 or end <= start or end > scan.timeline.duration.seconds:
        msg = "timestamp_drift"
        raise ValueError(msg)
    text = event.text.strip()
    return ContextCue(
        identifier=build_context_cue_id(
            video_fingerprint=source.fingerprint,
            source_kind="embedded_subtitle",
            stream_index=stream.index,
            start=start,
            end=end,
            text=text,
        ),
        video_fingerprint=source.fingerprint,
        source_kind="embedded_subtitle",
        stream_index=stream.index,
        start=start,
        end=end,
        timestamp_basis="source_pts",
        text=text,
        language=stream.language,
        provenance=ContextCueProvenance(
            codec_name=stream.codec_name,
            source_pts=event.pts,
            source_time_base=event.time_base,
            stream_language=stream.language,
            is_default=stream.is_default,
            is_forced=stream.is_forced,
            language_source="stream_metadata",
        ),
    )
