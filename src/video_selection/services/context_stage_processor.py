"""一つのVideo SourceのContext Collection Stage。"""

from dataclasses import replace

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
from .processing_stage_runner import ProcessingStageRunner
from .select_context_audio_stream import select_context_audio_stream
from .select_context_subtitle_stream import select_context_subtitle_stream
from .validate_video_set_snapshot import validate_video_source_snapshot


class ContextStageProcessor:
    """subtitleまたはaudioからsource-local Context Cueを確定する。"""

    def __init__(
        self,
        media_runtime: VideoStageMediaRuntime,
        speech_runtime: SpeechRuntime,
        observer: RunObserver,
    ) -> None:
        self._media_runtime = media_runtime
        self._speech_runtime = speech_runtime
        self._observer = observer

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
            before_stage=lambda: validate_video_source_snapshot(video_set, source),
            stage_order=(ProcessingStage.COLLECT_CONTEXT,),
        )
        cached = runner.reuse(
            ProcessingStage.COLLECT_CONTEXT,
            semantic_input,
            restore_context_stage,
        )
        if cached is not None:
            return replace(cached, completed_stage=runner.completed_stages[0])
        try:
            result = self._collect(
                source,
                scan,
                subtitle_stream,
                audio_stream,
                configuration,
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
        completed = runner.complete(
            ProcessingStage.COLLECT_CONTEXT,
            semantic_input,
            serialize_context_stage(result),
        )
        return replace(result, completed_stage=completed)

    def _collect(
        self,
        source: VideoSource,
        scan: VideoScanResult,
        subtitle_stream: MediaStream | None,
        audio_stream: MediaStream | None,
        configuration: EffectiveConfiguration,
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
            subtitle_cues = tuple(
                _subtitle_cue(source, scan, subtitle_stream, event)
                for event in self._media_runtime.read_embedded_subtitles(
                    source.path,
                    subtitle_stream.index,
                )
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
            configuration=configuration,
        )
        cues.extend(speech_result.cues)
        outcomes.extend(speech_result.outcomes)
        return ContextStageResult(
            cues=tuple(cues),
            outcomes=tuple(outcomes),
            rejected_speech_diagnostics=(speech_result.rejected_speech_diagnostics),
        )


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
