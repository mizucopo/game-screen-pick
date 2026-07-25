"""Completed Stageからcanonical report provenanceを構築する。"""

import json
from collections.abc import Mapping

from ..models.completed_stage import CompletedStage
from ..models.effective_configuration import EffectiveConfiguration
from ..models.processing_stage import ProcessingStage
from ..models.progress_event import ProgressEvent
from ..models.report_provenance import ReportProvenance
from ..models.report_stage_provenance import ReportStageProvenance
from ..models.vision_inference_diagnostics import VisionInferenceDiagnostics


def build_report_provenance(
    completed_stages: tuple[CompletedStage, ...],
    completed_stage_events: tuple[ProgressEvent, ...],
    configuration: EffectiveConfiguration,
    vision_diagnostics: Mapping[str, VisionInferenceDiagnostics],
    speech_runtime_identity: str,
) -> ReportProvenance:
    """実semantic inputと診断をprivacy-safeなStage provenanceへ変換する。"""
    events_by_fingerprint: dict[str, list[ProgressEvent]] = {}
    for event in completed_stage_events:
        if event.stage_fingerprint is None:
            continue
        events_by_fingerprint.setdefault(event.stage_fingerprint, []).append(event)
    counts: dict[ProcessingStage, int] = {}
    stages: list[ReportStageProvenance] = []
    for completed in completed_stages:
        events = events_by_fingerprint.get(completed.fingerprint.value, [])
        if not events or any(event.stage is not completed.stage for event in events):
            raise ValueError("Completed Stageのrun provenanceがありません")
        counts[completed.stage] = counts.get(completed.stage, 0) + 1
        ordinal = counts[completed.stage]
        name = f"{completed.stage.value.replace('-', '_')}_{ordinal:03d}"
        diagnostics = vision_diagnostics.get(completed.fingerprint.value)
        stages.append(
            ReportStageProvenance(
                name=name,
                fingerprint="stg_" + completed.fingerprint.value,
                upstream_fingerprints=tuple(
                    "stg_" + item.value for item in completed.upstream_fingerprints
                ),
                cache_hits=sum(event.cache_hit_count for event in events),
                cache_misses=sum(event.cache_miss_count for event in events),
                recomputed_items=sum(event.recompute_count for event in events),
                attempt_count=(
                    len(events) if diagnostics is None else diagnostics.attempt_count
                ),
                validation_failures=(
                    0
                    if diagnostics is None or diagnostics.validation_code is None
                    else 1
                ),
                effective_settings=_stage_effective_settings(
                    completed,
                    configuration,
                ),
                tool_refs=_stage_tool_refs(completed),
                model_refs=_stage_model_refs(completed),
                contract_refs=_stage_contract_refs(completed),
                duration_ms=round(
                    sum(event.elapsed_seconds or 0.0 for event in events) * 1000
                ),
                prompt_eval_tokens=(
                    None if diagnostics is None else diagnostics.prompt_eval_count
                ),
                eval_tokens=None if diagnostics is None else diagnostics.eval_count,
            )
        )
    return ReportProvenance(
        runtime={
            "application": "video_selection",
            "speech_runtime_identity": speech_runtime_identity,
        },
        tools=_report_tools(completed_stages, speech_runtime_identity),
        contracts=_report_contracts(completed_stages),
        stages=tuple(stages),
    )


def _report_tools(
    completed_stages: tuple[CompletedStage, ...],
    speech_runtime_identity: str,
) -> dict[str, str]:
    """Completed Stage semantic inputから実tool identity registryを返す。"""
    media_identities: set[str] = set()
    ollama_identities: set[str] = set()
    for completed in completed_stages:
        media = completed.semantic_input.get("media_runtime_identity")
        if isinstance(media, Mapping):
            media_identities.add(_canonical_mapping(media))
        model = completed.semantic_input.get("model")
        if isinstance(model, Mapping):
            runtime_identity = model.get("runtime_identity")
            if isinstance(runtime_identity, str):
                ollama_identities.add(runtime_identity)
    tools = {
        "video_selection": "application-v1",
        "speech_to_text": speech_runtime_identity,
    }
    if media_identities:
        tools["ffmpeg"] = _single_identity(media_identities, "FFmpeg")
    if ollama_identities:
        tools["ollama"] = _single_identity(ollama_identities, "Ollama")
    return tools


def _report_contracts(
    completed_stages: tuple[CompletedStage, ...],
) -> dict[str, str]:
    """Stage semantic inputから実行されたcontract versionを登録する。"""
    contracts = {
        "video_set_selection_policy": "video-set-selection-v2",
        "nearby_context_policy": "nearby-context-v1",
    }
    specifications = (
        (ProcessingStage.SCAN_VIDEO, "scan_algorithm", "video_scan"),
        (
            ProcessingStage.EXTRACT_FRAME_CANDIDATES,
            "candidate_extraction_algorithm",
            "frame_candidate_extraction",
        ),
        (ProcessingStage.COLLECT_CONTEXT, "policy_version", "context_collection"),
        (
            ProcessingStage.BUILD_SCENE_CATALOG,
            "stage_contract_version",
            "scene_catalog",
        ),
        (
            ProcessingStage.ANNOTATE_CANDIDATE,
            "stage_contract_version",
            "candidate_annotation",
        ),
    )
    for stage, semantic_key, registry_key in specifications:
        values: set[str] = set()
        for completed in completed_stages:
            if completed.stage is not stage:
                continue
            value = completed.semantic_input.get(semantic_key)
            if isinstance(value, str):
                values.add(value)
        if values:
            contracts[registry_key] = _single_identity(values, registry_key)
    return contracts


def _stage_effective_settings(
    completed: CompletedStage,
    configuration: EffectiveConfiguration,
) -> dict[str, object]:
    """Stageごとの公開可能な実semantic settingだけを返す。"""
    semantic_input = completed.semantic_input
    keys_by_stage = {
        ProcessingStage.SCAN_VIDEO: (
            "decode_backend",
            "heartbeat_interval_seconds",
            "scene_change_threshold",
            "scene_min_interval_seconds",
            "heartbeat_proxy_contract",
            "scan_algorithm",
            "timeline_algorithm",
            "scan_proxy_analysis",
        ),
        ProcessingStage.EXTRACT_FRAME_CANDIDATES: (
            "density_per_minute",
            "refinement_radius_seconds",
            "max_frame_candidates",
            "candidate_extraction_algorithm",
            "neutral_analysis_algorithm",
            "blur_reject_variance_min",
            "content_reject_algorithm",
            "source_local_dedupe_algorithm",
            "entity_id_algorithm",
            "candidate_proxy_contract",
        ),
        ProcessingStage.COLLECT_CONTEXT: (
            "policy_version",
            "subtitle_extraction_version",
            "timeline_contract",
            "language",
            "subtitle_stream_index",
            "audio_stream_index",
            "speech_device",
            "speech_compute_type",
            "speech_beam_size",
            "speech_vad_filter",
            "speech_chunk_seconds",
            "speech_overlap_seconds",
            "word_group_policy",
            "reliability_policy",
        ),
        ProcessingStage.SELECT_IMAGES: (
            "requested_count",
            "spoiler_sensitivity",
            "similarity_threshold",
        ),
    }
    settings = {
        key: semantic_input[key]
        for key in keys_by_stage.get(completed.stage, ())
        if key in semantic_input
    }
    if completed.stage in {
        ProcessingStage.BUILD_SCENE_CATALOG,
        ProcessingStage.ANNOTATE_CANDIDATE,
    }:
        settings.update(
            {
                key: value
                for key, value in semantic_input.items()
                if key.endswith("_version")
            }
        )
        model = semantic_input.get("model")
        if isinstance(model, Mapping) and "num_ctx" in model:
            settings["num_ctx"] = model["num_ctx"]
        generation = semantic_input.get("generation_options")
        if isinstance(generation, Mapping):
            settings["generation_options"] = dict(generation)
    if completed.stage is ProcessingStage.SELECT_IMAGES and not settings:
        settings = {
            "requested_count": configuration.image_count,
            "spoiler_sensitivity": configuration.spoiler_sensitivity,
            "similarity_threshold": configuration.similarity_threshold,
        }
    return settings


def _stage_tool_refs(completed: CompletedStage) -> tuple[str, ...]:
    """Stageが実際に利用するtool registry参照を返す。"""
    if completed.stage in {
        ProcessingStage.SCAN_VIDEO,
        ProcessingStage.EXTRACT_FRAME_CANDIDATES,
    }:
        return ("ffmpeg",)
    if completed.stage is ProcessingStage.COLLECT_CONTEXT:
        return (
            ("ffmpeg", "speech_to_text")
            if "speech_runtime_identity" in completed.semantic_input
            else ("ffmpeg",)
        )
    if completed.stage in {
        ProcessingStage.BUILD_SCENE_CATALOG,
        ProcessingStage.ANNOTATE_CANDIDATE,
    }:
        return ("ollama",)
    return ("video_selection",)


def _stage_model_refs(completed: CompletedStage) -> tuple[str, ...]:
    """Stageが依存するresolved model roleを返す。"""
    if (
        completed.stage is ProcessingStage.COLLECT_CONTEXT
        and "resolved_model_identity" in completed.semantic_input
    ):
        return ("speech_to_text",)
    if completed.stage is ProcessingStage.BUILD_SCENE_CATALOG:
        return ("scene_catalog",)
    if completed.stage is ProcessingStage.ANNOTATE_CANDIDATE:
        return ("candidate_annotation",)
    return ()


def _stage_contract_refs(completed: CompletedStage) -> tuple[str, ...]:
    """Stageが実行したcontract registry参照を返す。"""
    refs = {
        ProcessingStage.SCAN_VIDEO: ("video_scan",),
        ProcessingStage.EXTRACT_FRAME_CANDIDATES: ("frame_candidate_extraction",),
        ProcessingStage.COLLECT_CONTEXT: ("context_collection",),
        ProcessingStage.BUILD_SCENE_CATALOG: ("scene_catalog",),
        ProcessingStage.ANNOTATE_CANDIDATE: (
            "candidate_annotation",
            "nearby_context_policy",
        ),
        ProcessingStage.SELECT_IMAGES: ("video_set_selection_policy",),
    }
    return refs.get(completed.stage, ())


def _canonical_mapping(value: Mapping[object, object]) -> str:
    if not all(isinstance(key, str) for key in value):
        raise ValueError("Tool identity mappingが不正です")
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _single_identity(values: set[str], label: str) -> str:
    if len(values) != 1:
        raise ValueError(f"{label} identityがrun内で一致しません")
    return next(iter(values))
