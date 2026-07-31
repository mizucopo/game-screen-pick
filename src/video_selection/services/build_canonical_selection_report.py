"""確定済みdomain resultからCanonical Selection Reportを構築する。"""

from fractions import Fraction

from ..models.blog_candidate import BlogCandidate
from ..models.canonical_publication_request import CanonicalPublicationRequest
from ..models.context_cue import ContextCue
from ..models.rejected_blog_candidate import RejectedBlogCandidate
from ..models.report_stage_provenance import ReportStageProvenance
from ..models.selected_blog_image import SelectedBlogImage
from ..models.selected_image_artifact import SelectedImageArtifact
from ..models.selection_score import SelectionScore
from ..models.video_set_selection_result import (
    CONDITIONAL_COVERAGE_MINIMUM_REQUEST_COUNT,
)
from ..models.video_stage_result import VideoStageResult
from .build_video_source_ids import build_video_source_ids
from .report_time import (
    display_report_time,
    exact_seconds_string,
    rational_report_value,
    utc_report_datetime,
)

REPORT_SCHEMA_NAME = "game-screen-pick/report"
REPORT_SCHEMA_VERSION = "2.0.0"
SELECTION_POLICY_VERSION = "video-set-selection-v3"
SELECTION_EXPLANATION_RENDERER = "selection-explanation-ja-v1"

_REASON_LABELS = {
    "high_quality": "高い画質",
    "high_explanation_value": "高い説明価値",
    "strong_context_relevance": "強いContext Cue relevance",
    "normal_gameplay_coverage": "normal_gameplay coverage",
    "event_coverage": "event coverage",
    "menu_coverage": "menu coverage",
    "title_first_image_bonus": "最初のtitle bonus",
    "ordinary_combat_minimum_coverage": "通常戦闘の条件付き最低coverage",
    "event_minimum_coverage": "eventの条件付き最低coverage",
    "recurring_gameplay_variant": "recurring gameplayの状態差",
    "low_spoiler_penalty_applied": "low spoiler penalty適用後のutility",
    "medium_spoiler_penalty_applied": "medium spoiler penalty適用後のutility",
    "high_spoiler_penalty_applied": "high spoiler penalty適用後のutility",
    "stable_tie_break": "安定tie-break",
}


def build_canonical_selection_report(
    request: CanonicalPublicationRequest,
    image_artifacts: tuple[SelectedImageArtifact, ...],
) -> dict[str, object]:
    """画像artifactを含むreport@2.0.0 objectを返す。"""
    selection = request.selection_result
    artifacts_by_id = {item.image_id: item for item in image_artifacts}
    selected_ids = {item.candidate.identifier for item in selection.selected}
    if set(artifacts_by_id) != selected_ids:
        msg = "Selected Image artifactとselection resultが一致しません"
        raise ValueError(msg)
    stages_by_fingerprint = {
        item.source.fingerprint: item for item in request.video_stage_results
    }
    source_ids = build_video_source_ids(request.video_set.sources)
    near_misses = _published_near_misses(selection.rejected, selection.requested_count)
    referenced_cue_ids = {
        cue_id
        for candidate in (
            *(item.candidate for item in selection.selected),
            *(item.candidate for item in near_misses),
        )
        for cue_id in candidate.annotation.supporting_context_cue_ids
    }
    cues = tuple(
        cue
        for stage in request.video_stage_results
        for cue in stage.context.cues
        if cue.identifier in referenced_cue_ids
    )
    warnings = _warnings(request)
    selected_records = [
        _selected_record(
            item,
            artifacts_by_id[item.candidate.identifier],
            stages_by_fingerprint,
            source_ids,
            request,
        )
        for item in selection.selected
    ]
    near_miss_records = [
        _near_miss_record(
            item,
            stages_by_fingerprint,
            source_ids,
            request,
        )
        for item in near_misses
    ]
    sources = [
        _video_source_record(index, stage, source_ids[stage.source.fingerprint])
        for index, stage in enumerate(request.video_stage_results, start=1)
    ]
    total_duration = sum(
        (stage.scan.timeline.duration.seconds for stage in request.video_stage_results),
        start=Fraction(0),
    )
    models = {
        item.role.value: item.provenance()
        for item in sorted(
            request.resolved_models.items, key=lambda value: value.role.value
        )
    }
    contracts = dict(request.provenance.contracts)
    contracts["report_schema"] = REPORT_SCHEMA_VERSION
    contracts["video_set_selection_policy"] = SELECTION_POLICY_VERSION
    report: dict[str, object] = {
        "schema": {"name": REPORT_SCHEMA_NAME, "version": REPORT_SCHEMA_VERSION},
        "run": {
            "id": request.run_id,
            "status": "completed_with_warnings" if warnings else "completed",
            "started_at": utc_report_datetime(request.started_at),
            "completed_at": utc_report_datetime(request.completed_at),
            "requested_image_count": selection.requested_count,
            "selected_image_count": len(selection.selected),
            "warnings": warnings,
        },
        "artifacts": _artifact_contract(),
        "video_set": {
            "id": f"vset_{request.video_set.fingerprint[:12]}",
            "fingerprint_algorithm": "ordered-video-sha256-v1",
            "time_contract": {
                "authoritative_value": "offset_seconds_rational",
                "display_format": "unbounded_hours_HH:MM:SS.mmm",
                "display_rounding": "half_up",
                "frame_index": "omitted",
            },
            "source_path_policy": {
                "base": "video_input_folder",
                "separator": "/",
                "parent_segments": "forbidden",
                "absolute_paths": "omitted",
            },
            "duration": {
                "exact_seconds": exact_seconds_string(total_duration),
                "display": display_report_time(total_duration),
            },
            "sources": sources,
        },
        "selection_summary": _selection_summary(request),
        "rejection_summary": {
            "total": len(selection.rejected),
            "by_reason": selection.rejection_counts,
        },
        "selected": selected_records,
        "near_miss_publication": {
            "json_limit_formula": (
                "min(total_rejected, 100, max(20, requested_image_count * 2))"
            ),
            "json_limit_for_this_run": _near_miss_limit(
                len(selection.rejected), selection.requested_count
            ),
            "markdown_limit": 10,
            "coverage": "at_least_one_per_rejection_reason",
            "fill_order": (
                "counterfactual_marginal_utility_desc_then_selection_tie_break"
            ),
        },
        "near_misses": near_miss_records,
        "context_cues": [_context_cue_record(cue, source_ids) for cue in cues],
        "provenance": {
            "selection": {
                "policy_version": SELECTION_POLICY_VERSION,
                "spoiler_sensitivity": request.configuration.spoiler_sensitivity,
                "blog_image_type_target": {
                    "normal_gameplay": 0.70,
                    "event": 0.25,
                    "menu": 0.05,
                },
                "similarity_base": request.configuration.similarity_threshold,
                "similarity_final": selection.final_similarity_ceiling,
            },
            "runtime": dict(request.provenance.runtime),
            "tools": dict(request.provenance.tools),
            "models": models,
            "contracts": contracts,
            "stages": [_stage_record(item) for item in request.provenance.stages],
        },
        "privacy": {
            "absolute_paths": "omitted",
            "model_reasoning_trace": "omitted",
            "raw_model_responses": "omitted",
            "generated_screen_text_quotations": "omitted",
            "raw_context_cue_text": "processing_cache_only",
            "relative_source_paths": "included",
            "environment_variables": "omitted",
            "credentials": "omitted",
            "prompt_bodies": "omitted",
            "stack_traces": "omitted",
        },
    }
    return report


def _artifact_contract() -> dict[str, object]:
    return {
        "report_json": "report.json",
        "report_markdown": "report.md",
        "image_directory": "images",
        "image_contract": {
            "frame_candidate_id_algorithm": "video-fingerprint-video-time-sha256-v1",
            "frame_candidate_id_format": "frm_<64-lowercase-hex>",
            "filename_pattern": (
                "<selection-index-min-width-4>_<scene-slug>_"
                "<frame-digest-prefix-12-or-full>.webp"
            ),
            "format": "webp",
            "encoding": "lossy",
            "quality": 95,
            "color_space": "srgb",
            "metadata": "stripped",
            "size_policy": "source_dimensions",
            "configurable": False,
            "status": "fixed_for_v1",
        },
        "publication_contract": {
            "mode": "atomic_directory_rename",
            "staging": "hidden_sibling_same_filesystem",
            "validated_before_publish": True,
            "non_atomic_fallback": False,
        },
        "projection_contract": {
            "canonical_machine_report": "report.json",
            "markdown_source": "validated_canonical_report_object",
            "cache_or_model_reread": False,
            "json_key_order_significant": False,
            "cross_artifact_mismatch": "fatal",
            "selected_image_rendering": "clickable_inline_original",
            "thumbnail_artifacts": False,
            "alt_text": "selection_index_and_scene_display_name",
        },
    }


def _warnings(request: CanonicalPublicationRequest) -> list[dict[str, object]]:
    selection = request.selection_result
    warnings: list[dict[str, object]] = []
    if selection.shortfall:
        warnings.append(
            {
                "code": "selection_shortfall",
                "message": (
                    f"要求{selection.requested_count}枚に対して"
                    f"{len(selection.selected)}枚を選択しました。"
                ),
                "details": selection.rejection_counts,
            }
        )
    unavailable_roles = [
        role.value for role in request.resolved_models.unavailable_roles()
    ]
    if unavailable_roles:
        warnings.append(
            {
                "code": "model_update_unavailable",
                "message": (
                    "model更新確認を完了できず検証済みlocal artifactを使用しました。"
                ),
                "details": {"roles": unavailable_roles},
            }
        )
    return warnings


def _selection_summary(request: CanonicalPublicationRequest) -> dict[str, object]:
    selection = request.selection_result
    moment_count = sum(
        len(stage.extraction.moments) for stage in request.video_stage_results
    )
    zero_frame_count = sum(
        stage.extraction.zero_frame_moment_count
        for stage in request.video_stage_results
    )
    image_types = {
        name: {
            "target": selection.blog_image_type_targets[name],
            "actual": selection.blog_image_type_actuals[name],
        }
        for name in selection.blog_image_type_targets
    }
    eligible = selection.selection_coverage_eligible_counts
    minimums = selection.selection_coverage_minimums
    actuals = selection.selection_coverage_actuals
    reallocated = selection.selection_coverage_reallocated
    conditional_coverage = {
        "applies": (
            selection.requested_count >= CONDITIONAL_COVERAGE_MINIMUM_REQUEST_COUNT
        ),
        "minimum_requested_image_count": (CONDITIONAL_COVERAGE_MINIMUM_REQUEST_COUNT),
        "facets": {
            facet: {
                "eligible": eligible[facet],
                "minimum": minimums[facet],
                "actual": actuals[facet],
                "reallocated": reallocated[facet],
            }
            for facet in eligible
        },
    }
    return {
        "candidate_moments": moment_count,
        "moments_without_valid_frame": zero_frame_count,
        "candidate_annotations": selection.annotated_candidate_count,
        "candidate_annotation_failures": 0,
        "selected": len(selection.selected),
        "not_selected": len(selection.rejected),
        "final_similarity_ceiling": selection.final_similarity_ceiling,
        "shortlist_expansion_count": selection.shortlist_expansion_count,
        "shortfall": {
            "requested": selection.requested_count,
            "selected": len(selection.selected),
            "all_candidate_moments_exhausted": (
                selection.all_candidate_moments_exhausted
            ),
        },
        "blog_image_type": image_types,
        "conditional_coverage": conditional_coverage,
    }


def _video_source_record(
    order: int,
    stage: VideoStageResult,
    video_id: str,
) -> dict[str, object]:
    timeline = stage.scan.timeline
    duration = timeline.duration.seconds
    duration_pts = _source_pts(timeline.origin_pts, timeline.time_base, duration)
    return {
        "id": video_id,
        "order": order,
        "relative_path": stage.source.relative_path,
        "fingerprint": {
            "algorithm": "sha256-whole-file",
            "value": stage.source.fingerprint,
        },
        "duration": {
            "source_pts": duration_pts,
            "origin_pts": timeline.origin_pts,
            "time_base": rational_report_value(timeline.time_base),
            "exact_seconds": exact_seconds_string(duration),
            "display": display_report_time(duration),
        },
    }


def _selected_record(
    selected: SelectedBlogImage,
    artifact: SelectedImageArtifact,
    stages: dict[str, VideoStageResult],
    source_ids: dict[str, str],
    request: CanonicalPublicationRequest,
) -> dict[str, object]:
    candidate = selected.candidate
    return {
        "image_id": candidate.identifier,
        "selection_index": selected.selection_index,
        "output": {
            "relative_path": artifact.relative_path,
            "sha256": artifact.sha256,
            "width": artifact.width,
            "height": artifact.height,
            "bytes": artifact.size_bytes,
        },
        "source": _candidate_source(candidate, stages, source_ids),
        "classification": _classification(candidate, request),
        "annotation": _annotation(candidate),
        "selection": {
            **_selection_score(selected.score),
            "reason_codes": list(selected.reason_codes),
            "decision_explanation": {
                "renderer": SELECTION_EXPLANATION_RENDERER,
                "text": _selection_explanation(selected.reason_codes),
            },
            "variant_group_id": selected.variant_group_id,
            "tie_break_applied": selected.tie_break_applied,
        },
    }


def _near_miss_record(
    rejected: RejectedBlogCandidate,
    stages: dict[str, VideoStageResult],
    source_ids: dict[str, str],
    request: CanonicalPublicationRequest,
) -> dict[str, object]:
    rejection: dict[str, object] = {"reason_code": rejected.reason_code.value}
    if rejected.blocked_by_image_id is not None:
        rejection["blocked_by_image_id"] = rejected.blocked_by_image_id
    if rejected.nearest_selected_image_id is not None:
        rejection["nearest_selected_image_id"] = rejected.nearest_selected_image_id
    if rejected.similarity is not None:
        rejection["similarity"] = rejected.similarity
    return {
        "image_id": rejected.candidate.identifier,
        "source": _candidate_source(rejected.candidate, stages, source_ids),
        "classification": _classification(rejected.candidate, request),
        "annotation": _annotation(rejected.candidate),
        "counterfactual_selection": _selection_score(rejected.counterfactual_score),
        "rejection": rejection,
        "variant_group_id": rejected.variant_group_id,
    }


def _candidate_source(
    candidate: BlogCandidate,
    stages: dict[str, VideoStageResult],
    source_ids: dict[str, str],
) -> dict[str, object]:
    frame = candidate.annotation.candidate
    fingerprint = frame.video_fingerprint
    if fingerprint is None:
        raise ValueError("Report candidateにVideo Fingerprintがありません")
    stage = stages[fingerprint]
    moment = next(
        item
        for item in stage.extraction.moments
        if item.identifier == candidate.annotation.candidate_moment_id
    )
    video_time = frame.video_time
    if (
        frame.source_pts is None
        or frame.origin_pts is None
        or frame.time_base is None
        or video_time is None
    ):
        raise ValueError("Report candidateにexact Video Timeがありません")
    return {
        "video_id": source_ids[fingerprint],
        "candidate_moment_id": moment.identifier,
        "timeline_segment_id": moment.timeline_segment_id,
        "video_time": _video_time_record(
            frame.source_pts,
            frame.origin_pts,
            frame.time_base,
            video_time,
        ),
        "video_set_progress": float(candidate.video_set_progress),
    }


def _classification(
    candidate: BlogCandidate,
    request: CanonicalPublicationRequest,
) -> dict[str, object]:
    annotation = candidate.annotation
    scene_catalog = request.scene_catalog
    if scene_catalog is None:  # pragma: no cover - publication requestで保証される
        raise AssertionError
    scene = scene_catalog.for_slug(annotation.scene_slug)
    spoiler_evidence: dict[str, str] | None = None
    if annotation.spoiler_risk != "none":
        spoiler_evidence = {
            "source": "candidate_annotation",
            "summary": annotation.spoiler_evidence,
        }
    return {
        "scene_slug": annotation.scene_slug,
        "scene_display_name": scene.display_name,
        "scene_selection_role": candidate.scene_selection_role,
        "blog_image_type": annotation.blog_image_type,
        "explanation_value": annotation.explanation_value,
        "screen_text_kind": annotation.screen_text_kind,
        "spoiler_risk": annotation.spoiler_risk,
        "spoiler_evidence": spoiler_evidence,
    }


def _annotation(candidate: BlogCandidate) -> dict[str, object]:
    annotation = candidate.annotation
    return {
        "summary": annotation.summary,
        "representative_frame_reason": annotation.frame_choice_reason,
        "context_cue_relevance": annotation.context_relevance,
        "supporting_context_cue_ids": list(annotation.supporting_context_cue_ids),
    }


def _selection_score(score: SelectionScore) -> dict[str, object]:
    return {
        "base_utility": score.base_utility,
        "spoiler_penalty": score.spoiler_penalty,
        "coverage_bonus": score.coverage_bonus,
        "temporal_diversity_penalty": score.temporal_diversity_penalty,
        "marginal_utility": score.marginal_utility,
        "similarity_pass": score.similarity_pass,
        "nearest_selected_similarity": score.nearest_selected_similarity,
    }


def _selection_explanation(reason_codes: tuple[str, ...]) -> str:
    if not reason_codes:
        return "Marginal Selection Utilityと安定tie-breakにより選択された。"
    labels = [_REASON_LABELS.get(item, f"`{item}`") for item in reason_codes]
    return "、".join(labels) + "により選択された。"


def _published_near_misses(
    rejected: tuple[RejectedBlogCandidate, ...],
    requested_count: int,
) -> tuple[RejectedBlogCandidate, ...]:
    limit = _near_miss_limit(len(rejected), requested_count)
    representatives: list[RejectedBlogCandidate] = []
    for reason in sorted(
        {item.reason_code for item in rejected}, key=lambda item: item.value
    ):
        representatives.append(
            next(item for item in rejected if item.reason_code is reason)
        )
    represented_ids = {item.candidate.identifier for item in representatives}
    remainder = [
        item for item in rejected if item.candidate.identifier not in represented_ids
    ]
    return tuple((*representatives, *remainder)[:limit])


def _near_miss_limit(total_rejected: int, requested_count: int) -> int:
    return min(total_rejected, 100, max(20, requested_count * 2))


def _context_cue_record(
    cue: ContextCue,
    source_ids: dict[str, str],
) -> dict[str, object]:
    provenance = cue.provenance
    if provenance is None:
        raise ValueError("Report Context Evidenceにprovenanceがありません")
    language_origin = (
        "speech_recognition"
        if provenance.language_source == "speech_recognition"
        else "stream_metadata"
    )
    if cue.timestamp_basis == "source_pts":
        start, end = _source_pts_context_time_records(cue)
    elif cue.timestamp_basis == "asr_sample_grid_estimate":
        start = _sample_time_record(cue.start, provenance.source_time_base)
        end = _sample_time_record(cue.end, provenance.source_time_base)
    else:
        start = _offset_time_record(cue.start)
        end = _offset_time_record(cue.end)
    reliability: dict[str, object] = {"policy": cue.reliability}
    if cue.diagnostics is not None:
        reliability["average_log_probability"] = cue.diagnostics.average_log_probability
    return {
        "id": cue.identifier,
        "video_id": source_ids[cue.video_fingerprint],
        "source_kind": cue.source_kind,
        "stream_index": cue.stream_index,
        "language": {"value": cue.language, "origin": language_origin},
        "timestamp_basis": cue.timestamp_basis,
        "start": start,
        "end": end,
        "reliability": reliability,
        "text_included": False,
    }


def _source_pts_context_time_records(
    cue: ContextCue,
) -> tuple[dict[str, object], dict[str, object]]:
    """Context source stream固有PTSから開始・終了時刻を構築する。"""
    provenance = cue.provenance
    if provenance is None:
        raise ValueError("Report Context Evidenceにprovenanceがありません")
    time_base = provenance.source_time_base
    origin_pts = Fraction(provenance.source_pts) - cue.start / time_base
    end_pts = origin_pts + cue.end / time_base
    if origin_pts.denominator != 1 or end_pts.denominator != 1:
        return _offset_time_record(cue.start), _offset_time_record(cue.end)
    origin = origin_pts.numerator
    return (
        _video_time_record(
            provenance.source_pts,
            origin,
            time_base,
            cue.start,
        ),
        _video_time_record(
            end_pts.numerator,
            origin,
            time_base,
            cue.end,
        ),
    )


def _sample_time_record(
    value: Fraction,
    source_time_base: Fraction,
) -> dict[str, object]:
    sample_rate = Fraction(1, 1) / source_time_base
    sample_index = value * sample_rate
    if sample_rate.denominator == 1 and sample_index.denominator == 1:
        return {
            "sample_index": sample_index.numerator,
            "sample_rate_hz": sample_rate.numerator,
            "exact_seconds": exact_seconds_string(value),
            "display": display_report_time(value),
        }
    return {
        "offset_seconds": rational_report_value(value),
        "display": display_report_time(value),
    }


def _offset_time_record(value: Fraction) -> dict[str, object]:
    return {
        "offset_seconds": rational_report_value(value),
        "display": display_report_time(value),
    }


def _video_time_record(
    source_pts: int,
    origin_pts: int,
    time_base: Fraction,
    video_time: Fraction,
) -> dict[str, object]:
    if (source_pts - origin_pts) * time_base != video_time:
        msg = "source PTSとReport Video Timeが一致しません"
        raise ValueError(msg)
    return {
        "source_pts": source_pts,
        "origin_pts": origin_pts,
        "time_base": rational_report_value(time_base),
        "offset_seconds": rational_report_value(video_time),
        "display": display_report_time(video_time),
    }


def _source_pts(origin_pts: int, time_base: Fraction, value: Fraction) -> int:
    pts = Fraction(origin_pts) + value / time_base
    if pts.denominator != 1:
        msg = "Report Video Timeをsource PTSへlosslessに変換できません"
        raise ValueError(msg)
    return pts.numerator


def _stage_record(stage: ReportStageProvenance) -> dict[str, object]:
    record: dict[str, object] = {
        "name": stage.name,
        "status": "completed",
        "fingerprint": stage.fingerprint,
        "upstream_fingerprints": list(stage.upstream_fingerprints),
        "cache_hits": stage.cache_hits,
        "cache_misses": stage.cache_misses,
        "recomputed_items": stage.recomputed_items,
        "attempt_count": stage.attempt_count,
        "validation_failures": stage.validation_failures,
        "effective_settings": dict(stage.effective_settings),
        "tool_refs": list(stage.tool_refs),
        "model_refs": list(stage.model_refs),
        "contract_refs": list(stage.contract_refs),
        "duration_ms": stage.duration_ms,
    }
    if stage.prompt_eval_tokens is not None:
        record["prompt_eval_tokens"] = stage.prompt_eval_tokens
    if stage.eval_tokens is not None:
        record["eval_tokens"] = stage.eval_tokens
    return record
