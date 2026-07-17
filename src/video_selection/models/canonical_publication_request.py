"""Canonical Selection Report公開に必要な確定済みdomain input。"""

import re
from dataclasses import dataclass
from datetime import datetime

from .candidate_annotation import candidate_annotation_free_text_is_safe
from .effective_configuration import EffectiveConfiguration
from .report_provenance import ReportProvenance
from .resolved_models import ResolvedModels
from .scene_catalog import SceneCatalog
from .video_set import VideoSet
from .video_set_selection_result import VideoSetSelectionResult
from .video_stage_result import VideoStageResult

_RUN_ID = re.compile(r"run_[0-9A-Za-z][0-9A-Za-z._:-]{0,127}")
_CONTEXT_CUE_ID = re.compile(r"cue_[0-9a-f]{64}")


@dataclass(frozen=True)
class CanonicalPublicationRequest:
    """公開済みartifact以外の全Canonical report sourceをまとめる。"""

    video_set: VideoSet
    video_stage_results: tuple[VideoStageResult, ...]
    scene_catalog: SceneCatalog | None
    selection_result: VideoSetSelectionResult
    resolved_models: ResolvedModels
    configuration: EffectiveConfiguration
    run_id: str
    started_at: datetime
    completed_at: datetime
    provenance: ReportProvenance

    def __post_init__(self) -> None:
        """source、selection、Catalog、Cue、run metadataの整合を検証する。"""
        if (
            _RUN_ID.fullmatch(self.run_id) is None
            or self.started_at.tzinfo is None
            or self.completed_at.tzinfo is None
            or self.completed_at < self.started_at
            or self.configuration.video_input_folder.resolve(strict=False)
            != self.video_set.input_folder.resolve(strict=False)
            or self.configuration.image_count != self.selection_result.requested_count
            or tuple(item.source for item in self.video_stage_results)
            != self.video_set.sources
        ):
            msg = "Canonical Publication RequestのrunまたはVideo Setが不正です"
            raise ValueError(msg)
        selected = self.selection_result.selected
        rejected = self.selection_result.rejected
        if tuple(item.selection_index for item in selected) != tuple(
            range(1, len(selected) + 1)
        ):
            msg = "Selected Imageのselection indexが連続していません"
            raise ValueError(msg)
        selected_ids = tuple(item.candidate.identifier for item in selected)
        rejected_ids = tuple(item.candidate.identifier for item in rejected)
        if (
            len(selected_ids) != len(set(selected_ids))
            or len(rejected_ids) != len(set(rejected_ids))
            or set(selected_ids) & set(rejected_ids)
            or len(selected) + len(rejected)
            != self.selection_result.annotated_candidate_count
        ):
            msg = "Canonical Publication Requestの選定集合が不正です"
            raise ValueError(msg)
        stages_by_fingerprint = {
            item.source.fingerprint: item for item in self.video_stage_results
        }
        cues = tuple(
            cue for stage in self.video_stage_results for cue in stage.context.cues
        )
        cue_ids = tuple(item.identifier for item in cues)
        if len(cue_ids) != len(set(cue_ids)) or not all(
            _stage_context_cues_are_valid(stage) for stage in self.video_stage_results
        ):
            msg = "Canonical Publication RequestのContext Cueが不正です"
            raise ValueError(msg)
        candidates = (
            *(item.candidate for item in selected),
            *(item.candidate for item in rejected),
        )
        if bool(candidates) != (self.scene_catalog is not None):
            msg = "Scene CatalogはAnnotationがあるrunだけで必須です"
            raise ValueError(msg)
        raw_context_texts = tuple(cue.text for cue in cues if cue.text)
        for candidate in candidates:
            frame = candidate.annotation.candidate
            fingerprint = frame.video_fingerprint
            if fingerprint is None or fingerprint not in stages_by_fingerprint:
                msg = "Blog CandidateのVideo Sourceが見つかりません"
                raise ValueError(msg)
            stage = stages_by_fingerprint[fingerprint]
            source_index = self.video_set.sources.index(stage.source)
            moment_ids = {moment.identifier for moment in stage.extraction.moments}
            stage_cue_ids = {cue.identifier for cue in stage.context.cues}
            annotation = candidate.annotation
            scene_catalog = self.scene_catalog
            if scene_catalog is None:  # pragma: no cover - 上で保証される
                raise AssertionError
            scene = scene_catalog.for_slug(annotation.scene_slug)
            if (
                candidate.video_order != source_index
                or annotation.candidate_moment_id not in moment_ids
                or candidate.scene_selection_role != scene.selection_role
                or not set(annotation.supporting_context_cue_ids) <= stage_cue_ids
                or not candidate_annotation_free_text_is_safe(
                    (
                        annotation.summary,
                        annotation.frame_choice_reason or "",
                        annotation.spoiler_evidence,
                    ),
                    raw_context_texts,
                )
            ):
                msg = "Blog Candidateのsource、Moment、Scene、Context Cueが不正です"
                raise ValueError(msg)


def _stage_context_cues_are_valid(stage: VideoStageResult) -> bool:
    """一つのVideo Stageに属する公開対象Cueの基本契約を検証する。"""
    duration = stage.scan.timeline.duration.seconds
    for cue in stage.context.cues:
        provenance = cue.provenance
        if (
            _CONTEXT_CUE_ID.fullmatch(cue.identifier) is None
            or cue.video_fingerprint != stage.source.fingerprint
            or cue.source_kind not in {"embedded_subtitle", "speech_to_text"}
            or cue.timestamp_basis
            not in {"source_pts", "container_text_ms", "asr_sample_grid_estimate"}
            or cue.stream_index < 0
            or cue.start < 0
            or cue.end <= cue.start
            or cue.end > duration
            or not cue.text.strip()
            or cue.reliability != "usable"
            or provenance is None
            or provenance.source_time_base <= 0
            or provenance.language_source
            not in {"stream_metadata", "speech_recognition"}
        ):
            return False
    return True
