"""Processing Stageを依存順にatomic確定する。"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar

from ..models.completed_stage import CompletedStage
from ..models.completed_stage_bundle import CompletedStageBundle
from ..models.processing_stage import VIDEO_SET_STAGE_ORDER, ProcessingStage
from ..models.stage_fingerprint import StageFingerprint
from ..protocols.run_observer import RunObserver
from .build_stage_fingerprint import build_stage_fingerprint
from .completed_stage_writer import (
    ArtifactProducer,
    CacheNamespace,
    CompletedStageWriter,
    FaultInjector,
)
from .run_progress_tracker import RunProgressTracker

StageResult = TypeVar("StageResult")


class ProcessingStageRunner:
    """Stage順序、fingerprint、manifest確定、通知を一つに保つ。"""

    def __init__(
        self,
        cache_folder: Path,
        observer: RunObserver,
        *,
        subject_namespace: CacheNamespace,
        subject_fingerprint: str,
        before_stage: Callable[[], None] | None = None,
        fault_injector: FaultInjector | None = None,
        stage_order: tuple[ProcessingStage, ...] = VIDEO_SET_STAGE_ORDER,
        progress: RunProgressTracker | None = None,
        total_stage_count: int | None = None,
        video_order: int | None = None,
        video_count: int | None = None,
        video_relative_path: str | None = None,
        work_unit_kind: str = "processing_stage",
    ) -> None:
        self._writer = CompletedStageWriter(
            cache_folder,
            subject_namespace=subject_namespace,
            subject_fingerprint=subject_fingerprint,
            fault_injector=fault_injector,
        )
        self._observer = observer
        self._before_stage = before_stage or _skip_before_stage
        self._stage_order = stage_order
        self._completed_stages: list[CompletedStage] = []
        self._progress = progress
        self._total_stage_count = total_stage_count
        self._video_order = video_order
        self._video_count = video_count
        self._video_relative_path = video_relative_path
        self._work_unit_kind = work_unit_kind
        self._progress_stage: ProcessingStage | None = None
        self._cache_miss_observed = False
        if not stage_order or len(stage_order) != len(set(stage_order)):
            msg = "stage_orderには重複のない1件以上のStageが必要です"
            raise ValueError(msg)

    @property
    def completed_stages(self) -> tuple[CompletedStage, ...]:
        """確定済みStageを順番に返す。"""
        return tuple(self._completed_stages)

    def complete(
        self,
        stage: ProcessingStage,
        semantic_input: Mapping[str, object],
        artifact: dict[str, object],
        *,
        upstream_stages: tuple[ProcessingStage, ...] | None = None,
    ) -> CompletedStage:
        """次のStageだけをartifactとmanifestへ確定する。"""
        upstream_fingerprints, fingerprint = self._prepare_stage(
            stage,
            semantic_input,
            upstream_stages,
        )
        completed_stage = self._writer.write(
            stage,
            fingerprint,
            upstream_fingerprints,
            semantic_input,
            artifact,
        )
        self._record_completion(completed_stage, reused=False)
        return completed_stage

    def reuse(
        self,
        stage: ProcessingStage,
        semantic_input: Mapping[str, object],
        restore: Callable[[dict[str, object]], StageResult],
        *,
        upstream_stages: tuple[ProcessingStage, ...] | None = None,
    ) -> StageResult | None:
        """検証済みCompleted Stageがあれば復元して完了扱いにする。"""
        upstream_fingerprints, fingerprint = self._prepare_stage(
            stage,
            semantic_input,
            upstream_stages,
        )
        artifact = self._writer.read(
            stage,
            fingerprint,
            upstream_fingerprints,
            semantic_input,
        )
        if artifact is None:
            self._record_cache_miss()
            return None
        restored = restore(artifact)
        self._record_completion(
            CompletedStage(stage=stage, fingerprint=fingerprint),
            reused=True,
        )
        return restored

    def complete_artifacts(
        self,
        stage: ProcessingStage,
        semantic_input: Mapping[str, object],
        produce_artifacts: ArtifactProducer,
        *,
        upstream_stages: tuple[ProcessingStage, ...] | None = None,
    ) -> CompletedStageBundle:
        """複数artifactを生成して次のStageを確定する。"""
        upstream_fingerprints, fingerprint = self._prepare_stage(
            stage,
            semantic_input,
            upstream_stages,
        )
        completed_stage = self._writer.write_artifacts(
            stage,
            fingerprint,
            upstream_fingerprints,
            semantic_input,
            produce_artifacts,
        )
        bundle = self._writer.read_bundle(
            stage,
            fingerprint,
            upstream_fingerprints,
            semantic_input,
        )
        if bundle is None:
            msg = "確定直後のCompleted Stage artifactを検証できませんでした"
            raise RuntimeError(msg)
        self._record_completion(completed_stage, reused=False)
        return bundle

    def reuse_bundle(
        self,
        stage: ProcessingStage,
        semantic_input: Mapping[str, object],
        *,
        upstream_stages: tuple[ProcessingStage, ...] | None = None,
    ) -> CompletedStageBundle | None:
        """検証済みCompleted Stage bundleがあれば完了扱いにする。"""
        upstream_fingerprints, fingerprint = self._prepare_stage(
            stage,
            semantic_input,
            upstream_stages,
        )
        bundle = self._writer.read_bundle(
            stage,
            fingerprint,
            upstream_fingerprints,
            semantic_input,
        )
        if bundle is None:
            self._record_cache_miss()
            return None
        self._record_completion(
            CompletedStage(stage=stage, fingerprint=fingerprint),
            reused=True,
        )
        return bundle

    def adopt_prepared_bundle(
        self,
        stage: ProcessingStage,
        semantic_input: Mapping[str, object],
        *,
        reused: bool,
        duration_seconds: float,
        upstream_stages: tuple[ProcessingStage, ...] | None = None,
    ) -> CompletedStageBundle:
        """先行確定されたbundleを実際のdispositionと時間で完了扱いにする。"""
        upstream_fingerprints, fingerprint = self._prepare_stage(
            stage,
            semantic_input,
            upstream_stages,
        )
        bundle = self._writer.read_bundle(
            stage,
            fingerprint,
            upstream_fingerprints,
            semantic_input,
        )
        if bundle is None:
            msg = "先行確定されたCompleted Stage artifactを検証できませんでした"
            raise RuntimeError(msg)
        if not reused:
            self._record_cache_miss()
        self._record_completion(
            CompletedStage(stage=stage, fingerprint=fingerprint),
            reused=reused,
            duration_seconds=duration_seconds,
        )
        return bundle

    def _prepare_stage(
        self,
        stage: ProcessingStage,
        semantic_input: Mapping[str, object],
        upstream_stages: tuple[ProcessingStage, ...] | None,
    ) -> tuple[tuple[StageFingerprint, ...], StageFingerprint]:
        """次Stageを検証して上流とfingerprintを返す。"""
        stages = self._stage_order
        completed_count = len(self._completed_stages)
        if completed_count >= len(stages):
            msg = f"all Processing Stages are completed: actual={stage.value}"
            raise ValueError(msg)
        expected_stage = stages[completed_count]
        if stage is not expected_stage:
            msg = f"expected={expected_stage.value}, actual={stage.value}"
            raise ValueError(msg)
        self._before_stage()
        upstream_fingerprints = self._select_upstream_fingerprints(upstream_stages)
        fingerprint = build_stage_fingerprint(
            stage,
            upstream_fingerprints,
            semantic_input,
        )
        self._start_progress_stage(stage)
        return upstream_fingerprints, fingerprint

    def _select_upstream_fingerprints(
        self,
        upstream_stages: tuple[ProcessingStage, ...] | None,
    ) -> tuple[StageFingerprint, ...]:
        """完了済みStageから意味的に依存するfingerprintを順番に返す。"""
        if upstream_stages is None:
            return tuple(item.fingerprint for item in self._completed_stages)
        if len(upstream_stages) != len(set(upstream_stages)):
            msg = "upstream_stagesには重複しないStageが必要です"
            raise ValueError(msg)
        completed_by_stage = {
            item.stage: item.fingerprint for item in self._completed_stages
        }
        missing = [
            stage for stage in upstream_stages if stage not in completed_by_stage
        ]
        if missing:
            msg = f"upstream Stageが未完了です: {missing[0].value}"
            raise ValueError(msg)
        selected = set(upstream_stages)
        return tuple(
            item.fingerprint
            for item in self._completed_stages
            if item.stage in selected
        )

    def _record_completion(
        self,
        completed_stage: CompletedStage,
        *,
        reused: bool,
        duration_seconds: float | None = None,
    ) -> None:
        """Stage完了をrun stateへ追加して通知する。"""
        if self._progress is not None:
            self._progress.record_work_sample(
                "reuse" if reused else "recompute",
                duration_seconds,
            )
            self._progress.cache_observed(
                cache_hit_count=1 if reused else 0,
                cache_miss_count=0 if reused else 1,
                reuse_count=1 if reused else 0,
                recompute_count=0 if reused else 1,
                reason_code="cache_reused" if reused else "stage_recomputed",
            )
            self._progress.complete_stage(duration_seconds)
            self._progress_stage = None
            self._cache_miss_observed = False
        self._completed_stages.append(completed_stage)
        self._observer.stage_completed(completed_stage)

    def _start_progress_stage(self, stage: ProcessingStage) -> None:
        if self._progress is None:
            return
        if self._progress_stage is stage:
            return
        if self._progress_stage is not None:
            msg = "別のProcessing Stageがprogress上でactiveです"
            raise RuntimeError(msg)
        self._progress.start_stage(
            stage,
            stage_count=self._total_stage_count,
            video_order=self._video_order,
            video_count=self._video_count,
            video_relative_path=self._video_relative_path,
            work_unit_kind=self._work_unit_kind,
        )
        self._progress_stage = stage

    def _record_cache_miss(self) -> None:
        if self._progress is None or self._cache_miss_observed:
            return
        self._progress.cache_observed(
            cache_hit_count=0,
            cache_miss_count=1,
            reuse_count=0,
            recompute_count=0,
            reason_code="cache_miss",
        )
        self._cache_miss_observed = True


def _skip_before_stage() -> None:
    return
