"""Processing Stageを依存順にatomic確定する。"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar

from ..models.completed_stage import CompletedStage
from ..models.processing_stage import ProcessingStage
from ..models.stage_fingerprint import StageFingerprint
from ..protocols.run_observer import RunObserver
from .build_stage_fingerprint import build_stage_fingerprint
from .completed_stage_writer import (
    CacheNamespace,
    CompletedStageWriter,
    FaultInjector,
)

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
    ) -> None:
        self._writer = CompletedStageWriter(
            cache_folder,
            subject_namespace=subject_namespace,
            subject_fingerprint=subject_fingerprint,
            fault_injector=fault_injector,
        )
        self._observer = observer
        self._before_stage = before_stage or _skip_before_stage
        self._completed_stages: list[CompletedStage] = []

    @property
    def completed_stages(self) -> tuple[CompletedStage, ...]:
        """確定済みStageを順番に返す。"""
        return tuple(self._completed_stages)

    def complete(
        self,
        stage: ProcessingStage,
        semantic_input: Mapping[str, object],
        artifact: dict[str, object],
    ) -> CompletedStage:
        """次のStageだけをartifactとmanifestへ確定する。"""
        upstream_fingerprints, fingerprint = self._prepare_stage(stage, semantic_input)
        completed_stage = self._writer.write(
            stage,
            fingerprint,
            upstream_fingerprints,
            semantic_input,
            artifact,
        )
        self._record_completion(completed_stage)
        return completed_stage

    def reuse(
        self,
        stage: ProcessingStage,
        semantic_input: Mapping[str, object],
        restore: Callable[[dict[str, object]], StageResult],
    ) -> StageResult | None:
        """検証済みCompleted Stageがあれば復元して完了扱いにする。"""
        upstream_fingerprints, fingerprint = self._prepare_stage(stage, semantic_input)
        artifact = self._writer.read(
            stage,
            fingerprint,
            upstream_fingerprints,
            semantic_input,
        )
        if artifact is None:
            return None
        restored = restore(artifact)
        self._record_completion(CompletedStage(stage=stage, fingerprint=fingerprint))
        return restored

    def _prepare_stage(
        self,
        stage: ProcessingStage,
        semantic_input: Mapping[str, object],
    ) -> tuple[tuple[StageFingerprint, ...], StageFingerprint]:
        """次Stageを検証して上流とfingerprintを返す。"""
        stages = tuple(ProcessingStage)
        completed_count = len(self._completed_stages)
        if completed_count >= len(stages):
            msg = f"all Processing Stages are completed: actual={stage.value}"
            raise ValueError(msg)
        expected_stage = stages[completed_count]
        if stage is not expected_stage:
            msg = f"expected={expected_stage.value}, actual={stage.value}"
            raise ValueError(msg)
        self._before_stage()
        upstream_fingerprints = tuple(
            item.fingerprint for item in self._completed_stages
        )
        return upstream_fingerprints, build_stage_fingerprint(
            stage,
            upstream_fingerprints,
            semantic_input,
        )

    def _record_completion(self, completed_stage: CompletedStage) -> None:
        """Stage完了をrun stateへ追加して通知する。"""
        self._completed_stages.append(completed_stage)
        self._observer.stage_completed(completed_stage)


def _skip_before_stage() -> None:
    return
