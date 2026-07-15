"""Processing Stageを依存順にatomic確定する。"""

from collections.abc import Mapping
from pathlib import Path

from ..models.completed_stage import CompletedStage
from ..models.processing_stage import ProcessingStage
from ..protocols.run_observer import RunObserver
from .build_stage_fingerprint import build_stage_fingerprint
from .completed_stage_writer import CompletedStageWriter


class ProcessingStageRunner:
    """Stage順序、fingerprint、manifest確定、通知を一つに保つ。"""

    def __init__(self, cache_folder: Path, observer: RunObserver) -> None:
        self._writer = CompletedStageWriter(cache_folder)
        self._observer = observer
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
        stages = tuple(ProcessingStage)
        completed_count = len(self._completed_stages)
        if completed_count >= len(stages):
            msg = f"all Processing Stages are completed: actual={stage.value}"
            raise ValueError(msg)
        expected_stage = stages[completed_count]
        if stage is not expected_stage:
            msg = f"expected={expected_stage.value}, actual={stage.value}"
            raise ValueError(msg)

        upstream_fingerprints = tuple(
            item.fingerprint for item in self._completed_stages
        )
        fingerprint = build_stage_fingerprint(
            stage,
            upstream_fingerprints,
            semantic_input,
        )
        completed_stage = self._writer.write(
            stage,
            fingerprint,
            upstream_fingerprints,
            artifact,
        )
        self._completed_stages.append(completed_stage)
        self._observer.stage_completed(completed_stage)
        return completed_stage
