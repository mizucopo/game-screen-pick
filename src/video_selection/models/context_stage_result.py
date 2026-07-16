"""Completed Context Collection Stageのdomain result。"""

from dataclasses import dataclass

from .completed_stage import CompletedStage
from .context_cue import ContextCue
from .context_cue_equivalence_group import ContextCueEquivalenceGroup
from .context_source_outcome import ContextSourceOutcome
from .rejected_speech_diagnostic import RejectedSpeechDiagnostic


@dataclass(frozen=True)
class ContextStageResult:
    """Context Cueとsource別outcomeを保持する。"""

    cues: tuple[ContextCue, ...]
    outcomes: tuple[ContextSourceOutcome, ...]
    rejected_speech_diagnostics: tuple[RejectedSpeechDiagnostic, ...] = ()
    equivalence_groups: tuple[ContextCueEquivalenceGroup, ...] = ()
    completed_stage: CompletedStage | None = None

    @property
    def annotation_cues(self) -> tuple[ContextCue, ...]:
        """equivalent textを代表Cue一件へ畳んだannotation入力を返す。"""
        representative_ids = {
            group.representative_cue_id for group in self.equivalence_groups
        }
        equivalent_ids = {
            cue_id for group in self.equivalence_groups for cue_id in group.cue_ids
        }
        return tuple(
            cue
            for cue in self.cues
            if cue.identifier not in equivalent_ids
            or cue.identifier in representative_ids
        )
