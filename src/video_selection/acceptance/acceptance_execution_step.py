"""target acceptance suiteの型付き実行step。"""

from dataclasses import dataclass
from typing import Literal

from ..models.effective_configuration import EffectiveConfiguration

AcceptanceExecutionKind = Literal["phase", "comparison"]


@dataclass(frozen=True)
class AcceptanceExecutionStep:
    """Acceptance PhaseとComparison Runのstate境界を一か所に閉じ込める。"""

    kind: AcceptanceExecutionKind
    name: str
    configuration: EffectiveConfiguration

    def __post_init__(self) -> None:
        """kindごとに許されたubiquitous nameだけを受理する。"""
        valid_names = {
            "phase": {"cold", "warm"},
            "comparison": {"fixed3"},
        }
        names = valid_names.get(self.kind)
        if names is None or self.name not in names:
            raise ValueError("Acceptance execution stepが不正です")

    @property
    def records_state_key(self) -> str:
        """完了recordを保存するstate keyを返す。"""
        return "phases" if self.kind == "phase" else "comparison_runs"

    @property
    def active_state_key(self) -> str:
        """未確定runを示すstate keyを返す。"""
        return "active_phase" if self.kind == "phase" else "active_comparison_run"

    @property
    def attempts_state_key(self) -> str:
        """中断attemptを保存するstate keyを返す。"""
        return "phase_attempts" if self.kind == "phase" else "comparison_run_attempts"

    @property
    def attempts_label(self) -> str:
        """state validation error用のprivacy-safe labelを返す。"""
        return "phase attempts" if self.kind == "phase" else "comparison run attempts"

    @property
    def failure_context(self) -> dict[str, object]:
        """失敗した論理単位をphase/comparison別に返す。"""
        key = "phase" if self.kind == "phase" else "comparison_run"
        return {key: self.name}

    @property
    def is_cold_phase(self) -> bool:
        """auto cold Acceptance Phaseかを返す。"""
        return self.kind == "phase" and self.name == "cold"
