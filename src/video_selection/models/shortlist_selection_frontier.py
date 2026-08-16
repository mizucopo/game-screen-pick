"""Shortlist選定を継続した累積Candidate境界。"""

from dataclasses import dataclass

from .stage_fingerprint import StageFingerprint

_ARTIFACT_SCHEMA = "game-screen-pick/shortlist-selection-frontier@1.0.0"


@dataclass(frozen=True)
class ShortlistSelectionFrontier:
    """同じ選定意味入力で不足が確定したCandidate件数を保持する。"""

    selection_request_fingerprint: StageFingerprint
    annotated_candidate_count: int

    def __post_init__(self) -> None:
        """Candidate件数が正のbatch境界であることを検証する。"""
        if self.annotated_candidate_count < 1:
            raise ValueError("Shortlist FrontierのCandidate件数は1以上が必要です")

    @property
    def work_unit_key(self) -> str:
        """累積Candidate件数から安定したWork Unit keyを返す。"""
        return f"annotated-candidate-count-{self.annotated_candidate_count}"

    @property
    def semantic_input(self) -> dict[str, object]:
        """Durable Work Unit fingerprintへ渡す意味入力を返す。"""
        return {
            "selection_request_fingerprint": (self.selection_request_fingerprint.value),
            "annotated_candidate_count": self.annotated_candidate_count,
        }

    @property
    def artifact(self) -> dict[str, object]:
        """選択結果を含まない不足証明artifactを返す。"""
        return {
            "schema": _ARTIFACT_SCHEMA,
            "selection_request_fingerprint": (self.selection_request_fingerprint.value),
            "annotated_candidate_count": self.annotated_candidate_count,
            "selection_can_stop": False,
        }
