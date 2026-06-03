"""Ollama分類後の候補採点結果."""

from dataclasses import dataclass

from .scene_catalog_entry import SceneCatalogEntry
from .scored_candidate import ScoredCandidate


@dataclass(frozen=True)
class ScoredSceneCandidates:
    """scene分類と候補採点で得られる中間結果."""

    candidates: list[ScoredCandidate]
    resolved_profile: str
    scene_distribution: dict[str, int]
    scene_catalog: list[SceneCatalogEntry]
    classification_failed: int
    classification_failure_rate: float
