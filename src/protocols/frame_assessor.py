"""動画フレーム評価のproviderに依存しない境界."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from ..models.video_selection import FrameAssessment, FrameCandidate


class FrameAssessor(Protocol):
    """画像選定が必要とするmodel情報とframe評価を提供する."""

    host: str
    gpu_evidence: dict[str, dict[str, Any]]

    def fetch_model_metadata(
        self,
        requested_models: set[str],
    ) -> dict[str, dict[str, Any]]:
        """指定modelのmetadataを取得する."""

    def assess(
        self,
        *,
        model: str,
        model_digest: str,
        prompt: str,
        candidates: Sequence[FrameCandidate],
        contact_sheet: Path,
    ) -> list[FrameAssessment]:
        """候補IDに対応する評価を候補順に返す."""
