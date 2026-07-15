"""Video Set選定applicationの実行結果。"""

from dataclasses import dataclass
from pathlib import Path

from .completed_stage import CompletedStage


@dataclass(frozen=True)
class RunOutcome:
    """正常終了した内部Video Set選定の観測可能な結果。"""

    output_folder: Path
    selected_count: int
    completed_stages: tuple[CompletedStage, ...]
