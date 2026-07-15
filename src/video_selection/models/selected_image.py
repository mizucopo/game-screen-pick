"""walking skeletonで選ばれた画像。"""

from dataclasses import dataclass

from .candidate_annotation import CandidateAnnotation


@dataclass(frozen=True)
class SelectedImage:
    """出力へ公開するannotationと選定理由。"""

    annotation: CandidateAnnotation
    reason_codes: tuple[str, ...]
