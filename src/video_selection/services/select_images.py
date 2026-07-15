"""walking skeletonの最小画像選定。"""

from ..models.candidate_annotation import CandidateAnnotation
from ..models.selected_image import SelectedImage


def select_images(
    annotations: tuple[CandidateAnnotation, ...],
    image_count: int,
) -> tuple[SelectedImage, ...]:
    """先頭から要求数を選び、placeholder reason codeを付ける。"""
    return tuple(
        SelectedImage(
            annotation=annotation,
            reason_codes=("walking_skeleton_selected",),
        )
        for annotation in annotations[:image_count]
    )
