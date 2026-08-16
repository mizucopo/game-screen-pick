"""不足確認済みのShortlist境界を飛ばして決定的選定を再開する。"""

from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path

from ..models.blog_candidate import BlogCandidate
from ..models.checkpoint_operation import CheckpointOperation
from ..models.durable_work_unit_bundle import DurableWorkUnitBundle
from ..models.shortlist_selection_frontier import ShortlistSelectionFrontier
from ..models.stage_fingerprint import StageFingerprint
from ..models.video_set_selection_result import VideoSetSelectionResult
from .durable_work_unit_cache import DurableWorkUnitCache
from .select_video_set_images import (
    CandidateMomentTimelines,
    SpoilerSensitivity,
    select_from_shortlist_batches,
)


class ResumableShortlistSelector:
    """選定不能が確定したbatch境界を耐久checkpointから再利用する。"""

    def __init__(
        self,
        cache_folder: Path,
        *,
        video_set_fingerprint: str,
    ) -> None:
        self._checkpoints = DurableWorkUnitCache(
            cache_folder,
            subject_fingerprint=video_set_fingerprint,
            operation=CheckpointOperation.SHORTLIST_SELECTION_FRONTIER,
        )

    def select(
        self,
        batches: Iterable[tuple[BlogCandidate, ...]],
        *,
        selection_request_fingerprint: StageFingerprint,
        candidate_moment_timelines: CandidateMomentTimelines,
        requested_count: int,
        spoiler_sensitivity: SpoilerSensitivity,
        similarity_threshold: float,
    ) -> VideoSetSelectionResult:
        """未確認の境界だけを選定し、中断なしと同じ最終結果を返す。"""
        selected_boundary_indexes: list[int] = []
        result = select_from_shortlist_batches(
            self._unproven_batches(
                batches,
                selection_request_fingerprint=selection_request_fingerprint,
                selected_boundary_indexes=selected_boundary_indexes,
            ),
            candidate_moment_timelines=candidate_moment_timelines,
            requested_count=requested_count,
            spoiler_sensitivity=spoiler_sensitivity,
            similarity_threshold=similarity_threshold,
        )
        if not selected_boundary_indexes:
            return result
        try:
            original_expansion_count = selected_boundary_indexes[
                result.shortlist_expansion_count
            ]
        except IndexError:
            raise AssertionError("Shortlist batch境界の対応が失われました") from None
        return replace(
            result,
            shortlist_expansion_count=original_expansion_count,
        )

    def _unproven_batches(
        self,
        batches: Iterable[tuple[BlogCandidate, ...]],
        *,
        selection_request_fingerprint: StageFingerprint,
        selected_boundary_indexes: list[int],
    ) -> Iterator[tuple[BlogCandidate, ...]]:
        """確認済み不足境界を結合して次の未確認境界だけを公開する。"""
        pending: list[BlogCandidate] = []
        pending_checkpoint_count: int | None = None
        cumulative_count = 0
        last_boundary_index = -1
        for boundary_index, batch in enumerate(batches):
            if pending_checkpoint_count is not None:
                self._record_incomplete_boundary(
                    selection_request_fingerprint,
                    pending_checkpoint_count,
                )
                pending_checkpoint_count = None
            if not batch:
                selected_boundary_indexes.append(boundary_index)
                yield batch
                return
            cumulative_count += len(batch)
            last_boundary_index = boundary_index
            pending.extend(batch)
            frontier = ShortlistSelectionFrontier(
                selection_request_fingerprint=selection_request_fingerprint,
                annotated_candidate_count=cumulative_count,
            )
            if self._is_proven_incomplete(frontier):
                continue
            selected_boundary_indexes.append(boundary_index)
            yield tuple(pending)
            pending.clear()
            pending_checkpoint_count = cumulative_count
        if pending_checkpoint_count is not None:
            self._record_incomplete_boundary(
                selection_request_fingerprint,
                pending_checkpoint_count,
            )
        if pending:
            selected_boundary_indexes.append(last_boundary_index)
            yield tuple(pending)

    def _is_proven_incomplete(
        self,
        frontier: ShortlistSelectionFrontier,
    ) -> bool:
        """完全検証済みの不足checkpointが存在するかを返す。"""
        bundle = self._checkpoints.read(
            frontier.work_unit_key,
            frontier.semantic_input,
        )
        if bundle is None:
            return False
        try:
            _validate_bundle(bundle, frontier)
        except (TypeError, ValueError):
            self._checkpoints.discard(
                frontier.work_unit_key,
                frontier.semantic_input,
            )
            return False
        return True

    def _record_incomplete_boundary(
        self,
        request_fingerprint: StageFingerprint,
        candidate_count: int,
    ) -> None:
        """選定を継続した境界だけをatomicに確定する。"""
        frontier = ShortlistSelectionFrontier(
            selection_request_fingerprint=request_fingerprint,
            annotated_candidate_count=candidate_count,
        )

        def validate(bundle: DurableWorkUnitBundle) -> None:
            _validate_bundle(bundle, frontier)

        self._checkpoints.resolve(
            frontier.work_unit_key,
            frontier.semantic_input,
            lambda _folder: frontier.artifact,
            validate_bundle=validate,
        )


def _validate_bundle(
    bundle: DurableWorkUnitBundle,
    frontier: ShortlistSelectionFrontier,
) -> None:
    if bundle.artifact != frontier.artifact:
        raise ValueError("Shortlist Frontier artifactが不正です")
