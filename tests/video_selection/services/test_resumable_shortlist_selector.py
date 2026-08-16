"""再開可能なSelection Shortlist selectorのtest。"""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from time import perf_counter

import pytest

from src.video_selection.models.blog_candidate import BlogCandidate
from src.video_selection.models.stage_fingerprint import StageFingerprint
from src.video_selection.services.resumable_shortlist_selector import (
    ResumableShortlistSelector,
)
from src.video_selection.services.select_video_set_images import (
    select_video_set_images,
)
from tests.video_selection.services.test_select_video_set_images import (
    _candidate,
    _candidate_moment_timelines,
)


def test_resume_skips_proven_incomplete_boundaries_and_preserves_result(
    tmp_path: Path,
) -> None:
    """確認済み不足境界が再選定されず中断なしと同じ結果が返されること。

    Arrange:
        - 全batchで不足する300件の候補と耐久cacheが用意される
        - 最終batch評価後にprocessが中断された状態が作られる
    Act:
        - 新しいselector instanceで同じbatch列から再開される
    Assert:
        - 中断なしで全候補を一度選定した結果と一致すること
        - 再開時間が全batch境界を選び直す初回の3分の1未満になること
    """
    # Arrange
    candidates = _candidates(300)
    batches = _batches(candidates, requested_count=10)
    timelines = _candidate_moment_timelines(candidates)
    request_fingerprint = StageFingerprint("b" * 64)
    initial_selector = ResumableShortlistSelector(
        tmp_path,
        video_set_fingerprint="a" * 64,
    )

    def interrupted_batches() -> Iterator[tuple[BlogCandidate, ...]]:
        yield from batches
        raise RuntimeError("最終batch後の中断")

    initial_started = perf_counter()
    with pytest.raises(RuntimeError, match="最終batch後の中断"):
        initial_selector.select(
            interrupted_batches(),
            selection_request_fingerprint=request_fingerprint,
            candidate_moment_timelines=timelines,
            requested_count=10,
            spoiler_sensitivity="medium",
            similarity_threshold=0.72,
        )
    initial_elapsed = perf_counter() - initial_started
    single_selection_started = perf_counter()
    uninterrupted = select_video_set_images(
        candidates,
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )
    single_selection_elapsed = perf_counter() - single_selection_started
    expected = replace(
        uninterrupted,
        shortlist_expansion_count=len(batches) - 1,
        all_candidate_moments_exhausted=True,
    )

    # Act
    resumed_started = perf_counter()
    actual = ResumableShortlistSelector(
        tmp_path,
        video_set_fingerprint="a" * 64,
    ).select(
        batches,
        selection_request_fingerprint=request_fingerprint,
        candidate_moment_timelines=timelines,
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )
    resumed_elapsed = perf_counter() - resumed_started

    # Assert
    assert actual == expected
    assert resumed_elapsed < initial_elapsed / 3
    assert resumed_elapsed < single_selection_elapsed * 5


def test_hash_consistent_corrupt_frontier_is_recomputed_locally(
    tmp_path: Path,
) -> None:
    """内容とhashを改変されたFrontierだけが破棄され安全に再計算されること。

    Arrange:
        - 三つの不足batch境界が耐久保存される
        - 最初のartifact件数とmanifest hashが整合した別値へ改変される
    Act:
        - 新しいselector instanceで同じbatch列から再開される
    Assert:
        - 中断なしと同じ選定結果が返されること
        - 改変されたFrontierが正しい件数で再確定されること
    """
    # Arrange
    candidates = _candidates(40)
    batches = _batches(candidates, requested_count=10)
    timelines = _candidate_moment_timelines(candidates)
    request_fingerprint = StageFingerprint("d" * 64)

    def interrupted_batches() -> Iterator[tuple[BlogCandidate, ...]]:
        yield from batches
        raise RuntimeError("最終batch後の中断")

    with pytest.raises(RuntimeError, match="最終batch後の中断"):
        ResumableShortlistSelector(
            tmp_path,
            video_set_fingerprint="c" * 64,
        ).select(
            interrupted_batches(),
            selection_request_fingerprint=request_fingerprint,
            candidate_moment_timelines=timelines,
            requested_count=10,
            spoiler_sensitivity="medium",
            similarity_threshold=0.72,
        )
    checkpoint_folders = tuple(
        (
            tmp_path / "work-units" / ("c" * 64) / "shortlist-selection-frontier"
        ).iterdir()
    )
    first_folder = next(
        folder
        for folder in checkpoint_folders
        if json.loads((folder / "artifact.json").read_text(encoding="utf-8"))[
            "annotated_candidate_count"
        ]
        == 24
    )
    artifact_path = first_folder / "artifact.json"
    manifest_path = first_folder / "manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["annotated_candidate_count"] = 25
    artifact_bytes = (
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    artifact_path.write_bytes(artifact_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_record = next(
        item for item in manifest["artifacts"] if item["path"] == "artifact.json"
    )
    artifact_record["size_bytes"] = len(artifact_bytes)
    artifact_record["sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    expected = replace(
        select_video_set_images(
            candidates,
            requested_count=10,
            spoiler_sensitivity="medium",
            similarity_threshold=0.72,
        ),
        shortlist_expansion_count=len(batches) - 1,
        all_candidate_moments_exhausted=True,
    )

    # Act
    actual = ResumableShortlistSelector(
        tmp_path,
        video_set_fingerprint="c" * 64,
    ).select(
        batches,
        selection_request_fingerprint=request_fingerprint,
        candidate_moment_timelines=timelines,
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    repaired_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert actual == expected
    assert repaired_artifact["annotated_candidate_count"] == 24


def _candidates(count: int) -> tuple[BlogCandidate, ...]:
    """不足が最後まで解消されない相互に識別可能な候補を返す。"""
    return tuple(
        _candidate(
            f"resumable-{index}",
            quality=0.8,
            feature=(
                1.0,
                float(index % 17) / 17.0,
                float(index % 31) / 31.0,
            ),
            progress=Fraction(index + 1, count + 1),
            blog_image_type="normal_gameplay",
            explanation_value="high",
            context_relevance="none",
            candidate_moment_key=f"resumable-moment-{index}",
        )
        for index in range(count)
    )


def _batches(
    candidates: tuple[BlogCandidate, ...],
    *,
    requested_count: int,
) -> tuple[tuple[BlogCandidate, ...], ...]:
    """Applicationと同じ初期24件・以降要求枚数単位のbatchを返す。"""
    return (
        candidates[:24],
        *(
            candidates[offset : offset + requested_count]
            for offset in range(24, len(candidates), requested_count)
        ),
    )
