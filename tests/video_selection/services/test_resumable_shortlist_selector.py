"""再開可能なSelection Shortlist selectorのtest。"""

import hashlib
import json
import multiprocessing
from collections.abc import Callable, Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from time import perf_counter, sleep

import pytest

from src.video_selection.models.blog_candidate import BlogCandidate
from src.video_selection.models.checkpoint_operation import CheckpointOperation
from src.video_selection.models.shortlist_selection_frontier import (
    ShortlistSelectionFrontier,
)
from src.video_selection.models.stage_fingerprint import StageFingerprint
from src.video_selection.services.durable_work_unit_cache import DurableWorkUnitCache
from src.video_selection.services.resumable_shortlist_selector import (
    ResumableShortlistSelector,
)
from src.video_selection.services.select_video_set_images import (
    select_video_set_images,
)
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver
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
    observer = RecordingRunObserver()

    # Act
    actual = ResumableShortlistSelector(
        tmp_path,
        video_set_fingerprint="c" * 64,
        observer=observer,
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
    frontier_events = tuple(
        event
        for event in observer.progress_events
        if event.work_unit_kind == "shortlist-selection-frontier"
    )
    assert actual == expected
    assert repaired_artifact["annotated_candidate_count"] == 24
    assert sum(event.cache_hit_count for event in frontier_events) == 2
    assert sum(event.recompute_count for event in frontier_events) == 1


def test_process_kill_resumes_from_committed_frontier(tmp_path: Path) -> None:
    """process kill後も確定済みFrontierから同じ選定結果へ再開されること。

    Arrange:
        - 最初の不足Frontier確定後に待機する別processが用意される
    Act:
        - processが強制終了され同じcacheから選定が再開される
    Assert:
        - 中断なしと同じ選定結果が返されること
        - 確定済みFrontierが再開後も保持されること
    """
    # Arrange
    ready_path = tmp_path / "frontier-ready"
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_run_until_frontier_then_wait,
        args=(tmp_path, ready_path),
    )
    process.start()
    try:
        deadline = perf_counter() + 10
        while not ready_path.is_file() and perf_counter() < deadline:
            sleep(0.01)
        assert ready_path.is_file()
        assert process.is_alive()
        frontier_root = (
            tmp_path / "work-units" / ("e" * 64) / "shortlist-selection-frontier"
        )
        assert len(tuple(frontier_root.glob("*/manifest.json"))) == 2

        # Act
        process.kill()
        process.join(timeout=10)
        candidates = _candidates(40)
        batches = _batches(candidates, requested_count=10)
        actual = ResumableShortlistSelector(
            tmp_path,
            video_set_fingerprint="e" * 64,
        ).select(
            batches,
            selection_request_fingerprint=StageFingerprint("f" * 64),
            candidate_moment_timelines=_candidate_moment_timelines(candidates),
            requested_count=10,
            spoiler_sensitivity="medium",
            similarity_threshold=0.72,
        )
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=10)

    # Assert
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
    assert process.exitcode is not None and process.exitcode != 0
    assert actual == expected
    assert len(tuple(frontier_root.glob("*/manifest.json"))) == len(batches)


def test_7000_cached_candidates_are_selected_once_within_unit_budget(
    tmp_path: Path,
) -> None:
    """7,000件の確認済みFrontierが全件再選定なしで復元されること。

    Arrange:
        - 7,000件の候補と全batch分の検証可能なFrontierが用意される
    Act:
        - cache-onlyのShortlist選定が再開される
    Assert:
        - 全候補を一度選定した結果と一致すること
        - selector部分が30秒以内でFrontierを再計算せず完了すること
    """
    # Arrange
    candidates = _candidates(7_000)
    batches = _batches(candidates, requested_count=10)
    request_fingerprint = StageFingerprint("2" * 64)
    video_set_fingerprint = "1" * 64
    observer = RecordingRunObserver()
    frontier_root = _seed_frontiers(
        tmp_path,
        video_set_fingerprint=video_set_fingerprint,
        request_fingerprint=request_fingerprint,
        batches=batches,
    )
    manifest_mtimes = {
        path: path.stat().st_mtime_ns for path in frontier_root.glob("*/manifest.json")
    }
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
    started = perf_counter()
    actual = ResumableShortlistSelector(
        tmp_path,
        video_set_fingerprint=video_set_fingerprint,
        observer=observer,
    ).select(
        batches,
        selection_request_fingerprint=request_fingerprint,
        candidate_moment_timelines=_candidate_moment_timelines(candidates),
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )
    elapsed = perf_counter() - started

    # Assert
    assert actual == expected
    assert elapsed < 30
    assert {
        path: path.stat().st_mtime_ns for path in manifest_mtimes
    } == manifest_mtimes
    frontier_events = tuple(
        event
        for event in observer.progress_events
        if event.work_unit_kind == "shortlist-selection-frontier"
    )
    assert sum(event.cache_hit_count for event in frontier_events) == len(batches)
    assert sum(event.recompute_count for event in frontier_events) == 0


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


def _run_until_frontier_then_wait(cache_folder: Path, ready_path: Path) -> None:
    """最初のFrontier確定後に親processからkillされるまで待機する。"""
    candidates = _candidates(40)
    batches = _batches(candidates, requested_count=10)

    def blocking_batches() -> Iterator[tuple[BlogCandidate, ...]]:
        yield batches[0]
        yield batches[1]
        ready_path.write_text("ready\n", encoding="utf-8")
        while True:
            sleep(1)

    ResumableShortlistSelector(
        cache_folder,
        video_set_fingerprint="e" * 64,
    ).select(
        blocking_batches(),
        selection_request_fingerprint=StageFingerprint("f" * 64),
        candidate_moment_timelines=_candidate_moment_timelines(candidates),
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )


def _seed_frontiers(
    cache_folder: Path,
    *,
    video_set_fingerprint: str,
    request_fingerprint: StageFingerprint,
    batches: tuple[tuple[BlogCandidate, ...], ...],
) -> Path:
    """全batch境界へ不足Frontierを実cache writerで確定する。"""
    cache = DurableWorkUnitCache(
        cache_folder,
        subject_fingerprint=video_set_fingerprint,
        operation=CheckpointOperation.SHORTLIST_SELECTION_FRONTIER,
    )
    candidate_count = 0
    for batch in batches:
        candidate_count += len(batch)
        frontier = ShortlistSelectionFrontier(
            selection_request_fingerprint=request_fingerprint,
            annotated_candidate_count=candidate_count,
        )
        cache.resolve(
            frontier.work_unit_key,
            frontier.semantic_input,
            _frontier_artifact_producer(frontier),
        )
    return (
        cache_folder
        / "work-units"
        / video_set_fingerprint
        / "shortlist-selection-frontier"
    )


def _frontier_artifact_producer(
    frontier: ShortlistSelectionFrontier,
) -> Callable[[Path], dict[str, object]]:
    """指定Frontierのartifact producerを返す。"""

    def produce(_folder: Path) -> dict[str, object]:
        return frontier.artifact

    return produce
