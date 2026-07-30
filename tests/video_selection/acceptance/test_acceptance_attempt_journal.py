"""AcceptanceAttemptJournalのprocess kill回復test。"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.video_selection.acceptance.acceptance_attempt_journal import (
    AcceptanceAttemptJournal,
)


def test_recovery_combines_observed_and_committed_work_units(
    tmp_path: Path,
) -> None:
    """journal末尾前の確定work unitもabandoned attemptへ回収されること。

    Arrange:
        - 1件の観測済みhitとjournal未反映のCompleted manifestが用意される
    Act:
        - active attemptがjournalから回復される
    Assert:
        - hitとmanifest由来recomputeがともにmetricsへ含まれること
        - attempt execution contextが保持されること
    """
    # Arrange
    journal_path = tmp_path / "work" / "active-attempt.json"
    journal = AcceptanceAttemptJournal(journal_path)
    context = {"source_revision": {"commit": "a" * 40, "dirty": False}}
    journal.start(
        attempt_id="attempt-1",
        step_kind="phase",
        step_name="cold",
        started_at_epoch_seconds=1.0,
        execution_context=context,
    )
    observed_fingerprint = "1" * 64
    journal.record_snapshot(
        {
            "cache_hit_count": 1,
            "cache_miss_count": 0,
            "reuse_count": 1,
            "unexpected_recompute_count": 0,
            "stage_durations_seconds": {},
            "completed_stage_counts": {},
        },
        {observed_fingerprint: "reused"},
    )
    committed_fingerprint = "2" * 64
    manifest = (
        tmp_path
        / "cache"
        / "work-units"
        / ("a" * 64)
        / "pcm-audio-chunk"
        / committed_fingerprint
        / "manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "work_unit_fingerprint": committed_fingerprint,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    # Act
    recovered = journal.recover(
        attempt_id="attempt-1",
        step_kind="phase",
        step_name="cold",
        processing_cache_folder=tmp_path / "cache",
    )

    # Assert
    assert recovered is not None
    metrics, recovered_context = recovered
    assert metrics["cache_hit_count"] == 1
    assert metrics["cache_miss_count"] == 1
    assert metrics["reuse_count"] == 1
    assert metrics["unexpected_recompute_count"] == 1
    assert recovered_context == context


def test_out_of_order_snapshots_do_not_regress_durable_metrics(
    tmp_path: Path,
) -> None:
    """並行処理の古いsnapshotが後着してもjournalが後退しないこと。

    Arrange:
        - 2件まで進んだsnapshotと、それ以前の1件だけのsnapshotが用意される
    Act:
        - 新しいsnapshotの後に古いsnapshotがjournalへ記録される
    Assert:
        - 回復metricsは2件の進捗と両checkpoint状態を保持すること
    """
    # Arrange
    journal = AcceptanceAttemptJournal(tmp_path / "work" / "active-attempt.json")
    journal.start(
        attempt_id="attempt-1",
        step_kind="phase",
        step_name="cold",
        started_at_epoch_seconds=1.0,
        execution_context={},
    )
    first = "1" * 64
    second = "2" * 64

    # Act
    journal.record_snapshot(
        {
            "cache_hit_count": 2,
            "cache_miss_count": 0,
            "reuse_count": 2,
            "unexpected_recompute_count": 0,
            "stage_durations_seconds": {"scan-video": 2.0},
            "completed_stage_counts": {"scan-video": 2},
        },
        {first: "reused", second: "reused"},
    )
    journal.record_snapshot(
        {
            "cache_hit_count": 1,
            "cache_miss_count": 0,
            "reuse_count": 1,
            "unexpected_recompute_count": 0,
            "stage_durations_seconds": {"scan-video": 1.0},
            "completed_stage_counts": {"scan-video": 1},
        },
        {first: "reused"},
    )
    recovered = journal.recover(
        attempt_id="attempt-1",
        step_kind="phase",
        step_name="cold",
        processing_cache_folder=tmp_path / "cache",
    )

    # Assert
    assert recovered is not None
    metrics, _context = recovered
    assert metrics["cache_hit_count"] == 2
    assert metrics["reuse_count"] == 2
    assert metrics["stage_durations_seconds"] == {"scan-video": 2.0}
    assert metrics["completed_stage_counts"] == {"scan-video": 2}


def test_recovery_counts_stage_committed_after_miss_snapshot_once(
    tmp_path: Path,
) -> None:
    """miss記録後に確定したStageがmissを重複せずrecomputeへ回収されること。

    Arrange:
        - Stageのmiss開始だけがjournalへ記録される
        - そのStageのCompleted manifestがjournal更新前に確定される
    Act:
        - active attemptが回復される
    Assert:
        - cache missは1件のまま、recomputeだけが1件追加されること
    """
    # Arrange
    journal = AcceptanceAttemptJournal(tmp_path / "work" / "active-attempt.json")
    journal.start(
        attempt_id="attempt-1",
        step_kind="phase",
        step_name="cold",
        started_at_epoch_seconds=1.0,
        execution_context={},
    )
    fingerprint = "3" * 64
    journal.record_snapshot(
        {
            "cache_hit_count": 0,
            "cache_miss_count": 1,
            "reuse_count": 0,
            "unexpected_recompute_count": 0,
            "stage_durations_seconds": {},
            "completed_stage_counts": {},
        },
        {fingerprint: "miss_started"},
    )
    manifest = (
        tmp_path
        / "cache"
        / "videos"
        / ("a" * 64)
        / "scan-video"
        / fingerprint
        / "manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "stage_fingerprint": fingerprint,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    # Act
    recovered = journal.recover(
        attempt_id="attempt-1",
        step_kind="phase",
        step_name="cold",
        processing_cache_folder=tmp_path / "cache",
    )

    # Assert
    assert recovered is not None
    metrics, _context = recovered
    assert metrics["cache_miss_count"] == 1
    assert metrics["unexpected_recompute_count"] == 1


def test_recovery_includes_identity_committed_before_process_loss(
    tmp_path: Path,
) -> None:
    """journal更新直前に確定したVideo Identity SHAが回収されること。

    Arrange:
        - active attemptと完了時刻付きVideo Identity entryが用意される
    Act:
        - identity cacheも対象にattemptが回復される
    Assert:
        - 未観測のSHA計算がmissとrecomputeへ1件ずつ加算されること
    """
    # Arrange
    journal = AcceptanceAttemptJournal(tmp_path / "work" / "active-attempt.json")
    journal.start(
        attempt_id="attempt-1",
        step_kind="phase",
        step_name="cold",
        started_at_epoch_seconds=1.0,
        execution_context={},
    )
    identity_root = tmp_path / "video-identities"
    identity_root.mkdir()
    fingerprint = "4" * 64
    (identity_root / "source.json").write_text(
        json.dumps(
            {
                "work_unit_fingerprint": fingerprint,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    # Act
    recovered = journal.recover(
        attempt_id="attempt-1",
        step_kind="phase",
        step_name="cold",
        processing_cache_folder=tmp_path / "cache",
        video_identity_cache_folder=identity_root,
    )

    # Assert
    assert recovered is not None
    metrics, _context = recovered
    assert metrics["cache_miss_count"] == 1
    assert metrics["unexpected_recompute_count"] == 1


def test_nonfinite_journal_metric_is_rejected(tmp_path: Path) -> None:
    """非有限の途中計測値が再開時に拒否されること。

    Arrange:
        - hash整合したactive journalへNaNのStage時間が記録される
    Act:
        - active attemptがjournalから回復される
    Assert:
        - 非決定的な計測値として明示的に拒否されること
    """
    # Arrange
    journal_path = tmp_path / "work" / "active-attempt.json"
    journal = AcceptanceAttemptJournal(journal_path)
    journal.start(
        attempt_id="attempt-1",
        step_kind="phase",
        step_name="cold",
        started_at_epoch_seconds=1.0,
        execution_context={},
    )
    value = json.loads(journal_path.read_text(encoding="utf-8"))
    value["metrics"]["stage_durations_seconds"] = {"scan-video": float("nan")}
    journal_path.write_text(json.dumps(value), encoding="utf-8")

    # Act
    # Assert
    with pytest.raises(ValueError, match="不正"):
        journal.recover(
            attempt_id="attempt-1",
            step_kind="phase",
            step_name="cold",
            processing_cache_folder=tmp_path / "cache",
        )
