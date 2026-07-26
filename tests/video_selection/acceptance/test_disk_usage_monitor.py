"""Disk acceptance resource monitorのtest。"""

from pathlib import Path
from threading import Event

from src.video_selection.acceptance.disk_usage_monitor import DiskUsageMonitor


def test_persistent_cache_is_separate_from_peak_additional_storage(
    tmp_path: Path,
) -> None:
    """persistent cacheがtemporary/stagingのpeakへ二重計上されないこと。

    Arrange:
        - persistent cache、temporary work、hidden staging、final outputが用意される
    Act:
        - disk monitorが一回sampleされる
    Assert:
        - cacheは個別計測され、peakにはwork/stagingだけが数えられること
    """
    # Arrange
    working = tmp_path / "work"
    cache = working / "input" / ".game-screen-pick" / "cache"
    cache.mkdir(parents=True)
    (cache / "cache.bin").write_bytes(b"c" * 10)
    (working / "temporary.bin").write_bytes(b"w" * 7)
    outputs = tmp_path / "outputs"
    staging = outputs / ".cold.abc.staging"
    staging.mkdir(parents=True)
    (staging / "temporary.bin").write_bytes(b"t" * 5)
    final = outputs / "cold"
    final.mkdir()
    (final / "excluded.bin").write_bytes(b"x" * 1000)
    monitor = DiskUsageMonitor(
        working_root=working,
        output_parent=outputs,
        cache_folder=cache,
        interval_seconds=100,
    )

    # Act
    monitor.start()
    result = monitor.stop()

    # Assert
    assert result["persistent_cache_bytes"] == 10
    assert result["peak_additional_bytes"] == 12
    assert result["disk_sampling_complete"] is True


def test_blocked_sampler_marks_disk_evidence_incomplete_without_second_sample(
    tmp_path: Path,
) -> None:
    """停止timeoutを超えるsamplerが不完全証拠として返されること。

    Arrange:
        - backgroundのworking tree計測だけが停止までblockされる
    Act:
        - 短いjoin timeoutでdisk monitorが停止される
    Assert:
        - 同時に2回目のworking tree計測を開始せずsampling incompleteになること
    """
    # Arrange
    working = tmp_path / "work"
    cache = working / "cache"
    cache.mkdir(parents=True)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    background_started = Event()
    release_background = Event()
    working_sample_count = 0

    def tree_size(path: Path) -> int:
        nonlocal working_sample_count
        if path == working:
            working_sample_count += 1
            if working_sample_count == 2:
                background_started.set()
                release_background.wait(timeout=1)
        return 1

    monitor = DiskUsageMonitor(
        working_root=working,
        output_parent=outputs,
        cache_folder=cache,
        interval_seconds=0.001,
        join_timeout_seconds=0.01,
        tree_size_probe=tree_size,
        staging_size_probe=lambda _path: 0,
    )
    monitor.start()
    assert background_started.wait(timeout=1)

    # Act
    result = monitor.stop()
    release_background.set()

    # Assert
    assert result["disk_sampling_complete"] is False
    assert working_sample_count == 2


def test_background_sample_error_marks_disk_evidence_incomplete(
    tmp_path: Path,
) -> None:
    """background計測失敗が回復後も不完全証拠として残されること。

    Arrange:
        - 最初のbackground working tree計測だけがOSErrorになる
    Act:
        - error観測後にdisk monitorが停止される
    Assert:
        - samplerが停止して最終sampleに成功してもsampling incompleteになること
        - sample error件数がrecordへ残されること
    """
    # Arrange
    working = tmp_path / "work"
    cache = working / "cache"
    cache.mkdir(parents=True)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    background_failed = Event()
    working_sample_count = 0

    def tree_size(path: Path) -> int:
        nonlocal working_sample_count
        if path == working:
            working_sample_count += 1
            if working_sample_count == 2:
                background_failed.set()
                raise OSError("transient traversal failure")
        return 1

    monitor = DiskUsageMonitor(
        working_root=working,
        output_parent=outputs,
        cache_folder=cache,
        interval_seconds=0.001,
        tree_size_probe=tree_size,
        staging_size_probe=lambda _path: 0,
    )
    monitor.start()
    assert background_failed.wait(timeout=1)

    # Act
    result = monitor.stop()

    # Assert
    assert result["disk_sampling_complete"] is False
    assert result["disk_sample_count"] == 2
    assert result["disk_sample_error_count"] == 1
