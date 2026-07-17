"""Disk acceptance resource monitorのtest。"""

from pathlib import Path

from src.video_selection.acceptance.disk_usage_monitor import DiskUsageMonitor


def test_cache_and_hidden_staging_are_counted_but_final_output_is_excluded(
    tmp_path: Path,
) -> None:
    """persistent cacheとhidden stagingだけがbudget容量へ集計されること。

    Arrange:
        - work cache、hidden staging、大きなfinal outputが用意される
    Act:
        - disk monitorが一回sampleされる
    Assert:
        - persistent cacheとwork/stagingが数えられfinal outputが除外されること
    """
    # Arrange
    working = tmp_path / "work"
    cache = working / "input" / ".game-screen-pick" / "cache"
    cache.mkdir(parents=True)
    (cache / "cache.bin").write_bytes(b"c" * 10)
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
    assert result["peak_additional_bytes"] == 15
