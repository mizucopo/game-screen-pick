"""単一動画選定の純粋ロジックを検証する."""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from src.models.video_selection import FrameAssessment, FrameCandidate, VideoMetadata
from src.services.video_selector import (
    allocate_automatic_sample_counts,
    difference_hash_distance,
    infer_game_title,
    make_timestamps,
    measure_candidate,
    select_final_frames,
    select_primary_backfill_candidates,
    select_primary_candidates,
)


def test_infer_game_title_removes_common_episode_suffixes() -> None:
    """Part番号や#番号より前をゲーム名として使うこと."""
    assert (
        infer_game_title(Path("Clair Obscur： Expedition 33 Part12.mp4"))
        == "Clair Obscur： Expedition 33"
    )
    assert (
        infer_game_title(Path("かまいたちの夜×3 #04 エンディング有.mp4"))
        == "かまいたちの夜×3"
    )
    assert infer_game_title(Path("ゲーム本編.mp4")) == "ゲーム本編"


def test_make_timestamps_covers_the_whole_video() -> None:
    """自動sampleが動画のほぼ先頭から末尾までを覆うこと."""
    timestamps = make_timestamps(3600.0, 30, None)

    assert len(timestamps) == 1080
    assert timestamps[0] == 0.5
    assert timestamps[-1] == 3599.5
    assert any(850 < timestamp < 950 for timestamp in timestamps)
    assert any(1750 < timestamp < 1850 for timestamp in timestamps)
    assert any(2650 < timestamp < 2750 for timestamp in timestamps)


def test_make_timestamps_treats_requested_interval_as_a_maximum() -> None:
    """大きい最大間隔でも選択枚数以上のsample位置を確保すること."""
    timestamps = make_timestamps(3600.0, 30, 600.0)

    assert len(timestamps) == 30
    assert (
        max(
            right - left
            for left, right in zip(timestamps[:-1], timestamps[1:], strict=True)
        )
        <= 600.0
    )


def test_make_timestamps_rejects_an_interval_below_supported_floor() -> None:
    """0.25秒未満を黙って広げず、処理前に拒否すること."""
    with pytest.raises(ValueError, match="0.25秒以上"):
        make_timestamps(60.0, 30, 0.1)


def test_make_timestamps_rejects_interval_requiring_too_many_candidates() -> None:
    """候補上限によって明示した最大間隔を広げないこと."""
    with pytest.raises(ValueError, match="4,000件"):
        make_timestamps(3600.0, 30, 0.25)


def test_make_timestamps_accepts_exact_explicit_interval_candidate_limit() -> None:
    """浮動小数点誤差で4,000件ちょうどの候補を拒否しないこと."""
    timestamps = make_timestamps(1200.7, 30, 0.3)

    assert len(timestamps) == 4_000


def test_make_timestamps_adapts_endpoint_margin_for_short_video() -> None:
    """短い動画でも可能な選択枚数を固定余白で失わないこと."""
    timestamps = make_timestamps(8.0, 30, None)

    assert len(timestamps) == 30
    assert timestamps[0] < 0.5
    assert timestamps[-1] > 7.5
    assert (
        min(
            right - left
            for left, right in zip(timestamps[:-1], timestamps[1:], strict=True)
        )
        >= 0.25 - 1e-6
    )


def test_make_timestamps_stays_before_low_frame_rate_stream_end() -> None:
    """末尾sampleを最後のframe時刻より後へ置かないこと."""
    timestamps = make_timestamps(
        8.0,
        8,
        None,
        minimum_end_margin_seconds=1.0,
    )

    assert timestamps[-1] <= 7.0


def test_make_timestamps_stays_before_actual_last_vfr_frame() -> None:
    """平均frame rateでは推定できないVFRの最終frame位置を上限にすること."""
    timestamps = make_timestamps(
        8.0,
        8,
        None,
        minimum_end_margin_seconds=1 / 30,
        last_frame_timestamp_seconds=6.5,
    )

    assert timestamps[-1] <= 6.5


def test_make_timestamps_offsets_a_delayed_video_stream() -> None:
    """video streamが遅れて始まる場合も先頭から末尾までを覆うこと."""
    timestamps = make_timestamps(
        3.0,
        3,
        None,
        minimum_end_margin_seconds=1.0,
        start_time_seconds=5.0,
    )

    assert timestamps[0] == 5.5
    assert timestamps[-1] == 7.0
    assert len(timestamps) == 7


def test_make_timestamps_keeps_exact_minimum_interval_after_rounding() -> None:
    """浮動小数点誤差で配置可能な最小間隔のsampleを失わないこと."""
    timestamps = make_timestamps(0.45, 2, None)

    assert timestamps == (0.1, 0.35)


def test_automatic_sample_budget_is_allocated_across_all_videos() -> None:
    """自動sample数を各動画で増幅せず全入力の時間へ配分すること."""
    metadata = [VideoMetadata(3600.0, 320, 180, "fake", "30/1")] * 4

    counts = allocate_automatic_sample_counts(metadata, output_count=30)

    assert counts == (361, 361, 361, 361)
    assert sum(counts) <= 4_000
    assert all(
        len(
            make_timestamps(
                item.duration_seconds,
                1,
                None,
                automatic_sample_count=count,
            )
        )
        == count
        for item, count in zip(metadata, counts, strict=True)
    )


def test_automatic_sample_budget_caps_long_combined_inputs() -> None:
    """既定間隔の合計が上限を超えても自動modeは4,000件へ配分すること."""
    metadata = [VideoMetadata(3600.0, 320, 180, "fake", "30/1")] * 12

    counts = allocate_automatic_sample_counts(metadata, output_count=30)

    assert sum(counts) == 4_000
    assert all(count > 0 for count in counts)


def test_primary_shortlist_stays_bounded_with_more_sources_than_slots() -> None:
    """入力本数が多くても一次候補を出力枚数の12倍以内へ保つこと."""
    metadata = [VideoMetadata(4.0, 320, 180, "fake", "30/1")] * 100
    candidates = [
        FrameCandidate(
            frame_id=f"f{index:05d}",
            timestamp_seconds=1.0,
            path="",
            quality_score=200.0 - index,
            difference_hash=index,
            video_index=index,
        )
        for index in range(100)
    ]

    selected = select_primary_candidates(candidates, metadata, output_count=1)

    assert len(selected) == 12
    assert len({candidate.video_index for candidate in selected}) == 12


def test_primary_shortlist_keeps_fallbacks_for_each_representable_source() -> None:
    """全入力を出力可能なら各入力から一次評価候補を複数残すこと."""
    metadata = [VideoMetadata(8.0, 320, 180, "fake", "30/1")] * 2
    candidates = [
        FrameCandidate(
            frame_id=f"v{video_index}-f{candidate_index}",
            timestamp_seconds=float(candidate_index),
            path="",
            quality_score=100.0 - candidate_index,
            difference_hash=(video_index + 1) << (candidate_index * 8),
            video_index=video_index,
        )
        for video_index in range(2)
        for candidate_index in range(1, 4)
    ]

    selected = select_primary_candidates(candidates, metadata, output_count=2)

    assert len([item for item in selected if item.video_index == 0]) >= 2
    assert len([item for item in selected if item.video_index == 1]) >= 2


def test_primary_backfill_uses_next_candidate_after_reservations_fail() -> None:
    """予約候補が全滅した入力元から未評価の次候補を補充すること."""
    source_candidates = [
        FrameCandidate(
            frame_id=f"v1-f{index}",
            timestamp_seconds=float(index),
            path="",
            quality_score=100.0 - index,
            difference_hash=1 << (index * 8),
            video_index=0,
        )
        for index in range(1, 5)
    ]
    other_candidate = FrameCandidate(
        frame_id="v2-f1",
        timestamp_seconds=1.0,
        path="",
        quality_score=90.0,
        difference_hash=2,
        video_index=1,
    )
    assessed = [*source_candidates[:3], other_candidate]
    assessments = {
        candidate.frame_id: FrameAssessment(
            candidate.frame_id,
            80.0,
            candidate.video_index == 0,
            "探索",
            "test",
        )
        for candidate in assessed
    }

    backfill = select_primary_backfill_candidates(
        [*source_candidates, other_candidate],
        assessed,
        assessments,
        source_count=2,
        output_count=2,
    )

    assert [candidate.frame_id for candidate in backfill] == ["v1-f4"]


def test_measure_candidate_rejects_black_and_scores_visible_frame(
    tmp_path: Path,
) -> None:
    """暗転を落とし、情報量のあるframeへ品質とhashを付けること."""
    black_path = tmp_path / "black.jpg"
    Image.new("RGB", (320, 180), "black").save(black_path)
    visible_path = tmp_path / "visible.jpg"
    visible = Image.new("RGB", (320, 180), "navy")
    draw = ImageDraw.Draw(visible)
    draw.rectangle((20, 20, 300, 160), fill="white")
    draw.rectangle((80, 60, 240, 120), fill="red")
    visible.save(visible_path)

    assert measure_candidate(FrameCandidate("black", 1.0, str(black_path))) is None
    measured = measure_candidate(FrameCandidate("visible", 2.0, str(visible_path)))

    assert measured is not None
    assert measured.quality_score > 0
    assert measured.difference_hash != 0


def test_final_selection_excludes_transitions_and_avoids_soft_cap_bias() -> None:
    """遷移を除き、titleの過剰選択と近い重複を避けること."""
    candidates = [
        FrameCandidate("title-1", 10.0, "", difference_hash=0x0000000000000000),
        FrameCandidate("title-2", 20.0, "", difference_hash=0xFFFFFFFFFFFFFFFF),
        FrameCandidate("field-1", 100.0, "", difference_hash=0x00FF00FF00FF00FF),
        FrameCandidate(
            "field-duplicate",
            110.0,
            "",
            difference_hash=0x00FF00FF00FF00FF,
        ),
        FrameCandidate("battle", 200.0, "", difference_hash=0x0F0F0F0F0F0F0F0F),
        FrameCandidate("transition", 300.0, "", difference_hash=0xAAAAAAAAAAAAAAAA),
    ]
    scenes = {
        "title-1": "タイトル画面",
        "title-2": "タイトル画面",
        "field-1": "フィールド探索",
        "field-duplicate": "フィールド探索",
        "battle": "戦闘",
        "transition": "探索",
    }
    primary = {
        candidate.frame_id: FrameAssessment(
            candidate.frame_id,
            100.0 - index,
            candidate.frame_id == "transition",
            scenes[candidate.frame_id],
            "primary",
        )
        for index, candidate in enumerate(candidates)
    }
    secondary = {
        candidate.frame_id: FrameAssessment(
            candidate.frame_id,
            100.0 - index,
            candidate.frame_id == "transition",
            scenes[candidate.frame_id],
            "secondary",
        )
        for index, candidate in enumerate(candidates)
    }

    selected = select_final_frames(candidates, primary, secondary, 3)

    selected_ids = {item.candidate.frame_id for item in selected}
    assert "transition" not in selected_ids
    assert len(selected_ids & {"title-1", "title-2"}) <= 1
    assert len(selected_ids & {"field-1", "field-duplicate"}) <= 1
    selected_hashes = [item.candidate.difference_hash for item in selected]
    assert all(
        difference_hash_distance(left, right) >= 5
        for index, left in enumerate(selected_hashes)
        for right in selected_hashes[index + 1 :]
    )
