"""単一動画選定の純粋ロジックを検証する."""

from pathlib import Path

from PIL import Image, ImageDraw

from src.models.video_selection import FrameAssessment, FrameCandidate
from src.services.single_video_selector import (
    difference_hash_distance,
    infer_game_title,
    make_timestamps,
    measure_candidate,
    select_final_frames,
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
