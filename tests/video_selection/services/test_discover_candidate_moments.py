"""Candidate Moment discoveryのtest。"""

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock

from src.video_selection.models.heartbeat_proxy import HeartbeatProxy
from src.video_selection.models.scene_signal import SceneSignal
from src.video_selection.models.timeline_segment import TimelineSegment
from src.video_selection.models.video_duration import VideoDuration
from src.video_selection.models.video_timeline import VideoTimeline
from src.video_selection.services.discover_candidate_moments import (
    discover_candidate_moments,
)


def _timeline() -> VideoTimeline:
    return VideoTimeline(
        origin_pts=0,
        time_base=Fraction(1),
        duration=VideoDuration(Fraction(90)),
        segments=(
            TimelineSegment("seg_" + "1" * 64, Fraction(0), Fraction(45)),
            TimelineSegment("seg_" + "2" * 64, Fraction(45), Fraction(90)),
        ),
    )


def _heartbeat(
    second: int,
    quality_score: float,
    *,
    eligible: bool = True,
) -> HeartbeatProxy:
    return HeartbeatProxy(
        source_pts=second,
        video_time=Fraction(second),
        proxy_path=Path(f"heartbeats/{second}.jpg"),
        quality_score=quality_score,
        eligible=eligible,
    )


def test_density_keeps_best_anchor_and_allows_empty_windows() -> None:
    """各density windowで最大1件だけが選ばれ空区間が保持されること。

    Arrange:
        - 90秒timelineにheartbeatとscene signalが配置される
        - 最後の30秒には無効なheartbeatだけが配置される
    Act:
        - 毎分2件、refinement半径2秒でMomentが発見される
    Assert:
        - 最初のwindowではscene画像の高い画質を持つanchorが選ばれること
        - 2番目はheartbeat、最後は0件となること
    """
    # Arrange
    heartbeats = (
        _heartbeat(5, 0.60),
        _heartbeat(10, 0.80),
        _heartbeat(35, 0.70),
        _heartbeat(65, 0.99, eligible=False),
    )
    scenes = (
        SceneSignal(6, Fraction(6), 0.90, True),
        SceneSignal(66, Fraction(66), 1.00, False),
    )

    # Act
    discovery = discover_candidate_moments(
        video_fingerprint="a" * 64,
        timeline=_timeline(),
        heartbeats=heartbeats,
        scene_signals=scenes,
        density_per_minute=2.0,
        refinement_radius_seconds=2.0,
    )

    # Assert
    assert discovery.density_cap == 3
    assert [item.anchor_time for item in discovery.moments] == [
        Fraction(6),
        Fraction(35),
    ]
    assert discovery.moments[0].evidence == ("scene",)
    assert discovery.moments[0].timeline_segment_id == "seg_" + "1" * 64


def test_same_exact_anchor_merges_heartbeat_and_scene_evidence() -> None:
    """同じexact anchorの複数根拠が一つのCandidate Momentへ統合されること。

    Arrange:
        - 同じPTSにheartbeatとscene signalが用意される
    Act:
        - Candidate Momentが発見される
    Assert:
        - 一つのMomentが両方のevidenceを持つこと
        - IDがmom_と64桁SHA-256で構成されること
    """
    # Arrange / Act
    discovery = discover_candidate_moments(
        video_fingerprint="b" * 64,
        timeline=_timeline(),
        heartbeats=(_heartbeat(15, 0.70),),
        scene_signals=(SceneSignal(15, Fraction(15), 0.75, True),),
        density_per_minute=2.0,
        refinement_radius_seconds=1.0,
    )

    # Assert
    assert len(discovery.moments) == 1
    moment = discovery.moments[0]
    assert moment.evidence == ("heartbeat", "scene")
    assert moment.identifier.startswith("mom_")
    assert len(moment.identifier) == 68


def test_scene_quality_lookup_does_not_rescan_all_heartbeats() -> None:
    """scene品質参照でheartbeat全体がsceneごとに再走査されないこと。

    Arrange:
        - timeline順のheartbeatと多数のscene signalが用意される
        - heartbeat iterableの走査回数が記録される
    Act:
        - 各scene周辺の品質を使ってCandidate Momentが発見される
    Assert:
        - heartbeat iterableの全走査が定数回に抑えられること
    """
    # Arrange
    heartbeat_values = tuple(_heartbeat(second, 0.70) for second in range(50))
    scenes = tuple(
        SceneSignal(second, Fraction(second), 0.80, True) for second in range(50)
    )
    heartbeat_iteration_count = 0

    def iterate_heartbeats() -> Iterator[HeartbeatProxy]:
        nonlocal heartbeat_iteration_count
        heartbeat_iteration_count += 1
        return iter(heartbeat_values)

    heartbeats = MagicMock()
    heartbeats.__iter__.side_effect = iterate_heartbeats

    # Act
    discovery = discover_candidate_moments(
        video_fingerprint="c" * 64,
        timeline=_timeline(),
        heartbeats=heartbeats,
        scene_signals=scenes,
        density_per_minute=60.0,
        refinement_radius_seconds=1.0,
    )

    # Assert
    assert discovery.moments
    assert heartbeat_iteration_count <= 2
