"""scan signalからdensity制限済みCandidate Momentを発見する。"""

import math
from collections import deque
from fractions import Fraction
from typing import cast

from ..models.candidate_moment import CandidateMoment, MomentEvidence
from ..models.candidate_moment_discovery import CandidateMomentDiscovery
from ..models.heartbeat_proxy import HeartbeatProxy
from ..models.scene_signal import SceneSignal
from ..models.video_timeline import VideoTimeline
from .build_video_entity_id import build_video_entity_id

_MOMENT_ID_ALGORITHM = "candidate-moment-id-v1"


def discover_candidate_moments(
    *,
    video_fingerprint: str,
    timeline: VideoTimeline,
    heartbeats: tuple[HeartbeatProxy, ...],
    scene_signals: tuple[SceneSignal, ...],
    density_per_minute: float,
    refinement_radius_seconds: float,
) -> CandidateMomentDiscovery:
    """各density windowからproxy画質が最も高いanchorを最大1件返す。"""
    if not math.isfinite(density_per_minute) or density_per_minute <= 0:
        msg = "Candidate Moment Densityは正の有限値である必要があります"
        raise ValueError(msg)
    if not math.isfinite(refinement_radius_seconds) or refinement_radius_seconds < 0:
        msg = "Frame Refinement半径は0以上の有限値である必要があります"
        raise ValueError(msg)
    window_width = Fraction(60) / Fraction(str(density_per_minute))
    radius = Fraction(str(refinement_radius_seconds))
    density_cap = math.ceil(timeline.duration.seconds / window_width)
    anchors: dict[Fraction, tuple[int, set[MomentEvidence], float]] = {}
    eligible_heartbeats: list[HeartbeatProxy] = []

    for heartbeat in heartbeats:
        if not heartbeat.eligible or heartbeat.video_time >= timeline.duration.seconds:
            continue
        eligible_heartbeats.append(heartbeat)
        _merge_anchor(
            anchors,
            heartbeat.video_time,
            heartbeat.source_pts,
            "heartbeat",
            heartbeat.quality_score,
        )

    eligible_heartbeats.sort(key=lambda item: (item.video_time, item.source_pts))
    ordered_scenes = sorted(
        (
            scene
            for scene in scene_signals
            if scene.video_time < timeline.duration.seconds
        ),
        key=lambda item: (item.video_time, item.source_pts),
    )
    quality_window: deque[int] = deque()
    left_index = 0
    right_index = 0
    for scene in ordered_scenes:
        lower_bound = scene.video_time - radius
        upper_bound = scene.video_time + radius
        while (
            right_index < len(eligible_heartbeats)
            and eligible_heartbeats[right_index].video_time <= upper_bound
        ):
            while quality_window and (
                eligible_heartbeats[quality_window[-1]].quality_score
                <= eligible_heartbeats[right_index].quality_score
            ):
                quality_window.pop()
            quality_window.append(right_index)
            right_index += 1
        while (
            left_index < right_index
            and eligible_heartbeats[left_index].video_time < lower_bound
        ):
            if quality_window and quality_window[0] == left_index:
                quality_window.popleft()
            left_index += 1
        heartbeat_quality = (
            eligible_heartbeats[quality_window[0]].quality_score
            if quality_window
            else None
        )
        quality_score = heartbeat_quality
        if scene.eligible and (
            quality_score is None or scene.quality_score > quality_score
        ):
            quality_score = scene.quality_score
        if quality_score is None:
            continue
        _merge_anchor(
            anchors,
            scene.video_time,
            scene.source_pts,
            "scene",
            quality_score,
        )

    windows: dict[int, list[tuple[Fraction, int, set[MomentEvidence], float]]] = {}
    for video_time, (source_pts, evidence, quality_score) in anchors.items():
        window_index = int(video_time // window_width)
        windows.setdefault(window_index, []).append(
            (video_time, source_pts, evidence, quality_score)
        )

    moments: list[CandidateMoment] = []
    for window_index in sorted(windows):
        center = window_width * window_index + window_width / 2
        video_time, source_pts, evidence, quality_score = min(
            windows[window_index],
            key=lambda item: (
                -item[3],
                -int("scene" in item[2]),
                abs(item[0] - center),
                item[0],
            ),
        )
        moments.append(
            CandidateMoment(
                identifier=build_video_entity_id(
                    "mom",
                    _MOMENT_ID_ALGORITHM,
                    video_fingerprint,
                    video_time,
                ),
                source_pts=source_pts,
                anchor_time=video_time,
                timeline_segment_id=timeline.segment_at(video_time).identifier,
                evidence=cast(
                    tuple[MomentEvidence, ...],
                    tuple(sorted(evidence)),
                ),
                proxy_quality_score=quality_score,
            )
        )
    return CandidateMomentDiscovery(
        density_cap=density_cap,
        moments=tuple(moments),
    )


def _merge_anchor(
    anchors: dict[Fraction, tuple[int, set[MomentEvidence], float]],
    video_time: Fraction,
    source_pts: int,
    evidence: MomentEvidence,
    quality_score: float,
) -> None:
    current = anchors.get(video_time)
    if current is None:
        anchors[video_time] = (source_pts, {evidence}, quality_score)
        return
    current_source_pts, current_evidence, current_quality = current
    current_evidence.add(evidence)
    anchors[video_time] = (
        min(current_source_pts, source_pts),
        current_evidence,
        max(current_quality, quality_score),
    )
