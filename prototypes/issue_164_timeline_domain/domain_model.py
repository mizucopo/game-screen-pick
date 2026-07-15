"""Timeline domain model prototype用の純粋関数。"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import Any

Payload = dict[str, Any]


def stable_id(prefix: str, *parts: object) -> str:
    """domain値から短い安定IDを返す。"""
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{prefix}-{digest}"


def video_source(
    fingerprint: str,
    *,
    origin_pts: int,
    time_base: tuple[int, int],
    end_pts: int,
) -> Payload:
    """prototype用Video Sourceを返す。"""
    return {
        "id": fingerprint,
        "video_fingerprint": fingerprint,
        "origin_pts": origin_pts,
        "time_base": time_base,
        "duration": video_time(fingerprint, origin_pts, time_base, end_pts),
    }


def video_time(
    fingerprint: str,
    origin_pts: int,
    time_base: tuple[int, int],
    pts: int,
) -> Payload:
    """PTSとtime baseから正確なVideo Timeを返す。"""
    numerator, denominator = time_base
    value = Fraction((pts - origin_pts) * numerator, denominator)
    return {
        "video_source_id": fingerprint,
        "pts": pts,
        "time_base": f"{numerator}/{denominator}",
        "exact": f"{value.numerator}/{value.denominator}",
        "seconds": round(float(value), 6),
    }


def timeline_segments(source: Payload, boundary_pts: tuple[int, ...]) -> list[Payload]:
    """gapも重複もない半開区間を返す。"""
    return [
        {
            "id": stable_id(
                "segment",
                source["id"],
                start_pts,
                end_pts,
            ),
            "video_source_id": source["id"],
            "interval": "[start, end)",
            "start": time_at(source, start_pts),
            "end": time_at(source, end_pts),
        }
        for start_pts, end_pts in zip(
            boundary_pts,
            boundary_pts[1:],
            strict=False,
        )
    ]


def time_at(source: Payload, pts: int) -> Payload:
    """Video Source上のVideo Timeを返す。"""
    return video_time(
        source["id"],
        source["origin_pts"],
        source["time_base"],
        pts,
    )


def segment_id_at(segments: list[Payload], time: Payload) -> str:
    """半開区間規則でVideo Timeが属するsegment IDを返す。"""
    target = Fraction(time["exact"])
    for segment in segments:
        start = Fraction(segment["start"]["exact"])
        end = Fraction(segment["end"]["exact"])
        if start <= target < end:
            return str(segment["id"])
    raise ValueError("Video TimeがTimeline Segmentに属していません")


def upsert_candidate_moment(
    moments: list[Payload],
    source: Payload,
    segments: list[Payload],
    *,
    anchor_pts: int,
    evidence: str,
) -> list[Payload]:
    """同じanchorの根拠を一つのCandidate Momentへ統合する。"""
    anchor = time_at(source, anchor_pts)
    moment_id = stable_id(
        "moment",
        source["id"],
        anchor["pts"],
        anchor["time_base"],
    )
    updated: list[Payload] = []
    matched = False
    for moment in moments:
        if moment["id"] != moment_id:
            updated.append(moment)
            continue
        updated.append(
            {
                **moment,
                "evidence": sorted({*moment["evidence"], evidence}),
            }
        )
        matched = True
    if matched:
        return updated
    return [
        *updated,
        {
            "id": moment_id,
            "video_source_id": source["id"],
            "timeline_segment_id": segment_id_at(segments, anchor),
            "anchor": anchor,
            "evidence": [evidence],
            "frame_candidate_ids": [],
        },
    ]


def frame_candidate(
    source: Payload,
    segments: list[Payload],
    *,
    pts: int,
) -> Payload:
    """source frameに対してVideo Source内で一意な候補を返す。"""
    position = time_at(source, pts)
    return {
        "id": stable_id(
            "frame",
            source["id"],
            position["pts"],
            position["time_base"],
        ),
        "video_source_id": source["id"],
        "timeline_segment_id": segment_id_at(segments, position),
        "position": position,
    }


def attach_frame(
    moments: list[Payload],
    *,
    moment_id: str,
    frame_id: str,
) -> list[Payload]:
    """Candidate Momentから共有Frame Candidateへの参照を返す。"""
    return [
        (
            {
                **moment,
                "frame_candidate_ids": [
                    *moment["frame_candidate_ids"],
                    frame_id,
                ],
            }
            if moment["id"] == moment_id
            and frame_id not in moment["frame_candidate_ids"]
            else moment
        )
        for moment in moments
    ]


def candidate_moment_cap(source: Payload, density_per_minute: Fraction) -> int:
    """Video Durationと密度からCandidate Moment上限を返す。"""
    duration_seconds = Fraction(source["duration"]["exact"])
    exact_cap = duration_seconds * density_per_minute / 60
    return (exact_cap.numerator + exact_cap.denominator - 1) // exact_cap.denominator


def prototype_cases() -> list[Payload]:
    """人間が関係を確認する具体caseを返す。"""
    return [
        boundary_and_shared_frame_case(),
        vfr_case(),
        cross_video_identity_case(),
        shared_scene_catalog_case(),
        duration_normalized_candidate_cap_case(),
        video_set_stage_boundary_case(),
    ]


def boundary_and_shared_frame_case() -> Payload:
    """segment境界と共有frameのcaseを返す。"""
    source = video_source(
        "video-a-fingerprint",
        origin_pts=900_000,
        time_base=(1, 90_000),
        end_pts=2_700_000,
    )
    segments = timeline_segments(source, (900_000, 1_800_000, 2_700_000))
    moments: list[Payload] = []
    moments = upsert_candidate_moment(
        moments,
        source,
        segments,
        anchor_pts=1_800_000,
        evidence="heartbeat",
    )
    moments = upsert_candidate_moment(
        moments,
        source,
        segments,
        anchor_pts=1_800_000,
        evidence="scene_signal",
    )
    moments = upsert_candidate_moment(
        moments,
        source,
        segments,
        anchor_pts=1_845_000,
        evidence="subtitle_context",
    )
    frames = [
        frame_candidate(source, segments, pts=1_777_500),
        frame_candidate(source, segments, pts=1_822_500),
    ]
    shared_frame_id = frames[1]["id"]
    moments = attach_frame(
        moments,
        moment_id=moments[0]["id"],
        frame_id=frames[0]["id"],
    )
    moments = attach_frame(
        moments,
        moment_id=moments[0]["id"],
        frame_id=shared_frame_id,
    )
    moments = attach_frame(
        moments,
        moment_id=moments[1]["id"],
        frame_id=shared_frame_id,
    )
    return {
        "case": "segment boundary and shared frame",
        "expected": [
            "10.0秒のmomentは後側segmentに属する",
            "9.75秒のframeは前側segmentに属する",
            "10.25秒のframeを二つのmomentが同じIDで共有する",
            "同じanchorのheartbeatとscene signalは一つのmomentになる",
        ],
        "video_source": source,
        "timeline_segments": segments,
        "candidate_moments": moments,
        "frame_candidates": frames,
    }


def vfr_case() -> Payload:
    """VFR frameを有理Video Timeで表すcaseを返す。"""
    source = video_source(
        "video-vfr-fingerprint",
        origin_pts=1_000,
        time_base=(1, 90_000),
        end_pts=91_000,
    )
    segments = timeline_segments(source, (1_000, 91_000))
    frames = [
        frame_candidate(source, segments, pts=pts)
        for pts in (1_000, 4_003, 10_009, 19_019)
    ]
    return {
        "case": "VFR exact Video Time",
        "expected": [
            "不均一なframe間隔を正確な有理数で保持する",
            "float秒とframe indexをidentityに使わない",
        ],
        "video_source": source,
        "frame_candidates": frames,
    }


def cross_video_identity_case() -> Payload:
    """同じPTSでもVideoが違えば別frameになるcaseを返す。"""
    sources = [
        video_source(
            fingerprint,
            origin_pts=0,
            time_base=(1, 1_000),
            end_pts=20_000,
        )
        for fingerprint in ("video-first-fingerprint", "video-second-fingerprint")
    ]
    frames = []
    for source in sources:
        segments = timeline_segments(source, (0, 20_000))
        frames.append(frame_candidate(source, segments, pts=10_000))
    return {
        "case": "same PTS across different videos",
        "expected": ["同じ10秒でもVideo Fingerprintが違えば別frame IDになる"],
        "video_sources": sources,
        "frame_candidates": frames,
    }


def shared_scene_catalog_case() -> Payload:
    """Scene CatalogのVideo Set所有を示すcaseを返す。"""
    video_fingerprints = ("video-first-fingerprint", "video-second-fingerprint")
    video_set_id = stable_id("video-set", *video_fingerprints)
    return {
        "case": "shared Scene Catalog",
        "expected": [
            "Videoごとのcatalogは作らない",
            "全Video Sourceを同じslugとselection roleで分類する",
        ],
        "video_set": {
            "id": video_set_id,
            "ordered_video_source_ids": list(video_fingerprints),
        },
        "scene_catalog": {
            "owner": "Video Set Stage",
            "video_set_id": video_set_id,
            "shared_by_all_videos": True,
            "per_video_catalog_count": 0,
        },
    }


def duration_normalized_candidate_cap_case() -> Payload:
    """動画時間に比例し、採用を強制しない上限のcaseを返す。"""
    density_per_minute = Fraction(2, 1)
    samples = []
    for fingerprint, duration_seconds, eligible_count in (
        ("video-short-fingerprint", 30, 0),
        ("video-long-fingerprint", 90, 20),
    ):
        source = video_source(
            fingerprint,
            origin_pts=0,
            time_base=(1, 1_000),
            end_pts=duration_seconds * 1_000,
        )
        cap = candidate_moment_cap(source, density_per_minute)
        samples.append(
            {
                "video_source_id": source["id"],
                "duration": source["duration"],
                "eligible_count": eligible_count,
                "cap": cap,
                "retained_count": min(eligible_count, cap),
            }
        )
    return {
        "case": "duration-normalized Candidate Moment cap",
        "expected": [
            "同じ密度ならVideo Durationに比例して上限が増える",
            "上限が1でも適格候補0件なら保持数は0になる",
            "Video Setの本数、順序、要求出力枚数を計算に使わない",
        ],
        "illustrative_density_per_minute": str(density_per_minute),
        "samples": samples,
    }


def video_set_stage_boundary_case() -> Payload:
    """別動画の重複をVideo Set Stageだけが判断するcaseを返す。"""
    first_frame_id = stable_id("frame", "video-first-fingerprint", 10_000, "1/1000")
    second_frame_id = stable_id(
        "frame",
        "video-second-fingerprint",
        1_000,
        "1/1000",
    )
    return {
        "case": "cross-video comparison ownership",
        "expected": [
            "各Video Stageは別動画の存在に依存せず再利用できる",
            "別動画間の視覚的重複はVideo Set Stageだけが判断する",
            "最終選定は特定Video Sourceからの最低採用数を持たない",
        ],
        "video_stage_outputs": [
            {
                "video_source_id": "video-first-fingerprint",
                "frame_candidate_id": first_frame_id,
                "neutral_visual_signature": "loading-screen-example",
                "compared_with_other_videos": False,
            },
            {
                "video_source_id": "video-second-fingerprint",
                "frame_candidate_id": second_frame_id,
                "neutral_visual_signature": "loading-screen-example",
                "compared_with_other_videos": False,
            },
        ],
        "video_set_stage": {
            "cross_video_duplicate_group": [first_frame_id, second_frame_id],
            "kept_frame_candidate_ids": [first_frame_id],
            "rejected_frame_candidate_ids": [second_frame_id],
            "minimum_selected_per_video": 0,
        },
    }
