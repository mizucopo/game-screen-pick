"""native RGB frameをmodel-freeに一括解析する。"""

import math
from collections.abc import Iterable
from dataclasses import replace
from fractions import Fraction

import cv2
import numpy as np

from ..models.content_reject_reason import ContentRejectReason
from ..models.decoded_video_frame import DecodedVideoFrame
from ..models.neutral_image_analysis import NeutralImageAnalysis
from ..models.neutral_image_metrics import NeutralImageMetrics

NEUTRAL_ANALYSIS_ALGORITHM_VERSION = "neutral-image-analysis-v5"
BLUR_REJECT_VARIANCE_MIN = 12.0
_CLIPPED_WHITE_THRESHOLD = 235
_SOFT_WHITE_THRESHOLD = 230
_TRANSITION_BRIGHT_THRESHOLD = 220
_BRIGHT_SWEEP_WINDOW = Fraction(1, 4)
_BRIGHT_SWEEP_AFFECTED_REGION_MIN = 0.08
_BRIGHT_SWEEP_LOW_REGION_MAX = 0.20
_BRIGHT_SWEEP_DOMINANT_REGION_MIN = 0.60
_BRIGHT_SWEEP_REGION_CHANGE_MIN = 0.45
_CENTRAL_FLASH_CLIPPED_RATIO_MIN = 0.04
_CENTRAL_FLASH_CLIPPED_RATIO_MAX = 0.25
_CENTRAL_FLASH_CENTER_RATIO_MIN = 0.15
_CENTRAL_FLASH_REGION_RATIO_MIN = 0.035
_QUALITY_WEIGHTS = {
    "blur_score": 0.165,
    "contrast": 0.145,
    "color_richness": 0.06,
    "visual_balance": 0.10,
    "edge_density": 0.13,
    "action_intensity": 0.125,
    "ui_density": 0.225,
    "dramatic_score": 0.05,
}

_FrameTiming = tuple[int, int, int | None, Fraction]


def analyze_neutral_images(
    frames: Iterable[DecodedVideoFrame],
) -> tuple[NeutralImageAnalysis, ...]:
    """動画内分布と前後関係を使いnative frameを解析する。"""
    frame_timings: list[_FrameTiming] = []
    raw_rows: list[tuple[dict[str, float], np.ndarray, bytes]] = []
    for frame in frames:
        frame_timings.append(
            (
                frame.stream_index,
                frame.pts,
                frame.duration_ts,
                frame.time_base,
            )
        )
        raw_rows.append(_measure_frame(frame))
    if not raw_rows:
        return ()
    information_scores = _information_scores(raw_rows)
    visibility_scores = [_visibility_score(row) for row in raw_rows]
    analyses = [
        NeutralImageAnalysis(
            source_pts=frame_timings[index][1],
            metrics=_build_metrics(
                raw,
                information_scores[index],
                visibility_scores[index],
            ),
            quality_score=_quality_score(raw),
            visual_feature=tuple(float(value) for value in feature),
            grayscale_signature=signature,
            reject_reason=_absolute_reject_reason(raw),
        )
        for index, (raw, feature, signature) in enumerate(raw_rows)
    ]
    _apply_expanding_bright_sweep_rejections(
        analyses,
        frame_timings,
        raw_rows,
    )
    _apply_temporal_rejections(analyses, frame_timings)
    _apply_relative_fade_rejections(analyses)
    return tuple(analyses)


def _measure_frame(
    frame: DecodedVideoFrame,
) -> tuple[dict[str, float], np.ndarray, bytes]:
    rgb = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(
        frame.height,
        frame.width,
        3,
    )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray_flat = gray.reshape(-1)
    gray_size = gray.size
    gray_hist = np.bincount(gray_flat, minlength=256).astype(np.float32)
    gray_prob = gray_hist / gray_size
    non_zero_prob = gray_prob[gray_prob > 0]
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    _, contrast_std = cv2.meanStdDev(gray)
    edges = cv2.Canny(gray, 50, 150)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    _, saturation_std = cv2.meanStdDev(hsv[:, :, 1])
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=True)
    _, action_std = cv2.meanStdDev(
        cv2.filter2D(
            gray,
            -1,
            np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        )
    )
    high_saturation = hsv[:, :, 1] > 180
    high_value = hsv[:, :, 2] > 180
    luminance_p5, luminance_p95 = np.percentile(gray_flat, [5, 95])
    dominant_tone_hist = np.bincount(gray_flat // 16, minlength=16)
    brightness = float(cv2.mean(gray)[0])
    clipped_white = gray >= _CLIPPED_WHITE_THRESHOLD
    clipped_white_ratio = float(np.mean(clipped_white))
    height, width = gray.shape
    center_clipped_white_ratio = float(
        np.mean(
            clipped_white[
                height // 4 : height - height // 4,
                width // 4 : width - width // 4,
            ]
        )
    )
    largest_clipped_white_region_ratio = (
        _largest_connected_region_ratio(clipped_white)
        if clipped_white_ratio >= _CENTRAL_FLASH_CLIPPED_RATIO_MIN
        else 0.0
    )
    soft_white_ratio = float(np.mean(gray >= _SOFT_WHITE_THRESHOLD))
    transition_bright = gray >= _TRANSITION_BRIGHT_THRESHOLD
    transition_bright_ratio = float(np.mean(transition_bright))
    largest_transition_bright_region_ratio = (
        _largest_connected_region_ratio(transition_bright)
        if transition_bright_ratio >= _BRIGHT_SWEEP_AFFECTED_REGION_MIN
        else 0.0
    )
    raw = {
        "blur_score": float(laplacian.var()),
        "brightness": brightness,
        "contrast": float(contrast_std[0][0]),
        "edge_density": cv2.countNonZero(edges) / gray_size,
        "color_richness": float(saturation_std[0][0]),
        "ui_density": cv2.norm(sobel_x, cv2.NORM_L1) / gray_size,
        "action_intensity": float(action_std[0][0]),
        "visual_balance": float(max(0, 100 - abs(brightness - 128) * 0.5)),
        "dramatic_score": float(
            cv2.countNonZero((high_saturation & high_value).astype(np.uint8))
            / gray_size
            * 1000
        ),
        "luminance_entropy": float(-(non_zero_prob * np.log2(non_zero_prob)).sum()),
        "luminance_range": float(luminance_p95 - luminance_p5),
        "near_black_ratio": float(np.mean(gray_flat <= 12)),
        "near_white_ratio": float(np.mean(gray_flat >= 243)),
        "dominant_tone_ratio": float(dominant_tone_hist.max() / gray_size),
        "clipped_white_ratio": clipped_white_ratio,
        "center_clipped_white_ratio": center_clipped_white_ratio,
        "largest_clipped_white_region_ratio": (largest_clipped_white_region_ratio),
        "soft_white_ratio": soft_white_ratio,
        "largest_transition_bright_region_ratio": (
            largest_transition_bright_region_ratio
        ),
    }
    hsv_hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    luminance_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    edge_hist, _ = np.histogram(
        angle,
        bins=16,
        range=(0, 360),
        weights=magnitude,
    )
    feature = _safe_l2_normalize(
        np.concatenate(
            (
                cv2.normalize(hsv_hist, hsv_hist).flatten(),
                cv2.normalize(luminance_hist, luminance_hist).flatten(),
                edge_hist.astype(np.float32),
            )
        ).astype(np.float32)
    )
    signature = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).tobytes()
    return raw, feature, signature


def _information_scores(
    rows: list[tuple[dict[str, float], np.ndarray, bytes]],
) -> list[float]:
    weights = {
        "contrast": 0.20,
        "edge_density": 0.25,
        "action_intensity": 0.15,
        "luminance_entropy": 0.20,
        "luminance_range": 0.20,
    }
    sorted_values = {
        name: np.sort(np.asarray([row[0][name] for row in rows], dtype=np.float32))
        for name in weights
    }
    return [
        float(
            sum(
                weight * _percentile_rank(raw[name], sorted_values[name])
                for name, weight in weights.items()
            )
        )
        for raw, _feature, _signature in rows
    ]


def _visibility_score(
    row: tuple[dict[str, float], np.ndarray, bytes],
) -> float:
    raw = row[0]
    return float(
        0.30 * min(1.0, raw["luminance_range"] / 32.0)
        + 0.25 * min(1.0, raw["luminance_entropy"] / 1.5)
        + 0.20 * min(1.0, raw["edge_density"] / 0.08)
        + 0.15 * min(1.0, raw["contrast"] / 12.0)
        + 0.10 * (1.0 - max(raw["near_black_ratio"], raw["near_white_ratio"]))
    )


def _build_metrics(
    raw: dict[str, float],
    information_score: float,
    visibility_score: float,
) -> NeutralImageMetrics:
    return NeutralImageMetrics(
        blur_score=raw["blur_score"],
        brightness=raw["brightness"],
        contrast=raw["contrast"],
        edge_density=raw["edge_density"],
        color_richness=raw["color_richness"],
        ui_density=raw["ui_density"],
        action_intensity=raw["action_intensity"],
        visual_balance=raw["visual_balance"],
        dramatic_score=raw["dramatic_score"],
        luminance_entropy=raw["luminance_entropy"],
        luminance_range=raw["luminance_range"],
        near_black_ratio=raw["near_black_ratio"],
        near_white_ratio=raw["near_white_ratio"],
        dominant_tone_ratio=raw["dominant_tone_ratio"],
        information_score=information_score,
        visibility_score=visibility_score,
    )


def _absolute_reject_reason(
    raw: dict[str, float],
) -> ContentRejectReason | None:
    if (
        raw["near_black_ratio"] >= 0.98
        and raw["luminance_entropy"] <= 0.6
        and raw["luminance_range"] <= 16
    ):
        return ContentRejectReason.BLACKOUT
    if (
        raw["near_white_ratio"] >= 0.98
        and raw["luminance_entropy"] <= 0.6
        and raw["luminance_range"] <= 16
    ):
        return ContentRejectReason.WHITEOUT
    if (
        raw["clipped_white_ratio"] >= 0.25
        and raw["clipped_white_ratio"] <= 0.65
        and raw["center_clipped_white_ratio"] >= 0.50
        and raw["largest_clipped_white_region_ratio"] >= 0.20
    ):
        return ContentRejectReason.WHITEOUT
    if (
        raw["clipped_white_ratio"] >= _CENTRAL_FLASH_CLIPPED_RATIO_MIN
        and raw["clipped_white_ratio"] < _CENTRAL_FLASH_CLIPPED_RATIO_MAX
        and raw["center_clipped_white_ratio"] >= _CENTRAL_FLASH_CENTER_RATIO_MIN
        and raw["largest_clipped_white_region_ratio"] >= _CENTRAL_FLASH_REGION_RATIO_MIN
    ):
        return ContentRejectReason.WHITEOUT
    if (
        raw["soft_white_ratio"] >= 0.85
        and raw["dominant_tone_ratio"] >= 0.85
        and raw["luminance_entropy"] <= 2.0
        and raw["edge_density"] <= 0.02
    ):
        return ContentRejectReason.WHITEOUT
    if (
        raw["dominant_tone_ratio"] >= 0.97
        and raw["luminance_range"] <= 20
        and raw["contrast"] <= 10
    ):
        return ContentRejectReason.SINGLE_TONE
    if raw["blur_score"] < BLUR_REJECT_VARIANCE_MIN and raw["edge_density"] < 0.03:
        return ContentRejectReason.BLUR
    return None


def _largest_connected_region_ratio(mask: np.ndarray) -> float:
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max() / mask.size)


def _apply_temporal_rejections(
    analyses: list[NeutralImageAnalysis],
    frame_timings: list[_FrameTiming],
) -> None:
    for index in range(1, len(analyses) - 1):
        if not (
            _are_adjacent(frame_timings[index - 1], frame_timings[index])
            and _are_adjacent(frame_timings[index], frame_timings[index + 1])
        ):
            continue
        previous = analyses[index - 1]
        current = analyses[index]
        following = analyses[index + 1]
        if any(
            item.reject_reason is not None for item in (previous, current, following)
        ):
            continue
        similarity = float(
            np.clip(
                np.dot(previous.visual_feature, following.visual_feature),
                -1.0,
                1.0,
            )
        )
        if similarity >= 0.92 and current.metrics.visibility_score + 0.15 < min(
            previous.metrics.visibility_score,
            following.metrics.visibility_score,
        ):
            analyses[index] = replace(
                current,
                reject_reason=ContentRejectReason.TEMPORAL_TRANSITION,
            )


def _apply_expanding_bright_sweep_rejections(
    analyses: list[NeutralImageAnalysis],
    frame_timings: list[_FrameTiming],
    raw_rows: list[tuple[dict[str, float], np.ndarray, bytes]],
) -> None:
    """短時間に画面を覆う淡い連結領域の拡大・縮小を遷移として除外する。"""
    region_ratios = [
        row[0]["largest_transition_bright_region_ratio"] for row in raw_rows
    ]
    rejected_indices: set[int] = set()
    for start in range(len(analyses)):
        end = start
        while end + 1 < len(analyses) and _are_adjacent(
            frame_timings[end],
            frame_timings[end + 1],
        ):
            start_timing = frame_timings[start]
            next_timing = frame_timings[end + 1]
            elapsed = (next_timing[1] - start_timing[1]) * start_timing[3]
            if elapsed > _BRIGHT_SWEEP_WINDOW:
                break
            end += 1
        affected = [
            index
            for index in range(start, end + 1)
            if region_ratios[index] >= _BRIGHT_SWEEP_AFFECTED_REGION_MIN
        ]
        if len(affected) < 3:
            continue
        affected_ratios = [region_ratios[index] for index in affected]
        minimum = min(affected_ratios)
        maximum = max(affected_ratios)
        if (
            minimum <= _BRIGHT_SWEEP_LOW_REGION_MAX
            and maximum >= _BRIGHT_SWEEP_DOMINANT_REGION_MIN
            and maximum - minimum >= _BRIGHT_SWEEP_REGION_CHANGE_MIN
        ):
            rejected_indices.update(affected)
    for index in rejected_indices:
        current = analyses[index]
        if current.reject_reason is None:
            analyses[index] = replace(
                current,
                reject_reason=ContentRejectReason.TEMPORAL_TRANSITION,
            )


def _are_adjacent(
    previous: _FrameTiming,
    following: _FrameTiming,
) -> bool:
    previous_stream, previous_pts, previous_duration, previous_time_base = previous
    following_stream, following_pts, _following_duration, following_time_base = (
        following
    )
    return (
        previous_stream == following_stream
        and previous_time_base == following_time_base
        and previous_duration is not None
        and previous_duration > 0
        and previous_pts + previous_duration == following_pts
    )


def _apply_relative_fade_rejections(analyses: list[NeutralImageAnalysis]) -> None:
    eligible = [item for item in analyses if item.reject_reason is None]
    if len(eligible) < 3:
        return
    brightness = np.asarray(
        [item.metrics.brightness for item in eligible],
        dtype=np.float32,
    )
    visibility = np.asarray(
        [item.metrics.visibility_score for item in eligible],
        dtype=np.float32,
    )
    luminance_range = np.asarray(
        [item.metrics.luminance_range for item in eligible],
        dtype=np.float32,
    )
    brightness_p10, brightness_p90 = np.percentile(brightness, [10, 90])
    visibility_median = float(np.percentile(visibility, 50))
    range_p25 = float(np.percentile(luminance_range, 25))
    for index, analysis in enumerate(analyses):
        metrics = analysis.metrics
        if analysis.reject_reason is not None:
            continue
        is_exposure_extreme = (
            metrics.brightness < brightness_p10 or metrics.brightness > brightness_p90
        )
        if (
            is_exposure_extreme
            and metrics.visibility_score + 0.20 < visibility_median
            and metrics.luminance_range < max(24.0, range_p25 * 0.75)
        ):
            analyses[index] = replace(
                analysis,
                reject_reason=ContentRejectReason.FADE_TRANSITION,
            )


def _quality_score(raw: dict[str, float]) -> float:
    normalized = {
        "blur_score": _sigmoid(raw["blur_score"], 500, 0.005),
        "contrast": _sigmoid(raw["contrast"], 50, 0.1),
        "color_richness": _sigmoid(raw["color_richness"], 40, 0.1),
        "edge_density": min(1.0, raw["edge_density"] * 5.0),
        "dramatic_score": min(1.0, raw["dramatic_score"] / 100.0),
        "visual_balance": raw["visual_balance"] / 100.0,
        "action_intensity": _sigmoid(raw["action_intensity"], 30, 0.2),
        "ui_density": _sigmoid(raw["ui_density"], 10, 0.3),
    }
    return float(
        sum(normalized[name] * weight for name, weight in _QUALITY_WEIGHTS.items())
    )


def _sigmoid(value: float, center: float, steepness: float) -> float:
    try:
        return 1 / (1 + math.exp(-steepness * (value - center)))
    except OverflowError:
        return 1.0 if value > center else 0.0


def _percentile_rank(value: float, sorted_values: np.ndarray) -> float:
    left = np.searchsorted(sorted_values, value, side="left")
    right = np.searchsorted(sorted_values, value, side="right")
    return float((left + right) / 2.0 / sorted_values.size)


def _safe_l2_normalize(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 0:
        return np.zeros_like(values)
    return values / norm
