"""pytestの共通fixture設定.

scene mix ベースのテストで再利用する入力画像、解析結果、
採点済み候補のヘルパーをまとめて提供する。
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.constants.scene_label import SceneLabel
from src.models.analyzed_image import AnalyzedImage
from src.models.layout_heuristics import LayoutHeuristics
from src.models.normalized_metrics import NormalizedMetrics
from src.models.raw_metrics import RawMetrics
from src.models.scene_assessment import SceneAssessment
from src.models.scored_candidate import ScoredCandidate
from src.models.whole_input_profile import WholeInputProfile
from src.services.whole_input_profiler import WholeInputProfiler


def _create_test_image(
    tmp_path: Path, filename: str, size: tuple[int, int], pixel_range: tuple[int, int]
) -> str:
    """テスト画像を作成するヘルパー関数.

    Args:
        tmp_path: 一時ディレクトリのルート。
        filename: 生成する画像ファイル名。
        size: 画像サイズ。 `(height, width)` の順で渡す。
        pixel_range: 画素値の最小値と最大値。

    Returns:
        作成された画像ファイルの絶対パス。
    """
    np.random.seed(42)
    img_array = np.random.randint(
        pixel_range[0], pixel_range[1], (*size, 3), dtype=np.uint8
    )
    img_path = tmp_path / filename
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture(
    params=[
        "test_image.jpg",
        ("dark_image.jpg", (0, 50)),
        "test_image.png",
    ]
)
def sample_image_path(tmp_path: Path, request: pytest.FixtureRequest) -> str:
    """標準的なテスト画像（640x480）を作成する.

    Parametrize で明るさの異なるJPEGやPNGを切り替え、
    画像系テストが複数フォーマットでも同様に動くことを確認しやすくする。
    """
    param = request.param
    if isinstance(param, tuple):
        filename, pixel_range = param
    else:
        filename = param
        pixel_range = (0, 255)
    return _create_test_image(tmp_path, filename, (480, 640), pixel_range)


def create_analyzed_image(
    path: str,
    raw_metrics_dict: dict[str, float] | None = None,
    normalized_metrics_dict: dict[str, float] | None = None,
    clip_features: np.ndarray[Any, Any] | None = None,
    combined_features: np.ndarray[Any, Any] | None = None,
    content_features: np.ndarray[Any, Any] | None = None,
    layout_dict: dict[str, float] | None = None,
) -> AnalyzedImage:
    """`AnalyzedImage` を作成する共通ヘルパー.

    Args:
        path: 候補画像のパス。
        raw_metrics_dict: `RawMetrics` の上書き値。
        normalized_metrics_dict: `NormalizedMetrics` の上書き値。
        clip_features: scene 判定に使うCLIP特徴。
        combined_features: 類似度判定に使う結合特徴。
        layout_dict: レイアウトヒューリスティクスの上書き値。

    Returns:
        テスト用の `AnalyzedImage` インスタンス。
    """
    if clip_features is None:
        np.random.seed(42)
        clip_features = np.random.rand(512)
    if combined_features is None:
        np.random.seed(24)
        combined_features = np.random.rand(576)

    raw_metrics_dict = raw_metrics_dict or {}
    raw = RawMetrics(
        blur_score=raw_metrics_dict.get("blur_score", 100),
        brightness=raw_metrics_dict.get("brightness", 100),
        contrast=raw_metrics_dict.get("contrast", 50),
        edge_density=raw_metrics_dict.get("edge_density", 0.1),
        color_richness=raw_metrics_dict.get("color_richness", 50),
        ui_density=raw_metrics_dict.get("ui_density", 10),
        action_intensity=raw_metrics_dict.get("action_intensity", 30),
        visual_balance=raw_metrics_dict.get("visual_balance", 80),
        dramatic_score=raw_metrics_dict.get("dramatic_score", 50),
        luminance_entropy=raw_metrics_dict.get("luminance_entropy", 1.0),
        luminance_range=raw_metrics_dict.get("luminance_range", 40.0),
        near_black_ratio=raw_metrics_dict.get("near_black_ratio", 0.0),
        near_white_ratio=raw_metrics_dict.get("near_white_ratio", 0.0),
        dominant_tone_ratio=raw_metrics_dict.get("dominant_tone_ratio", 0.5),
    )

    normalized_metrics_dict = normalized_metrics_dict or {}
    norm = NormalizedMetrics(
        blur_score=normalized_metrics_dict.get("blur_score", 0.5),
        contrast=normalized_metrics_dict.get("contrast", 0.5),
        color_richness=normalized_metrics_dict.get("color_richness", 0.5),
        edge_density=normalized_metrics_dict.get("edge_density", 0.5),
        dramatic_score=normalized_metrics_dict.get("dramatic_score", 0.5),
        visual_balance=normalized_metrics_dict.get("visual_balance", 0.5),
        action_intensity=normalized_metrics_dict.get("action_intensity", 0.5),
        ui_density=normalized_metrics_dict.get("ui_density", 0.5),
    )

    layout_dict = layout_dict or {}
    heuristics = LayoutHeuristics(
        dialogue_overlay_score=layout_dict.get("dialogue_overlay_score", 0.1),
        menu_layout_score=layout_dict.get("menu_layout_score", 0.1),
        title_layout_score=layout_dict.get("title_layout_score", 0.1),
        game_over_layout_score=layout_dict.get("game_over_layout_score", 0.1),
    )
    if content_features is None:
        content_features = np.asarray(combined_features[:101], dtype=np.float32)

    return AnalyzedImage(
        path=path,
        raw_metrics=raw,
        normalized_metrics=norm,
        clip_features=clip_features,
        combined_features=combined_features,
        content_features=content_features,
        layout_heuristics=heuristics,
    )


def build_whole_input_profile(*images: AnalyzedImage) -> WholeInputProfile:
    """入力画像群からWholeInputProfileを作る."""
    return WholeInputProfiler().build_profile(list(images))


def create_scored_candidate(
    path: str,
    scene_label: SceneLabel = SceneLabel.PLAY,
    play_score: float = 0.8,
    event_score: float = 0.3,
    density_score: float = 0.8,
    scene_confidence: float = 0.5,
    quality_score: float = 0.6,
    selection_score: float = 0.6,
    resolved_profile: str = "active",
    combined_features: np.ndarray[Any, Any] | None = None,
    score_band: str | None = None,
    outlier_rejected: bool = False,
) -> ScoredCandidate:
    """`ScoredCandidate` を作成する共通ヘルパー.

    Args:
        path: 候補画像のパス。
        scene_label: 最終的に属する scene label 。
        play_score: play 向けスコア。
        event_score: event 向けスコア。
        density_score: 類似度密度スコア。
        scene_confidence: scene判定の信頼度。
        quality_score: 画質スコア。
        selection_score: 最終選定スコア。
        resolved_profile: 解決済みプロファイル名。
        combined_features: 類似度判定用の結合特徴。

    Returns:
        scene 判定とスコアが付与済みの `ScoredCandidate` 。
    """
    analyzed = create_analyzed_image(
        path=path,
        combined_features=combined_features,
    )
    assessment = SceneAssessment(
        play_score=play_score,
        event_score=event_score,
        density_score=density_score,
        scene_label=scene_label,
        scene_confidence=scene_confidence,
    )
    return ScoredCandidate(
        analyzed_image=analyzed,
        scene_assessment=assessment,
        resolved_profile=resolved_profile,
        quality_score=quality_score,
        selection_score=selection_score,
        score_band=score_band,
        outlier_rejected=outlier_rejected,
    )
