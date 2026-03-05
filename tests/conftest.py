"""pytestの共通fixture設定.

テスト用ヘルパー関数とfixtureを提供する。
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.models.image_metrics import ImageMetrics
from src.models.normalized_metrics import NormalizedMetrics
from src.models.raw_metrics import RawMetrics


def _create_test_image(
    tmp_path: Path, filename: str, size: tuple[int, int], pixel_range: tuple[int, int]
) -> str:
    """テスト画像を作成するヘルパー関数.

    Args:
        tmp_path: 一時ディレクトリパス
        filename: 画像ファイル名
        size: 画像サイズ（高さ、幅）
        pixel_range: ピクセル値の範囲（最小、最大）

    Returns:
        作成された画像の絶対パス
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

    Parametrizeで様々なバリエーション（暗い画像、PNGなど）をカバー。
    """
    param = request.param
    if isinstance(param, tuple):
        filename, pixel_range = param
    else:
        filename = param
        pixel_range = (0, 255)
    return _create_test_image(tmp_path, filename, (480, 640), pixel_range)


def create_image_metrics(
    path: str,
    raw_metrics_dict: dict[str, float] | None = None,
    normalized_metrics_dict: dict[str, float] | None = None,
    semantic_score: float = 0.8,
    total_score: float = 100.0,
    features: np.ndarray | None = None,
) -> ImageMetrics:
    """ImageMetricsを作成する共通ヘルパー関数.

    Args:
        path: 画像パス
        raw_metrics_dict: 生メトリクスの辞書（省略時はデフォルト値）
        normalized_metrics_dict: 正規化メトリクスの辞書（省略時はデフォルト値）
        semantic_score: セマンティックスコア
        total_score: 総合スコア
        features: 特徴ベクトル（省略時はランダム生成）

    Returns:
        ImageMetricsインスタンス
    """
    if features is None:
        np.random.seed(42)
        features = np.random.rand(128)

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

    return ImageMetrics(
        path=path,
        raw_metrics=raw,
        normalized_metrics=norm,
        semantic_score=semantic_score,
        total_score=total_score,
        features=features,
    )


def create_sample_metrics(
    count: int, base_path: str = "/fake/path"
) -> list[ImageMetrics]:
    """テスト用のサンプルImageMetricsリストを作成する.

    Args:
        count: 作成するメトリクス数
        base_path: 画像パスのベース（デフォルト: "/fake/path"）

    Returns:
        ImageMetricsのリスト
    """
    metrics = []
    for i in range(count):
        np.random.seed(i)
        features = np.random.rand(128)
        metrics.append(
            create_image_metrics(
                path=f"{base_path}/image{i}.jpg",
                raw_metrics_dict={"blur_score": 100.0 - i * 10},
                normalized_metrics_dict={"blur_score": 1.0 - i * 0.1},
                semantic_score=0.8,
                total_score=100.0 - i * 10,
                features=features,
            )
        )
    return metrics
