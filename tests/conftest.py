"""pytestの共通fixture設定.

複雑なモック設定を一箇所に集約し、メンテナンス性とデバッグ性を向上させる。
CI環境でのハング問題を防ぐため、極力シンプルなモック構造を採用する。
"""

from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from src.models.image_metrics import ImageMetrics
from src.models.normalized_metrics import NormalizedMetrics
from src.models.raw_metrics import RawMetrics


class _SimpleDict(dict[str, Any]):
    """辞書風アクセスと.to()メソッドをサポートするシンプルなクラス.

    CI環境でのハング問題を防ぐため、MagicMockを使わずに実装。
    """

    def to(self, device: str) -> "_SimpleDict":  # noqa: ARG002
        """デバイス移動のモック（自分自身を返す）."""
        return self


@pytest.fixture(scope="function")
def mock_clip_model() -> Generator[Any, Any, Any]:
    """CLIPモデルのモック.

    極力シンプルな実装にし、CI環境でのハングを防止する。
    明示的に使用するテストのみで適用する。
    """

    # モデルオブジェクト（MagicMockではなく普通のクラス）
    class _MockModel:
        """CLIPモデルのモック."""

        device = "cpu"
        _eval_called = False
        _to_called_with = []

        def get_text_features(self, **_kwargs: object) -> torch.Tensor:
            """テキスト特徴を返す."""
            return torch.ones(1, 512) / torch.sqrt(torch.tensor(512.0))

        def get_image_features(self, **kwargs: object) -> torch.Tensor:
            """画像特徴を返す."""
            inputs = kwargs.get("pixel_values")
            if inputs is not None and hasattr(inputs, "shape"):
                batch_size = inputs.shape[0]
            else:
                batch_size = 1
            return torch.ones(batch_size, 512) / torch.sqrt(torch.tensor(512.0))

        def to(self, device: str) -> "_MockModel":
            """デバイス移動のモック（自分自身を返す）."""
            self._to_called_with.append(device)
            return self

        def eval(self) -> None:
            """evalモードのモック（何もしない）."""
            self._eval_called = True

        # MagicMock互換のプロパティ
        @property
        def called(self) -> bool:
            """MagicMock互換プロパティ."""
            return True

    with patch("transformers.CLIPModel.from_pretrained") as mock:
        mock.return_value = _MockModel()
        yield mock


@pytest.fixture(scope="function")
def mock_clip_processor() -> Generator[Any, Any, Any]:
    """CLIPプロセッサのモック.

    _SimpleDictを使い、CI環境でのハングを防止する。
    明示的に使用するテストのみで適用する。
    """

    def mock_processor_func(**kwargs: object) -> _SimpleDict:
        """プロセッサの呼び出しをモックする."""
        images = kwargs.get("images")
        batch_size = len(images) if isinstance(images, list) else 1

        return _SimpleDict(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "pixel_values": torch.ones(batch_size, 3, 224, 224) * 0.5,
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        )

    # プロセッサオブジェクト（callable）
    class _MockProcessor:
        """CLIPプロセッサのモック."""

        def __call__(self, **kwargs: object) -> _SimpleDict:
            return mock_processor_func(**kwargs)

    with patch("transformers.CLIPProcessor.from_pretrained") as mock:
        mock.return_value = _MockProcessor()
        yield mock


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


@pytest.fixture
def multiple_image_paths(tmp_path: Path) -> list[str]:
    """複数のテスト画像を作成する."""
    paths = []
    for i in range(3):
        np.random.seed(42 + i)
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = tmp_path / f"test_image_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        paths.append(str(img_path))
    return paths


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
