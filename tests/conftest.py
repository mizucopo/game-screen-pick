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


class _SimpleDict(dict[str, Any]):
    """辞書風アクセスと.to()メソッドをサポートするシンプルなクラス.

    CI環境でのハング問題を防ぐため、MagicMockを使わずに実装。
    """

    def to(self, device: str) -> "_SimpleDict":  # noqa: ARG002
        """デバイス移動のモック（自分自身を返す）."""
        return self


@pytest.fixture(scope="function", autouse=True)
def mock_clip_model() -> Generator[Any, Any, Any]:
    """CLIPモデルのモック.

    極力シンプルな実装にし、CI環境でのハングを防止する。
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


@pytest.fixture(scope="function", autouse=True)
def mock_clip_processor() -> Generator[Any, Any, Any]:
    """CLIPプロセッサのモック.

    _SimpleDictを使い、CI環境でのハングを防止する。
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


@pytest.fixture
def sample_image_path(tmp_path: Path) -> str:
    """標準的なテスト画像（640x480 JPG）を作成する."""
    return _create_test_image(tmp_path, "test_image.jpg", (480, 640), (0, 255))


@pytest.fixture
def dark_image_path(tmp_path: Path) -> str:
    """輝度ペナルティのテスト用に暗いテスト画像（640x480 JPG）を作成する."""
    return _create_test_image(tmp_path, "dark_image.jpg", (480, 640), (0, 50))


@pytest.fixture
def png_image_path(tmp_path: Path) -> str:
    """PNG形式のテスト画像（640x480）を作成する."""
    return _create_test_image(tmp_path, "test_image.png", (480, 640), (0, 255))


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
