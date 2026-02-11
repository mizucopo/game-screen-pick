"""CLIPモデル関連の型定義."""

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from PIL import Image


class CLIPProcessorProtocol(Protocol):
    """CLIPProcessor の期待されるインターフェースを定義する Protocol."""

    def __call__(
        self,
        *,
        images: Any,
        text: "str | list[str] | None",
        return_tensors: str = "pt",
        padding: bool = True,
    ) -> "BatchFeatureProtocol":
        """画像またはテキストを処理してバッチ特徴量を返す.

        Args:
            images: 画像または画像リスト
            text: テキストまたはテキストリスト
            return_tensors: 戻り値のテンソル型
            padding: パディングを行うかどうか

        Returns:
            バッチ特徴量
        """
        ...


class BatchFeatureProtocol(Protocol):
    """BatchFeature の期待されるインターフェースを定義する Protocol."""

    def to(self, device: str) -> "BatchFeatureProtocol":
        """バッチ特徴量を指定デバイスに転送する.

        Args:
            device: 転送先デバイス

        Returns:
            転送後のバッチ特徴量
        """
        ...


# 型エイリアス
# Sequence は共変なため、list[Image] も list[Image | None] も受け取れる
SingleImageInput = Image.Image
BatchImageInput = Sequence[Image.Image]
OptionalBatchImageInput = Sequence[Image.Image | None]
CLIPFeaturesOutput = list[np.ndarray | None]
