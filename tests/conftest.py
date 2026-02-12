"""pytestの共通fixture設定.

複雑なモック設定を一箇所に集約し、メンテナンス性とデバッグ性を向上させる。
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture(scope="function", autouse=True)
def mock_clip_model() -> Generator[Any, Any, Any]:
    """CLIPモデルのモック.

    複雑なside_effectを排除し、単純で決定論的な振る舞いのみを実装。
    """
    with patch("transformers.CLIPModel.from_pretrained") as mock:
        model = MagicMock()

        # 単純な固定値を返す（バッチサイズ対応なし）
        def mock_get_text_features(**_kwargs: object) -> torch.Tensor:
            return torch.ones(1, 512) / torch.sqrt(torch.tensor(512.0))

        def mock_get_image_features(**kwargs: object) -> torch.Tensor:
            # pixel_valuesからバッチサイズを取得
            inputs = kwargs.get("pixel_values")
            if inputs is not None and hasattr(inputs, "shape"):
                batch_size = inputs.shape[0]
            else:
                batch_size = 1
            return torch.ones(batch_size, 512) / torch.sqrt(torch.tensor(512.0))

        model.get_text_features = MagicMock(side_effect=mock_get_text_features)
        model.get_image_features = MagicMock(side_effect=mock_get_image_features)
        model.to = MagicMock(return_value=model)
        model.eval = MagicMock()

        mock.return_value = model
        yield mock


@pytest.fixture(scope="function", autouse=True)
def mock_clip_processor() -> Generator[Any, Any, Any]:
    """CLIPプロセッサのモック.

    辞書のようなオブジェクトを返し、.to()メソッドをサポート。
    バッチサイズに応じた形状を動的に返す。
    """
    with patch("transformers.CLIPProcessor.from_pretrained") as mock:
        processor = MagicMock()

        def mock_processor(**kwargs: object) -> MagicMock:
            # 単純な辞書を返す
            images = kwargs.get("images")
            batch_size = len(images) if isinstance(images, list) else 1

            input_ids = torch.tensor([[1, 2, 3]])
            pixel_values = torch.ones(batch_size, 3, 224, 224) * 0.5
            attention_mask = torch.tensor([[1, 1, 1]])

            result_obj = MagicMock()
            result_obj.input_ids = input_ids
            result_obj.pixel_values = pixel_values
            result_obj.attention_mask = attention_mask

            # 辞書のように振る舞うための__getitem__を実装
            def getitem(_self: MagicMock, key: str) -> torch.Tensor:
                if key == "input_ids":
                    return input_ids
                elif key == "pixel_values":
                    return pixel_values
                elif key == "attention_mask":
                    return attention_mask
                else:
                    raise KeyError(key)

            import types

            result_obj.__getitem__ = types.MethodType(getitem, result_obj)
            result_obj.to = MagicMock(return_value=result_obj)

            return result_obj

        processor.side_effect = mock_processor
        mock.return_value = processor
        yield mock
