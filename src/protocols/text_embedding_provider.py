from collections.abc import Sequence
from typing import Protocol

import torch


class TextEmbeddingProvider(Protocol):
    """SceneScorerが必要とする最小の埋め込みAPI."""

    def get_text_embeddings(self, texts: Sequence[str]) -> torch.Tensor:
        """テキスト埋め込みを返す."""
