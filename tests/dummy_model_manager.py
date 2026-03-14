from collections.abc import Sequence

import torch


class DummyModelManager:
    """固定テキスト埋め込みを返すダミーモデル."""

    def get_text_embeddings(self, texts: Sequence[str]) -> torch.Tensor:
        """promptに応じた固定ベクトルを返す."""
        embeddings = []
        for text in texts:
            lowered = text.lower()
            if "dialogue" in lowered or "cutscene" in lowered or "event" in lowered:
                embeddings.append(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32))
            elif (
                "menu" in lowered
                or "title" in lowered
                or "game over" in lowered
                or "result" in lowered
                or "reward" in lowered
                or "loading" in lowered
            ):
                embeddings.append(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))
            else:
                embeddings.append(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))
        return torch.stack(embeddings)
