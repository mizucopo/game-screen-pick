"""CLIPモデルのライフサイクルを管理するマネージャー."""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class CLIPModelManager:
    """CLIPモデルのライフサイクルとデバイス管理を行うクラス."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        target_text: str = "epic game scenery",
        device: Optional[str] = None,
    ):
        """CLIPモデルマネージャーを初期化する.

        Args:
            model_name: 使用するCLIPモデル名
            target_text: セマンティック検索用のターゲットテキスト
            device: 使用するデバイス（Noneの場合は自動検出）
        """
        self.model_name = model_name
        self._target_text = target_text

        # デバイス設定
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # モデルとプロセッサのロード
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # デバイスにモデルを転送
        self.model.to(self.device)
        self.model.eval()

        # GPU最適化: TF32を許可（Ampere GPU以上で有効）
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # テキスト埋め込みの事前計算とキャッシュ
        self._text_embeddings = self._precompute_text_embeddings()

    def _precompute_text_embeddings(self) -> torch.Tensor:
        """テキスト埋め込みを事前計算してキャッシュする.

        Returns:
            L2正規化済みのテキスト埋め込みテンソル（1次元の512要素）
        """
        with torch.inference_mode():
            inputs = self.processor(
                text=[self._target_text],
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            text_features: torch.Tensor = self.model.get_text_features(**inputs)
            # L2正規化を適用してコサイン類似度計算を可能にする
            text_features_normalized = F.normalize(text_features, p=2, dim=-1)
            return text_features_normalized

    def get_image_features(
        self, pil_image: object, batch_mode: bool = False
    ) -> torch.Tensor:
        """PIL画像からCLIP画像特徴を抽出する.

        Args:
            pil_image: PIL画像オブジェクト（バッチモード時はリスト）
            batch_mode: バッチ処理モードかどうか

        Returns:
            CLIP画像特徴テンソル
        """
        with torch.inference_mode():
            if batch_mode:
                # バッチ処理用の入力を作成
                inputs = self.processor(
                    images=pil_image,  # type: ignore[arg-type]
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                image_features: torch.Tensor = self.model.get_image_features(**inputs)
                return image_features
            else:
                # 単一画像処理
                inputs = self.processor(
                    images=pil_image,  # type: ignore[arg-type]
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                single_image_features = self.model.get_image_features(**inputs)
                assert isinstance(single_image_features, torch.Tensor)
                return single_image_features

    def get_text_embeddings(self) -> torch.Tensor:
        """キャッシュされたテキスト埋め込みを返す.

        Returns:
            テキスト埋め込みテンソル
        """
        return self._text_embeddings

    @property
    def target_text(self) -> str:
        """ターゲットテキストを返す."""
        return self._target_text
