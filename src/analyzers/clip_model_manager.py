"""CLIPモデルのライフサイクルを管理するマネージャー."""

import logging
import os
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class CLIPModelManager:
    """CLIPモデルのライフサイクルとデバイス管理を行うクラス.

    モデルロード、テキスト埋め込みのキャッシュ、画像特徴抽出を提供する。
    """

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

        # デバイス設定（CUDA → MPS → CPUの優先順位で自動検出）
        self.device = device or self._detect_device()

        # モデルとプロセッサのロード
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # デバイスにモデルを転送
        self.model.to(self.device)
        self.model.eval()

        # GPU最適化: TF32を許可
        # 理由: Ampere GPU以上ではTF32演算を使用することで、
        #       精度をほぼ維持したまま行列演算を高速化できる
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # テキスト埋め込みの事前計算とキャッシュ
        self._text_embeddings = self._precompute_text_embeddings()

    @staticmethod
    def _detect_device() -> str:
        """利用可能な最適なデバイスを自動検出する.

        Returns:
            検出されたデバイス名（"cuda", "mps", "cpu"のいずれか）
        """
        # 診断情報の出力
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA が利用可能です (device: {device_name})")
        else:
            if torch.version.cuda:
                ld_path = os.environ.get("LD_LIBRARY_PATH", "未設定")
                logger.warning(
                    f"CUDA が検出されません。"
                    f"torch.version.cuda={torch.version.cuda}, "
                    f"LD_LIBRARY_PATH={ld_path}"
                )
            else:
                logger.warning(
                    "CUDA が検出されません。"
                    "CPU-only版PyTorchがインストールされています。"
                    "CUDA版をインストールしてください。"
                )

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

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
        self, pil_image: Image.Image | Sequence[Image.Image], batch_mode: bool = False
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
                # Sequence -> list 変換（transformers は list を期待）
                images_arg = (
                    list(pil_image)
                    if not isinstance(pil_image, Image.Image)
                    else [pil_image]
                )
                inputs = self.processor(
                    images=images_arg,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                image_features: torch.Tensor = self.model.get_image_features(**inputs)
                return image_features
            else:
                # 単一画像処理（型を明示的に単一画像に制限）
                if isinstance(pil_image, Image.Image):
                    inputs = self.processor(
                        images=pil_image,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)
                    single_image_features = self.model.get_image_features(**inputs)
                    assert isinstance(single_image_features, torch.Tensor)
                    return single_image_features
                else:
                    # 非バッチモードでシーケンスが渡された場合のフォールバック
                    raise ValueError("非バッチモードでは単一のPIL画像を渡してください")

    def get_normalized_image_features(self, pil_image: Image.Image) -> torch.Tensor:
        """PIL画像から正規化済みCLIP画像特徴を抽出する.

        単一画像の特徴抽出をL2正規化済みで返すユーティリティメソッド。
        feature_extractor.py と metric_calculator.py の重複コードを統合する。

        Args:
            pil_image: PIL画像オブジェクト（RGB形式）

        Returns:
            L2正規化済みのCLIP画像特徴テンソル（1次元の512要素）
        """
        with torch.inference_mode():
            inputs = self.processor(
                images=pil_image,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            image_features = self.model.get_image_features(**inputs)
            # L2正規化して返す（最初の要素を抽出）
            return F.normalize(image_features, p=2, dim=-1)[0]

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
