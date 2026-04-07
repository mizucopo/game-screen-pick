"""CLIPモデルのライフサイクルを管理するマネージャー."""

import logging
import os

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
        device: str | None = None,
    ):
        """CLIPモデルマネージャーを初期化する.

        Args:
            model_name: 使用するCLIPモデル名
            device: 使用するデバイス（Noneの場合は自動検出）
        """
        self.model_name = model_name

        # デバイス設定（CUDA → MPS → CPUの優先順位で自動検出）
        self.device = device or self._detect_device()

        # モデルとプロセッサは遅延初期化（実際に使用されるまでロードしない）
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None

    @property
    def model(self) -> CLIPModel:
        """モデルを返す（必要に応じてロード）."""
        self._ensure_model_loaded()
        assert self._model is not None
        return self._model

    @property
    def processor(self) -> CLIPProcessor:
        """プロセッサを返す（必要に応じてロード）."""
        self._ensure_model_loaded()
        assert self._processor is not None
        return self._processor

    def _ensure_model_loaded(self) -> None:
        """モデルが未ロードならロードする."""
        if self._model is not None:
            return

        logger.info(f"CLIPモデルをロードしています ({self.model_name})...")
        self._model = CLIPModel.from_pretrained(self.model_name)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

        # デバイスにモデルを転送
        logger.info(f"モデルをデバイスに転送しています ({self.device})...")
        assert self._model is not None  # 型チェッカー用
        self._model.to(torch.device(self.device))  # type: ignore[arg-type]
        self._model.eval()

        # GPU最適化: TF32を許可
        # 理由: Ampere GPU以上ではTF32演算を使用することで、
        #       精度をほぼ維持したまま行列演算を高速化できる
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

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
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        logger.info(f"デバイス: {device} を使用します")
        return device

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
