"""特徴抽出器 - HSV, CLIP, 統合特徴の抽出を行う."""

import contextlib
import logging
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..utils.vector_utils import VectorUtils
from .clip_model_manager import CLIPModelManager

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """画像特徴抽出器.

    CLIPモデルマネージャーを使用して、HSV特徴、CLIP特徴、
    および統合特徴を抽出する。
    """

    def __init__(self, model_manager: "CLIPModelManager"):
        """特徴抽出器を初期化する.

        Args:
            model_manager: CLIPモデルマネージャー
        """
        self.model_manager = model_manager

    @staticmethod
    def extract_hsv_features(img: np.ndarray) -> np.ndarray:
        """HSV色空間のヒストグラム特徴を抽出する.

        注: 呼出し元（batch_pipeline.py）で既にmax_dim以下に縮小された画像を
        受け取ることを想定している。

        Args:
            img: OpenCV画像（BGR形式、既に縮小されている）

        Returns:
            正規化されたHSVヒストグラム特徴（64次元）
        """
        small = cv2.resize(img, (128, 128))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def extract_clip_features(self, pil_img: Image.Image) -> np.ndarray:
        """CLIP画像埋め込みを抽出する.

        CLIPModelManager.get_normalized_image_features を使用し、
        重複する推論コードを排除する。

        Args:
            pil_img: PIL画像（RGB形式）

        Returns:
            正規化されたCLIP画像埋め込み（512次元）
        """
        # CLIPModelManagerから正規化済み特徴を取得（torch.Tensor）
        features_tensor = self.model_manager.get_normalized_image_features(pil_img)
        # CPUに転送してNumPy配列に変換
        features = features_tensor.cpu().numpy()
        return VectorUtils.safe_l2_normalize(features)

    def extract_combined_features(
        self,
        img: np.ndarray,
        clip_features: np.ndarray,
    ) -> np.ndarray:
        """HSV特徴とCLIP特徴を結合する.

        注: imgは呼出し元（batch_pipeline.py）で既にmax_dim以下に
        縮小された画像であることを想定している。

        Args:
            img: OpenCV画像（BGR形式、既に縮小されている）
            clip_features: CLIP画像埋め込み（512次元、正規化済み、np.ndarray）

        Returns:
            結合された特徴ベクトル（576次元、np.ndarray）
        """
        hsv_features = FeatureExtractor.extract_hsv_features(img)
        # L2正規化（既に正規化されているが、安全のため再正規化）
        hsv_normalized = VectorUtils.safe_l2_normalize(hsv_features)

        # 結合
        return np.concatenate([hsv_normalized, clip_features])

    def extract_clip_features_batch(
        self,
        pil_images: Sequence[Image.Image | None],
        initial_batch_size: int = 32,
    ) -> list[torch.Tensor | None]:
        """複数のPIL画像に対してCLIP推論をバッチ実行して特徴を抽出.

        OOM対策（失敗したバッチのみ再試行）:
        - initial_batch_sizeから開始（デフォルト32）
        - torch.cuda.OutOfMemoryError発生時に失敗したバッチのみを分割してリトライ
        - 未処理のバッチは縮小されたバッチサイズで処理
        - 最小バッチサイズ1まで試行（32→16→8→4→2→1）
        - それでも失敗した画像はNoneとして返す

        パフォーマンス最適化:
        - CPU転送を削除し、torch.Tensorのまま返す
        - バッチ処理でもtensor形式を維持することで、
          後続のsemantic計算でのtensor→numpy→tensor往復を回避

        Args:
            pil_images: PIL画像のリスト（失敗した画像はNone）
            initial_batch_size: 初期バッチサイズ

        Returns:
            CLIP画像埋め込みのリスト（512次元、正規化済み、失敗した画像はNone）
            ※torch.Tensorのまま返す（デバイス上に配置）
        """
        # 有効な画像のインデックスと画像を収集
        valid_indices = [i for i, img in enumerate(pil_images) if img is not None]
        valid_images: Sequence[Image.Image] = [
            img for img in pil_images if img is not None
        ]

        if not valid_images:
            return [None] * len(pil_images)

        # 結果を格納する配列（初期値はNone）
        results: list[torch.Tensor | None] = [None] * len(pil_images)

        # 現在のバッチサイズ
        current_batch_size = initial_batch_size

        # 処理位置を追跡
        i = 0
        while i < len(valid_images):
            # 現在のバッチを取得
            batch_start = i
            batch_end = min(i + current_batch_size, len(valid_images))
            batch = valid_images[batch_start:batch_end]

            try:
                # GPU/CUDA/MPSの場合はautocastでfp16推論を使用（高速化）
                if self.model_manager.device in ("cuda", "mps"):
                    autocast_context: Any = torch.autocast(
                        device_type=self.model_manager.device,
                        dtype=torch.float16,
                    )
                else:
                    autocast_context = contextlib.nullcontext()
                with autocast_context, torch.inference_mode():
                    inputs = self.model_manager.processor(
                        images=list(batch),  # Sequence -> list 変換
                        return_tensors="pt",
                        padding=True,
                    ).to(self.model_manager.device)
                    image_features = self.model_manager.model.get_image_features(
                        **inputs
                    )

                    # バッチ単位でL2正規化（テンソルのまま処理して高速化）
                    batch_features_normalized = F.normalize(image_features, p=2, dim=-1)

                # 結果を元のインデックスにマッピング
                for j in range(len(batch_features_normalized)):
                    original_idx = valid_indices[batch_start + j]
                    results[original_idx] = batch_features_normalized[j]

                # バッチ処理成功、次へ進む
                i = batch_end

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # MPSのOOMはRuntimeErrorとして報告される場合がある
                error_msg = str(e).lower()
                is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or (
                    isinstance(e, RuntimeError)
                    and ("out of memory" in error_msg or "mps" in error_msg)
                )

                if is_oom:
                    # バッチサイズを半分にしてリトライ
                    new_batch_size = current_batch_size // 2
                    device_name = self.model_manager.device.upper()
                    if new_batch_size >= 1:
                        logger.warning(
                            f"{device_name} OOM発生（位置{batch_start}/"
                            f"{len(valid_images)}）。バッチサイズを{new_batch_size}に"
                            f"縮小してリトライします。"
                        )
                        current_batch_size = new_batch_size
                        # i は変更せず、同じ位置から小さいバッチサイズでリトライ
                    else:
                        logger.error(
                            f"バッチサイズ1でもOOMが発生しました（位置{batch_start}）。"
                            "残りの画像の処理をスキップします。"
                        )
                        # 処理済みの結果は残すが、未処理の画像はNoneのまま
                        break
                else:
                    # OOM以外のRuntimeErrorはそのまま再raise
                    raise

        return results
