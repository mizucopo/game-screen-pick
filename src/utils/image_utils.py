"""画像処理ユーティリティ."""

import cv2
import numpy as np
from PIL import Image


class ImageUtils:
    """画像処理ユーティリティクラス.

    PILとOpenCV間の画像変換機能を提供する。
    """

    @staticmethod
    def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
        """PIL画像をOpenCV形式（BGR）に変換する.

        Args:
            pil_img: PIL画像（RGB形式）

        Returns:
            OpenCV画像（BGR形式）のNumPy配列
        """
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
