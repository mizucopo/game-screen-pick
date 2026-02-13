"""画像処理ユーティリティ."""

import cv2
import numpy as np
from PIL import Image

from .exception_handler import ExceptionHandler


class ImageUtils:
    """画像処理ユーティリティクラス.

    PILとOpenCV間の画像変換機能と画像読み込み機能を提供する。
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

    @staticmethod
    def load_as_rgb(path: str) -> Image.Image | None:
        """画像を読み込み、RGB形式に変換してコピーを返す.

        Args:
            path: 画像ファイルパス

        Returns:
            RGB形式のPIL画像（失敗時はNone）
        """
        try:
            with Image.open(path) as img:
                if img.mode != "RGB":
                    return img.convert("RGB").copy()
                return img.copy()
        except ExceptionHandler.get_expected_image_errors():
            return None
