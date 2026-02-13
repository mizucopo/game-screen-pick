"""共通例外定義."""

from PIL import UnidentifiedImageError
import cv2


class ExceptionHandler:
    """例外ハンドリングユーティリティクラス.

    画像処理で正常な失敗として扱う例外型を提供する。
    """

    @staticmethod
    def get_expected_image_errors() -> tuple[type[Exception], ...]:
        """画像処理で正常な失敗として扱うエラー型.

        Returns:
            正常な失敗として扱う例外型のタプル
        """
        return (
            FileNotFoundError,
            UnidentifiedImageError,
            OSError,
            cv2.error,
            ValueError,
        )
