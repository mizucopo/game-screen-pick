"""ベクトル操作ユーティリティ."""

from typing import Any

import numpy as np


class VectorUtils:
    """ベクトル操作に関するユーティリティクラス."""

    @staticmethod
    def safe_l2_normalize(
        vec: np.ndarray[Any, Any], eps: float = 1e-8
    ) -> np.ndarray[Any, Any]:
        """ゼロ割れ安全なL2正規化を行う.

        Args:
            vec: 正規化するベクトル
            eps: ゼロ割れ防止用の微小値

        Returns:
            L2正規化されたベクトル（元のノルムが0の場合はゼロベクトル）
        """
        norm = float(np.linalg.norm(vec))
        if norm < eps:
            return np.zeros_like(vec)
        return vec / norm
