"""レイアウトヒューリスティクス解析器。"""

import cv2
import numpy as np

from ..models.layout_heuristics import LayoutHeuristics


class LayoutAnalyzer:
    """画像のレイアウト傾向を簡易的に推定する."""

    BOTTOM_REGION_RATIO: float = 0.65
    STD_NORMALIZATION_DIVISOR: float = 64.0
    DIALOGUE_EDGE_MULTIPLIER: float = 4.0
    DIALOGUE_BRIGHTNESS_FACTOR: float = 0.2
    MENU_EDGE_MULTIPLIER: float = 4.5
    TITLE_EDGE_MULTIPLIER: float = 3.0
    GAME_OVER_BRIGHTNESS_MULTIPLIER: float = 1.2
    GAME_OVER_EDGE_MULTIPLIER: float = 3.0

    @staticmethod
    def analyze(img: np.ndarray) -> LayoutHeuristics:
        """OpenCV画像からレイアウトヒューリスティクスを算出する."""
        LA = LayoutAnalyzer
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        bottom_start = int(height * LA.BOTTOM_REGION_RATIO)
        bottom_region = gray[bottom_start:, :]
        center_region = gray[
            height // 4 : (height * 3) // 4, width // 4 : (width * 3) // 4
        ]
        upper_region = gray[: height // 3, :]

        bottom_edges = cv2.Canny(bottom_region, 50, 150)
        whole_edges = cv2.Canny(gray, 50, 150)
        center_edges = cv2.Canny(center_region, 50, 150)

        bottom_edge_density = cv2.countNonZero(bottom_edges) / max(
            1, bottom_region.size
        )
        whole_edge_density = cv2.countNonZero(whole_edges) / max(1, gray.size)
        center_edge_density = cv2.countNonZero(center_edges) / max(
            1, center_region.size
        )

        _, bottom_std_dev = cv2.meanStdDev(bottom_region)
        bottom_std = float(bottom_std_dev[0][0]) / LA.STD_NORMALIZATION_DIVISOR
        _, upper_std_dev = cv2.meanStdDev(upper_region)
        upper_std = float(upper_std_dev[0][0]) / LA.STD_NORMALIZATION_DIVISOR
        _, global_std_dev = cv2.meanStdDev(gray)
        global_std = float(global_std_dev[0][0]) / LA.STD_NORMALIZATION_DIVISOR
        brightness = float(np.mean(gray)) / 255.0
        center_brightness = float(np.mean(center_region)) / 255.0

        dialogue_overlay_score = float(
            min(
                1.0,
                max(0.0, bottom_edge_density * LA.DIALOGUE_EDGE_MULTIPLIER)
                * max(0.0, 1.0 - min(1.0, bottom_std))
                * max(0.0, 1.0 - brightness * LA.DIALOGUE_BRIGHTNESS_FACTOR),
            )
        )
        menu_layout_score = float(
            min(
                1.0,
                max(0.0, whole_edge_density * LA.MENU_EDGE_MULTIPLIER)
                * max(0.0, 1.0 - min(1.0, abs(bottom_std - upper_std))),
            )
        )
        title_layout_score = float(
            min(
                1.0,
                max(0.0, center_edge_density * LA.TITLE_EDGE_MULTIPLIER)
                * max(0.0, 1.0 - min(1.0, global_std))
                * max(0.0, center_brightness),
            )
        )
        game_over_layout_score = float(
            min(
                1.0,
                max(0.0, 1.0 - brightness * LA.GAME_OVER_BRIGHTNESS_MULTIPLIER)
                * max(0.0, center_edge_density * LA.GAME_OVER_EDGE_MULTIPLIER),
            )
        )

        return LayoutHeuristics(
            dialogue_overlay_score=dialogue_overlay_score,
            menu_layout_score=menu_layout_score,
            title_layout_score=title_layout_score,
            game_over_layout_score=game_over_layout_score,
        )
