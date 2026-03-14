"""Layout heuristic analyzer."""

import cv2
import numpy as np

from ..models.layout_heuristics import LayoutHeuristics


class LayoutAnalyzer:
    """画像のレイアウト傾向を簡易的に推定する."""

    @staticmethod
    def analyze(img: np.ndarray) -> LayoutHeuristics:
        """OpenCV画像からレイアウトヒューリスティクスを算出する."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        bottom_start = int(height * 0.65)
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

        bottom_std = float(np.std(bottom_region)) / 64.0
        upper_std = float(np.std(upper_region)) / 64.0
        global_std = float(np.std(gray)) / 64.0
        brightness = float(np.mean(gray)) / 255.0
        center_brightness = float(np.mean(center_region)) / 255.0

        dialogue_overlay_score = float(
            min(
                1.0,
                max(0.0, bottom_edge_density * 4.0)
                * max(0.0, 1.0 - min(1.0, bottom_std))
                * max(0.0, 1.0 - brightness * 0.2),
            )
        )
        menu_layout_score = float(
            min(
                1.0,
                max(0.0, whole_edge_density * 4.5)
                * max(0.0, 1.0 - min(1.0, abs(bottom_std - upper_std))),
            )
        )
        title_layout_score = float(
            min(
                1.0,
                max(0.0, center_edge_density * 3.0)
                * max(0.0, 1.0 - min(1.0, global_std))
                * max(0.0, center_brightness),
            )
        )
        game_over_layout_score = float(
            min(
                1.0,
                max(0.0, 1.0 - brightness * 1.2) * max(0.0, center_edge_density * 3.0),
            )
        )

        return LayoutHeuristics(
            dialogue_overlay_score=dialogue_overlay_score,
            menu_layout_score=menu_layout_score,
            title_layout_score=title_layout_score,
            game_over_layout_score=game_over_layout_score,
        )
