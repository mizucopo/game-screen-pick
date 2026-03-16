"""解析済み画像群から選定プロファイルを解決する."""

from ..models.analyzed_image import AnalyzedImage


class ProfileResolver:
    """active / static プロファイルを決める."""

    def resolve(
        self,
        requested_profile: str,
        analyzed_images: list[AnalyzedImage],
    ) -> tuple[str, dict[str, float]]:
        """実行プロファイルを決定する."""
        if requested_profile in {"active", "static"}:
            return requested_profile, {requested_profile: 1.0}

        if not analyzed_images:
            return "active", {"active": 0.5, "static": 0.5}

        avg_action = sum(
            image.normalized_metrics.action_intensity for image in analyzed_images
        ) / len(analyzed_images)
        avg_edge = sum(
            image.normalized_metrics.edge_density for image in analyzed_images
        ) / len(analyzed_images)
        avg_ui = sum(
            image.normalized_metrics.ui_density for image in analyzed_images
        ) / len(analyzed_images)

        active_score = 0.45 * avg_action + 0.35 * avg_edge + 0.20 * (1.0 - avg_ui)
        static_score = 0.50 * avg_ui + 0.25 * (1.0 - avg_action) + 0.25 * (
            1.0 - avg_edge
        )

        resolved = "active" if active_score >= static_score else "static"
        return resolved, {"active": active_score, "static": static_score}
