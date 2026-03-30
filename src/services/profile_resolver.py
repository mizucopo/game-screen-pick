"""解析済み画像群から選定プロファイルを解決する."""

from ..constants.profile_weights import ProfileWeights
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

        active_score = (
            ProfileWeights.ACTION * avg_action
            + ProfileWeights.EDGE * avg_edge
            + ProfileWeights.UI_INVERSE * (1.0 - avg_ui)
        )
        static_score = (
            ProfileWeights.UI * avg_ui
            + ProfileWeights.ACTION_INVERSE * (1.0 - avg_action)
            + ProfileWeights.EDGE_INVERSE * (1.0 - avg_edge)
        )

        resolved = "active" if active_score >= static_score else "static"
        return resolved, {"active": active_score, "static": static_score}
