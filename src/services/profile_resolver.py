"""解析済み画像群から選定プロファイルを解決する."""

from ..models.analyzed_image import AnalyzedImage
from ..models.scene_assessment import SceneAssessment


class ProfileResolver:
    """active / static プロファイルを決める.

    明示指定がある場合はそれを優先し、`auto` の場合だけ
    入力画像群の平均的な活動量とUI量から適したプロファイルを推定する。
    """

    def resolve(
        self,
        requested_profile: str,
        analyzed_images: list[AnalyzedImage],
        assessments: list[SceneAssessment],
    ) -> tuple[str, dict[str, float]]:
        """実行プロファイルを決定する.

        明示的に `active` / `static` が指定されていればその値を返す。
        `auto` の場合は action、edge、UI、gameplay、event の平均値から
        active寄りかstatic寄りかを数値化し、より高い方を採用する。

        Args:
            requested_profile: CLIまたは設定から渡された希望プロファイル。
            analyzed_images: 中立解析済み画像の一覧。
            assessments: 各画像に対応するscene判定結果。

        Returns:
            1. 解決済みプロファイル名
            2. 判定に使った profile ごとのスコア
        """
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
        avg_gameplay = sum(
            assessment.gameplay_score for assessment in assessments
        ) / len(assessments)
        avg_event = sum(assessment.event_score for assessment in assessments) / len(
            assessments
        )

        active_score = (
            0.35 * avg_action + 0.25 * avg_edge + 0.15 * avg_gameplay + 0.25 * avg_event
        )
        static_score = (
            0.45 * avg_ui
            + 0.20 * avg_gameplay
            + 0.20 * (1.0 - avg_action)
            + 0.15 * (1.0 - avg_event)
        )

        resolved = "active" if active_score >= static_score else "static"
        return resolved, {"active": active_score, "static": static_score}
