"""copy / report / console に渡す出力record。"""

from dataclasses import dataclass, replace

from .metric_distribution import MetricDistribution
from .output_candidate_record import OutputCandidateRecord
from .picker_statistics import PickerStatistics
from .scored_candidate import ScoredCandidate
from .whole_input_profile import WholeInputProfile


@dataclass(frozen=True)
class OutputRecord:
    """出力adapterが共有して利用する選定結果record."""

    selected: list[OutputCandidateRecord]
    rejected: list[OutputCandidateRecord]
    total_files: int
    analyzed_ok: int
    analyzed_fail: int
    rejected_by_similarity: int
    rejected_by_content_filter: int
    selected_count: int
    resolved_profile: str
    scene_distribution: dict[str, int]
    scene_mix_target: dict[str, int]
    scene_mix_actual: dict[str, int]
    threshold_relaxation_steps: list[float]
    content_filter_breakdown: dict[str, int]
    whole_input_profile: dict[str, dict[str, float]] | None
    scene_catalog: list[dict[str, str]]
    ollama_classification_failed: int
    ollama_classification_failure_rate: float

    @classmethod
    def from_selection(
        cls,
        selected: list[ScoredCandidate],
        rejected: list[ScoredCandidate],
        stats: PickerStatistics,
    ) -> "OutputRecord":
        """選定結果と統計情報を出力用recordへ射影する."""
        return cls(
            selected=[
                OutputCandidateRecord.from_scored_candidate(
                    candidate,
                    stats.selection_annotations_by_path.get(candidate.path),
                )
                for candidate in selected
            ],
            rejected=[
                OutputCandidateRecord.from_scored_candidate(
                    candidate,
                    stats.selection_annotations_by_path.get(candidate.path),
                )
                for candidate in rejected
            ],
            total_files=stats.total_files,
            analyzed_ok=stats.analyzed_ok,
            analyzed_fail=stats.analyzed_fail,
            rejected_by_similarity=stats.rejected_by_similarity,
            rejected_by_content_filter=stats.rejected_by_content_filter,
            selected_count=stats.selected_count,
            resolved_profile=stats.resolved_profile,
            scene_distribution=dict(stats.scene_distribution),
            scene_mix_target=dict(stats.scene_mix_target),
            scene_mix_actual=dict(stats.scene_mix_actual),
            threshold_relaxation_steps=list(stats.threshold_relaxation_steps),
            content_filter_breakdown=dict(stats.content_filter_breakdown),
            whole_input_profile=cls._serialize_whole_input_profile(
                stats.whole_input_profile
            ),
            scene_catalog=[
                {
                    "slug": scene.slug,
                    "display_name": scene.display_name,
                    "description": scene.description,
                }
                for scene in stats.scene_catalog
            ],
            ollama_classification_failed=stats.ollama_classification_failed,
            ollama_classification_failure_rate=round(
                stats.ollama_classification_failure_rate,
                4,
            ),
        )

    def with_selected_output_paths(
        self,
        output_paths_by_source_path: dict[str, str],
    ) -> "OutputRecord":
        """コピー後の出力先パスを選択候補へ反映したrecordを返す."""
        return replace(
            self,
            selected=[
                candidate.with_output_path(
                    output_paths_by_source_path.get(candidate.source_path)
                )
                for candidate in self.selected
            ],
        )

    @staticmethod
    def _serialize_distribution(
        distribution: MetricDistribution,
    ) -> dict[str, float]:
        """分布プロフィールをJSON互換の値へ変換する."""
        return {
            "p10": round(distribution.p10, 4),
            "p25": round(distribution.p25, 4),
            "p50": round(distribution.p50, 4),
            "p90": round(distribution.p90, 4),
        }

    @classmethod
    def _serialize_whole_input_profile(
        cls,
        profile: WholeInputProfile | None,
    ) -> dict[str, dict[str, float]] | None:
        """入力全体分布プロフィールを出力record向けに変換する."""
        if profile is None:
            return None
        return {
            "brightness": cls._serialize_distribution(profile.brightness),
            "contrast": cls._serialize_distribution(profile.contrast),
            "edge_density": cls._serialize_distribution(profile.edge_density),
            "action_intensity": cls._serialize_distribution(profile.action_intensity),
            "luminance_entropy": cls._serialize_distribution(profile.luminance_entropy),
            "luminance_range": cls._serialize_distribution(profile.luminance_range),
            "near_black_ratio": cls._serialize_distribution(profile.near_black_ratio),
            "near_white_ratio": cls._serialize_distribution(profile.near_white_ratio),
            "dominant_tone_ratio": cls._serialize_distribution(
                profile.dominant_tone_ratio
            ),
        }
