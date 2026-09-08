"""動画理解で見つけた場面から最終画像までの統合契約."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import Mock

import pytest

from src.models.semantic_video import SemanticVideoPlan
from src.models.video_selection import FrameAssessment, FrameCandidate
from src.models.video_selection_request import VideoSelectionRequest
from src.models.vllm_config import VllmConfig
from src.services.video_selector import VideoSelector
from src.services.vllm_client import VllmClient
from tests.services.test_video_pipeline import FakeAssessor, FakeFrameExtractor


class SemanticAssessor(FakeAssessor):
    """実効vLLMモデルと動画から得た文脈を検証する外部境界."""

    def __init__(
        self, config: VllmConfig, *, transition_stage: str | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.transition_stage = transition_stage
        self.primary_batches: list[list[float]] = []
        self.quality_scores: list[float] = []

    def fetch_model_metadata(
        self, requested_models: set[str]
    ) -> dict[str, dict[str, Any]]:
        assert requested_models == {self.config.model}
        return {self.config.model: VllmClient(self.config).model_metadata()}

    def assess(
        self,
        *,
        model: str,
        model_digest: str,
        prompt: str,
        candidates: Sequence[FrameCandidate],
        contact_sheet: Path,
    ) -> list[FrameAssessment]:
        assert model == self.config.model
        assert model_digest == VllmClient(self.config).model_metadata()["digest"]
        assert "ボス撃破" in prompt
        primary = "中央だけを採点" not in prompt
        if primary:
            self.primary_batches.append(
                [candidate.timestamp_seconds for candidate in candidates]
            )
        self.quality_scores.extend(candidate.quality_score for candidate in candidates)
        assessments = super().assess(
            model=model,
            model_digest=f"digest-{model}",
            prompt=prompt,
            candidates=candidates,
            contact_sheet=contact_sheet,
        )
        if self.transition_stage is None:
            return assessments
        return [
            replace(
                assessment,
                is_transition=(
                    primary == (self.transition_stage == "primary")
                    and candidate.timestamp_seconds <= 1.11
                ),
            )
            for assessment, candidate in zip(assessments, candidates, strict=True)
        ]


class SameImageExtractor(FakeFrameExtractor):
    """機械的品質と見た目を揃え、場面の重要度だけが異なる動画."""

    def extract_frame(
        self,
        video: Path,
        timestamp_seconds: float,
        output_path: Path,
        *,
        max_width: int | None,
        video_stream_index: int = 0,
    ) -> None:
        del timestamp_seconds
        super().extract_frame(
            video,
            0.6,
            output_path,
            max_width=max_width,
            video_stream_index=video_stream_index,
        )


def semantic_request(tmp_path: Path) -> VideoSelectionRequest:
    video = tmp_path / "game.mp4"
    video.write_bytes(b"test video")
    return VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "out"),
        output_count=2,
        game_title=None,
        game_context="探索とボス戦",
        primary_model="unused-primary",
        secondary_model="unused-secondary",
        ollama_host="http://unused-ollama:bad",
        ollama_timeout=1.0,
        allow_cpu=False,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
        selection_method="semantic_video",
        vllm_config=VllmConfig(model="video-model"),
    )


def semantic_plan() -> SemanticVideoPlan:
    return SemanticVideoPlan(
        timestamps=(0.6, 1.1, 1.6, 2.1, 2.6, 3.1),
        cache_key="semantic-plan-key",
        evidence={
            "events": [
                {
                    "chunk_key": "chunk-1",
                    "start_seconds": 0.5,
                    "end_seconds": 3.2,
                    "timestamp_seconds": 2.1,
                    "summary": "ボス撃破",
                    "importance": 95,
                }
            ]
        },
    )


def test_video_meaning_drives_extraction_assessment_and_report(tmp_path: Path) -> None:
    request = semantic_request(tmp_path)
    assert request.vllm_config is not None
    assessor = SemanticAssessor(request.vllm_config)
    extractor = FakeFrameExtractor()
    planner = Mock()
    planner.plan.return_value = semantic_plan()
    selector = VideoSelector(
        request,
        frame_extractor=extractor,
        assessor=assessor,
        semantic_planner=planner,
    )
    assert selector.run().is_file()
    assert selector.sources[0].timestamps == semantic_plan().timestamps
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["selection_method"] == "semantic_video"
    assert report["models"]["primary"]["name"] == "video-model"
    assert report["models"]["secondary"]["name"] == "video-model"
    for selected in report["selected"]:
        assert selected["timestamp_seconds"] in semantic_plan().timestamps
        assert selected["semantic_provenance"][0]["summary"] == "ボス撃破"
    calls = assessor.assess_calls
    extraction_calls = extractor.extract_calls
    assert selector.run().is_file()
    assert assessor.assess_calls == calls
    assert extractor.extract_calls == extraction_calls
    VideoSelector(
        replace(request, allow_cpu=True),
        frame_extractor=extractor,
        assessor=assessor,
        semantic_planner=planner,
    ).run()
    assert assessor.assess_calls == calls
    assert extractor.extract_calls == extraction_calls


def test_whitespace_title_does_not_activate_unused_ollama(tmp_path: Path) -> None:
    request = replace(
        semantic_request(tmp_path),
        game_title="   ",
        game_context_provider="ollama",
    )
    assert request.vllm_config is not None
    planner = Mock()
    planner.plan.return_value = semantic_plan()
    context_generator = Mock()

    assert (
        VideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=SemanticAssessor(request.vllm_config),
            semantic_planner=planner,
            context_generator=context_generator,
        )
        .run()
        .is_file()
    )
    context_generator.generate.assert_not_called()


def test_semantic_failure_preserves_previous_completed_images(tmp_path: Path) -> None:
    request = semantic_request(tmp_path)
    assert request.vllm_config is not None
    planner = Mock()
    planner.plan.return_value = semantic_plan()
    VideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=SemanticAssessor(request.vllm_config),
        semantic_planner=planner,
    ).run()
    output = tmp_path / "out"
    previous = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    planner.plan.side_effect = RuntimeError("video understanding unavailable")
    with pytest.raises(RuntimeError, match="video understanding unavailable"):
        VideoSelector(
            replace(request, output_count=3),
            frame_extractor=FakeFrameExtractor(),
            assessor=SemanticAssessor(request.vllm_config),
            semantic_planner=planner,
        ).run()
    assert {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    } == previous


def test_sampled_frames_rejects_injected_semantic_planner(tmp_path: Path) -> None:
    request = replace(
        semantic_request(tmp_path),
        selection_method="sampled_frames",
        vllm_config=None,
        ollama_host="http://fake-ollama",
    )
    extractor = FakeFrameExtractor()
    planner = Mock()
    planner.plan.return_value = semantic_plan()
    selector = VideoSelector(
        request,
        frame_extractor=extractor,
        assessor=FakeAssessor(),
        semantic_planner=planner,
    )

    with pytest.raises(ValueError, match="semantic_planner.*semantic_video"):
        selector.run()

    assert extractor.probe_calls == 0
    assert extractor.extract_calls == 0
    planner.plan.assert_not_called()


@pytest.mark.parametrize("transition_stage", ["primary", "secondary"])
def test_transition_backfill_keeps_semantic_importance_ranking(
    tmp_path: Path, transition_stage: str
) -> None:
    request = replace(semantic_request(tmp_path), output_count=1)
    assert request.vllm_config is not None
    timestamps = tuple(round(1.0 + index / 100, 2) for index in range(20))
    planner = Mock()
    planner.plan.return_value = SemanticVideoPlan(
        timestamps=timestamps,
        cache_key="semantic-backfill-plan",
        evidence={
            "events": [
                {
                    "start_seconds": timestamp,
                    "end_seconds": timestamp,
                    "timestamp_seconds": timestamp,
                    "summary": "ボス撃破",
                    "importance": (
                        100
                        if timestamp <= 1.11
                        else 80
                        if 1.15 <= timestamp <= 1.17
                        else 10
                    ),
                }
                for timestamp in timestamps
            ]
        },
    )
    assessor = SemanticAssessor(request.vllm_config, transition_stage=transition_stage)

    assert (
        VideoSelector(
            request,
            frame_extractor=SameImageExtractor(),
            assessor=assessor,
            semantic_planner=planner,
        )
        .run()
        .is_file()
    )

    assert assessor.primary_batches[1] == [1.15, 1.16, 1.17]
    assert len(set(assessor.quality_scores)) == 1
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["selected"][0]["semantic_provenance"][0]["importance"] == 80
