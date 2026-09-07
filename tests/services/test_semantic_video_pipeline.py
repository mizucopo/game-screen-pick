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

    def __init__(self, config: VllmConfig) -> None:
        super().__init__()
        self.config = config

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
        return super().assess(
            model=model,
            model_digest=f"digest-{model}",
            prompt=prompt,
            candidates=candidates,
            contact_sheet=contact_sheet,
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
