"""単一動画production pipelineの小さな結合テスト."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import pytest
from PIL import Image, ImageDraw

from src.models.video_selection import (
    FrameAssessment,
    FrameCandidate,
    VideoMetadata,
)
from src.models.video_selection_request import VideoSelectionRequest
from src.services.ollama_frame_assessor import OllamaFrameAssessor
from src.services.single_video_selector import SingleVideoSelector
from src.services.video_frame_extractor import VideoFrameExtractor
from src.utils.video_selection_files import file_sha256


class FakeFrameExtractor(VideoFrameExtractor):
    """実動画を使わず決定的なframeを生成するfake."""

    def __init__(self) -> None:
        """外部command確認を省略する."""
        self.extract_calls = 0

    def probe(self, video: Path) -> VideoMetadata:
        """短いtest動画のmetadataを返す."""
        assert video.is_file()
        return VideoMetadata(4.0, 320, 180, "fake", "30/1")

    def extract_frame(
        self,
        video: Path,
        timestamp_seconds: float,
        output_path: Path,
        *,
        max_width: int | None,
    ) -> None:
        """時刻に応じた見た目のJPEGを生成する."""
        assert video.is_file()
        assert max_width is None or max_width == 960
        self.extract_calls += 1
        marker = round(timestamp_seconds * 100)
        image = Image.new(
            "RGB",
            (320, 180),
            (20 + marker * 7 % 180, 30 + marker * 11 % 180, 40),
        )
        draw = ImageDraw.Draw(image)
        left = marker * 13 % 240
        draw.rectangle((left, 20, left + 70, 160), fill="white")
        draw.line((0, marker % 180, 319, 179 - marker % 180), fill="red", width=8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="JPEG", quality=95)


class FakeAssessor(OllamaFrameAssessor):
    """全候補へ決定的な評価を返すfake."""

    def __init__(self) -> None:
        """network設定を省略する."""
        self.host = "http://fake"
        self.timeout_seconds = 1.0
        self.require_gpu = False
        self.gpu_evidence: dict[str, dict[str, Any]] = {}
        self.assess_calls = 0

    def fetch_model_metadata(
        self,
        requested_models: set[str],
    ) -> dict[str, dict[str, Any]]:
        """固定digestを返す."""
        return {
            model: {
                "digest": f"digest-{model}",
                "capabilities": ["vision"],
                "details": {},
            }
            for model in sorted(requested_models)
        }

    def assess(
        self,
        *,
        model: str,
        model_digest: str,
        prompt: str,
        candidates: Sequence[FrameCandidate],
        contact_sheet: Path,
    ) -> list[FrameAssessment]:
        """frame IDごとの固定scoreとsceneを返す."""
        assert model_digest == f"digest-{model}"
        assert "全編録画" in prompt
        assert contact_sheet.is_file()
        self.assess_calls += 1
        return [
            FrameAssessment(
                frame_id=candidate.frame_id,
                blog_score=70.0 + candidate.timestamp_seconds,
                is_transition=False,
                scene=("探索" if int(candidate.frame_id[1:]) % 2 else "会話"),
                reason="test",
            )
            for candidate in candidates
        ]


def test_pipeline_outputs_artifacts_and_reuses_completed_run(tmp_path: Path) -> None:
    """成果物を揃え、同条件再実行ではmodel評価を繰り返さないこと."""
    video = tmp_path / "Sample Game Part3.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    extractor = FakeFrameExtractor()
    assessor = FakeAssessor()

    contact_sheet = SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=assessor,
    ).run()

    selected_paths = [output_dir / "selected-01.jpg", output_dir / "selected-02.jpg"]
    assert contact_sheet == output_dir / "selected-contact-sheet.jpg"
    assert contact_sheet.is_file()
    assert all(path.is_file() for path in selected_paths)
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["game_title"] == "Sample Game"
    assert report["output_count"] == 2
    assert len(report["selected"]) == 2
    completion = json.loads(
        (output_dir / ".game-screen-pick" / "completion.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(completion["artifacts"]) == 4
    first_hashes = [file_sha256(path) for path in selected_paths]
    calls_after_first_run = assessor.assess_calls
    extraction_after_first_run = extractor.extract_calls

    resumed_sheet = SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=assessor,
    ).run()

    assert resumed_sheet == contact_sheet
    assert assessor.assess_calls == calls_after_first_run
    assert extractor.extract_calls == extraction_after_first_run
    assert [file_sha256(path) for path in selected_paths] == first_hashes


def test_pipeline_does_not_reuse_cpu_allowed_cache_for_gpu_required_run(
    tmp_path: Path,
) -> None:
    """GPU保証が異なる既存outputを同じrunとして再開しないこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    cpu_allowed_request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    extractor = FakeFrameExtractor()
    assessor = FakeAssessor()
    SingleVideoSelector(
        cpu_allowed_request,
        frame_extractor=extractor,
        assessor=assessor,
    ).run()

    with pytest.raises(RuntimeError, match="実行条件が今回と異なります"):
        SingleVideoSelector(
            replace(cpu_allowed_request, allow_cpu=False),
            frame_extractor=extractor,
            assessor=assessor,
        ).run()
