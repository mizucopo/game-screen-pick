"""1本以上の動画を扱うproduction pipelineの小さな結合テスト."""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any, Sequence

import pytest
from PIL import Image, ImageDraw

from src.models.video_selection import (
    FrameAssessment,
    FrameCandidate,
    VideoMetadata,
)
from src.models.video_selection_request import VideoSelectionRequest
from src.services.ollama_frame_assessor import (
    OllamaFrameAssessor,
    OllamaModelValidationError,
)
from src.services.video_frame_extractor import VideoFrameExtractor
from src.services.video_selector import VideoSelector as SingleVideoSelector
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
        video_stream_index: int = 0,
    ) -> None:
        """時刻に応じた見た目のJPEGを生成する."""
        assert video.is_file()
        assert max_width is None or max_width == 960
        assert video_stream_index == 0
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
                "resolved_name": model,
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


class InterruptingFrameExtractor(FakeFrameExtractor):
    """最初のframe抽出でCtrl+C相当を発生させるfake."""

    def extract_frame(
        self,
        video: Path,
        timestamp_seconds: float,
        output_path: Path,
        *,
        max_width: int | None,
        video_stream_index: int = 0,
    ) -> None:
        """queued job取消を検証するためKeyboardInterruptを送出する."""
        del video, timestamp_seconds, output_path, max_width, video_stream_index
        raise KeyboardInterrupt


class AliasFakeAssessor(FakeAssessor):
    """untagged modelを`:latest`へ解決するfake."""

    def __init__(self) -> None:
        """実際に評価へ渡されたmodel名を記録する."""
        super().__init__()
        self.assessed_models: list[str] = []

    def fetch_model_metadata(
        self,
        requested_models: set[str],
    ) -> dict[str, dict[str, Any]]:
        """requested keyごとにcanonical nameを返す."""
        assert requested_models == {"llava"}
        return {
            "llava": {
                "digest": "digest-llava:latest",
                "resolved_name": "llava:latest",
                "capabilities": ["vision"],
                "details": {},
            }
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
        """canonical model名を記録して固定評価を返す."""
        self.assessed_models.append(model)
        return super().assess(
            model=model,
            model_digest=model_digest,
            prompt=prompt,
            candidates=candidates,
            contact_sheet=contact_sheet,
        )


class UnavailableAssessor(FakeAssessor):
    """Ollamaへ接続できない状態を表すfake."""

    def __init__(self) -> None:
        """metadata取得回数を記録する."""
        super().__init__()
        self.metadata_calls = 0

    def fetch_model_metadata(
        self,
        requested_models: set[str],
    ) -> dict[str, dict[str, Any]]:
        """metadata取得が呼ばれた場合は接続失敗にする."""
        del requested_models
        self.metadata_calls += 1
        raise ConnectionError("Ollama is unavailable")


class GpuValidationFailingAssessor(FakeAssessor):
    """決定的なGPU検証失敗を返すfake."""

    def assess(
        self,
        *,
        model: str,
        model_digest: str,
        prompt: str,
        candidates: Sequence[FrameCandidate],
        contact_sheet: Path,
    ) -> list[FrameAssessment]:
        """呼出回数を記録してmodel検証errorを送出する."""
        del model, model_digest, prompt, candidates, contact_sheet
        self.assess_calls += 1
        raise OllamaModelValidationError("GPU利用を確認できません")


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

    unavailable_assessor = UnavailableAssessor()
    resumed_sheet = SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=unavailable_assessor,
    ).run()

    assert resumed_sheet == contact_sheet
    assert assessor.assess_calls == calls_after_first_run
    assert unavailable_assessor.assess_calls == 0
    assert unavailable_assessor.metadata_calls == 0
    assert extractor.extract_calls == extraction_after_first_run
    assert [file_sha256(path) for path in selected_paths] == first_hashes


def test_pipeline_logs_concrete_processing_without_generic_status(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """具体的な処理だけを出力し汎用状態と自動決定詳細を出さないこと."""
    video = tmp_path / "Sample Game\nPart3.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title="Sample Game",
        game_context="",
        primary_model="primary\nmodel",
        secondary_model="secondary\x1bmodel",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    caplog.set_level(logging.INFO)

    SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    ).run()

    messages = [record.getMessage() for record in caplog.records]
    assert '入力動画の情報を確認しています: 1/1件 "Sample Game\\nPart3.mp4"' in messages
    assert (
        '入力動画の同一性を確認しています: 1/1件 "Sample Game\\nPart3.mp4"' in messages
    )
    assert (
        'Ollamaモデル情報を確認しています: "primary\\nmodel, '
        'secondary\\u001bmodel"' in messages
    )
    assert "候補フレームを抽出します: 13/13件" in messages
    assert all(
        control not in message
        for message in messages
        for control in ("\n", "\r", "\x1b")
    )
    assert all("画像選定処理は動作中です" not in message for message in messages)
    assert all(not message.startswith("自動決定オプション:") for message in messages)


def test_pipeline_logs_assessment_completion_and_failure_without_batch_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """評価開始は省略し、完了と一時失敗は引き続き出力すること."""

    class RetryOnceAssessor(FakeAssessor):
        """最初の評価だけ一時失敗するfake."""

        def assess(
            self,
            *,
            model: str,
            model_digest: str,
            prompt: str,
            candidates: Sequence[FrameCandidate],
            contact_sheet: Path,
        ) -> list[FrameAssessment]:
            """一度だけ失敗し、その後は固定評価を返す."""
            if self.assess_calls == 0:
                self.assess_calls += 1
                raise ConnectionError("temporary failure")
            return super().assess(
                model=model,
                model_digest=model_digest,
                prompt=prompt,
                candidates=candidates,
                contact_sheet=contact_sheet,
            )

    monkeypatch.setattr("src.services.video_selector.time.sleep", lambda _: None)
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title="Sample Game",
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
    caplog.set_level(logging.INFO)

    SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=RetryOnceAssessor(),
    ).run()

    messages = [record.getMessage() for record in caplog.records]
    assert all("評価を開始します" not in message for message in messages)
    assert any(message.startswith("primary評価: ") for message in messages)
    assert any(message.startswith("secondary評価: ") for message in messages)
    assert (
        "primary評価batch 1の試行1が失敗しました: temporary failure" in messages
    )


def test_pipeline_selects_from_multiple_videos_and_reports_each_source(
    tmp_path: Path,
) -> None:
    """複数入力を一つのrunとして選定し各入力元をreportへ残すこと."""
    videos = (
        tmp_path / "Sample Game Part1.mp4",
        tmp_path / "Sample Game Part2.mp4",
    )
    for video in videos:
        video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_videos=tuple(str(video) for video in videos),
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

    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert [item["path"] for item in report["videos"]] == [
        str(video.resolve()) for video in videos
    ]
    assert {item["video_index"] for item in report["selected"]} == {1, 2}
    manifest = json.loads(
        (output_dir / ".game-screen-pick" / "run-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert [item["path"] for item in manifest["inputs"]] == [
        str(video.resolve()) for video in videos
    ]
    extraction_count = extractor.extract_calls
    unavailable_assessor = UnavailableAssessor()

    resumed_sheet = SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=unavailable_assessor,
    ).run()

    assert resumed_sheet == contact_sheet
    assert extractor.extract_calls == extraction_count
    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


def test_pipeline_backfills_source_when_reserved_primary_candidates_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """入力元の予約候補が全遷移なら未評価候補を追加評価すること."""
    failed_primary_ids: set[str] = set()

    def select_initial_candidates(
        candidates: Sequence[FrameCandidate],
        _metadata: Sequence[VideoMetadata],
        _output_count: int,
    ) -> list[FrameCandidate]:
        by_source = {
            video_index: [
                candidate
                for candidate in candidates
                if candidate.video_index == video_index
            ]
            for video_index in range(2)
        }
        failed_primary_ids.update(candidate.frame_id for candidate in by_source[0][:3])
        return [*by_source[0][:3], *by_source[1][:6]]

    class ReservedCandidateFailingAssessor(FakeAssessor):
        """最初の入力元で予約された一次候補だけを遷移と判定するfake."""

        def assess(
            self,
            *,
            model: str,
            model_digest: str,
            prompt: str,
            candidates: Sequence[FrameCandidate],
            contact_sheet: Path,
        ) -> list[FrameAssessment]:
            assessments = super().assess(
                model=model,
                model_digest=model_digest,
                prompt=prompt,
                candidates=candidates,
                contact_sheet=contact_sheet,
            )
            return [
                replace(
                    assessment,
                    is_transition=assessment.frame_id in failed_primary_ids,
                )
                for assessment in assessments
            ]

    monkeypatch.setattr(
        "src.services.video_selector.select_primary_candidates",
        select_initial_candidates,
    )
    videos = (
        tmp_path / "Sample Game Part1.mp4",
        tmp_path / "Sample Game Part2.mp4",
    )
    for video in videos:
        video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_videos=tuple(str(video) for video in videos),
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
    SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=ReservedCandidateFailingAssessor(),
    ).run()

    work_dir = output_dir / ".game-screen-pick"
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert (work_dir / "assessments-primary-backfill-0001.json").is_file()
    assert {item["video_index"] for item in report["selected"]} == {1, 2}
    assert not failed_primary_ids & {item["frame_id"] for item in report["selected"]}

    (work_dir / "completion.json").unlink()
    (output_dir / "selected-01.jpg").unlink()
    unavailable_assessor = UnavailableAssessor()
    SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=unavailable_assessor,
    ).run()

    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


def test_pipeline_backfills_source_when_secondary_representative_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """二次評価の唯一の代表が遷移なら同じ入力元から追補すること."""
    failed_secondary_ids: set[str] = set()

    def select_single_source_representative(
        candidates: Sequence[FrameCandidate],
        _assessments: dict[str, FrameAssessment],
        count: int,
    ) -> list[FrameCandidate]:
        by_source = {
            video_index: [
                candidate
                for candidate in candidates
                if candidate.video_index == video_index
            ]
            for video_index in range(2)
        }
        return [by_source[0][0], *by_source[1][: count - 1]]

    class SecondaryRepresentativeFailingAssessor(FakeAssessor):
        """最初の入力元で最初の二次候補だけを遷移と判定するfake."""

        def assess(
            self,
            *,
            model: str,
            model_digest: str,
            prompt: str,
            candidates: Sequence[FrameCandidate],
            contact_sheet: Path,
        ) -> list[FrameAssessment]:
            assessments = super().assess(
                model=model,
                model_digest=model_digest,
                prompt=prompt,
                candidates=candidates,
                contact_sheet=contact_sheet,
            )
            if "厳しい再評価" in prompt and not failed_secondary_ids:
                failed_secondary_ids.add(
                    next(
                        candidate.frame_id
                        for candidate in candidates
                        if candidate.video_index == 0
                    )
                )
            return [
                replace(
                    assessment,
                    is_transition=assessment.frame_id in failed_secondary_ids,
                )
                for assessment in assessments
            ]

    monkeypatch.setattr(
        "src.services.video_selector.select_diverse_candidates",
        select_single_source_representative,
    )
    videos = (
        tmp_path / "Sample Game Part1.mp4",
        tmp_path / "Sample Game Part2.mp4",
    )
    for video in videos:
        video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_videos=tuple(str(video) for video in videos),
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

    SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=SecondaryRepresentativeFailingAssessor(),
    ).run()

    work_dir = output_dir / ".game-screen-pick"
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert (work_dir / "assessments-secondary-backfill-0001.json").is_file()
    assert {item["video_index"] for item in report["selected"]} == {1, 2}
    assert not failed_secondary_ids & {item["frame_id"] for item in report["selected"]}

    (work_dir / "completion.json").unlink()
    (output_dir / "selected-01.jpg").unlink()
    unavailable_assessor = UnavailableAssessor()
    SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=unavailable_assessor,
    ).run()

    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


def test_pipeline_backfills_when_secondary_survivors_cannot_fill_output(
    tmp_path: Path,
) -> None:
    """二次評価後の生存候補総数が出力枚数未満なら候補を追補すること."""
    failed_secondary_ids: set[str] = set()
    kept_secondary_sources: set[int] = set()

    class SecondaryCountFailingAssessor(FakeAssessor):
        """最初の二次評価で各入力元の1件以外を遷移にするfake."""

        def assess(
            self,
            *,
            model: str,
            model_digest: str,
            prompt: str,
            candidates: Sequence[FrameCandidate],
            contact_sheet: Path,
        ) -> list[FrameAssessment]:
            assessments = super().assess(
                model=model,
                model_digest=model_digest,
                prompt=prompt,
                candidates=candidates,
                contact_sheet=contact_sheet,
            )
            if contact_sheet.parent.name == "secondary":
                for candidate in candidates:
                    if candidate.video_index in kept_secondary_sources:
                        failed_secondary_ids.add(candidate.frame_id)
                    else:
                        kept_secondary_sources.add(candidate.video_index)
            return [
                replace(
                    assessment,
                    is_transition=assessment.frame_id in failed_secondary_ids,
                )
                for assessment in assessments
            ]

    videos = (
        tmp_path / "Sample Game Part1.mp4",
        tmp_path / "Sample Game Part2.mp4",
    )
    for video in videos:
        video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_videos=tuple(str(video) for video in videos),
        output_dir=str(output_dir),
        output_count=3,
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

    SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=SecondaryCountFailingAssessor(),
    ).run()

    work_dir = output_dir / ".game-screen-pick"
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert (work_dir / "assessments-secondary-backfill-0001.json").is_file()
    assert len(report["selected"]) == 3
    assert {item["video_index"] for item in report["selected"]} == {1, 2}


def test_pipeline_assesses_new_primary_candidate_for_secondary_backfill(
    tmp_path: Path,
) -> None:
    """二次追補poolが尽きた入力元は未評価候補を一次評価から補充すること."""
    kept_primary_source = False
    failed_primary_ids: set[str] = set()
    failed_secondary_ids: set[str] = set()

    class ExhaustedSecondaryPoolAssessor(FakeAssessor):
        """片方の一次生存候補を1件にし、その二次評価を遷移にするfake."""

        def assess(
            self,
            *,
            model: str,
            model_digest: str,
            prompt: str,
            candidates: Sequence[FrameCandidate],
            contact_sheet: Path,
        ) -> list[FrameAssessment]:
            nonlocal kept_primary_source
            assessments = super().assess(
                model=model,
                model_digest=model_digest,
                prompt=prompt,
                candidates=candidates,
                contact_sheet=contact_sheet,
            )
            if contact_sheet.parent.name == "primary":
                for candidate in candidates:
                    if candidate.video_index != 0:
                        continue
                    if kept_primary_source:
                        failed_primary_ids.add(candidate.frame_id)
                    else:
                        kept_primary_source = True
            if contact_sheet.parent.name == "secondary":
                failed_secondary_ids.update(
                    candidate.frame_id
                    for candidate in candidates
                    if candidate.video_index == 0
                )
            return [
                replace(
                    assessment,
                    is_transition=(
                        assessment.frame_id in failed_primary_ids
                        or assessment.frame_id in failed_secondary_ids
                    ),
                )
                for assessment in assessments
            ]

    videos = (
        tmp_path / "Sample Game Part1.mp4",
        tmp_path / "Sample Game Part2.mp4",
    )
    for video in videos:
        video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_videos=tuple(str(video) for video in videos),
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
    SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=ExhaustedSecondaryPoolAssessor(),
    ).run()

    work_dir = output_dir / ".game-screen-pick"
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert (work_dir / "assessments-primary-secondary-backfill-0001.json").is_file()
    assert (work_dir / "assessments-secondary-backfill-0001.json").is_file()
    assert {item["video_index"] for item in report["selected"]} == {1, 2}

    (work_dir / "completion.json").unlink()
    (output_dir / "selected-01.jpg").unlink()
    unavailable_assessor = UnavailableAssessor()
    SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=unavailable_assessor,
    ).run()

    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


def test_pipeline_rejects_resume_when_any_input_video_changes(tmp_path: Path) -> None:
    """2本目だけの内容変更も全体SHA-256で検出すること."""
    videos = (
        tmp_path / "Sample Game Part1.mp4",
        tmp_path / "Sample Game Part2.mp4",
    )
    for video in videos:
        video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_videos=tuple(str(video) for video in videos),
        output_dir=str(tmp_path / "selected"),
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
    SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    ).run()
    changed_video = videos[1]
    original_stat = changed_video.stat()
    with changed_video.open("r+b") as file:
        file.seek(1024)
        file.write(b"changed")
    os.utime(
        changed_video,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    unavailable_assessor = UnavailableAssessor()

    with pytest.raises(RuntimeError, match="実行条件が今回と異なります"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=unavailable_assessor,
        ).run()

    assert unavailable_assessor.metadata_calls == 0


def test_pipeline_rejects_duplicate_input_video_before_ollama(tmp_path: Path) -> None:
    """同じ入力pathの重複は外部接続より前に拒否すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(b"video")
    request = VideoSelectionRequest(
        input_videos=(str(video), str(video)),
        output_dir=str(tmp_path / "selected"),
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
    unavailable_assessor = UnavailableAssessor()

    with pytest.raises(ValueError, match="重複"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=unavailable_assessor,
        ).run()

    assert unavailable_assessor.metadata_calls == 0


def test_pipeline_requires_title_when_input_names_infer_different_games(
    tmp_path: Path,
) -> None:
    """入力名から同じgame titleを得られない場合は明示指定を求めること."""
    videos = (tmp_path / "Game A Part1.mp4", tmp_path / "Game B Part1.mp4")
    for video in videos:
        video.write_bytes(b"video")
    request = VideoSelectionRequest(
        input_videos=tuple(str(video) for video in videos),
        output_dir=str(tmp_path / "selected"),
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
    unavailable_assessor = UnavailableAssessor()

    with pytest.raises(ValueError, match="--game-title"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=unavailable_assessor,
        ).run()

    assert unavailable_assessor.metadata_calls == 0


def test_pipeline_rejects_resume_when_unsampled_video_bytes_change(
    tmp_path: Path,
) -> None:
    """sizeとmtimeが同じでも入力動画全体の変更を見逃さないこと."""
    video = tmp_path / "Sample Game.mp4"
    with video.open("wb") as file:
        file.truncate(8 * 1024 * 1024)
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
    SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    ).run()
    original_stat = video.stat()
    with video.open("r+b") as file:
        file.seek(2 * 1024 * 1024)
        file.write(b"changed")
    os.utime(
        video,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    unavailable_assessor = UnavailableAssessor()

    with pytest.raises(RuntimeError, match="実行条件が今回と異なります"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=unavailable_assessor,
        ).run()

    assert unavailable_assessor.metadata_calls == 0


def test_pipeline_finishes_fully_assessed_run_without_ollama(tmp_path: Path) -> None:
    """評価後の成果物生成を中断してもOllamaなしで完了できること."""
    video = tmp_path / "Sample Game.mp4"
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
    SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=FakeAssessor(),
    ).run()
    (output_dir / ".game-screen-pick" / "completion.json").unlink()
    (output_dir / "selected-01.jpg").unlink()
    unavailable_assessor = UnavailableAssessor()

    contact_sheet = SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=unavailable_assessor,
    ).run()

    assert contact_sheet.is_file()
    assert (output_dir / "selected-01.jpg").is_file()
    assert (output_dir / ".game-screen-pick" / "completion.json").is_file()
    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


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


def test_pipeline_uses_resolved_ollama_model_name(tmp_path: Path) -> None:
    """untagged modelはresolved nameでchatとGPU確認へ渡すこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="",
        primary_model="llava",
        secondary_model="llava",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    assessor = AliasFakeAssessor()

    SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=assessor,
    ).run()

    assert assessor.assessed_models
    assert set(assessor.assessed_models) == {"llava:latest"}


def test_pipeline_rejects_output_count_above_contact_sheet_limit(
    tmp_path: Path,
) -> None:
    """JPEG一覧を生成できない枚数を高価な処理前に拒否すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=601,
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

    with pytest.raises(ValueError, match="600以下"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=FakeAssessor(),
        ).run()


def test_pipeline_rejects_nonempty_output_before_contacting_ollama(
    tmp_path: Path,
) -> None:
    """再開manifestのない非空folderはOllama接続前に拒否すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
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

    with pytest.raises(RuntimeError, match="再開manifestもありません"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=UnavailableAssessor(),
        ).run()


def test_pipeline_rejects_concurrent_run_for_same_output(tmp_path: Path) -> None:
    """同じoutputを処理中の別pipelineへ書き込ませないこと."""
    metadata_started = Event()
    release_metadata = Event()

    class BlockingAssessor(FakeAssessor):
        """最初のrunをmanifest作成前で待機させるfake."""

        def fetch_model_metadata(
            self,
            requested_models: set[str],
        ) -> dict[str, dict[str, Any]]:
            metadata_started.set()
            if not release_metadata.wait(timeout=5):
                raise TimeoutError("concurrency test timed out")
            return super().fetch_model_metadata(requested_models)

    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
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

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_run = executor.submit(
            SingleVideoSelector(
                request,
                frame_extractor=FakeFrameExtractor(),
                assessor=BlockingAssessor(),
            ).run
        )
        assert metadata_started.wait(timeout=2)
        try:
            with pytest.raises(RuntimeError, match="同じ出力フォルダ.*実行中"):
                SingleVideoSelector(
                    request,
                    frame_extractor=FakeFrameExtractor(),
                    assessor=FakeAssessor(),
                ).run()
        finally:
            release_metadata.set()

        assert first_run.result(timeout=10).is_file()


def test_candidate_extraction_cancels_queued_jobs_on_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ctrl+C時にexecutorが未開始jobを取消して待ち続けないこと."""
    shutdown_calls: list[tuple[bool, bool]] = []

    class RecordingExecutor(ThreadPoolExecutor):
        """shutdown引数を記録するexecutor."""

        def shutdown(
            self,
            wait: bool = True,
            *,
            cancel_futures: bool = False,
        ) -> None:
            shutdown_calls.append((wait, cancel_futures))
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(
        "src.services.video_selector.ThreadPoolExecutor",
        RecordingExecutor,
    )
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=1,
        sample_interval_seconds=None,
        debug=False,
    )
    selector = SingleVideoSelector(
        request,
        frame_extractor=InterruptingFrameExtractor(),
        assessor=FakeAssessor(),
    )
    selector._prepare_run()

    with pytest.raises(KeyboardInterrupt):
        selector._extract_candidates()

    assert (False, True) in shutdown_calls


def test_context_extraction_cancels_queued_jobs_on_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """遷移判定frameの抽出中も未開始jobを取消すこと."""
    shutdown_calls: list[tuple[bool, bool]] = []

    class RecordingExecutor(ThreadPoolExecutor):
        """shutdown引数を記録するexecutor."""

        def shutdown(
            self,
            wait: bool = True,
            *,
            cancel_futures: bool = False,
        ) -> None:
            shutdown_calls.append((wait, cancel_futures))
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(
        "src.services.video_selector.ThreadPoolExecutor",
        RecordingExecutor,
    )
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=1,
        sample_interval_seconds=None,
        debug=False,
    )
    selector = SingleVideoSelector(
        request,
        frame_extractor=InterruptingFrameExtractor(),
        assessor=FakeAssessor(),
    )
    selector._prepare_run()

    with pytest.raises(KeyboardInterrupt):
        selector._extract_context_frames(
            [FrameCandidate("f00001", 2.0, str(tmp_path / "candidate.jpg"))]
        )

    assert (False, True) in shutdown_calls


def test_mechanical_preselection_cancels_queued_jobs_on_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """機械評価中も未開始jobを取消して中断を伝播すること."""
    shutdown_calls: list[tuple[bool, bool]] = []

    class RecordingExecutor(ThreadPoolExecutor):
        """shutdown引数を記録するexecutor."""

        def shutdown(
            self,
            wait: bool = True,
            *,
            cancel_futures: bool = False,
        ) -> None:
            shutdown_calls.append((wait, cancel_futures))
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    def interrupt(_candidate: FrameCandidate) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "src.services.video_selector.ThreadPoolExecutor",
        RecordingExecutor,
    )
    monkeypatch.setattr(
        "src.services.video_selector.measure_candidate",
        interrupt,
    )
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=1,
        sample_interval_seconds=None,
        debug=False,
    )
    selector = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    )
    selector._prepare_run()

    with pytest.raises(KeyboardInterrupt):
        selector._preselect_candidates([FrameCandidate("f00001", 1.0, "unused")])

    assert (False, True) in shutdown_calls


def test_pipeline_does_not_retry_model_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GPUやdigestの決定的な検証失敗をinference再試行しないこと."""
    monkeypatch.setattr("src.services.video_selector.time.sleep", lambda _: None)
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=False,
        ffmpeg_workers=1,
        sample_interval_seconds=None,
        debug=False,
    )
    assessor = GpuValidationFailingAssessor()

    with pytest.raises(OllamaModelValidationError):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=assessor,
        ).run()

    assert assessor.assess_calls == 1
