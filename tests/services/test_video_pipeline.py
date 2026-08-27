"""1本以上の動画を扱うproduction pipelineの小さな結合テスト."""

import errno
import json
import logging
import os
import shutil
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
from src.services.game_context_generator import (
    GameContextGenerationError,
    GameContextGenerator,
    GeneratedGameContext,
)
from src.services.ollama_frame_assessor import (
    OllamaFrameAssessor,
    OllamaModelValidationError,
)
from src.services.video_frame_extractor import VideoFrameExtractor
from src.services.video_phase_cache import (
    CACHE_DIRECTORY_NAME,
    CACHE_INFO_FILENAME,
    prepare_cache_root,
)
from src.services.video_selector import VideoSelector as SingleVideoSelector
from src.services.video_selector import measure_candidate
from src.utils.contact_sheet import context_frame_path
from src.utils.video_selection_files import file_sha256, json_digest


def _cache_root(input_directory: Path) -> Path:
    """test用Input Video Directory cache rootを返す."""
    return input_directory / CACHE_DIRECTORY_NAME


def _single_run_cache(input_directory: Path) -> Path:
    """testが作成した唯一のrun cache directoryを返す."""
    runs = list((_cache_root(input_directory) / "runs").iterdir())
    assert len(runs) == 1
    return runs[0]


def _completion_path(run_cache: Path) -> Path:
    """test runの唯一のOutput Folder完了記録を返す."""
    completions = list(run_cache.glob("completion-*.json"))
    assert len(completions) == 1
    return completions[0]


def _assessment_files(input_directory: Path, stage: str) -> list[Path]:
    """全Input Videoの指定評価phase cacheを返す."""
    return list(
        (_cache_root(input_directory) / "videos").glob(f"*/assessments/{stage}/*.json")
    )


GENERATED_GAME_CONTEXT = (
    "ジャンル: ロールプレイングゲーム\n"
    "基本的なゲーム進行と主なプレイ要素: 世界を探索して戦う。\n"
    "代表的な画面や場面: フィールド探索と戦闘。\n"
    "画像選定で重視する視覚的要素: 景観と戦況が明瞭な画面。"
)


class FakeFrameExtractor(VideoFrameExtractor):
    """実動画を使わず決定的なframeを生成するfake."""

    def __init__(self) -> None:
        """外部command確認を省略する."""
        self.extract_calls = 0
        self.probe_calls = 0

    def probe(self, video: Path) -> VideoMetadata:
        """短いtest動画のmetadataを返す."""
        assert video.is_file()
        self.probe_calls += 1
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


class TrackingFrameExtractor(FakeFrameExtractor):
    """probeとcandidate/output抽出をInput Video別に記録するfake."""

    def __init__(self) -> None:
        """記録用collectionを初期化する."""
        super().__init__()
        self.probed_videos: list[str] = []
        self.candidate_videos: list[str] = []
        self.output_videos: list[str] = []

    def probe(self, video: Path) -> VideoMetadata:
        """probe対象の相対名を記録する."""
        self.probed_videos.append(video.name)
        return super().probe(video)

    def extract_frame(
        self,
        video: Path,
        timestamp_seconds: float,
        output_path: Path,
        *,
        max_width: int | None,
        video_stream_index: int = 0,
    ) -> None:
        """candidateと最終Outputを分けて記録する."""
        target = self.candidate_videos if max_width == 960 else self.output_videos
        target.append(video.name)
        super().extract_frame(
            video,
            timestamp_seconds,
            output_path,
            max_width=max_width,
            video_stream_index=video_stream_index,
        )


class FakeAssessor(OllamaFrameAssessor):
    """全候補へ決定的な評価を返すfake."""

    def __init__(self) -> None:
        """network設定を省略する."""
        self.host = "http://fake"
        self.timeout_seconds = 1.0
        self.require_gpu = False
        self.gpu_evidence: dict[str, dict[str, Any]] = {}
        self.assess_calls = 0
        self.prompts: list[str] = []

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
        self.prompts.append(prompt)
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


class FakeContextGenerator(GameContextGenerator):
    """Web検索なしで固定Game Contextを返すfake."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        game_title: str,
        provider: str,
        model: str,
        ollama_host: str,
        timeout_seconds: float,
    ) -> GeneratedGameContext:
        self.calls.append(
            {
                "game_title": game_title,
                "provider": provider,
                "model": model,
                "ollama_host": ollama_host,
                "timeout_seconds": timeout_seconds,
            }
        )
        return GeneratedGameContext(
            game_context=GENERATED_GAME_CONTEXT,
            provider=provider,
            model=f"{model}:resolved",
        )


class ExplodingContextGenerator(GameContextGenerator):
    """呼び出されてはならないcontext generator."""

    def generate(
        self,
        *,
        game_title: str,
        provider: str,
        model: str,
        ollama_host: str,
        timeout_seconds: float,
    ) -> GeneratedGameContext:
        del game_title, provider, model, ollama_host, timeout_seconds
        raise AssertionError("context generatorは呼ばれないこと")


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
        game_context="テスト用のGame Context",
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
        context_generator=ExplodingContextGenerator(),
    ).run()

    selected_paths = [output_dir / "selected-01.jpg", output_dir / "selected-02.jpg"]
    assert contact_sheet == output_dir / "selected-contact-sheet.jpg"
    assert contact_sheet.is_file()
    assert all(path.is_file() for path in selected_paths)
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert "game_title" not in report
    assert report["game_context"] == "テスト用のGame Context"
    assert "game_context_generation" not in report
    assert report["output_count"] == 2
    assert len(report["selected"]) == 2
    run_cache = _single_run_cache(tmp_path)
    completion = json.loads(_completion_path(run_cache).read_text(encoding="utf-8"))
    assert len(completion["artifacts"]) == 4
    first_hashes = [file_sha256(path) for path in selected_paths]
    calls_after_first_run = assessor.assess_calls
    extraction_after_first_run = extractor.extract_calls

    unavailable_assessor = UnavailableAssessor()
    resumed_sheet = SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=unavailable_assessor,
        context_generator=ExplodingContextGenerator(),
    ).run()

    assert resumed_sheet == contact_sheet
    assert assessor.assess_calls == calls_after_first_run
    assert unavailable_assessor.assess_calls == 0
    assert unavailable_assessor.metadata_calls == 0
    assert extractor.extract_calls == extraction_after_first_run
    assert [file_sha256(path) for path in selected_paths] == first_hashes
    manifest = json.loads((run_cache / "run-manifest.json").read_text(encoding="utf-8"))
    assert "game_title" not in manifest
    assert manifest["game_context"] == "テスト用のGame Context"
    assert "game_context_generation" not in manifest
    assert all("ゲーム『" not in prompt for prompt in assessor.prompts)
    assert all("テスト用のGame Context" in prompt for prompt in assessor.prompts)


@pytest.mark.parametrize("cache_relative_output", [".", "output"])
def test_pipeline_rejects_output_folder_inside_regenerable_cache(
    tmp_path: Path,
    cache_relative_output: str,
) -> None:
    """Output Folderを削除可能なcache root以下に置かないこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = _cache_root(tmp_path) / cache_relative_output
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )

    with pytest.raises(ValueError, match="Output Folder.*cache"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=FakeAssessor(),
        )._prepare_paths()


@pytest.mark.parametrize(
    "corruption",
    ["truncated", "invalid-artifacts", "missing-artifact"],
)
def test_pipeline_regenerates_corrupt_completion_record(
    tmp_path: Path,
    corruption: str,
) -> None:
    """読取不能または不正な完了記録をmissとして成果物を再生成すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    completion_path = _completion_path(_single_run_cache(tmp_path))
    if corruption == "truncated":
        completion_path.write_text("{", encoding="utf-8")
    elif corruption == "invalid-artifacts":
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["artifacts"] = {}
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
    else:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["artifacts"].pop()
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
    unavailable_assessor = UnavailableAssessor()

    resumed_sheet = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=unavailable_assessor,
    ).run()

    assert resumed_sheet == output_dir / "selected-contact-sheet.jpg"
    regenerated = json.loads(completion_path.read_text(encoding="utf-8"))
    assert len(regenerated["artifacts"]) == 4
    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


def test_pipeline_rewrites_undecodable_cache_info_without_losing_phase_cache(
    tmp_path: Path,
) -> None:
    """decode不能なcache説明fileだけを書き直し、高価なphaseを再利用すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    primary_cache = _assessment_files(tmp_path, "primary")[0]
    primary_cache_bytes = primary_cache.read_bytes()
    info_path = _cache_root(tmp_path) / "CACHE_INFO.txt"
    info_path.write_bytes(b"\xff")
    unavailable_assessor = UnavailableAssessor()

    resumed_sheet = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=unavailable_assessor,
    ).run()

    assert resumed_sheet == output_dir / "selected-contact-sheet.jpg"
    assert "game-screen-pick" in info_path.read_text(encoding="utf-8")
    assert primary_cache.read_bytes() == primary_cache_bytes
    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


def test_prepare_cache_root_replaces_fifo_cache_info_without_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIFOのcache説明fileをopenせず通常fileへ置換すること."""
    cache_root = tmp_path / CACHE_DIRECTORY_NAME
    cache_root.mkdir()
    info_path = cache_root / CACHE_INFO_FILENAME
    os.mkfifo(info_path)
    original_read_text = Path.read_text

    def reject_fifo_read(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == info_path:
            raise AssertionError("FIFOをread_textしてはなりません")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_fifo_read)

    prepared = prepare_cache_root(tmp_path)

    assert prepared == cache_root
    assert info_path.is_file()
    assert "game-screen-pick" in original_read_text(info_path, encoding="utf-8")


def test_pipeline_generates_context_before_video_processing_and_reuses_it(
    tmp_path: Path,
) -> None:
    """title指定時だけ事前生成し、再開時は保存済みcontextを再利用すること."""
    video = tmp_path / "recording.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title="ドラクエ11",
        game_context="",
        game_context_provider="openai",
        game_context_model="gpt-context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    generator = FakeContextGenerator()
    assessor = FakeAssessor()

    SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=assessor,
        context_generator=generator,
    ).run()

    assert generator.calls == [
        {
            "game_title": "ドラクエ11",
            "provider": "openai",
            "model": "gpt-context",
            "ollama_host": "fake",
            "timeout_seconds": 1.0,
        }
    ]
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["game_context"] == GENERATED_GAME_CONTEXT
    assert report["game_context_generation"] == {
        "provider": "openai",
        "model": "gpt-context:resolved",
    }
    assert "game_title" not in report
    assert all("ドラクエ11" not in prompt for prompt in assessor.prompts)
    assert all(GENERATED_GAME_CONTEXT in prompt for prompt in assessor.prompts)

    unavailable_assessor = UnavailableAssessor()
    SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=unavailable_assessor,
        context_generator=ExplodingContextGenerator(),
    ).run()

    assert unavailable_assessor.metadata_calls == 0


def test_ollama_context_checkpoint_identity_includes_normalized_host(
    tmp_path: Path,
) -> None:
    """Ollama host変更時だけGame Contextを再生成すること."""

    class HostContextGenerator(GameContextGenerator):
        """正規化前hostをcontextへ埋め込むgenerator."""

        def __init__(self) -> None:
            self.hosts: list[str] = []

        def generate(
            self,
            *,
            game_title: str,
            provider: str,
            model: str,
            ollama_host: str,
            timeout_seconds: float,
        ) -> GeneratedGameContext:
            del game_title, timeout_seconds
            self.hosts.append(ollama_host)
            return GeneratedGameContext(
                game_context=f"{GENERATED_GAME_CONTEXT}\nEndpoint: {ollama_host}",
                provider=provider,
                model=model,
            )

    video = tmp_path / "recording.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected-first"),
        output_count=2,
        game_title="テストゲーム",
        game_context="",
        game_context_provider="ollama",
        game_context_model="context-model",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="first-host:11434",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    generator = HostContextGenerator()
    SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
        context_generator=generator,
    ).run()
    second_request = replace(
        request,
        output_dir=str(tmp_path / "selected-second"),
        ollama_host="second-host:11434",
    )

    SingleVideoSelector(
        second_request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
        context_generator=generator,
    ).run()

    assert generator.hosts == ["first-host:11434", "second-host:11434"]
    third_output = tmp_path / "selected-third"
    SingleVideoSelector(
        replace(
            request,
            output_dir=str(third_output),
            ollama_host="http://first-host:11434/",
        ),
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
        context_generator=ExplodingContextGenerator(),
    ).run()
    report = json.loads((third_output / "report.json").read_text(encoding="utf-8"))
    assert report["game_context"] == (
        f"{GENERATED_GAME_CONTEXT}\nEndpoint: first-host:11434"
    )


def test_assessment_identity_includes_normalized_ollama_host(tmp_path: Path) -> None:
    """評価endpoint変更時だけ完了runと評価cacheを再利用しないこと."""
    video = tmp_path / "recording.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="first-host:11434",
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
    changed_assessor = FakeAssessor()

    SingleVideoSelector(
        replace(request, ollama_host="second-host:11434"),
        frame_extractor=FakeFrameExtractor(),
        assessor=changed_assessor,
    ).run()

    assert changed_assessor.assess_calls > 0
    assert len(list((_cache_root(tmp_path) / "runs").iterdir())) == 2

    unavailable_assessor = UnavailableAssessor()
    SingleVideoSelector(
        replace(request, ollama_host="http://second-host:11434/"),
        frame_extractor=FakeFrameExtractor(),
        assessor=unavailable_assessor,
    ).run()

    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


def test_context_generation_failure_stops_before_video_probe(tmp_path: Path) -> None:
    """検索・生成失敗では長い動画処理へ進まないこと."""

    class FailingContextGenerator(GameContextGenerator):
        def generate(
            self,
            *,
            game_title: str,
            provider: str,
            model: str,
            ollama_host: str,
            timeout_seconds: float,
        ) -> GeneratedGameContext:
            del game_title, model, ollama_host, timeout_seconds
            raise GameContextGenerationError(f"{provider}: 認証error")

    video = tmp_path / "recording.mp4"
    video.write_bytes(b"video")
    extractor = FakeFrameExtractor()
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title="Game",
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

    with pytest.raises(GameContextGenerationError, match="ollama.*認証"):
        SingleVideoSelector(
            request,
            frame_extractor=extractor,
            assessor=FakeAssessor(),
            context_generator=FailingContextGenerator(),
        ).run()

    assert extractor.probe_calls == 0


def test_generated_context_is_checkpointed_before_video_probe(
    tmp_path: Path,
) -> None:
    """probe失敗後の再実行では課金済みcontext生成を繰り返さないこと."""

    class FailingProbeExtractor(FakeFrameExtractor):
        def probe(self, video: Path) -> VideoMetadata:
            del video
            raise RuntimeError("probe failed")

    video = tmp_path / "recording.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title="ドラクエ11",
        game_context="",
        game_context_provider="openai",
        game_context_model="gpt-context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    generator = FakeContextGenerator()

    with pytest.raises(RuntimeError, match="probe failed"):
        SingleVideoSelector(
            request,
            frame_extractor=FailingProbeExtractor(),
            assessor=FakeAssessor(),
            context_generator=generator,
        ).run()

    checkpoints = list((_cache_root(tmp_path) / "game-context").glob("*.json"))
    assert len(checkpoints) == 1
    checkpoint_path = checkpoints[0]
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["request"] == {
        "game_title": "ドラクエ11",
        "provider": "openai",
        "model": "gpt-context",
    }
    assert checkpoint["result"] == {
        "game_context": GENERATED_GAME_CONTEXT,
        "provider": "openai",
        "model": "gpt-context:resolved",
    }
    assert checkpoint["payload_digest"] == json_digest(
        {"request": checkpoint["request"], "result": checkpoint["result"]}
    )

    SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
        context_generator=ExplodingContextGenerator(),
    ).run()

    assert len(generator.calls) == 1
    assert checkpoint_path.exists()


def test_pipeline_does_not_reuse_legacy_output_cache(tmp_path: Path) -> None:
    """旧Output Folder cacheは現行phase cacheとして再利用しないこと."""
    video = tmp_path / "recording.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    legacy_cache = output_dir / ".game-screen-pick"
    legacy_cache.mkdir(parents=True)
    (legacy_cache / "run-manifest.json").write_text(
        json.dumps({"prompt_version": "blog-image-selection-v4"}),
        encoding="utf-8",
    )
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="context",
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

    with pytest.raises(RuntimeError, match="対応する完了記録"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=unavailable_assessor,
            context_generator=ExplodingContextGenerator(),
        ).run()

    assert unavailable_assessor.metadata_calls == 0


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
        game_title=None,
        game_context="テスト用のGame Context",
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
    assert any(
        message.startswith("入力動画cacheの実行状態を確認しています:")
        for message in messages
    )
    assert (
        'Ollamaモデル情報を確認しています: "primary\\nmodel, '
        'secondary\\u001bmodel"' in messages
    )
    assert "候補フレームを抽出します: 13/13件" in messages
    assert (
        "処理予定: 全候補数=13件, 一次評価予定数=13件（上限）, "
        "二次評価予定数=6件（上限）" in messages
    )
    assert "一次評価対象が確定しました: 13件" in messages
    assert "二次評価対象が確定しました: 6件" in messages
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
        game_title=None,
        game_context="テスト用のGame Context",
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
    assert "primary評価batch 1の試行1が失敗しました: temporary failure" in messages


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
        game_context="テスト用のGame Context",
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
        (_single_run_cache(tmp_path) / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert [item["relative_path"] for item in manifest["inputs"]] == [
        video.name for video in videos
    ]
    assert all("sha256" not in item for item in manifest["inputs"])
    assert all("mtime_ns" not in item for item in manifest["inputs"])
    assert all("path" not in item for item in manifest["inputs"])
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


def test_pipeline_reuses_video_phase_cache_after_input_directory_move(
    tmp_path: Path,
) -> None:
    """Input Video Directoryとcacheを移動しても高価なphaseを再実行しないこと."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    video = input_dir / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    first_request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected-first"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
        first_request,
        frame_extractor=TrackingFrameExtractor(),
        assessor=FakeAssessor(),
    ).run()
    first_primary_cache = _assessment_files(input_dir, "primary")[0].read_bytes()
    moved_input_dir = tmp_path / "moved-recordings"
    input_dir.rename(moved_input_dir)
    moved_video = moved_input_dir / video.name
    second_request = replace(
        first_request,
        input_videos=(str(moved_video),),
        output_dir=str(tmp_path / "selected-second"),
    )
    extractor = TrackingFrameExtractor()
    unavailable_assessor = UnavailableAssessor()

    SingleVideoSelector(
        second_request,
        frame_extractor=extractor,
        assessor=unavailable_assessor,
    ).run()

    assert extractor.probed_videos == []
    assert extractor.candidate_videos == []
    assert extractor.output_videos == [video.name, video.name]
    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0
    assert _assessment_files(moved_input_dir, "primary")[0].read_bytes() == (
        first_primary_cache
    )
    assert (_cache_root(moved_input_dir) / "CACHE_INFO.txt").is_file()
    assert not (tmp_path / "selected-second" / ".game-screen-pick").exists()


def test_pipeline_processes_only_added_video_before_global_reselection(
    tmp_path: Path,
) -> None:
    """動画追加時は既存動画phaseを保ち、新規動画だけを処理すること."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    first_video = input_dir / "Sample Game Part1.mp4"
    first_video.write_bytes(bytes(range(256)) * 16)
    first_request = VideoSelectionRequest(
        input_video=str(first_video),
        output_dir=str(tmp_path / "selected-first"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
        first_request,
        frame_extractor=TrackingFrameExtractor(),
        assessor=FakeAssessor(),
    ).run()
    first_primary_cache_path = _assessment_files(input_dir, "primary")[0]
    first_primary_cache = first_primary_cache_path.read_bytes()

    second_video = input_dir / "Sample Game Part2.mp4"
    second_video.write_bytes(bytes(range(256)) * 16)
    second_request = replace(
        first_request,
        input_videos=(str(first_video), str(second_video)),
        output_dir=str(tmp_path / "selected-second"),
    )
    extractor = TrackingFrameExtractor()
    assessor = FakeAssessor()

    SingleVideoSelector(
        second_request,
        frame_extractor=extractor,
        assessor=assessor,
    ).run()

    assert extractor.probed_videos == [second_video.name]
    assert extractor.candidate_videos
    assert set(extractor.candidate_videos) == {second_video.name}
    assert assessor.prompts
    assert all(second_video.name in prompt for prompt in assessor.prompts)
    assert first_primary_cache_path.read_bytes() == first_primary_cache
    report = json.loads(
        (tmp_path / "selected-second" / "report.json").read_text(encoding="utf-8")
    )
    assert [item["video_name"] for item in report["selected"]] == [
        first_video.name,
        second_video.name,
    ]


def test_secondary_phase_version_change_reuses_preceding_video_phases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """二次phase version変更ではprobe・抽出・一次評価を再利用すること."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    video = input_dir / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    first_request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected-first"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
        first_request,
        frame_extractor=TrackingFrameExtractor(),
        assessor=FakeAssessor(),
    ).run()
    monkeypatch.setattr(
        "src.services.video_selector.SECONDARY_ASSESSMENT_PHASE_VERSION",
        3,
    )
    extractor = TrackingFrameExtractor()
    assessor = FakeAssessor()

    SingleVideoSelector(
        replace(first_request, output_dir=str(tmp_path / "selected-second")),
        frame_extractor=extractor,
        assessor=assessor,
    ).run()

    assert extractor.probed_videos == []
    assert extractor.candidate_videos == []
    assert assessor.prompts
    assert all("厳しい再評価" in prompt for prompt in assessor.prompts)


def test_deleted_cache_directory_is_safely_regenerated(tmp_path: Path) -> None:
    """cache-game-screen-pick削除後は全phaseを新規生成すること."""
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    video = input_dir / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    first_request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected-first"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
        first_request,
        frame_extractor=TrackingFrameExtractor(),
        assessor=FakeAssessor(),
    ).run()
    shutil.rmtree(_cache_root(input_dir))
    extractor = TrackingFrameExtractor()
    assessor = FakeAssessor()

    SingleVideoSelector(
        replace(first_request, output_dir=str(tmp_path / "selected-second")),
        frame_extractor=extractor,
        assessor=assessor,
    ).run()

    assert extractor.probed_videos == [video.name]
    assert extractor.candidate_videos
    assert assessor.assess_calls > 0
    assert (_cache_root(input_dir) / "CACHE_INFO.txt").is_file()


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
        game_context="テスト用のGame Context",
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

    work_dir = _single_run_cache(tmp_path)
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert _assessment_files(tmp_path, "primary")
    assert {item["video_index"] for item in report["selected"]} == {1, 2}
    assert not failed_primary_ids & {item["frame_id"] for item in report["selected"]}

    _completion_path(work_dir).unlink()
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
        present_sources = {candidate.video_index for candidate in candidates}
        if present_sources == {0}:
            return by_source[0][:1]
        if present_sources == {1}:
            return by_source[1][:count]
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
        game_context="テスト用のGame Context",
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

    work_dir = _single_run_cache(tmp_path)
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert _assessment_files(tmp_path, "secondary")
    assert {item["video_index"] for item in report["selected"]} == {1, 2}
    assert not failed_secondary_ids & {item["frame_id"] for item in report["selected"]}

    _completion_path(work_dir).unlink()
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
        game_context="テスト用のGame Context",
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

    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert _assessment_files(tmp_path, "secondary")
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
        game_context="テスト用のGame Context",
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

    work_dir = _single_run_cache(tmp_path)
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert _assessment_files(tmp_path, "primary")
    assert _assessment_files(tmp_path, "secondary")
    assert {item["video_index"] for item in report["selected"]} == {1, 2}

    _completion_path(work_dir).unlink()
    (output_dir / "selected-01.jpg").unlink()
    unavailable_assessor = UnavailableAssessor()
    SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=unavailable_assessor,
    ).run()

    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


def test_pipeline_reuses_cache_when_video_content_changes_at_same_size(
    tmp_path: Path,
) -> None:
    """相対名とsizeが同じ動画の内容変更はidentityの検出対象外とすること."""
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
        game_context="テスト用のGame Context",
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
        game_context="テスト用のGame Context",
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


def test_pipeline_requires_exactly_one_title_or_context(
    tmp_path: Path,
) -> None:
    """programmatic requestでもtitleとcontextの両方未指定を拒否すること."""
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

    with pytest.raises(ValueError, match="どちらか一方"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=unavailable_assessor,
        ).run()

    assert unavailable_assessor.metadata_calls == 0


def test_pipeline_does_not_hash_unsampled_video_bytes_on_cache_reuse(
    tmp_path: Path,
) -> None:
    """same-name/size動画は別Output Folderでも動画byteを再検証しないこと."""
    video = tmp_path / "Sample Game.mp4"
    with video.open("wb") as file:
        file.truncate(8 * 1024 * 1024)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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

    second_request = replace(request, output_dir=str(tmp_path / "selected-again"))
    SingleVideoSelector(
        second_request,
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
        game_context="テスト用のGame Context",
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
    work_dir = _single_run_cache(tmp_path)
    _completion_path(work_dir).unlink()
    (output_dir / "selected-01.jpg").unlink()
    unavailable_assessor = UnavailableAssessor()

    contact_sheet = SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=unavailable_assessor,
    ).run()

    assert contact_sheet.is_file()
    assert (output_dir / "selected-01.jpg").is_file()
    assert _completion_path(work_dir).is_file()
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
        game_context="テスト用のGame Context",
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

    previous_assess_calls = assessor.assess_calls
    SingleVideoSelector(
        replace(cpu_allowed_request, allow_cpu=False),
        frame_extractor=extractor,
        assessor=assessor,
    ).run()

    assert assessor.assess_calls > previous_assess_calls


def test_pipeline_uses_resolved_ollama_model_name(tmp_path: Path) -> None:
    """untagged modelはresolved nameでchatとGPU確認へ渡すこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
        game_context="テスト用のGame Context",
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
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )

    with pytest.raises(RuntimeError, match="対応する完了記録もありません"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=UnavailableAssessor(),
        ).run()


def test_pipeline_rejects_nonempty_output_before_generating_context(
    tmp_path: Path,
) -> None:
    """未登録の非空Output FolderはWeb検索より前に拒否すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title="テストゲーム",
        game_context="",
        game_context_provider="openai",
        game_context_model="gpt-context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    context_generator = FakeContextGenerator()

    with pytest.raises(RuntimeError, match="対応する完了記録もありません"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=FakeAssessor(),
            context_generator=context_generator,
        ).run()

    assert context_generator.calls == []


@pytest.mark.parametrize(
    "checkpoint_corruption",
    [
        "truncated",
        "empty",
        "missing-heading",
        "whitespace",
        "oversized",
        "digest-context",
    ],
)
def test_pipeline_regenerates_corrupt_game_context_checkpoint(
    tmp_path: Path,
    checkpoint_corruption: str,
) -> None:
    """読取不能または不正なGame Context checkpointをmissとして再生成すること."""

    class FailingProbeExtractor(FakeFrameExtractor):
        """checkpoint保存後に最初のrunを停止するfake."""

        def probe(self, video: Path) -> VideoMetadata:
            del video
            raise RuntimeError("probe failed")

    video = tmp_path / "recording.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title="テストゲーム",
        game_context="",
        game_context_provider="openai",
        game_context_model="gpt-context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    with pytest.raises(RuntimeError, match="probe failed"):
        SingleVideoSelector(
            request,
            frame_extractor=FailingProbeExtractor(),
            assessor=FakeAssessor(),
            context_generator=FakeContextGenerator(),
        ).run()
    checkpoint_path = next((_cache_root(tmp_path) / "game-context").glob("*.json"))
    if checkpoint_corruption == "truncated":
        checkpoint_path.write_text("{", encoding="utf-8")
    elif checkpoint_corruption == "empty":
        checkpoint_path.write_text("{}", encoding="utf-8")
    else:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        invalid_contexts = {
            "missing-heading": "ジャンル: RPG",
            "whitespace": f" {GENERATED_GAME_CONTEXT}\n",
            "oversized": f"{GENERATED_GAME_CONTEXT}\n{'x' * 2_400}",
            "digest-context": GENERATED_GAME_CONTEXT.replace(
                "世界を探索して戦う。", "町を探索して戦う。"
            ),
        }
        checkpoint["result"]["game_context"] = invalid_contexts[checkpoint_corruption]
        if checkpoint_corruption != "digest-context":
            checkpoint["payload_digest"] = json_digest(
                {"request": checkpoint["request"], "result": checkpoint["result"]}
            )
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    generator = FakeContextGenerator()

    SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
        context_generator=generator,
    ).run()

    assert len(generator.calls) == 1
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["result"]["game_context"] == GENERATED_GAME_CONTEXT


def test_pipeline_treats_unreadable_run_manifest_as_cache_miss(
    tmp_path: Path,
) -> None:
    """parse不能なrun manifestは再生成して処理を継続すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    manifest_path = _single_run_cache(tmp_path) / "run-manifest.json"
    manifest_path.write_text("{", encoding="utf-8")
    assessor = FakeAssessor()

    resumed_sheet = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=assessor,
    ).run()

    assert resumed_sheet == output_dir / "selected-contact-sheet.jpg"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["run_key"]
    assert assessor.assess_calls == 0


def test_secondary_context_reextracts_changed_frame(tmp_path: Path) -> None:
    """記録済みdigestと異なる遷移JPEGを動画から再抽出すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    selector = SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=FakeAssessor(),
    )
    selector.run()
    context_frames = sorted(
        (_cache_root(tmp_path) / "videos").glob("*/secondary-context/*/frames/*.jpg")
    )
    assert len(context_frames) >= 2
    changed_path = context_frames[0]
    original_hash = file_sha256(changed_path)
    shutil.copyfile(context_frames[1], context_frames[0])
    changed_frame_id = changed_path.name.removesuffix("-before.jpg").removesuffix(
        "-after.jpg"
    )
    candidate = next(
        candidate
        for candidate in selector._extract_candidates()
        if candidate.frame_id == changed_frame_id
    )
    extractor.extract_calls = 0

    selector._extract_context_frames([candidate])

    assert extractor.extract_calls == 1
    assert file_sha256(changed_path) == original_hash


def test_same_model_name_keeps_stage_specific_digests_on_partial_resume(
    tmp_path: Path,
) -> None:
    """同名model更新時も完了済み一次評価のdigestを正確に保持すること."""

    class DigestAssessor(FakeAssessor):
        """同じmodel名へ任意のdigestを返すfake."""

        def __init__(self, digest: str) -> None:
            super().__init__()
            self.digest = digest
            self.assessed_digests: list[str] = []

        def fetch_model_metadata(
            self,
            requested_models: set[str],
        ) -> dict[str, dict[str, Any]]:
            return {
                model: {
                    "digest": self.digest,
                    "resolved_name": model,
                    "capabilities": ["vision"],
                    "details": {},
                }
                for model in requested_models
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
            del model, prompt
            assert contact_sheet.is_file()
            self.assess_calls += 1
            self.assessed_digests.append(model_digest)
            return [
                FrameAssessment(
                    frame_id=candidate.frame_id,
                    blog_score=75.0,
                    is_transition=False,
                    scene="探索",
                    reason="test",
                )
                for candidate in candidates
            ]

    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="shared",
        secondary_model="shared",
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
        assessor=DigestAssessor("digest-old"),
    ).run()
    _completion_path(_single_run_cache(tmp_path)).unlink()
    for state_path in _assessment_files(tmp_path, "secondary"):
        state_path.unlink()
    assessor = DigestAssessor("digest-new")

    SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=assessor,
    ).run()

    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["models"]["primary"]["digest"] == "digest-old"
    assert report["models"]["secondary"]["digest"] == "digest-new"
    assert set(assessor.assessed_digests) == {"digest-new"}


def test_pipeline_replaces_managed_output_registered_by_different_run(
    tmp_path: Path,
) -> None:
    """別条件のrun登録では古い生成物を除去してから再生成すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=3,
        game_title=None,
        game_context="テスト用のGame Context",
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
    stale_output = output_dir / "selected-03.jpg"
    assessor = FakeAssessor()

    SingleVideoSelector(
        replace(request, output_count=2),
        frame_extractor=FakeFrameExtractor(),
        assessor=assessor,
    ).run()

    assert not stale_output.exists()
    assert assessor.assess_calls > 0
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["output_count"] == 2

    unavailable_assessor = UnavailableAssessor()
    SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=unavailable_assessor,
    ).run()

    assert stale_output.is_file()
    restored_report = json.loads(
        (output_dir / "report.json").read_text(encoding="utf-8")
    )
    assert restored_report["output_count"] == 3
    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


def test_pipeline_publishes_replacement_without_cross_device_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cacheとOutput Folderが別filesystemでも成果物を置換できること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=3,
        game_title=None,
        game_context="テスト用のGame Context",
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
    cache_root = _cache_root(tmp_path)
    original_replace = Path.replace

    def reject_cross_device_replace(source: Path, target: Path) -> Path:
        target_path = Path(target)
        if cache_root in source.parents and target_path.parent == output_dir:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return original_replace(source, target_path)

    monkeypatch.setattr(Path, "replace", reject_cross_device_replace)

    SingleVideoSelector(
        replace(request, output_count=2),
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    ).run()

    assert not (output_dir / "selected-03.jpg").exists()
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["output_count"] == 2


def test_pipeline_recovers_abandoned_publication_staging_directory(
    tmp_path: Path,
) -> None:
    """強制終了で残ったpublication stagingを回収して再開すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    abandoned = output_dir / ".game-screen-pick-publication-abandoned"
    abandoned.mkdir()
    (abandoned / "selected-01.jpg").write_bytes(b"partial")
    unavailable_assessor = UnavailableAssessor()

    resumed_sheet = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=unavailable_assessor,
    ).run()

    assert resumed_sheet == output_dir / "selected-contact-sheet.jpg"
    assert not abandoned.exists()
    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


def test_pipeline_regenerates_output_after_visible_cache_is_deleted(
    tmp_path: Path,
) -> None:
    """cache削除後も既存成果物から所有を再確立して再生成すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    shutil.rmtree(_cache_root(tmp_path))
    replacement_bytes = bytes(reversed(range(256))) * 16
    assert len(replacement_bytes) == video.stat().st_size
    video.write_bytes(replacement_bytes)
    extractor = FakeFrameExtractor()
    assessor = FakeAssessor()

    regenerated_sheet = SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=assessor,
    ).run()

    assert regenerated_sheet == output_dir / "selected-contact-sheet.jpg"
    assert extractor.extract_calls > 0
    assert assessor.assess_calls > 0


def test_pipeline_rejects_tampered_output_after_visible_cache_is_deleted(
    tmp_path: Path,
) -> None:
    """cache削除後もreportと不一致な既存成果物を所有済みにしないこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    shutil.rmtree(_cache_root(tmp_path))
    selected_path = output_dir / "selected-01.jpg"
    shutil.copyfile(output_dir / "selected-02.jpg", selected_path)
    tampered_hash = file_sha256(selected_path)
    unavailable_assessor = UnavailableAssessor()

    with pytest.raises(RuntimeError, match="対応する完了記録もありません"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=unavailable_assessor,
        ).run()

    assert file_sha256(selected_path) == tampered_hash
    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


def test_pipeline_rejects_unrecorded_managed_artifact_after_cache_is_deleted(
    tmp_path: Path,
) -> None:
    """reportに記録のないmanaged風成果物を所有済みにしないこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    shutil.rmtree(_cache_root(tmp_path))
    unrecorded = output_dir / "selected-03.jpg"
    shutil.copyfile(output_dir / "selected-02.jpg", unrecorded)
    unrecorded_hash = file_sha256(unrecorded)
    unavailable_assessor = UnavailableAssessor()

    with pytest.raises(RuntimeError, match="対応する完了記録もありません"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=unavailable_assessor,
        ).run()

    assert file_sha256(unrecorded) == unrecorded_hash
    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


def test_pipeline_does_not_follow_publication_staging_symlink(
    tmp_path: Path,
) -> None:
    """publication staging風のsymlinkを回収対象にしないこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "user-owned.txt"
    external.write_text("user-owned", encoding="utf-8")
    staging_symlink = output_dir / ".game-screen-pick-publication-symlink"
    staging_symlink.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="出力フォルダが空ではなく"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=FakeAssessor(),
        ).run()

    assert staging_symlink.is_symlink()
    assert external.read_text(encoding="utf-8") == "user-owned"


def test_pipeline_rejects_unmanaged_output_before_game_context_generation(
    tmp_path: Path,
) -> None:
    """登録済みOutput Folderの未管理fileを外部context生成前に拒否すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    (output_dir / "notes.txt").write_text("user-owned", encoding="utf-8")
    generator = FakeContextGenerator()

    with pytest.raises(RuntimeError, match="出力フォルダが空ではなく"):
        SingleVideoSelector(
            replace(request, game_title="Sample Game", game_context=""),
            frame_extractor=FakeFrameExtractor(),
            assessor=FakeAssessor(),
            context_generator=generator,
        ).run()

    assert generator.calls == []
    assert (output_dir / "notes.txt").read_text(encoding="utf-8") == "user-owned"


@pytest.mark.parametrize(
    "registration_payload",
    ["{", "{}", '{"output_path":"/different"}'],
)
def test_pipeline_rejects_invalid_output_registration_without_deleting_files(
    tmp_path: Path,
    registration_payload: str,
) -> None:
    """不正な所有記録だけでは管理対象風の既存fileを削除しないこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    run_cache = _single_run_cache(tmp_path)
    _completion_path(run_cache).unlink()
    registration_path = next(run_cache.glob("output-*.json"))
    registration_path.write_text(registration_payload, encoding="utf-8")
    report_path = output_dir / "report.json"
    report_path.write_bytes(b"user-owned")

    with pytest.raises(RuntimeError, match="対応する完了記録もありません"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=FakeAssessor(),
        ).run()

    assert report_path.read_bytes() == b"user-owned"
    assert (output_dir / "selected-01.jpg").is_file()


def test_pipeline_reuses_valid_completion_when_output_registration_is_corrupt(
    tmp_path: Path,
) -> None:
    """所有記録が破損しても正常な完了記録から成果物を再利用すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    registration_path = next(_single_run_cache(tmp_path).glob("output-*.json"))
    registration_path.write_text("{", encoding="utf-8")
    unavailable_assessor = UnavailableAssessor()

    resumed_sheet = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=unavailable_assessor,
    ).run()

    assert resumed_sheet == output_dir / "selected-contact-sheet.jpg"
    assert json.loads(registration_path.read_text(encoding="utf-8")) == {
        "output_path": str(output_dir)
    }
    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


def test_pipeline_rejects_unrecorded_artifact_with_only_completion_ownership(
    tmp_path: Path,
) -> None:
    """完了記録にないmanaged風成果物を所有対象へ含めないこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    next(_single_run_cache(tmp_path).glob("output-*.json")).write_text(
        "{",
        encoding="utf-8",
    )
    unrecorded = output_dir / "selected-03.jpg"
    shutil.copyfile(output_dir / "selected-02.jpg", unrecorded)
    unrecorded_hash = file_sha256(unrecorded)
    unavailable_assessor = UnavailableAssessor()

    with pytest.raises(RuntimeError, match="対応する完了記録もありません"):
        SingleVideoSelector(
            request,
            frame_extractor=FakeFrameExtractor(),
            assessor=unavailable_assessor,
        ).run()

    assert file_sha256(unrecorded) == unrecorded_hash
    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


def test_pipeline_regenerates_completion_with_extra_managed_artifact(
    tmp_path: Path,
) -> None:
    """現runの完了記録にないmanaged風成果物を残して即returnしないこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    unrecorded = output_dir / "selected-03.jpg"
    shutil.copyfile(output_dir / "selected-02.jpg", unrecorded)
    unavailable_assessor = UnavailableAssessor()

    resumed_sheet = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=unavailable_assessor,
    ).run()

    assert resumed_sheet == output_dir / "selected-contact-sheet.jpg"
    assert not unrecorded.exists()
    assert unavailable_assessor.metadata_calls == 0
    assert unavailable_assessor.assess_calls == 0


@pytest.mark.parametrize(
    "corruption",
    ["missing-manifest", "manifest-digest", "incomplete-artifacts"],
)
def test_pipeline_ignores_completion_without_matching_run_manifest(
    tmp_path: Path,
    corruption: str,
) -> None:
    """run manifestと完全一致しない完了記録自体は所有根拠にしないこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    run_cache = _single_run_cache(tmp_path)
    next(run_cache.glob("output-*.json")).write_text("{", encoding="utf-8")
    manifest_path = run_cache / "run-manifest.json"
    if corruption == "missing-manifest":
        manifest_path.unlink()
    elif corruption == "manifest-digest":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["output_count"] = 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        completion_path = _completion_path(run_cache)
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["artifacts"].pop()
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
    original_artifacts = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    selector = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    )
    selector._prepare_paths()

    assert not selector._completion_establishes_output_ownership(
        _completion_path(run_cache)
    )

    resumed_sheet = selector.run()

    assert resumed_sheet == output_dir / "selected-contact-sheet.jpg"
    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == (
        original_artifacts
    )


def test_pipeline_preserves_completed_output_when_replacement_probe_fails(
    tmp_path: Path,
) -> None:
    """条件変更runの事前検証失敗では使用可能な旧成果物を保持すること."""

    class AddedVideoFailingExtractor(FakeFrameExtractor):
        """追加動画のprobeだけを失敗させるextractor."""

        def probe(self, video: Path) -> VideoMetadata:
            if video.name == "Sample Game Part2.mp4":
                raise RuntimeError("added video probe failed")
            return super().probe(video)

    first_video = tmp_path / "Sample Game Part1.mp4"
    first_video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(first_video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    original_artifacts = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    second_video = tmp_path / "Sample Game Part2.mp4"
    second_video.write_bytes(bytes(range(255, -1, -1)) * 16)

    with pytest.raises(RuntimeError, match="added video probe failed"):
        SingleVideoSelector(
            replace(
                request,
                input_videos=(str(first_video), str(second_video)),
            ),
            frame_extractor=AddedVideoFailingExtractor(),
            assessor=FakeAssessor(),
        ).run()

    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == (
        original_artifacts
    )


def test_pipeline_rejects_concurrent_run_for_same_output_folder(
    tmp_path: Path,
) -> None:
    """異なるInput Video Directoryでも同じOutput Folderへ同時出力しないこと."""
    metadata_started = Event()
    release_metadata = Event()

    class BlockingAssessor(FakeAssessor):
        """最初のrunをOutput Folder lock保持中に待機させるfake."""

        def fetch_model_metadata(
            self,
            requested_models: set[str],
        ) -> dict[str, dict[str, Any]]:
            metadata_started.set()
            if not release_metadata.wait(timeout=5):
                raise TimeoutError("concurrency test timed out")
            return super().fetch_model_metadata(requested_models)

    first_input = tmp_path / "first-input"
    second_input = tmp_path / "second-input"
    first_input.mkdir()
    second_input.mkdir()
    first_video = first_input / "Sample Game.mp4"
    second_video = second_input / "Sample Game.mp4"
    first_video.write_bytes(bytes(range(256)) * 16)
    second_video.write_bytes(bytes(range(256)) * 16)
    output_dir = tmp_path / "selected"
    request = VideoSelectionRequest(
        input_video=str(first_video),
        output_dir=str(output_dir),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
            with pytest.raises(RuntimeError, match="同じOutput Folder.*実行中"):
                SingleVideoSelector(
                    replace(request, input_videos=(str(second_video),)),
                    frame_extractor=FakeFrameExtractor(),
                    assessor=FakeAssessor(),
                ).run()
        finally:
            release_metadata.set()

        assert first_run.result(timeout=10).is_file()


def test_pipeline_allocates_explicit_sample_minimum_across_long_input_videos(
    tmp_path: Path,
) -> None:
    """明示intervalの最低sample数を動画ごとに増幅せず全入力へ配分すること."""

    class LongVideoExtractor(FakeFrameExtractor):
        """長時間動画のmetadataを返すfake."""

        def probe(self, video: Path) -> VideoMetadata:
            assert video.is_file()
            self.probe_calls += 1
            return VideoMetadata(3600.0, 320, 180, "fake", "30/1")

    video_count = 7
    videos = tuple(
        tmp_path / f"Sample Game Part{index}.mp4" for index in range(video_count)
    )
    for video in videos:
        video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_videos=tuple(str(video) for video in videos),
        output_dir=str(tmp_path / "selected"),
        output_count=600,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=600.0,
        debug=False,
    )
    selector = SingleVideoSelector(
        request,
        frame_extractor=LongVideoExtractor(),
        assessor=FakeAssessor(),
    )

    selector._prepare_run()

    assert sum(len(source.timestamps) for source in selector.sources) == 600


def test_pipeline_plans_more_than_legacy_combined_candidate_limit(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """1時間動画4本の自動modeで4,320候補と評価上限を計画すること."""

    class LongVideoExtractor(FakeFrameExtractor):
        """1時間動画のmetadataを返すfake."""

        def probe(self, video: Path) -> VideoMetadata:
            assert video.is_file()
            self.probe_calls += 1
            return VideoMetadata(3600.0, 320, 180, "fake", "30/1")

    videos = tuple(tmp_path / f"Sample Game Part{index}.mp4" for index in range(4))
    for video in videos:
        video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_videos=tuple(str(video) for video in videos),
        output_dir=str(tmp_path / "selected"),
        output_count=30,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    selector = SingleVideoSelector(
        request,
        frame_extractor=LongVideoExtractor(),
        assessor=FakeAssessor(),
    )
    caplog.set_level(logging.INFO)

    selector._prepare_run()

    assert sum(len(source.timestamps) for source in selector.sources) == 4_320
    assert (
        "処理予定: 全候補数=4320件, 一次評価予定数=1440件（上限）, "
        "二次評価予定数=360件（上限）"
        in [record.getMessage() for record in caplog.records]
    )


def test_automatic_sample_positions_stay_stable_when_video_is_added(
    tmp_path: Path,
) -> None:
    """自動sample位置を入力集合に依存させず既存動画の抽出cacheを保つこと."""

    class MediumVideoExtractor(TrackingFrameExtractor):
        """入力集合依存のsample配分差が出る長さを返すfake."""

        def probe(self, video: Path) -> VideoMetadata:
            self.probed_videos.append(video.name)
            self.probe_calls += 1
            return VideoMetadata(600.0, 320, 180, "fake", "30/1")

    first_video = tmp_path / "Sample Game Part1.mp4"
    second_video = tmp_path / "Sample Game Part2.mp4"
    first_video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(first_video),
        output_dir=str(tmp_path / "selected-first"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    first_selector = SingleVideoSelector(
        request,
        frame_extractor=MediumVideoExtractor(),
        assessor=FakeAssessor(),
    )
    first_selector._prepare_run()
    first_candidates = first_selector._extract_candidates()
    first_selector._preselect_candidates(first_candidates)
    first_timestamps = first_selector.sources[0].timestamps
    second_video.write_bytes(bytes(range(256)) * 16)
    extractor = MediumVideoExtractor()
    second_selector = SingleVideoSelector(
        replace(
            request,
            input_videos=(str(first_video), str(second_video)),
            output_dir=str(tmp_path / "selected-second"),
        ),
        frame_extractor=extractor,
        assessor=FakeAssessor(),
    )

    second_selector._prepare_run()
    second_selector._extract_candidates()

    assert second_selector.sources[0].timestamps == first_timestamps
    assert set(extractor.candidate_videos) == {second_video.name}


@pytest.mark.parametrize("corruption", ["endpoint", "sample-count", "payload"])
def test_probe_cache_rejects_corrupt_payload(
    tmp_path: Path,
    corruption: str,
) -> None:
    """digest不一致または後続演算overflowのprobe cacheをmissにすること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    )._prepare_run()
    probe_path = next((_cache_root(tmp_path) / "videos").glob("*/probe/*.json"))
    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    if corruption == "payload":
        payload["data"]["metadata"]["video_stream_index"] = 1
    else:
        payload["data"]["metadata"]["duration_seconds"] = 1e308
        if corruption == "endpoint":
            payload["data"]["metadata"]["start_time_seconds"] = 1e308
    probe_path.write_text(json.dumps(payload), encoding="utf-8")
    extractor = FakeFrameExtractor()
    selector = SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=FakeAssessor(),
    )

    selector._prepare_run()

    assert extractor.probe_calls == 1
    assert selector.sources[0].metadata.duration_seconds == 4.0


def test_mechanical_cache_reextracts_changed_candidate_image(tmp_path: Path) -> None:
    """記録済みdigestと異なる候補JPEGを動画から再抽出すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    first_selector = SingleVideoSelector(
        request,
        frame_extractor=TrackingFrameExtractor(),
        assessor=FakeAssessor(),
    )
    first_selector._prepare_run()
    first_candidates = first_selector._extract_candidates()
    first_selector._preselect_candidates(first_candidates)
    first_key = first_selector._assessment_cache_key(
        request.primary_model,
        "primary",
        first_selector.sources[0],
        first_candidates,
    )
    shutil.copyfile(first_candidates[1].path, first_candidates[0].path)
    extractor = TrackingFrameExtractor()
    second_selector = SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=FakeAssessor(),
    )

    second_selector._prepare_run()
    second_candidates = second_selector._extract_candidates()
    second_selector._preselect_candidates(second_candidates)
    second_key = second_selector._assessment_cache_key(
        request.primary_model,
        "primary",
        second_selector.sources[0],
        second_candidates,
    )

    assert extractor.candidate_videos == [video.name]
    assert second_key == first_key


def test_candidate_cache_reextracts_without_recorded_digests(tmp_path: Path) -> None:
    """機械評価記録のない候補JPEGを抽出cache hitにしないこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    first_selector = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    )
    first_selector._prepare_run()
    first_candidates = first_selector._extract_candidates()
    shutil.copyfile(first_candidates[1].path, first_candidates[0].path)
    extractor = TrackingFrameExtractor()
    second_selector = SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=FakeAssessor(),
    )
    second_selector._prepare_run()

    second_candidates = second_selector._extract_candidates()

    assert len(extractor.candidate_videos) == len(second_candidates)
    assert set(extractor.candidate_videos) == {video.name}


def test_primary_cache_key_tracks_batch_composition(tmp_path: Path) -> None:
    """一次評価cacheを候補の順序・画像byteが等しい評価単位だけで再利用すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    selector = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    )
    selector._prepare_run()
    candidates = selector._extract_candidates()
    selector._preselect_candidates(candidates)

    first_key = selector._assessment_cache_key(
        request.primary_model,
        "primary",
        selector.sources[0],
        candidates[:3],
    )
    changed_key = selector._assessment_cache_key(
        request.primary_model,
        "primary",
        selector.sources[0],
        candidates[1:4],
    )

    assert changed_key != first_key


@pytest.mark.parametrize("quality_corruption", ["out-of-range", "in-range"])
def test_mechanical_cache_rejects_changed_quality_score(
    tmp_path: Path,
    quality_corruption: str,
) -> None:
    """JPEGから再計算した値と異なるquality scoreをcache hitにしないこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    first_selector = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    )
    first_selector._prepare_run()
    candidates = first_selector._extract_candidates()
    first_selector._preselect_candidates(candidates)
    mechanical_path = next(
        (_cache_root(tmp_path) / "videos").glob("*/mechanical-analysis/*.json")
    )
    payload = json.loads(mechanical_path.read_text(encoding="utf-8"))
    original_quality = payload["data"]["candidates"][0]["quality_score"]
    if quality_corruption == "out-of-range":
        changed_quality = 1e300
    elif original_quality == 0.0:
        changed_quality = 1.0
    else:
        changed_quality = 0.0
    payload["data"]["candidates"][0]["quality_score"] = changed_quality
    mechanical_path.write_text(json.dumps(payload), encoding="utf-8")
    second_selector = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    )
    second_selector._prepare_run()
    second_candidates = second_selector._extract_candidates()

    restored = second_selector._load_mechanical_candidates(
        second_selector.sources[0],
        second_candidates,
    )

    assert restored is None


@pytest.mark.parametrize(
    "candidate_corruption",
    ["duplicate", "missing", "reclassified"],
)
def test_mechanical_cache_rejects_candidate_membership_corruption(
    tmp_path: Path,
    candidate_corruption: str,
) -> None:
    """機械評価cacheのframe ID欠落・重複・分類変更をmissにすること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    first_selector = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    )
    first_selector._prepare_run()
    candidates = first_selector._extract_candidates()
    first_selector._preselect_candidates(candidates)
    mechanical_path = next(
        (_cache_root(tmp_path) / "videos").glob("*/mechanical-analysis/*.json")
    )
    payload = json.loads(mechanical_path.read_text(encoding="utf-8"))
    if candidate_corruption == "duplicate":
        payload["data"]["candidates"].append(payload["data"]["candidates"][0])
    elif candidate_corruption == "missing":
        payload["data"]["candidates"].pop(0)
    else:
        moved_id = payload["data"]["candidates"].pop(0)["frame_id"]
        rejected_ids = {*payload["data"]["rejected_frame_ids"], moved_id}
        payload["data"]["rejected_frame_ids"] = [
            frame["frame_id"]
            for frame in payload["data"]["source_frames"]
            if frame["frame_id"] in rejected_ids
        ]
    mechanical_path.write_text(json.dumps(payload), encoding="utf-8")
    second_selector = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    )
    second_selector._prepare_run()
    second_candidates = second_selector._extract_candidates()

    restored = second_selector._load_mechanical_candidates(
        second_selector.sources[0],
        second_candidates,
    )

    assert restored is None


@pytest.mark.parametrize(
    "difference_hash",
    ["-000000000000001", "000000000000000g"],
)
def test_mechanical_cache_rejects_non_hexadecimal_difference_hash(
    tmp_path: Path,
    difference_hash: str,
) -> None:
    """16桁のunsigned hexadecimalではないdifference hashを拒否すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    first_selector = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    )
    first_selector._prepare_run()
    candidates = first_selector._extract_candidates()
    first_selector._preselect_candidates(candidates)
    mechanical_path = next(
        (_cache_root(tmp_path) / "videos").glob("*/mechanical-analysis/*.json")
    )
    payload = json.loads(mechanical_path.read_text(encoding="utf-8"))
    payload["data"]["candidates"][0]["difference_hash"] = difference_hash
    mechanical_path.write_text(json.dumps(payload), encoding="utf-8")
    second_selector = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    )
    second_selector._prepare_run()
    second_candidates = second_selector._extract_candidates()

    restored = second_selector._load_mechanical_candidates(
        second_selector.sources[0],
        second_candidates,
    )

    assert restored is None


def test_mechanical_cache_rejects_difference_hash_not_matching_image(
    tmp_path: Path,
) -> None:
    """JPEGと一致しないvalidなdifference hashを拒否すること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    first_selector = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    )
    first_selector._prepare_run()
    candidates = first_selector._extract_candidates()
    first_selector._preselect_candidates(candidates)
    mechanical_path = next(
        (_cache_root(tmp_path) / "videos").glob("*/mechanical-analysis/*.json")
    )
    payload = json.loads(mechanical_path.read_text(encoding="utf-8"))
    stored_hash = payload["data"]["candidates"][0]["difference_hash"]
    payload["data"]["candidates"][0]["difference_hash"] = (
        f"{int(stored_hash, 16) ^ 1:016x}"
    )
    mechanical_path.write_text(json.dumps(payload), encoding="utf-8")
    second_selector = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    )
    second_selector._prepare_run()
    second_candidates = second_selector._extract_candidates()

    restored = second_selector._load_mechanical_candidates(
        second_selector.sources[0],
        second_candidates,
    )

    assert restored is None


@pytest.mark.parametrize(
    "checkpoint_corruption",
    ["non-prefix", "scene", "reason", "digest-score", "digest-transition"],
)
def test_assessment_cache_rejects_corrupt_checkpoint(
    tmp_path: Path,
    checkpoint_corruption: str,
) -> None:
    """batch prefixまたはtext契約が不正な評価cacheをmissにすること."""

    class BatchTrackingAssessor(FakeAssessor):
        """評価batchの候補数を記録するfake."""

        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def assess(
            self,
            *,
            model: str,
            model_digest: str,
            prompt: str,
            candidates: Sequence[FrameCandidate],
            contact_sheet: Path,
        ) -> list[FrameAssessment]:
            self.batch_sizes.append(len(candidates))
            return super().assess(
                model=model,
                model_digest=model_digest,
                prompt=prompt,
                candidates=candidates,
                contact_sheet=contact_sheet,
            )

    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    assessor = BatchTrackingAssessor()
    selector = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=assessor,
    )
    selector._prepare_run()
    candidates = selector._extract_candidates()
    primary_candidates = selector._preselect_candidates(candidates)
    selector._assess_candidates(
        model=request.primary_model,
        stage="primary",
        candidates=primary_candidates,
    )
    state_path = _assessment_files(tmp_path, "primary")[0]
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    first_frame_id = primary_candidates[0].frame_id
    if checkpoint_corruption == "non-prefix":
        payload["assessments"].pop(first_frame_id)
    elif checkpoint_corruption == "digest-score":
        original_score = payload["assessments"][first_frame_id]["blog_score"]
        payload["assessments"][first_frame_id]["blog_score"] = (
            0.0 if original_score != 0.0 else 1.0
        )
    elif checkpoint_corruption == "digest-transition":
        original_transition = payload["assessments"][first_frame_id]["is_transition"]
        payload["assessments"][first_frame_id][
            "is_transition"
        ] = not original_transition
    else:
        field = checkpoint_corruption
        limit = 80 if field == "scene" else 300
        payload["assessments"][first_frame_id][field] = "x" * (limit + 1)
    if not checkpoint_corruption.startswith("digest-"):
        assessment_payload = {
            "cache_key": payload["cache_key"],
            "assessments": payload["assessments"],
        }
        payload["payload_digest"] = json_digest(assessment_payload)
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    assessor.batch_sizes.clear()

    selector._assess_candidates(
        model=request.primary_model,
        stage="primary",
        candidates=primary_candidates,
    )

    assert sum(assessor.batch_sizes) == len(primary_candidates)
    assert assessor.batch_sizes[0] > 1


@pytest.mark.parametrize("cache_image", ["candidate", "context"])
def test_pipeline_reextracts_symlinked_cached_image(
    tmp_path: Path,
    cache_image: str,
) -> None:
    """candidateとcontextのleaf symlinkをcache hitにしないこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    selector = SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=FakeAssessor(),
    )
    selector._prepare_run()
    candidates = selector._extract_candidates()
    context_candidate: FrameCandidate | None = None
    if cache_image == "candidate":
        selector._preselect_candidates(candidates)
        cached_path = Path(candidates[0].path)
    else:
        primary_candidates = selector._preselect_candidates(candidates)
        context_candidate = primary_candidates[0]
        selector._extract_context_frames([context_candidate])
        cached_path = context_frame_path(
            selector._context_directory(selector._source_for(context_candidate)),
            context_candidate,
            "before",
        )
    external = tmp_path / f"external-{cache_image}.jpg"
    shutil.copyfile(cached_path, external)
    external_hash = file_sha256(external)
    cached_path.unlink()
    cached_path.symlink_to(external)
    extractor.extract_calls = 0

    if cache_image == "candidate":
        selector._extract_candidates()
    else:
        assert context_candidate is not None
        selector._extract_context_frames([context_candidate])

    assert extractor.extract_calls == 1
    assert not cached_path.is_symlink()
    assert file_sha256(external) == external_hash


def test_pipeline_reextracts_cached_image_over_decompression_bomb_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pillowの展開上限を超えるcache画像をabortせずcache missにすること."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    selector = SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=FakeAssessor(),
    )
    selector._prepare_run()
    candidates = selector._extract_candidates()
    extractor.extract_calls = 0
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    selector._extract_candidates()

    assert extractor.extract_calls == len(candidates)


@pytest.mark.parametrize("invalid_cache_image", ["too-wide", "not-jpeg"])
def test_pipeline_reextracts_cached_image_outside_extraction_contract(
    tmp_path: Path,
    invalid_cache_image: str,
) -> None:
    """最大幅を超える画像と非JPEGをcache hitにしないこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    selector = SingleVideoSelector(
        request,
        frame_extractor=extractor,
        assessor=FakeAssessor(),
    )
    selector._prepare_run()
    candidates = selector._extract_candidates()
    selector._preselect_candidates(candidates)
    cached_path = Path(candidates[0].path)
    if invalid_cache_image == "too-wide":
        Image.new("RGB", (961, 10), "white").save(cached_path, format="JPEG")
    else:
        Image.new("RGB", (320, 180), "white").save(cached_path, format="PNG")
    extractor.extract_calls = 0

    selector._extract_candidates()

    assert extractor.extract_calls == 1


@pytest.mark.parametrize("cache_directory", ["run", "video"])
def test_pipeline_rejects_symlinked_cache_directories(
    tmp_path: Path,
    cache_directory: str,
) -> None:
    """managed cache directoryのsymlinkを辿って外部へ書き込まないこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    selector = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    )
    selector._prepare_paths()
    outside = tmp_path / "outside"
    outside.mkdir()
    if cache_directory == "video":
        symlink_path = selector.cache_root / "videos" / selector.video_identities[0].key
    else:
        selector.game_context = request.game_context
        run_key = json_digest(selector._run_identity_conditions())
        symlink_path = selector.cache_root / "runs" / run_key
    symlink_path.parent.mkdir(parents=True, exist_ok=True)
    if symlink_path.is_dir():
        symlink_path.rmdir()
    symlink_path.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        selector._prepare_run()

    assert list(outside.iterdir()) == []


def test_primary_assessment_rejects_symlinked_cache_directory(
    tmp_path: Path,
) -> None:
    """評価cache directoryのsymlinkを辿って外部へ書き込まないこと."""
    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host="fake",
        ollama_timeout=1.0,
        allow_cpu=True,
        ffmpeg_workers=2,
        sample_interval_seconds=None,
        debug=False,
    )
    selector = SingleVideoSelector(
        request,
        frame_extractor=FakeFrameExtractor(),
        assessor=FakeAssessor(),
    )
    selector._prepare_run()
    candidates = selector._extract_candidates()
    primary_candidates = selector._preselect_candidates(candidates)
    outside = tmp_path / "outside"
    outside.mkdir()
    assessments_dir = selector.sources[0].cache_dir / "assessments"
    assessments_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        selector._assess_candidates(
            model=request.primary_model,
            stage="primary",
            candidates=primary_candidates,
        )

    assert list(outside.iterdir()) == []


def test_pipeline_rejects_concurrent_run_for_same_input_cache(tmp_path: Path) -> None:
    """同じInput Video Directory cacheへ同時に書き込ませないこと."""
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
        game_context="テスト用のGame Context",
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
            with pytest.raises(
                RuntimeError,
                match="同じInput Video Directory.*実行中",
            ):
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
    """Ctrl+C時に未開始jobを取消し、active cache writerの終了を待つこと."""
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
        game_context="テスト用のGame Context",
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

    assert (True, True) in shutdown_calls


def test_candidate_extraction_keeps_pending_jobs_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """候補総数分の抽出jobを一度にexecutorへ投入しないこと."""
    extraction_started = Event()
    release_extraction = Event()
    submission_window_filled = Event()
    submitted_futures: list[object] = []

    class BlockingFrameExtractor(FakeFrameExtractor):
        """releaseまで候補抽出を停止するfake."""

        def extract_frame(
            self,
            video: Path,
            timestamp_seconds: float,
            output_path: Path,
            *,
            max_width: int | None,
            video_stream_index: int = 0,
        ) -> None:
            extraction_started.set()
            assert release_extraction.wait(timeout=2)
            super().extract_frame(
                video,
                timestamp_seconds,
                output_path,
                max_width=max_width,
                video_stream_index=video_stream_index,
            )

    class RecordingExecutor(ThreadPoolExecutor):
        """投入されたfutureを記録するexecutor."""

        def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
            future = super().submit(fn, *args, **kwargs)
            submitted_futures.append(future)
            if len(submitted_futures) == 2:
                submission_window_filled.set()
            return future

    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
        frame_extractor=BlockingFrameExtractor(),
        assessor=FakeAssessor(),
    )
    selector._prepare_run()
    monkeypatch.setattr(
        "src.services.video_selector.ThreadPoolExecutor",
        RecordingExecutor,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        extraction = executor.submit(selector._extract_candidates)
        assert extraction_started.wait(timeout=2)
        assert submission_window_filled.wait(timeout=2)
        try:
            assert len(submitted_futures) == 2
            assert len(submitted_futures) < len(selector.sources[0].timestamps)
        finally:
            release_extraction.set()
        assert len(extraction.result(timeout=10)) == 13


def test_mechanical_evaluation_keeps_pending_jobs_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """候補総数分の機械評価jobを一度にexecutorへ投入しないこと."""
    evaluation_started = Event()
    release_evaluation = Event()
    submission_window_filled = Event()
    submitted_futures: list[object] = []

    class RecordingExecutor(ThreadPoolExecutor):
        """投入されたfutureを記録するexecutor."""

        def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
            future = super().submit(fn, *args, **kwargs)
            submitted_futures.append(future)
            if len(submitted_futures) == 4:
                submission_window_filled.set()
            return future

    video = tmp_path / "Sample Game.mp4"
    video.write_bytes(bytes(range(256)) * 16)
    request = VideoSelectionRequest(
        input_video=str(video),
        output_dir=str(tmp_path / "selected"),
        output_count=2,
        game_title=None,
        game_context="テスト用のGame Context",
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
    candidates = selector._extract_candidates()
    original_measure_candidate = measure_candidate

    def blocking_measure_candidate(candidate: FrameCandidate) -> FrameCandidate | None:
        evaluation_started.set()
        assert release_evaluation.wait(timeout=2)
        return original_measure_candidate(candidate)

    monkeypatch.setattr(
        "src.services.video_selector.ThreadPoolExecutor",
        RecordingExecutor,
    )
    monkeypatch.setattr(
        "src.services.video_selector.measure_candidate",
        blocking_measure_candidate,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        evaluation = executor.submit(selector._preselect_candidates, candidates)
        assert evaluation_started.wait(timeout=2)
        assert submission_window_filled.wait(timeout=2)
        try:
            assert len(submitted_futures) == 4
            assert len(submitted_futures) < len(candidates)
        finally:
            release_evaluation.set()
        assert evaluation.result(timeout=10)


def test_context_extraction_cancels_queued_jobs_on_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """遷移frame抽出中も未開始jobを取消し、active writerの終了を待つこと."""
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
        game_context="テスト用のGame Context",
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

    assert (True, True) in shutdown_calls


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
        game_context="テスト用のGame Context",
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
        game_context="テスト用のGame Context",
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
