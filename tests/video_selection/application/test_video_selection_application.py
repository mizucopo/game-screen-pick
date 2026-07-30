"""実Processorを接続するVideo Selection Applicationのtest。"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.video_selection.application.video_selection_application import (
    VideoSelectionApplication,
)
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.run_status import RunStatus
from src.video_selection.services.run_progress_tracker import RunProgressTracker
from tests.video_selection.fakes.echo_structured_vision_runtime import (
    EchoStructuredVisionRuntime,
)
from tests.video_selection.fakes.fake_model_runtime import FakeModelRuntime
from tests.video_selection.fakes.fake_speech_runtime import FakeSpeechRuntime
from tests.video_selection.fakes.fake_video_stage_media_runtime import (
    FakeVideoStageMediaRuntime,
)
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver


def test_real_processors_publish_canonical_output_and_reuse_warm_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """実Processor列がcanonical outputを公開しwarm時に推論cacheが再利用されること。

    Arrange:
        - 一つのVideo Sourceと決定的なMedia、Model、Speech、Vision fakeが用意される
    Act:
        - cold runと別Output Folderへのexact warm runが実行される
    Assert:
        - 選択画像とcanonical reportがatomicに公開されること
        - warm runでscan、refinement、Vision推論が再実行されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mkv").write_bytes(b"video-content")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "cold-output",
        image_count=1,
    )
    cold_media = FakeVideoStageMediaRuntime()
    cold_vision = EchoStructuredVisionRuntime()
    cold_observer = RecordingRunObserver()
    cold_progress = RunProgressTracker(cold_observer)
    cold_progress.start_run()
    cold_application = _application(
        cold_media,
        cold_vision,
        observer=cold_observer,
        progress=cold_progress,
    )

    # Act
    cold = cold_application.run(configuration)
    selection_indexes = tuple(
        (configuration.processing_cache_folder / ".indexes" / "video-sets").rglob(
            "*.json"
        )
    )
    assert len(selection_indexes) == 1
    selection_indexes[0].unlink()
    monkeypatch.setattr(
        "src.video_selection.application.video_selection_application."
        "select_from_shortlist_batches",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("warm runでselectorが再実行されました")
        ),
    )
    warm_media = FakeVideoStageMediaRuntime()
    warm_vision = EchoStructuredVisionRuntime()
    warm_observer = RecordingRunObserver()
    warm_progress = RunProgressTracker(warm_observer)
    warm_progress.start_run()
    warm = _application(
        warm_media,
        warm_vision,
        observer=warm_observer,
        progress=warm_progress,
    ).run(replace(configuration, output_folder=tmp_path / "warm-output"))

    # Assert
    cold_report = json.loads(
        (cold.output_folder / "report.json").read_text(encoding="utf-8")
    )
    warm_report = json.loads(
        (warm.output_folder / "report.json").read_text(encoding="utf-8")
    )
    assert cold.status is RunStatus.COMPLETED
    assert cold.selected_count == 1
    assert len(tuple((cold.output_folder / "images").glob("*.webp"))) == 1
    assert cold_report["run"]["selected_image_count"] == 1
    assert warm_report["run"]["selected_image_count"] == 1
    assert cold_media.scan_calls
    assert cold_media.range_calls
    assert cold_vision.scene_catalog_calls
    assert cold_vision.candidate_annotation_calls
    assert warm_media.scan_calls == []
    assert warm_media.range_calls == []
    assert warm_vision.scene_catalog_calls == []
    assert warm_vision.candidate_annotation_calls == []
    cold_selection_cache = tuple(
        event
        for event in cold_observer.progress_events
        if event.kind == "cache" and event.stage is ProcessingStage.SELECT_IMAGES
    )
    warm_selection_cache = tuple(
        event
        for event in warm_observer.progress_events
        if event.kind == "cache" and event.stage is ProcessingStage.SELECT_IMAGES
    )
    assert len(cold_selection_cache) == 1
    assert cold_selection_cache[0].recompute_count == 1
    assert len(warm_selection_cache) == 1
    assert warm_selection_cache[0].reuse_count == 1
    assert warm_selection_cache[0].recompute_count == 0
    assert selection_indexes[0].is_file()
    cold_stages = {
        stage["name"]: stage for stage in cold_report["provenance"]["stages"]
    }
    warm_stages = {
        stage["name"]: stage for stage in warm_report["provenance"]["stages"]
    }
    cold_scan = cold_stages["scan_video_001"]
    cold_extraction = cold_stages["extract_frame_candidates_001"]
    cold_context = cold_stages["collect_context_001"]
    cold_catalog = cold_stages["build_scene_catalog_001"]
    cold_annotation = cold_stages["annotate_candidate_001"]
    cold_selection = cold_stages["select_images_001"]
    warm_scan = warm_stages["scan_video_001"]
    assert cold_scan["cache_misses"] == 1
    assert cold_scan["recomputed_items"] == 1
    assert cold_scan["attempt_count"] == 1
    assert cold_extraction["upstream_fingerprints"] == [cold_scan["fingerprint"]]
    assert warm_scan["cache_hits"] == 1
    assert warm_scan["cache_misses"] == 0
    assert warm_scan["recomputed_items"] == 0
    assert warm_scan["attempt_count"] == 1
    provenance = cold_report["provenance"]
    assert "speech_runtime_identity" not in provenance["runtime"]
    assert "speech_to_text" not in provenance["tools"]
    assert provenance["runtime"]["video_scan_parallelism"]["mode"] == "auto"
    assert provenance["runtime"]["video_scan_parallelism"]["initial_workers"] == 1
    assert cold_scan["effective_settings"]["decode_backend"] == "cpu"
    assert cold_scan["tool_refs"] == ["ffmpeg"]
    assert cold_scan["contract_refs"] == ["video_scan"]
    assert cold_extraction["tool_refs"] == ["ffmpeg"]
    assert cold_extraction["contract_refs"] == ["frame_candidate_extraction"]
    assert cold_context["tool_refs"] == ["video_selection"]
    assert cold_context["model_refs"] == []
    assert cold_context["contract_refs"] == ["context_collection"]
    assert cold_catalog["tool_refs"] == ["ollama"]
    assert cold_catalog["model_refs"] == ["scene_catalog"]
    assert cold_catalog["effective_settings"]["stage_contract_version"]
    assert cold_catalog["prompt_eval_tokens"] == 10
    assert cold_catalog["eval_tokens"] == 5
    assert cold_annotation["tool_refs"] == ["ollama"]
    assert cold_annotation["model_refs"] == ["candidate_annotation"]
    assert cold_annotation["contract_refs"] == [
        "candidate_annotation",
        "nearby_context_policy",
    ]
    assert cold_selection["tool_refs"] == ["video_selection"]
    assert cold_selection["contract_refs"] == ["video_set_selection_policy"]


def test_restart_after_atomic_publication_reuses_exact_output(
    tmp_path: Path,
) -> None:
    """atomic公開後に呼出元が中断されても同じoutputで正常終了されること。

    Arrange:
        - 一つのVideo Sourceが最後まで処理されCanonical outputが公開される
        - 呼出元の完了記録だけが失われ同じ設定でapplicationが再起動される
    Act:
        - 同じOutput Folderを指定したrunが再実行される
    Assert:
        - 全StageとSelected WebP checkpointが再利用されること
        - 公開済みreportと画像のbyteが変更されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mkv").write_bytes(b"video-content")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        image_count=1,
    )
    first_outcome = _application(
        FakeVideoStageMediaRuntime(),
        EchoStructuredVisionRuntime(),
    ).run(configuration)
    before = {
        path.relative_to(first_outcome.output_folder): path.read_bytes()
        for path in first_outcome.output_folder.rglob("*")
        if path.is_file()
    }
    resumed_media = FakeVideoStageMediaRuntime()
    resumed_vision = EchoStructuredVisionRuntime()

    # Act
    resumed_outcome = _application(
        resumed_media,
        resumed_vision,
    ).run(configuration)

    # Assert
    after = {
        path.relative_to(resumed_outcome.output_folder): path.read_bytes()
        for path in resumed_outcome.output_folder.rglob("*")
        if path.is_file()
    }
    assert resumed_outcome.status is first_outcome.status
    assert resumed_outcome.selected_count == first_outcome.selected_count
    assert resumed_outcome.reused_completed_publication is True
    assert resumed_media.scan_calls == []
    assert resumed_media.range_calls == []
    assert resumed_media.extracted_original_frame_calls == []
    assert resumed_vision.scene_catalog_calls == []
    assert resumed_vision.candidate_annotation_calls == []
    assert after == before
    assert not tuple(tmp_path.glob(".output.*.staging"))


def test_restart_reuses_exact_output_after_unused_speech_dependency_update(
    tmp_path: Path,
) -> None:
    """未使用STT依存が更新されても完成済みoutputがbyte変更なしで再利用されること。

    Arrange:
        - subtitle・audioのない動画でCanonical outputが公開される
        - STT modelとSpeech Runtimeだけが更新された再起動runが用意される
    Act:
        - 同じOutput Folderを指定して再実行される
    Assert:
        - 完成済みpublicationが再利用され全output byteが保持されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mkv").write_bytes(b"video-content")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        image_count=1,
    )
    first_outcome = _application(
        FakeVideoStageMediaRuntime(),
        EchoStructuredVisionRuntime(),
        speech_identity_seed="speech-model-a",
        speech_runtime_identity="speech-runtime-a",
    ).run(configuration)
    before = {
        path.relative_to(first_outcome.output_folder): path.read_bytes()
        for path in first_outcome.output_folder.rglob("*")
        if path.is_file()
    }

    # Act
    resumed_outcome = _application(
        FakeVideoStageMediaRuntime(),
        EchoStructuredVisionRuntime(),
        speech_identity_seed="speech-model-b",
        speech_runtime_identity="speech-runtime-b",
    ).run(configuration)

    # Assert
    after = {
        path.relative_to(resumed_outcome.output_folder): path.read_bytes()
        for path in resumed_outcome.output_folder.rglob("*")
        if path.is_file()
    }
    assert resumed_outcome.reused_completed_publication is True
    assert after == before


def test_hash_consistent_corrupt_selection_cache_is_locally_recomputed(
    tmp_path: Path,
) -> None:
    """hash整合した決定不一致Selectionだけが破棄され再選定されること。

    Arrange:
        - 正常runのSelection scoreが算術整合を保った別値へ書き換えられる
        - manifestのartifact hashも書き換え後の内容へ一致させられる
    Act:
        - 同じ入力と別Output Folderでapplicationが再実行される
    Assert:
        - Video、Context、Visionの健全なcheckpointが再利用されること
        - Selectionだけが再計算され中断なしと同じ選択結果と画像が公開されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mkv").write_bytes(b"video-content")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "first-output",
        image_count=1,
    )
    first_outcome = _application(
        FakeVideoStageMediaRuntime(),
        EchoStructuredVisionRuntime(),
    ).run(configuration)
    first_report = json.loads(
        (first_outcome.output_folder / "report.json").read_text(encoding="utf-8")
    )
    selection_stage_folders = tuple(
        (configuration.processing_cache_folder / "video-sets").glob(
            f"*/{ProcessingStage.SELECT_IMAGES.value}/*"
        )
    )
    assert len(selection_stage_folders) == 1
    selection_stage_folder = selection_stage_folders[0]
    artifact_path = selection_stage_folder / "artifact.json"
    manifest_path = selection_stage_folder / "manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    score = artifact["selected"][0]["score"]
    score["base_utility"] += 0.25
    score["marginal_utility"] += 0.25
    artifact_bytes = (
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    artifact_path.write_bytes(artifact_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_record = next(
        item for item in manifest["artifacts"] if item["path"] == "artifact.json"
    )
    artifact_record["size_bytes"] = len(artifact_bytes)
    artifact_record["sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resumed_media = FakeVideoStageMediaRuntime()
    resumed_vision = EchoStructuredVisionRuntime()
    resumed_observer = RecordingRunObserver()
    resumed_progress = RunProgressTracker(resumed_observer)
    resumed_progress.start_run()

    # Act
    resumed_outcome = _application(
        resumed_media,
        resumed_vision,
        observer=resumed_observer,
        progress=resumed_progress,
    ).run(
        replace(
            configuration,
            output_folder=tmp_path / "resumed-output",
        )
    )

    # Assert
    resumed_report = json.loads(
        (resumed_outcome.output_folder / "report.json").read_text(encoding="utf-8")
    )
    assert resumed_media.scan_calls == []
    assert resumed_media.range_calls == []
    assert resumed_vision.scene_catalog_calls == []
    assert resumed_vision.candidate_annotation_calls == []
    assert resumed_report["selected"] == first_report["selected"]
    first_images = {
        path.name: path.read_bytes()
        for path in (first_outcome.output_folder / "images").glob("*.webp")
    }
    resumed_images = {
        path.name: path.read_bytes()
        for path in (resumed_outcome.output_folder / "images").glob("*.webp")
    }
    assert resumed_images == first_images
    selection_events = tuple(
        event
        for event in resumed_observer.progress_events
        if event.kind == "cache" and event.stage is ProcessingStage.SELECT_IMAGES
    )
    assert len(selection_events) == 1
    assert selection_events[0].recompute_count == 1


def test_similarity_threshold_change_reuses_vision_and_recomputes_selection(
    tmp_path: Path,
) -> None:
    """類似度閾値変更がVisionまで遡らず最終選定だけを失効させること。

    Arrange:
        - 一つのVideo Sourceを既定類似度閾値で完了したrunが用意される
    Act:
        - 同じcacheで類似度閾値だけを変更して再実行される
    Assert:
        - Video StageとVision推論は再利用されること
        - Select Images Stageだけが異なるfingerprintで再計算されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mkv").write_bytes(b"video-content")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "first-output",
        image_count=1,
        similarity_threshold=0.72,
    )
    first_outcome = _application(
        FakeVideoStageMediaRuntime(),
        EchoStructuredVisionRuntime(),
    ).run(configuration)
    changed_media = FakeVideoStageMediaRuntime()
    changed_vision = EchoStructuredVisionRuntime()
    changed_observer = RecordingRunObserver()
    changed_progress = RunProgressTracker(changed_observer)
    changed_progress.start_run()

    # Act
    changed_outcome = _application(
        changed_media,
        changed_vision,
        observer=changed_observer,
        progress=changed_progress,
    ).run(
        replace(
            configuration,
            output_folder=tmp_path / "changed-output",
            similarity_threshold=0.90,
        )
    )

    # Assert
    first_report = json.loads(
        (first_outcome.output_folder / "report.json").read_text(encoding="utf-8")
    )
    changed_report = json.loads(
        (changed_outcome.output_folder / "report.json").read_text(encoding="utf-8")
    )
    first_stages = {
        stage["name"]: stage["fingerprint"]
        for stage in first_report["provenance"]["stages"]
    }
    changed_stages = {
        stage["name"]: stage["fingerprint"]
        for stage in changed_report["provenance"]["stages"]
    }
    assert changed_media.scan_calls == []
    assert changed_media.range_calls == []
    assert changed_vision.scene_catalog_calls == []
    assert changed_vision.candidate_annotation_calls == []
    assert (
        changed_stages["build_scene_catalog_001"]
        == first_stages["build_scene_catalog_001"]
    )
    assert (
        changed_stages["annotate_candidate_001"]
        == first_stages["annotate_candidate_001"]
    )
    assert changed_stages["select_images_001"] != first_stages["select_images_001"]
    selection_cache_events = tuple(
        event
        for event in changed_observer.progress_events
        if event.kind == "cache" and event.stage is ProcessingStage.SELECT_IMAGES
    )
    assert len(selection_cache_events) == 1
    assert selection_cache_events[0].recompute_count == 1


def test_report_timestamps_cover_the_pipeline_lifecycle(tmp_path: Path) -> None:
    """reportの開始・完了時刻がpipeline全体の境界から記録されること。

    Arrange:
        - run開始時刻とpublication時刻を返すUTC clockが用意される
    Act:
        - Video Selection Applicationが実行される
    Assert:
        - reportへ異なる開始・完了時刻がそのまま記録されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mkv").write_bytes(b"video-content")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        image_count=1,
    )
    media = FakeVideoStageMediaRuntime()
    clock_call_count = 0

    def lifecycle_clock() -> datetime:
        nonlocal clock_call_count
        clock_call_count += 1
        if clock_call_count == 1:
            return datetime(2026, 7, 21, 1, 2, 3, tzinfo=timezone.utc)
        assert media.extracted_original_frame_calls
        return datetime(2026, 7, 21, 1, 3, 4, tzinfo=timezone.utc)

    application = _application(
        media,
        EchoStructuredVisionRuntime(),
        clock=lifecycle_clock,
    )

    # Act
    outcome = application.run(configuration)

    # Assert
    report = json.loads(
        (outcome.output_folder / "report.json").read_text(encoding="utf-8")
    )
    assert report["run"]["started_at"] == "2026-07-21T01:02:03Z"
    assert report["run"]["completed_at"] == "2026-07-21T01:03:04Z"
    assert clock_call_count == 2


def test_speech_runtime_is_closed_before_vision_inference(tmp_path: Path) -> None:
    """STT model資源がVision推論開始前に解放されること。

    Arrange:
        - close状態を記録するSpeech RuntimeとVision callbackが用意される
    Act:
        - Video Selection Applicationが実行される
    Assert:
        - 最初のScene Catalog推論時にSpeech Runtimeがclose済みであること
        - close前に取得したSpeech Runtime Identityがreportへ保持されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mkv").write_bytes(b"video-content")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        image_count=1,
    )
    speech_runtime = FakeSpeechRuntime(runtime_identity="speech-runtime-before-close")
    close_states_at_vision: list[bool] = []
    vision = EchoStructuredVisionRuntime(
        on_create_scene_catalog=lambda: close_states_at_vision.append(
            speech_runtime.closed
        )
    )
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer)
    progress.start_run()
    application = VideoSelectionApplication(
        media_runtime=FakeVideoStageMediaRuntime(),
        model_runtime=FakeModelRuntime("application-test"),
        speech_runtime_factory=lambda _model, _configuration: speech_runtime,
        vision_runtime=vision,
        observer=observer,
        progress=progress,
    )

    # Act
    outcome = application.run(configuration)

    # Assert
    report = json.loads(
        (outcome.output_folder / "report.json").read_text(encoding="utf-8")
    )
    assert close_states_at_vision == [True]
    assert speech_runtime.close_call_count == 1
    assert "speech_runtime_identity" not in report["provenance"]["runtime"]


@pytest.mark.parametrize("failure_point", ("model", "speech", "media"))
def test_runtime_preflight_failure_preserves_cache_requested_for_reset(
    tmp_path: Path,
    failure_point: str,
) -> None:
    """runtime preflight失敗時にreset対象のprocessing cacheが保持されること。

    Arrange:
        - 既存processing cacheとmodel、speech、mediaの失敗境界が用意される
    Act:
        - reset_cacheを有効にして実Video Selection Applicationが実行される
    Assert:
        - runtime検証完了前には既存cacheが削除されないこと
        - 構築済みSpeech Runtimeだけが確実にcloseされること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mkv").write_bytes(b"video-content")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        image_count=1,
        reset_cache=True,
    )
    cache_sentinel = configuration.processing_cache_folder / "keep.json"
    cache_sentinel.parent.mkdir(parents=True)
    cache_sentinel.write_text("keep", encoding="utf-8")
    speech_runtime = FakeSpeechRuntime()
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer)
    progress.start_run()

    def fail_media_preflight() -> None:
        if failure_point == "media":
            raise RuntimeError("media preflight failed")

    def speech_factory(
        _model: object,
        _configuration: EffectiveConfiguration,
    ) -> FakeSpeechRuntime:
        if failure_point == "speech":
            raise RuntimeError("speech preflight failed")
        return speech_runtime

    application = VideoSelectionApplication(
        media_runtime=FakeVideoStageMediaRuntime(
            on_preflight=fail_media_preflight,
        ),
        model_runtime=FakeModelRuntime(
            "application-test",
            resolution_error=(
                RuntimeError("model preflight failed")
                if failure_point == "model"
                else None
            ),
        ),
        speech_runtime_factory=speech_factory,
        vision_runtime=EchoStructuredVisionRuntime(),
        observer=observer,
        progress=progress,
    )

    # Act
    # Assert
    with pytest.raises(RuntimeError, match="preflight failed"):
        application.run(configuration)
    assert cache_sentinel.read_text(encoding="utf-8") == "keep"
    assert observer.legacy_cache_diagnostics == []
    assert speech_runtime.close_call_count == (1 if failure_point == "media" else 0)


def test_media_preflight_failure_preserves_recognized_legacy_cache(
    tmp_path: Path,
) -> None:
    """media preflight失敗時に認識済みlegacy cacheも保持されること。

    Arrange:
        - 認識済みlegacy cacheと失敗するmedia preflightが用意される
    Act:
        - resetなしで実Video Selection Applicationが実行される
    Assert:
        - runtime検証前にはlegacy cache cleanupが実行されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mkv").write_bytes(b"video-content")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        image_count=1,
    )
    legacy_artifact = (
        configuration.processing_cache_folder / "neutral-analysis" / "keep.json"
    )
    legacy_artifact.parent.mkdir(parents=True)
    legacy_artifact.write_text("keep", encoding="utf-8")
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer)
    progress.start_run()

    def fail_media_preflight() -> None:
        raise RuntimeError("media preflight failed")

    application = VideoSelectionApplication(
        media_runtime=FakeVideoStageMediaRuntime(
            on_preflight=fail_media_preflight,
        ),
        model_runtime=FakeModelRuntime("application-test"),
        speech_runtime_factory=lambda model, _configuration: FakeSpeechRuntime(
            resolved_model_identity=model.execution_identity.identifier
        ),
        vision_runtime=EchoStructuredVisionRuntime(),
        observer=observer,
        progress=progress,
    )

    # Act
    # Assert
    with pytest.raises(RuntimeError, match="media preflight failed"):
        application.run(configuration)
    assert legacy_artifact.read_text(encoding="utf-8") == "keep"
    assert observer.legacy_cache_diagnostics == []


def test_no_valid_candidate_skips_vision_and_publishes_zero_shortfall(
    tmp_path: Path,
) -> None:
    """有効Candidateが0件ならVisionを呼ばず0枚warning outputが公開されること。

    Arrange:
        - refinement frameがすべてblackoutになるVideo Sourceが用意される
    Act:
        - Video Selection Applicationが実行される
    Assert:
        - Scene CatalogとCandidate Annotationが呼ばれないこと
        - requested N、selected 0、exhaustedを持つwarning reportが公開されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "black.mkv").write_bytes(b"video-content")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        image_count=3,
    )
    vision = EchoStructuredVisionRuntime()
    application = _application(
        FakeVideoStageMediaRuntime(zero_valid_frames=True),
        vision,
    )

    # Act
    outcome = application.run(configuration)

    # Assert
    report = json.loads(
        (outcome.output_folder / "report.json").read_text(encoding="utf-8")
    )
    assert outcome.status is RunStatus.COMPLETED_WITH_WARNINGS
    assert outcome.selected_count == 0
    assert vision.scene_catalog_calls == []
    assert vision.candidate_annotation_calls == []
    assert report["selection_summary"]["shortfall"] == {
        "requested": 3,
        "selected": 0,
        "all_candidate_moments_exhausted": True,
    }
    assert report["selected"] == []
    assert report["run"]["warnings"][0]["code"] == "selection_shortfall"


def test_shortfall_expands_annotation_without_fixed_maximum(tmp_path: Path) -> None:
    """initial batch不足時に要求枚数単位で全有効Momentまで注釈されること。

    Arrange:
        - 26件の有効Momentと要求2枚、近似画像を返す13 Videoが用意される
    Act:
        - Video Selection Applicationが実行される
    Assert:
        - initial 24件の後に2件が追加され全26件が注釈されること
        - Scene Catalog代表が要求枚数に依存せず24枚に制限されること
        - 全Moment消費後のshortfallが記録されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    for index in range(13):
        (input_folder / f"{index:02d}.mkv").write_bytes(f"video-{index}".encode())
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        image_count=2,
        candidate_density_per_minute=3.0,
    )
    vision = EchoStructuredVisionRuntime()
    application = _application(
        FakeVideoStageMediaRuntime(distant_moments=True),
        vision,
    )

    # Act
    outcome = application.run(configuration)

    # Assert
    report = json.loads(
        (outcome.output_folder / "report.json").read_text(encoding="utf-8")
    )
    assert len(vision.candidate_annotation_calls) == 26
    assert len(vision.scene_catalog_calls) == 1
    assert len(vision.scene_catalog_calls[0].representatives) == 24
    assert report["selection_summary"]["shortlist_expansion_count"] == 1
    assert (
        report["selection_summary"]["shortfall"]["all_candidate_moments_exhausted"]
        is True
    )


def _application(
    media_runtime: FakeVideoStageMediaRuntime,
    vision_runtime: EchoStructuredVisionRuntime,
    *,
    observer: RecordingRunObserver | None = None,
    progress: RunProgressTracker | None = None,
    clock: Callable[[], datetime] | None = None,
    speech_identity_seed: str = "speech-model",
    speech_runtime_identity: str = "fake-speech-runtime-v1",
) -> VideoSelectionApplication:
    """実Processorをtest fake境界へ接続する。"""
    actual_observer = observer or RecordingRunObserver()
    actual_progress = progress or RunProgressTracker(actual_observer)
    if progress is None:
        actual_progress.start_run()
    return VideoSelectionApplication(
        media_runtime=media_runtime,
        model_runtime=FakeModelRuntime(
            "application-test",
            speech_identity_seed=speech_identity_seed,
        ),
        speech_runtime_factory=lambda model, _configuration: FakeSpeechRuntime(
            runtime_identity=speech_runtime_identity,
            resolved_model_identity=model.execution_identity.identifier,
        ),
        vision_runtime=vision_runtime,
        observer=actual_observer,
        progress=actual_progress,
        clock=clock,
    )
