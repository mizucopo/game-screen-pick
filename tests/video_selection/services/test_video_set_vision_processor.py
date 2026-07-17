import threading
from fractions import Fraction
from pathlib import Path

import pytest

from src.video_selection.models.candidate_annotation import CandidateAnnotation
from src.video_selection.models.candidate_annotation_request import (
    CandidateAnnotationRequest,
)
from src.video_selection.models.candidate_moment import CandidateMoment
from src.video_selection.models.context_cue import ContextCue
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.frame_candidate import FrameCandidate
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.scene_catalog import SceneCatalog
from src.video_selection.models.scene_catalog_entry import SceneCatalogEntry
from src.video_selection.models.stage_fingerprint import StageFingerprint
from src.video_selection.models.video_set import VideoSet
from src.video_selection.services.discover_video_set import discover_video_set
from src.video_selection.services.run_progress_tracker import RunProgressTracker
from src.video_selection.services.video_set_vision_processor import (
    VideoSetVisionProcessor,
)
from tests.video_selection.fakes.fake_model_runtime import FakeModelRuntime
from tests.video_selection.fakes.fake_structured_vision_runtime import (
    FakeStructuredVisionRuntime,
)
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver


def test_matching_fingerprints_reuse_catalog_and_each_annotation(
    tmp_path: Path,
) -> None:
    """一致するStage Fingerprintでmodel contentが再生成されないこと。

    Arrange:
        - Scene Catalogと2件のCandidate Annotationを返すfakeが用意される
        - 初回と選択枚数・spoiler感度・outputだけ異なる2回目が用意される
    Act:
        - 同じVision Stage semantic入力が2回処理される
    Assert:
        - 2回目はCatalogとAnnotationをすべて独立cacheから復元すること
        - raw Context Cue、prompt body、raw response、reasoningがcacheされないこと
    """
    # Arrange
    video_set, configuration = _video_set_and_configuration(tmp_path)
    requests = _requests()
    catalog = _catalog()
    annotations = _annotations(requests)
    cold_runtime = FakeStructuredVisionRuntime(catalog, annotations)
    cold_processor = VideoSetVisionProcessor(
        cold_runtime,
        RecordingRunObserver(),
    )
    models = FakeModelRuntime("vision-model").resolve_models(configuration)

    # Act
    cold_result = cold_processor.process(
        video_set=video_set,
        representatives=tuple(request.frame_candidates[0] for request in requests),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=requests,
        configuration=configuration,
        resolved_models=models,
    )
    warm_runtime = FakeStructuredVisionRuntime(
        catalog,
        annotations,
        reject_all_calls=True,
    )
    warm_result = VideoSetVisionProcessor(
        warm_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        representatives=tuple(request.frame_candidates[0] for request in requests),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=requests,
        configuration=EffectiveConfiguration(
            video_input_folder=configuration.video_input_folder,
            output_folder=tmp_path / "different-output",
            image_count=999,
            spoiler_sensitivity="high",
        ),
        resolved_models=models,
    )

    # Assert
    assert len(cold_runtime.scene_catalog_calls) == 1
    assert len(cold_runtime.candidate_annotation_calls) == 2
    assert warm_runtime.scene_catalog_calls == []
    assert warm_runtime.candidate_annotation_calls == []
    assert warm_result.catalog == cold_result.catalog
    assert warm_result.annotations == cold_result.annotations
    assert cold_result.catalog_diagnostics.cache_hit is False
    assert all(not item.cache_hit for item in cold_result.annotation_diagnostics)
    assert warm_result.catalog_diagnostics.cache_hit is True
    assert all(item.cache_hit for item in warm_result.annotation_diagnostics)
    assert len(warm_result.completed_stages) == 3
    cache_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in configuration.processing_cache_folder.rglob("*.json")
    )
    assert "正体を明かす秘密の台詞" not in cache_text
    assert '"messages"' not in cache_text
    assert '"raw_response"' not in cache_text
    assert '"reasoning"' not in cache_text


def test_same_catalog_fingerprint_runs_inference_once_under_lock(
    tmp_path: Path,
) -> None:
    """同じCatalog fingerprintの並行処理でmodel推論が一度だけ実行されること。

    Arrange:
        - 最初のCatalog推論を一時停止するfake runtimeが用意される
        - 同じVideo Setとsemantic入力を処理する2つのthreadが用意される
    Act:
        - 最初の推論中に2つ目の処理が開始される
    Assert:
        - 2つ目が同じ推論を開始せず最初のCompleted Stageを再利用すること
    """
    # Arrange
    video_set, configuration = _video_set_and_configuration(tmp_path)
    requests = _requests()
    request = requests[0]
    annotation = _annotations(requests)[0]
    first_call_started = threading.Event()
    release_first_call = threading.Event()
    second_process_started = threading.Event()
    runtime = FakeStructuredVisionRuntime(
        _catalog(),
        (annotation,),
        scene_catalog_call_started=first_call_started,
        release_scene_catalog_call=release_first_call,
    )
    models = FakeModelRuntime("vision-model").resolve_models(configuration)
    errors: list[BaseException] = []
    results: list[object] = []

    def process(second: bool = False) -> None:
        try:
            if second:
                second_process_started.set()
            results.append(
                VideoSetVisionProcessor(runtime, RecordingRunObserver()).process(
                    video_set=video_set,
                    representatives=request.frame_candidates,
                    representative_source_fingerprints=(StageFingerprint("c" * 64),),
                    annotation_requests=(request,),
                    configuration=configuration,
                    resolved_models=models,
                )
            )
        except BaseException as error:
            errors.append(error)

    first_thread = threading.Thread(target=process, name="first-vision-processor")
    second_thread = threading.Thread(
        target=process,
        args=(True,),
        name="second-vision-processor",
    )

    # Act
    first_thread.start()
    assert first_call_started.wait(timeout=5)
    second_thread.start()
    assert second_process_started.wait(timeout=5)
    try:
        second_thread.join(timeout=0.2)
        second_finished_early = not second_thread.is_alive()
        inference_count_while_locked = len(runtime.scene_catalog_calls)
    finally:
        release_first_call.set()
        first_thread.join(timeout=5)
        second_thread.join(timeout=5)

    # Assert
    assert second_finished_early is False
    assert inference_count_while_locked == 1
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []
    assert len(results) == 2
    assert len(runtime.scene_catalog_calls) == 1


@pytest.mark.parametrize("failure_position", [0, 1, 2])
def test_completed_annotations_survive_first_middle_last_failure(
    tmp_path: Path,
    failure_position: int,
) -> None:
    """先頭・途中・末尾Annotation失敗後も先行Momentが再利用されること。

    Arrange:
        - 3件のうち指定位置だけ失敗するfake VisionRuntimeが用意される
    Act:
        - 失敗runの後に同じ入力が再実行される
    Assert:
        - Scene Catalogと先行Annotationは再実行されず未完了分だけ生成されること
    """
    # Arrange
    video_set, configuration = _video_set_and_configuration(tmp_path)
    requests = _three_requests()
    catalog = _catalog()
    annotations = _three_annotations(requests)
    first_runtime = FakeStructuredVisionRuntime(
        catalog,
        annotations,
        failure_moment_id=requests[failure_position].moment.identifier,
    )
    models = FakeModelRuntime("vision-model").resolve_models(configuration)

    # Act / Assert
    with pytest.raises(RuntimeError, match="fake raw response"):
        VideoSetVisionProcessor(
            first_runtime,
            RecordingRunObserver(),
        ).process(
            video_set=video_set,
            representatives=tuple(request.frame_candidates[0] for request in requests),
            representative_source_fingerprints=(StageFingerprint("c" * 64),),
            annotation_requests=requests,
            configuration=configuration,
            resolved_models=models,
        )

    # Act
    retry_runtime = FakeStructuredVisionRuntime(catalog, annotations)
    result = VideoSetVisionProcessor(
        retry_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        representatives=tuple(request.frame_candidates[0] for request in requests),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=requests,
        configuration=configuration,
        resolved_models=models,
    )

    # Assert
    assert retry_runtime.scene_catalog_calls == []
    assert [
        item.moment.identifier for item in retry_runtime.candidate_annotation_calls
    ] == [request.moment.identifier for request in requests[failure_position:]]
    assert result.annotations == annotations
    assert [item.cache_hit for item in result.annotation_diagnostics] == [
        *([True] * failure_position),
        *([False] * (len(requests) - failure_position)),
    ]


def test_failed_scene_catalog_is_recomputed_on_rerun(tmp_path: Path) -> None:
    """失敗したScene Catalogが未完了として再生成されること。

    Arrange:
        - Scene Catalogだけが失敗するfake Vision Runtimeが用意される
    Act:
        - 失敗runの後に同じ入力が再実行される
    Assert:
        - 失敗Catalogは再生成され、その後のAnnotationがすべて確定されること
    """
    # Arrange
    video_set, configuration = _video_set_and_configuration(tmp_path)
    requests = _three_requests()
    catalog = _catalog()
    annotations = _three_annotations(requests)
    failing_runtime = FakeStructuredVisionRuntime(
        catalog,
        annotations,
        fail_scene_catalog=True,
    )
    models = FakeModelRuntime("vision-model").resolve_models(configuration)

    # Act
    with pytest.raises(RuntimeError, match="fake raw catalog response"):
        VideoSetVisionProcessor(
            failing_runtime,
            RecordingRunObserver(),
        ).process(
            video_set=video_set,
            representatives=tuple(request.frame_candidates[0] for request in requests),
            representative_source_fingerprints=(StageFingerprint("c" * 64),),
            annotation_requests=requests,
            configuration=configuration,
            resolved_models=models,
        )
    retry_runtime = FakeStructuredVisionRuntime(catalog, annotations)
    result = VideoSetVisionProcessor(
        retry_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        representatives=tuple(request.frame_candidates[0] for request in requests),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=requests,
        configuration=configuration,
        resolved_models=models,
    )

    # Assert
    assert len(failing_runtime.scene_catalog_calls) == 1
    assert failing_runtime.candidate_annotation_calls == []
    assert len(retry_runtime.scene_catalog_calls) == 1
    assert [
        item.moment.identifier for item in retry_runtime.candidate_annotation_calls
    ] == [request.moment.identifier for request in requests]
    assert result.annotations == annotations


def test_vision_stage_progress_reports_catalog_and_each_annotation(
    tmp_path: Path,
) -> None:
    """Scene Catalogと各Candidate Annotationが個別Stageとして通知されること。

    Arrange:
        - 2件のAnnotation requestとrun開始済みProgress Trackerが用意される
    Act:
        - cold Video Set Vision processingが実行される
    Assert:
        - Catalogと各Annotationが単調なStage番号とrecompute結果で通知されること
    """
    # Arrange
    video_set, configuration = _video_set_and_configuration(tmp_path)
    requests = _requests()
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer, clock=lambda: 10.0)
    progress.start_run()
    processor = VideoSetVisionProcessor(
        FakeStructuredVisionRuntime(_catalog(), _annotations(requests)),
        observer,
        progress=progress,
    )

    # Act
    processor.process(
        video_set=video_set,
        representatives=tuple(request.frame_candidates[0] for request in requests),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=requests,
        configuration=configuration,
        resolved_models=FakeModelRuntime("vision-model").resolve_models(configuration),
    )

    # Assert
    assert tuple(
        (event.stage, event.stage_index, event.work_unit_kind)
        for event in observer.progress_events
        if event.kind == "stage_started"
    ) == (
        (ProcessingStage.BUILD_SCENE_CATALOG, 1, "scene_catalog"),
        (ProcessingStage.ANNOTATE_CANDIDATE, 2, "candidate"),
        (ProcessingStage.ANNOTATE_CANDIDATE, 3, "candidate"),
    )
    assert tuple(
        (
            event.cache_miss_count,
            event.recompute_count,
            event.cache_hit_count,
            event.reuse_count,
        )
        for event in observer.progress_events
        if event.kind == "cache"
    ) == ((1, 1, 0, 0), (1, 1, 0, 0), (1, 1, 0, 0))
    assert tuple(
        event.reason_code
        for event in observer.progress_events
        if event.kind == "external_work_started"
    ) == (
        "scene_catalog_inference_started",
        "candidate_annotation_inference_started",
        "candidate_annotation_inference_started",
    )


def test_warm_vision_progress_reports_reuse_without_external_work(
    tmp_path: Path,
) -> None:
    """warm Vision Stageがcache reuseだけを通知し外部処理を開始しないこと。

    Arrange:
        - Catalogと2件のAnnotationがCompleted Stageとして確定済みである
    Act:
        - 同じ入力がProgress Tracker付きで再実行される
    Assert:
        - 各Stageがhit/reuseとして通知され外部処理eventがないこと
    """
    # Arrange
    video_set, configuration = _video_set_and_configuration(tmp_path)
    requests = _requests()
    catalog = _catalog()
    annotations = _annotations(requests)
    models = FakeModelRuntime("vision-model").resolve_models(configuration)
    VideoSetVisionProcessor(
        FakeStructuredVisionRuntime(catalog, annotations),
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        representatives=tuple(request.frame_candidates[0] for request in requests),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=requests,
        configuration=configuration,
        resolved_models=models,
    )
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer, clock=lambda: 10.0)
    progress.start_run()

    # Act
    VideoSetVisionProcessor(
        FakeStructuredVisionRuntime(
            catalog,
            annotations,
            reject_all_calls=True,
        ),
        observer,
        progress=progress,
    ).process(
        video_set=video_set,
        representatives=tuple(request.frame_candidates[0] for request in requests),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=requests,
        configuration=configuration,
        resolved_models=models,
    )

    # Assert
    cache_events = tuple(
        (
            event.cache_hit_count,
            event.cache_miss_count,
            event.reuse_count,
            event.recompute_count,
            event.reason_code,
        )
        for event in observer.progress_events
        if event.kind == "cache"
    )
    assert cache_events == (
        (1, 0, 1, 0, "cache_reused"),
        (1, 0, 1, 0, "cache_reused"),
        (1, 0, 1, 0, "cache_reused"),
    )
    assert not any(
        event.kind == "external_work_started" for event in observer.progress_events
    )


def test_vision_processor_records_annotation_duration_for_eta(
    tmp_path: Path,
) -> None:
    """Vision processorのAnnotation完了時間がETA sampleへ記録されること。

    Arrange:
        - 同じAnnotation系列を異なるVideo Setで5回再計算するprocessorが用意される
    Act:
        - 6件目のAnnotation Stageで残り1件のETAが通知される
    Assert:
        - atomic completionまでの実時間によるETAが通知されること
    """
    # Arrange
    observer = RecordingRunObserver()
    current_time = [0.0]
    progress = RunProgressTracker(observer, clock=lambda: current_time[0])
    progress.start_run()
    requests = _requests()
    request = requests[0]
    annotation = _annotations(requests)[0]

    def advance_annotation_clock() -> None:
        current_time[0] += 10.0

    for run_index in range(5):
        run_folder = tmp_path / f"run-{run_index}"
        run_folder.mkdir()
        video_set, configuration = _video_set_and_configuration(run_folder)
        VideoSetVisionProcessor(
            FakeStructuredVisionRuntime(
                _catalog(),
                (annotation,),
                on_candidate_annotation=advance_annotation_clock,
            ),
            observer,
            progress=progress,
        ).process(
            video_set=video_set,
            representatives=request.frame_candidates,
            representative_source_fingerprints=(StageFingerprint("c" * 64),),
            annotation_requests=(request,),
            configuration=configuration,
            resolved_models=FakeModelRuntime("vision-model").resolve_models(
                configuration
            ),
        )
    progress.start_stage(
        ProcessingStage.ANNOTATE_CANDIDATE,
        work_unit_kind="candidate",
    )
    current_time[0] += 30.0

    # Act
    progress.progress(
        remaining_reuse_count=0,
        remaining_recompute_count=1,
    )

    # Assert
    event = observer.progress_events[-1]
    assert (event.estimation_state, event.eta_seconds) == ("available", 10.0)


def test_verbatim_context_cue_is_rejected_before_annotation_cache(
    tmp_path: Path,
) -> None:
    """Context Cue本文を含むAnnotationがcache保存前に拒否されること。

    Arrange:
        - Context Cue本文をsummaryへ逐語再出力するfake runtimeが用意される
    Act:
        - 一つのCandidate Annotationが処理される
    Assert:
        - runtime境界で失敗しraw Context Cue本文がcacheへ保存されないこと
    """
    # Arrange
    video_set, configuration = _video_set_and_configuration(tmp_path)
    request = _requests()[1]
    raw_context_text = request.context_cues[0].text
    unsafe_annotation = CandidateAnnotation(
        candidate=request.frame_candidates[0],
        summary=raw_context_text,
        candidate_moment_id=request.moment.identifier,
        scene_slug="climax",
        blog_image_type="event",
        explanation_value="high",
        frame_choice_reason="対決する人物が明確に写る",
        screen_text_kind="dialogue",
        context_relevance="strong",
        supporting_context_cue_ids=("cue-b",),
        spoiler_risk="high",
        spoiler_evidence="主要人物の正体が画面で明示される",
    )
    runtime = FakeStructuredVisionRuntime(_catalog(), (unsafe_annotation,))
    models = FakeModelRuntime("vision-model").resolve_models(configuration)

    # Act
    # Assert
    with pytest.raises(ValueError, match="Candidate Annotation"):
        VideoSetVisionProcessor(runtime, RecordingRunObserver()).process(
            video_set=video_set,
            representatives=request.frame_candidates,
            representative_source_fingerprints=(StageFingerprint("c" * 64),),
            annotation_requests=(request,),
            configuration=configuration,
            resolved_models=models,
        )
    cache_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in configuration.processing_cache_folder.rglob("*.json")
    )
    assert raw_context_text not in cache_text


def test_foreign_frame_payload_is_rejected_before_annotation_cache(
    tmp_path: Path,
) -> None:
    """同じIDで異なるpayloadを持つFrame Candidateがcache前に拒否されること。

    Arrange:
        - request frameと同じIDに異なる画像bytesを持つAnnotationが用意される
    Act:
        - 一つのCandidate Annotationが処理される
    Assert:
        - runtime境界で失敗しforeign frameがcold resultへ返されないこと
    """
    # Arrange
    video_set, configuration = _video_set_and_configuration(tmp_path)
    request = _requests()[0]
    foreign_frame = FrameCandidate(
        request.frame_candidates[0].identifier,
        b"foreign-image",
    )
    foreign_annotation = CandidateAnnotation(
        candidate=foreign_frame,
        summary="フィールドを探索する場面",
        candidate_moment_id=request.moment.identifier,
        scene_slug="exploration",
        blog_image_type="normal_gameplay",
        explanation_value="medium",
        frame_choice_reason="探索場所が明確に写る",
        screen_text_kind="hud",
        context_relevance="unavailable",
        spoiler_risk="none",
    )
    runtime = FakeStructuredVisionRuntime(_catalog(), (foreign_annotation,))
    models = FakeModelRuntime("vision-model").resolve_models(configuration)

    # Act
    # Assert
    with pytest.raises(ValueError, match="Candidate Annotation"):
        VideoSetVisionProcessor(runtime, RecordingRunObserver()).process(
            video_set=video_set,
            representatives=request.frame_candidates,
            representative_source_fingerprints=(StageFingerprint("c" * 64),),
            annotation_requests=(request,),
            configuration=configuration,
            resolved_models=models,
        )


def _video_set_and_configuration(
    tmp_path: Path,
) -> tuple[VideoSet, EffectiveConfiguration]:
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mp4").write_bytes(b"video-content")
    return (
        discover_video_set(input_folder),
        EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=tmp_path / "output",
        ),
    )


def _requests() -> tuple[CandidateAnnotationRequest, ...]:
    first_frame = FrameCandidate("frame-a", b"image-a")
    second_frame = FrameCandidate("frame-b", b"image-b")
    return (
        CandidateAnnotationRequest(
            moment=_moment("a", first_frame.identifier, Fraction(10)),
            frame_candidates=(first_frame,),
            context_cues=(),
            video_set_progress=Fraction(1, 4),
            selection_intent="ブログ本文を説明できる画像を選ぶ",
            cue_selection_policy_version="nearby-context-v1",
        ),
        CandidateAnnotationRequest(
            moment=_moment("b", second_frame.identifier, Fraction(20)),
            frame_candidates=(second_frame,),
            context_cues=(
                ContextCue(
                    identifier="cue-b",
                    start=Fraction(19),
                    end=Fraction(21),
                    text="正体を明かす秘密の台詞",
                ),
            ),
            video_set_progress=Fraction(3, 4),
            selection_intent="ブログ本文を説明できる画像を選ぶ",
            cue_selection_policy_version="nearby-context-v1",
        ),
    )


def _three_requests() -> tuple[CandidateAnnotationRequest, ...]:
    first, second = _requests()
    third_frame = FrameCandidate("frame-c", b"image-c")
    third = CandidateAnnotationRequest(
        moment=_moment("c", third_frame.identifier, Fraction(30)),
        frame_candidates=(third_frame,),
        context_cues=(),
        video_set_progress=Fraction(7, 8),
        selection_intent="ブログ本文を説明できる画像を選ぶ",
        cue_selection_policy_version="nearby-context-v1",
    )
    return (first, second, third)


def _moment(seed: str, frame_id: str, time: Fraction) -> CandidateMoment:
    return CandidateMoment(
        identifier="mom_" + seed * 64,
        source_pts=int(time),
        anchor_time=time,
        timeline_segment_id="seg_" + seed * 64,
        evidence=("scene",),
        proxy_quality_score=0.9,
        frame_candidate_ids=(frame_id,),
    )


def _catalog() -> SceneCatalog:
    return SceneCatalog(
        (
            SceneCatalogEntry("exploration", "探索", "フィールド探索", "ordinary"),
            SceneCatalogEntry("climax", "終盤", "重要な対決", "cinematic"),
            SceneCatalogEntry("other", "その他", "分類不能", "ordinary"),
        )
    )


def _annotations(
    requests: tuple[CandidateAnnotationRequest, ...],
) -> tuple[CandidateAnnotation, ...]:
    return (
        CandidateAnnotation(
            candidate=requests[0].frame_candidates[0],
            summary="フィールドを探索する場面",
            candidate_moment_id=requests[0].moment.identifier,
            scene_slug="exploration",
            blog_image_type="normal_gameplay",
            explanation_value="medium",
            frame_choice_reason="探索場所が明確に写る",
            screen_text_kind="hud",
            context_relevance="unavailable",
            spoiler_risk="none",
        ),
        CandidateAnnotation(
            candidate=requests[1].frame_candidates[0],
            summary="終盤の重要な対決",
            candidate_moment_id=requests[1].moment.identifier,
            scene_slug="climax",
            blog_image_type="event",
            explanation_value="high",
            frame_choice_reason="対決する人物が明確に写る",
            screen_text_kind="dialogue",
            context_relevance="strong",
            supporting_context_cue_ids=("cue-b",),
            spoiler_risk="high",
            spoiler_evidence="主要人物の正体が画面で明示される",
        ),
    )


def _three_annotations(
    requests: tuple[CandidateAnnotationRequest, ...],
) -> tuple[CandidateAnnotation, ...]:
    first, second = _annotations(requests[:2])
    third = CandidateAnnotation(
        candidate=requests[2].frame_candidates[0],
        summary="終盤の探索場面",
        candidate_moment_id=requests[2].moment.identifier,
        scene_slug="exploration",
        blog_image_type="normal_gameplay",
        explanation_value="medium",
        frame_choice_reason="探索対象が明確に写る",
        screen_text_kind="hud",
        context_relevance="unavailable",
        spoiler_risk="none",
    )
    return (first, second, third)
