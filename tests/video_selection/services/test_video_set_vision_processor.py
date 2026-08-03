import threading
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from src.video_selection.models.candidate_annotation import (
    CandidateAnnotation,
    ExplanationValue,
)
from src.video_selection.models.candidate_annotation_request import (
    CandidateAnnotationRequest,
)
from src.video_selection.models.candidate_moment import CandidateMoment
from src.video_selection.models.combat_encounter_basis import CombatEncounterBasis
from src.video_selection.models.combat_encounter_kind import CombatEncounterKind
from src.video_selection.models.context_cue import ContextCue
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.frame_candidate import FrameCandidate
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.representative_frame_evidence import (
    RepresentativeFrameEvidence,
)
from src.video_selection.models.scene_catalog import SceneCatalog
from src.video_selection.models.scene_catalog_entry import SceneCatalogEntry
from src.video_selection.models.stage_fingerprint import StageFingerprint
from src.video_selection.models.video_set import VideoSet
from src.video_selection.models.vision_stage_result import VisionStageResult
from src.video_selection.services.discover_video_set import discover_video_set
from src.video_selection.services.run_progress_tracker import RunProgressTracker
from src.video_selection.services.video_set_vision_processor import (
    VideoSetVisionProcessor,
    plan_vision_stage_fingerprints,
)
from tests.video_selection.fakes.fake_model_runtime import FakeModelRuntime
from tests.video_selection.fakes.fake_structured_vision_runtime import (
    FakeStructuredVisionRuntime,
)
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver


def test_combat_primary_without_explanation_uses_best_same_moment_fallback(
    tmp_path: Path,
) -> None:
    """不適格な戦闘Primaryだけが独立評価された同一Momentの最良frameへ置換されること。

    Arrange:
        - 説明価値なしの戦闘Primaryと説明価値が異なる二つの代替frameが用意される
    Act:
        - 一つのCandidate MomentがVision processorで処理される
    Assert:
        - 三つのframeが一枚ずつ独立requestで評価されること
        - 最も説明価値が高い代替frameがRepresentative Frameになること
        - 各frameの診断とCompleted Stageが保持されること
    """
    # Arrange
    video_set, base_configuration = _video_set_and_configuration(tmp_path)
    configuration = EffectiveConfiguration(
        video_input_folder=base_configuration.video_input_folder,
        output_folder=base_configuration.output_folder,
        ollama_max_parallel_requests=2,
    )
    primary = FrameCandidate("frame-primary", b"primary")
    first_fallback = FrameCandidate("frame-fallback-low", b"fallback-low")
    best_fallback = FrameCandidate("frame-fallback-high", b"fallback-high")
    moment = CandidateMoment(
        identifier="mom_" + "d" * 64,
        source_pts=10,
        anchor_time=Fraction(10),
        timeline_segment_id="seg_" + "d" * 64,
        evidence=("scene",),
        proxy_quality_score=0.9,
        frame_candidate_ids=(
            primary.identifier,
            first_fallback.identifier,
            best_fallback.identifier,
        ),
    )
    request = CandidateAnnotationRequest(
        moment=moment,
        frame_candidates=(primary, first_fallback, best_fallback),
        context_cues=(),
        video_set_progress=Fraction(1, 2),
        selection_intent="ブログ本文を説明できる画像を選ぶ",
        cue_selection_policy_version="nearby-context-v1",
    )
    annotations = (
        CandidateAnnotation(
            candidate=primary,
            summary="攻撃effectで敵が見えない戦闘",
            candidate_moment_id=moment.identifier,
            scene_slug="exploration",
            blog_image_type="normal_gameplay",
            explanation_value="none",
            combat_encounter_kind="ordinary",
            combat_encounter_basis="ordinary_opponent_presentation",
        ),
        CandidateAnnotation(
            candidate=first_fallback,
            summary="敵が見える通常戦闘",
            candidate_moment_id=moment.identifier,
            scene_slug="exploration",
            blog_image_type="normal_gameplay",
            explanation_value="low",
            combat_encounter_kind="ordinary",
            combat_encounter_basis="ordinary_opponent_presentation",
        ),
        CandidateAnnotation(
            candidate=best_fallback,
            summary="敵と主人公が明瞭な通常戦闘",
            candidate_moment_id=moment.identifier,
            scene_slug="exploration",
            blog_image_type="normal_gameplay",
            explanation_value="high",
            combat_encounter_kind="ordinary",
            combat_encounter_basis="ordinary_opponent_presentation",
        ),
    )
    runtime = FakeStructuredVisionRuntime(_catalog(), annotations)

    # Act
    result = VideoSetVisionProcessor(runtime, RecordingRunObserver()).process(
        video_set=video_set,
        representatives=(primary,),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=(request,),
        configuration=configuration,
        resolved_models=FakeModelRuntime("vision-model").resolve_models(configuration),
    )

    # Assert
    evaluated_frame_ids = tuple(
        call.frame_candidates[0].identifier
        for call in runtime.candidate_annotation_calls
    )
    assert evaluated_frame_ids[0] == primary.identifier
    assert set(evaluated_frame_ids[1:]) == {
        first_fallback.identifier,
        best_fallback.identifier,
    }
    assert all(
        len(call.frame_candidates) == 1 for call in runtime.candidate_annotation_calls
    )
    assert result.annotations == (annotations[2],)
    assert len(result.annotation_diagnostics) == 3
    assert len(result.completed_stages) == 4


def test_combat_fallback_frames_are_evaluated_concurrently(
    tmp_path: Path,
) -> None:
    """同一Momentのfallback frameが独立性を保って設定上限内で並列評価されること。

    Arrange:
        - 説明価値なしの戦闘Primaryと、互いの開始を待つ二つの代替frameが用意される
        - Ollama同時request上限2が設定される
    Act:
        - 一つのCandidate MomentがVision processorで処理される
    Assert:
        - 二つの代替frameが別requestとして同時実行されること
        - 完了順に依存せず最良frameがRepresentative Frameになること
    """
    # Arrange
    video_set, base_configuration = _video_set_and_configuration(tmp_path)
    configuration = EffectiveConfiguration(
        video_input_folder=base_configuration.video_input_folder,
        output_folder=base_configuration.output_folder,
        ollama_max_parallel_requests=2,
    )
    request, annotations = _combat_fallback_fixture()
    primary_frame_id = request.frame_candidates[0].identifier
    state_lock = threading.Lock()
    both_fallbacks_started = threading.Event()
    active_fallback_count = 0
    maximum_active_fallback_count = 0

    def synchronize_fallbacks(call: CandidateAnnotationRequest) -> None:
        nonlocal active_fallback_count, maximum_active_fallback_count
        if call.frame_candidates[0].identifier == primary_frame_id:
            return
        with state_lock:
            active_fallback_count += 1
            maximum_active_fallback_count = max(
                maximum_active_fallback_count,
                active_fallback_count,
            )
            if active_fallback_count == 2:
                both_fallbacks_started.set()
        if not both_fallbacks_started.wait(timeout=1.0):
            raise RuntimeError("二つのfallback requestが並列に開始されませんでした")
        with state_lock:
            active_fallback_count -= 1

    runtime = FakeStructuredVisionRuntime(
        _catalog(),
        annotations,
        on_candidate_annotation_request=synchronize_fallbacks,
    )

    # Act
    result = VideoSetVisionProcessor(runtime, RecordingRunObserver()).process(
        video_set=video_set,
        representatives=(request.frame_candidates[0],),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=(request,),
        configuration=configuration,
        resolved_models=FakeModelRuntime("vision-model").resolve_models(configuration),
    )

    # Assert
    assert maximum_active_fallback_count == 2
    assert all(
        len(call.frame_candidates) == 1 for call in runtime.candidate_annotation_calls
    )
    assert result.annotations == (annotations[2],)


def test_primary_candidate_moments_are_evaluated_concurrently_in_input_order(
    tmp_path: Path,
) -> None:
    """主候補が設定上限まで並列評価され結果は入力順に固定されること。

    Arrange:
        - 完了順が逆転する3件の主候補とOllama同時request上限2が用意される
    Act:
        - Video Set Vision processingが実行される
    Assert:
        - 最大2件だけが同時実行され、結果は入力順で返されること
    """
    # Arrange
    video_set, base_configuration = _video_set_and_configuration(tmp_path)
    configuration = EffectiveConfiguration(
        video_input_folder=base_configuration.video_input_folder,
        output_folder=base_configuration.output_folder,
        ollama_max_parallel_requests=2,
    )
    requests = _three_requests()
    annotations = _three_annotations(requests)
    first_moment_id = requests[0].moment.identifier
    second_moment_id = requests[1].moment.identifier
    state_lock = threading.Lock()
    first_started = threading.Event()
    second_completed_first = threading.Event()
    active_count = 0
    maximum_active_count = 0

    def reverse_first_two_completion(call: CandidateAnnotationRequest) -> None:
        nonlocal active_count, maximum_active_count
        with state_lock:
            active_count += 1
            maximum_active_count = max(maximum_active_count, active_count)
        try:
            if call.moment.identifier == first_moment_id:
                first_started.set()
                if not second_completed_first.wait(timeout=1.0):
                    raise RuntimeError("二つ目の主候補が並列に完了しませんでした")
            elif call.moment.identifier == second_moment_id:
                if not first_started.wait(timeout=1.0):
                    raise RuntimeError("一つ目の主候補が開始されませんでした")
                second_completed_first.set()
        finally:
            with state_lock:
                active_count -= 1

    runtime = FakeStructuredVisionRuntime(
        _catalog(),
        annotations,
        on_candidate_annotation_request=reverse_first_two_completion,
    )

    # Act
    result = VideoSetVisionProcessor(runtime, RecordingRunObserver()).process(
        video_set=video_set,
        representatives=tuple(request.frame_candidates[0] for request in requests),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=requests,
        configuration=configuration,
        resolved_models=FakeModelRuntime("vision-model").resolve_models(configuration),
    )

    # Assert
    assert maximum_active_count == 2
    assert result.annotations == annotations
    assert all(
        len(call.frame_candidates) == 1 for call in runtime.candidate_annotation_calls
    )


def test_candidate_worker_slot_is_refilled_before_slow_sibling_finishes(
    tmp_path: Path,
) -> None:
    """完了したworkerへ次のCandidate Momentが直ちに補充されること。

    Arrange:
        - 上限2、長時間停止する先頭Moment、すぐ完了する2番目が用意される
    Act:
        - 3件のCandidate Annotationが実行される
    Assert:
        - 先頭Momentの解放前に3番目のMomentが開始されること
    """
    # Arrange
    video_set, base_configuration = _video_set_and_configuration(tmp_path)
    configuration = EffectiveConfiguration(
        video_input_folder=base_configuration.video_input_folder,
        output_folder=base_configuration.output_folder,
        ollama_max_parallel_requests=2,
    )
    requests = _three_requests()
    annotations = _three_annotations(requests)
    first_started = threading.Event()
    third_started = threading.Event()
    release_first = threading.Event()

    def block_first_moment(call: CandidateAnnotationRequest) -> None:
        if call.moment.identifier == requests[0].moment.identifier:
            first_started.set()
            if not release_first.wait(timeout=5.0):
                raise RuntimeError("先頭Momentを解放できませんでした")
        elif call.moment.identifier == requests[2].moment.identifier:
            third_started.set()

    runtime = FakeStructuredVisionRuntime(
        _catalog(),
        annotations,
        on_candidate_annotation_request=block_first_moment,
    )
    errors: list[BaseException] = []
    results: list[VisionStageResult] = []

    def process() -> None:
        try:
            results.append(
                VideoSetVisionProcessor(runtime, RecordingRunObserver()).process(
                    video_set=video_set,
                    representatives=tuple(
                        request.frame_candidates[0] for request in requests
                    ),
                    representative_source_fingerprints=(StageFingerprint("c" * 64),),
                    annotation_requests=requests,
                    configuration=configuration,
                    resolved_models=FakeModelRuntime("vision-model").resolve_models(
                        configuration
                    ),
                )
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=process, name="candidate-worker-refill")

    # Act
    thread.start()
    try:
        assert first_started.wait(timeout=5.0)
        third_started_before_release = third_started.wait(timeout=1.0)
    finally:
        release_first.set()
        thread.join(timeout=5.0)

    # Assert
    assert third_started_before_release is True
    assert errors == []
    assert not thread.is_alive()
    assert len(results) == 1
    assert results[0].annotations == annotations


def test_primary_and_fallback_annotations_share_the_parallel_limit(
    tmp_path: Path,
) -> None:
    """複数Momentの主候補とfallbackで同じ同時実行上限が共有されること。

    Arrange:
        - fallbackが必要な2件のMomentとOllama同時request上限2が用意される
    Act:
        - 両Momentが並列にCandidate Annotationされる
    Assert:
        - 主候補とfallbackのどちらも3件目が同時実行されないこと
    """
    # Arrange
    video_set, base_configuration = _video_set_and_configuration(tmp_path)
    configuration = EffectiveConfiguration(
        video_input_folder=base_configuration.video_input_folder,
        output_folder=base_configuration.output_folder,
        ollama_max_parallel_requests=2,
    )
    first_request, first_annotations = _combat_fallback_fixture()
    second_request, second_annotations = _combat_fallback_fixture(
        seed="e",
        anchor_time=Fraction(20),
    )
    requests = (first_request, second_request)
    annotations = (*first_annotations, *second_annotations)
    primary_ids = {
        first_request.frame_candidates[0].identifier,
        second_request.frame_candidates[0].identifier,
    }
    state_lock = threading.Lock()
    two_primaries_started = threading.Event()
    release_primaries = threading.Event()
    two_fallbacks_started = threading.Event()
    third_fallback_started = threading.Event()
    release_fallbacks = threading.Event()
    active_primary_count = 0
    active_fallback_count = 0

    def observe_parallel_limit(call: CandidateAnnotationRequest) -> None:
        nonlocal active_primary_count, active_fallback_count
        frame_id = call.frame_candidates[0].identifier
        is_primary = frame_id in primary_ids
        with state_lock:
            if is_primary:
                active_primary_count += 1
                if active_primary_count == 2:
                    two_primaries_started.set()
            else:
                active_fallback_count += 1
                if active_fallback_count == 2:
                    two_fallbacks_started.set()
                elif active_fallback_count >= 3:
                    third_fallback_started.set()
        try:
            release = release_primaries if is_primary else release_fallbacks
            if not release.wait(timeout=5.0):
                raise RuntimeError("Candidate Annotationを解放できませんでした")
        finally:
            with state_lock:
                if is_primary:
                    active_primary_count -= 1
                else:
                    active_fallback_count -= 1

    runtime = FakeStructuredVisionRuntime(
        _catalog(),
        annotations,
        on_candidate_annotation_request=observe_parallel_limit,
    )
    errors: list[BaseException] = []
    results: list[object] = []

    def process() -> None:
        try:
            results.append(
                VideoSetVisionProcessor(runtime, RecordingRunObserver()).process(
                    video_set=video_set,
                    representatives=tuple(
                        request.frame_candidates[0] for request in requests
                    ),
                    representative_source_fingerprints=(StageFingerprint("c" * 64),),
                    annotation_requests=requests,
                    configuration=configuration,
                    resolved_models=FakeModelRuntime("vision-model").resolve_models(
                        configuration
                    ),
                )
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=process, name="parallel-candidate-moments")

    # Act
    thread.start()
    try:
        assert two_primaries_started.wait(timeout=5.0)
        release_primaries.set()
        assert two_fallbacks_started.wait(timeout=5.0)
        third_started_before_release = third_fallback_started.wait(timeout=0.2)
    finally:
        release_primaries.set()
        release_fallbacks.set()
        thread.join(timeout=5.0)

    # Assert
    assert third_started_before_release is False
    assert errors == []
    assert not thread.is_alive()
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, VisionStageResult)
    assert result.annotations == (first_annotations[2], second_annotations[2])


def test_interrupt_cancels_active_candidate_annotations(tmp_path: Path) -> None:
    """Candidate Annotation待機中の割り込みでactive推論が中止されること。

    Arrange:
        - 一方がKeyboardInterruptとなり他方が中止要求を待つ2件のMomentが用意される
    Act:
        - Candidate Annotationが並列実行される
    Assert:
        - KeyboardInterruptが維持され、runtimeへ中止が一度要求されること
    """
    # Arrange
    video_set, base_configuration = _video_set_and_configuration(tmp_path)
    configuration = EffectiveConfiguration(
        video_input_folder=base_configuration.video_input_folder,
        output_folder=base_configuration.output_folder,
        ollama_max_parallel_requests=2,
    )
    requests = _requests()
    annotations = _annotations(requests)
    both_started = threading.Event()
    cancellation_requested = threading.Event()
    state_lock = threading.Lock()
    started_count = 0

    def interrupt_first_moment(call: CandidateAnnotationRequest) -> None:
        nonlocal started_count
        with state_lock:
            started_count += 1
            if started_count == 2:
                both_started.set()
        if not both_started.wait(timeout=1.0):
            raise RuntimeError("二つのCandidate Momentが開始されませんでした")
        if call.moment.identifier == requests[0].moment.identifier:
            raise KeyboardInterrupt
        if not cancellation_requested.wait(timeout=1.0):
            raise RuntimeError("active Candidate Annotationが中止されませんでした")

    runtime = FakeStructuredVisionRuntime(
        _catalog(),
        annotations,
        on_candidate_annotation_request=interrupt_first_moment,
        on_cancel_candidate_annotations=cancellation_requested.set,
    )

    # Act
    # Assert
    with pytest.raises(KeyboardInterrupt):
        VideoSetVisionProcessor(runtime, RecordingRunObserver()).process(
            video_set=video_set,
            representatives=tuple(request.frame_candidates[0] for request in requests),
            representative_source_fingerprints=(StageFingerprint("c" * 64),),
            annotation_requests=requests,
            configuration=configuration,
            resolved_models=FakeModelRuntime("vision-model").resolve_models(
                configuration
            ),
        )
    assert runtime.cancel_candidate_annotations_call_count == 1
    assert cancellation_requested.is_set()


def test_parallel_failure_reuses_successful_sibling_without_blocking_retry_batch(
    tmp_path: Path,
) -> None:
    """成功済み兄弟Momentが再利用され未完了Momentは並列再開されること。

    Arrange:
        - 先頭と3件目が失敗し2件目だけが成功する3件の主候補が用意される
    Act:
        - 失敗runの後に同じ入力が再実行される
    Assert:
        - 成功済み2件目を飛ばし、先頭と未開始3件目が並列推論されること
    """
    # Arrange
    video_set, base_configuration = _video_set_and_configuration(tmp_path)
    configuration = EffectiveConfiguration(
        video_input_folder=base_configuration.video_input_folder,
        output_folder=base_configuration.output_folder,
        ollama_max_parallel_requests=2,
    )
    requests = _three_requests()
    annotations = _three_annotations(requests)
    first_two_started = threading.Event()
    state_lock = threading.Lock()
    first_two_call_count = 0

    def synchronize_first_pair(call: CandidateAnnotationRequest) -> None:
        nonlocal first_two_call_count
        if call.moment.identifier not in {
            requests[0].moment.identifier,
            requests[1].moment.identifier,
        }:
            return
        with state_lock:
            first_two_call_count += 1
            if first_two_call_count == 2:
                first_two_started.set()
        if not first_two_started.wait(timeout=1.0):
            raise RuntimeError("先頭2件が並列に開始されませんでした")

    failing_runtime = FakeStructuredVisionRuntime(
        _catalog(),
        annotations,
        failure_moment_ids=frozenset(
            (
                requests[0].moment.identifier,
                requests[2].moment.identifier,
            )
        ),
        on_candidate_annotation_request=synchronize_first_pair,
    )
    models = FakeModelRuntime("vision-model").resolve_models(configuration)
    retry_pair_started = threading.Event()
    retry_state_lock = threading.Lock()
    retry_call_count = 0

    def synchronize_retry_misses(call: CandidateAnnotationRequest) -> None:
        nonlocal retry_call_count
        if call.moment.identifier not in {
            requests[0].moment.identifier,
            requests[2].moment.identifier,
        }:
            return
        with retry_state_lock:
            retry_call_count += 1
            if retry_call_count == 2:
                retry_pair_started.set()
        if not retry_pair_started.wait(timeout=1.0):
            raise RuntimeError("cache hitを越えた未完了Momentが並列化されませんでした")

    # Act
    with pytest.raises(RuntimeError, match="fake raw response"):
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
    retry_runtime = FakeStructuredVisionRuntime(
        _catalog(),
        annotations,
        on_candidate_annotation_request=synchronize_retry_misses,
    )
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
    assert {
        call.moment.identifier for call in failing_runtime.candidate_annotation_calls
    } == {
        requests[0].moment.identifier,
        requests[1].moment.identifier,
        requests[2].moment.identifier,
    }
    assert {
        call.moment.identifier for call in retry_runtime.candidate_annotation_calls
    } == {
        requests[0].moment.identifier,
        requests[2].moment.identifier,
    }
    assert retry_pair_started.is_set()
    assert result.annotations == annotations
    assert tuple(item.cache_hit for item in result.annotation_diagnostics) == (
        False,
        True,
        False,
    )


def test_parallel_limit_does_not_change_vision_identity_or_result(
    tmp_path: Path,
) -> None:
    """worker数1と2でVision Stage identityと意味結果が同一になること。

    Arrange:
        - 同じVideo Setと候補に独立した直列cacheと並列cacheが用意される
    Act:
        - worker数1と2でcold Vision processingが実行される
    Assert:
        - 計画fingerprint、Annotation順、Completed Stageが一致すること
    """
    # Arrange
    video_set, base_configuration = _video_set_and_configuration(tmp_path)
    parallel_input = tmp_path / "parallel-input"
    parallel_input.mkdir()
    sequential_configuration = EffectiveConfiguration(
        video_input_folder=base_configuration.video_input_folder,
        output_folder=tmp_path / "sequential-output",
        ollama_max_parallel_requests=1,
    )
    parallel_configuration = EffectiveConfiguration(
        video_input_folder=parallel_input,
        output_folder=tmp_path / "parallel-output",
        ollama_max_parallel_requests=2,
    )
    requests = _three_requests()
    annotations = _three_annotations(requests)
    representatives = tuple(request.frame_candidates[0] for request in requests)
    source_fingerprints = (StageFingerprint("c" * 64),)
    sequential_models = FakeModelRuntime("vision-model").resolve_models(
        sequential_configuration
    )
    parallel_models = FakeModelRuntime("vision-model").resolve_models(
        parallel_configuration
    )

    # Act
    sequential_fingerprints = plan_vision_stage_fingerprints(
        video_set=video_set,
        representatives=representatives,
        representative_source_fingerprints=source_fingerprints,
        annotation_requests=requests,
        configuration=sequential_configuration,
        resolved_models=sequential_models,
    )
    parallel_fingerprints = plan_vision_stage_fingerprints(
        video_set=video_set,
        representatives=representatives,
        representative_source_fingerprints=source_fingerprints,
        annotation_requests=requests,
        configuration=parallel_configuration,
        resolved_models=parallel_models,
    )
    sequential_result = VideoSetVisionProcessor(
        FakeStructuredVisionRuntime(_catalog(), annotations),
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        representatives=representatives,
        representative_source_fingerprints=source_fingerprints,
        annotation_requests=requests,
        configuration=sequential_configuration,
        resolved_models=sequential_models,
    )
    parallel_result = VideoSetVisionProcessor(
        FakeStructuredVisionRuntime(_catalog(), annotations),
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        representatives=representatives,
        representative_source_fingerprints=source_fingerprints,
        annotation_requests=requests,
        configuration=parallel_configuration,
        resolved_models=parallel_models,
    )

    # Assert
    assert sequential_fingerprints == parallel_fingerprints
    assert sequential_result.annotations == parallel_result.annotations == annotations
    assert sequential_result.completed_stages == parallel_result.completed_stages
    assert tuple(
        item.request_fingerprint for item in sequential_result.annotation_diagnostics
    ) == tuple(
        item.request_fingerprint for item in parallel_result.annotation_diagnostics
    )


def test_failed_combat_fallback_resumes_only_unfinished_frame(
    tmp_path: Path,
) -> None:
    """一部失敗したfallbackが成功済みframeを保持して未完了分だけ再開されること。

    Arrange:
        - 二つの代替frameのうち後者だけ推論に失敗する初回runtimeが用意される
    Act:
        - 失敗runの後に同じ入力が正常なruntimeで再実行される
    Assert:
        - 初回runが部分成功を採用せず失敗すること
        - 再開時は失敗したframeだけが推論されること
        - 全frameがそろってから最良のRepresentative Frameが返されること
    """
    # Arrange
    video_set, base_configuration = _video_set_and_configuration(tmp_path)
    configuration = EffectiveConfiguration(
        video_input_folder=base_configuration.video_input_folder,
        output_folder=base_configuration.output_folder,
        ollama_max_parallel_requests=2,
    )
    request, annotations = _combat_fallback_fixture()
    failed_frame_id = request.frame_candidates[2].identifier
    models = FakeModelRuntime("vision-model").resolve_models(configuration)
    failing_runtime = FakeStructuredVisionRuntime(
        _catalog(),
        annotations,
        failure_frame_id=failed_frame_id,
    )
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer, clock=lambda: 10.0)
    progress.start_run()

    # Act
    with pytest.raises(RuntimeError, match="fake raw response"):
        VideoSetVisionProcessor(
            failing_runtime,
            observer,
            progress=progress,
        ).process(
            video_set=video_set,
            representatives=(request.frame_candidates[0],),
            representative_source_fingerprints=(StageFingerprint("c" * 64),),
            annotation_requests=(request,),
            configuration=configuration,
            resolved_models=models,
        )
    retry_runtime = FakeStructuredVisionRuntime(_catalog(), annotations)
    result = VideoSetVisionProcessor(
        retry_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        representatives=(request.frame_candidates[0],),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=(request,),
        configuration=configuration,
        resolved_models=models,
    )

    # Assert
    assert tuple(
        call.frame_candidates[0].identifier
        for call in retry_runtime.candidate_annotation_calls
    ) == (failed_frame_id,)
    assert result.annotations == (annotations[2],)
    assert tuple(
        diagnostics.cache_hit for diagnostics in result.annotation_diagnostics
    ) == (True, True, False)
    assert len(progress.completed_stage_events) == 3
    assert len(observer.completed_stages) == 3


@pytest.mark.parametrize(
    ("explanation_value", "combat_encounter_kind", "combat_encounter_basis"),
    (
        ("medium", "ordinary", "ordinary_opponent_presentation"),
        ("none", "not_combat", "none"),
    ),
)
def test_fallback_is_skipped_without_failed_combat_primary(
    tmp_path: Path,
    explanation_value: ExplanationValue,
    combat_encounter_kind: CombatEncounterKind,
    combat_encounter_basis: CombatEncounterBasis,
) -> None:
    """説明価値のある戦闘または非戦闘Primaryではfallbackされないこと。

    Arrange:
        - 代替frameを持つがfallback開始条件を満たさないPrimaryが用意される
    Act:
        - 一つのCandidate MomentがVision processorで処理される
    Assert:
        - Primaryだけが評価され、そのままRepresentative Frameになること
    """
    # Arrange
    video_set, configuration = _video_set_and_configuration(tmp_path)
    request, annotations = _combat_fallback_fixture()
    primary = replace(
        annotations[0],
        explanation_value=explanation_value,
        combat_encounter_kind=combat_encounter_kind,
        combat_encounter_basis=combat_encounter_basis,
    )
    runtime = FakeStructuredVisionRuntime(
        _catalog(),
        (primary, *annotations[1:]),
    )

    # Act
    result = VideoSetVisionProcessor(runtime, RecordingRunObserver()).process(
        video_set=video_set,
        representatives=(request.frame_candidates[0],),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=(request,),
        configuration=configuration,
        resolved_models=FakeModelRuntime("vision-model").resolve_models(configuration),
    )

    # Assert
    assert tuple(
        call.frame_candidates[0].identifier
        for call in runtime.candidate_annotation_calls
    ) == (request.frame_candidates[0].identifier,)
    assert result.annotations == (primary,)
    assert len(result.annotation_diagnostics) == 1


def test_expanded_combat_request_reuses_existing_primary_annotation_cache(
    tmp_path: Path,
) -> None:
    """従来の一枚Primary cacheがfallback導入後も再利用されること。

    Arrange:
        - Primaryだけを評価したCompleted Stageと同一Momentの代替frameが用意される
    Act:
        - 同じPrimaryを先頭にした拡張requestが処理される
    Assert:
        - Primaryはcache hitになり、新しい代替frameだけが推論されること
    """
    # Arrange
    video_set, configuration = _video_set_and_configuration(tmp_path)
    request, annotations = _combat_fallback_fixture()
    primary_frame = request.frame_candidates[0]
    primary_request = replace(
        request,
        moment=replace(
            request.moment,
            frame_candidate_ids=(primary_frame.identifier,),
        ),
        frame_candidates=(primary_frame,),
    )
    models = FakeModelRuntime("vision-model").resolve_models(configuration)
    VideoSetVisionProcessor(
        FakeStructuredVisionRuntime(_catalog(), (annotations[0],)),
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        representatives=(primary_frame,),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=(primary_request,),
        configuration=configuration,
        resolved_models=models,
    )
    expanded_runtime = FakeStructuredVisionRuntime(_catalog(), annotations)

    # Act
    result = VideoSetVisionProcessor(
        expanded_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        representatives=(primary_frame,),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=(request,),
        configuration=configuration,
        resolved_models=models,
    )

    # Assert
    assert tuple(
        call.frame_candidates[0].identifier
        for call in expanded_runtime.candidate_annotation_calls
    ) == tuple(frame.identifier for frame in request.frame_candidates[1:])
    assert tuple(
        diagnostics.cache_hit for diagnostics in result.annotation_diagnostics
    ) == (True, False, False)
    assert result.annotations == (annotations[2],)


def test_combat_fallback_prefers_visible_unobstructed_subjects_before_frame_id(
    tmp_path: Path,
) -> None:
    """同じ説明価値では敵と主対象が明瞭で遮蔽の少ないframeが優先されること。

    Arrange:
        - 同じExplanation Valueだが視認性と遮蔽が異なる二つの代替frameが用意される
        - Frame ID順では視認性の低いframeが先になる
    Act:
        - 戦闘PrimaryからCombat Representative Fallbackが実行される
    Assert:
        - Frame IDではなく構造化された視認性と遮蔽でRepresentativeが選ばれること
    """
    # Arrange
    video_set, configuration = _video_set_and_configuration(tmp_path)
    request, original_annotations = _combat_fallback_fixture()
    clear = replace(
        original_annotations[1],
        explanation_value="high",
        representative_frame_evidence=RepresentativeFrameEvidence(
            content_kind="gameplay_action",
            primary_subject_visibility="clear",
            opponent_body_visibility="clear",
            transient_obstruction="none",
        ),
    )
    obstructed = replace(
        original_annotations[2],
        explanation_value="high",
        representative_frame_evidence=RepresentativeFrameEvidence(
            content_kind="gameplay_action",
            primary_subject_visibility="partial",
            opponent_body_visibility="clear",
            transient_obstruction="partial",
        ),
    )
    annotations = (original_annotations[0], clear, obstructed)

    # Act
    result = VideoSetVisionProcessor(
        FakeStructuredVisionRuntime(_catalog(), annotations),
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        representatives=(request.frame_candidates[0],),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=(request,),
        configuration=configuration,
        resolved_models=FakeModelRuntime("vision-model").resolve_models(configuration),
    )
    warm_result = VideoSetVisionProcessor(
        FakeStructuredVisionRuntime(
            _catalog(),
            annotations,
            reject_all_calls=True,
        ),
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        representatives=(request.frame_candidates[0],),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=(request,),
        configuration=configuration,
        resolved_models=FakeModelRuntime("vision-model").resolve_models(configuration),
    )

    # Assert
    assert result.annotations == (clear,)
    assert warm_result.annotations[0].candidate == clear.candidate
    assert (
        warm_result.annotations[0].representative_frame_evidence
        == clear.representative_frame_evidence
    )


def test_combat_fallback_keeps_primary_when_every_frame_has_no_explanation(
    tmp_path: Path,
) -> None:
    """全frameが説明価値なしの場合に代替frameが強制採用されないこと。

    Arrange:
        - Primaryと二つの代替frameがすべてExplanation Valueなしで用意される
    Act:
        - Combat Representative Fallbackが実行される
    Assert:
        - Primary Representative Frameが維持されること
    """
    # Arrange
    video_set, configuration = _video_set_and_configuration(tmp_path)
    request, original_annotations = _combat_fallback_fixture()
    annotations = (
        original_annotations[0],
        replace(original_annotations[1], explanation_value="none"),
        replace(original_annotations[2], explanation_value="none"),
    )

    # Act
    result = VideoSetVisionProcessor(
        FakeStructuredVisionRuntime(_catalog(), annotations),
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        representatives=(request.frame_candidates[0],),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=(request,),
        configuration=configuration,
        resolved_models=FakeModelRuntime("vision-model").resolve_models(configuration),
    )

    # Assert
    assert result.annotations == (annotations[0],)


def test_combat_fallback_does_not_select_noncombat_annotation(
    tmp_path: Path,
) -> None:
    """説明価値が高くても非戦闘frameが戦闘Representativeにされないこと。

    Arrange:
        - 説明価値なしの戦闘Primary、低い説明価値の戦闘frame、
          高い説明価値の非戦闘frameが用意される
    Act:
        - Combat Representative Fallbackが実行される
    Assert:
        - 戦闘を示す代替frameだけからRepresentativeが選択されること
    """
    # Arrange
    video_set, configuration = _video_set_and_configuration(tmp_path)
    request, original_annotations = _combat_fallback_fixture()
    combat_fallback = original_annotations[1]
    noncombat_fallback = replace(
        original_annotations[2],
        blog_image_type="event",
        explanation_value="high",
        combat_encounter_kind="not_combat",
        combat_encounter_basis="none",
    )
    annotations = (
        original_annotations[0],
        combat_fallback,
        noncombat_fallback,
    )

    # Act
    result = VideoSetVisionProcessor(
        FakeStructuredVisionRuntime(_catalog(), annotations),
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        representatives=(request.frame_candidates[0],),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=(request,),
        configuration=configuration,
        resolved_models=FakeModelRuntime("vision-model").resolve_models(configuration),
    )

    # Assert
    assert result.annotations == (combat_fallback,)


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
        - Scene Kindと全生成条件・監査versionがfingerprint入力へ保存されること
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
    assert warm_result.annotations[0].combat_action is True
    assert warm_result.annotations[0].combat_encounter_kind == "ordinary"
    assert (
        warm_result.annotations[0].combat_encounter_basis
        == "ordinary_opponent_presentation"
    )
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
    assert '"scene_kind": "exploration"' in cache_text
    assert '"combat_encounter_kind": "ordinary"' in cache_text
    assert '"combat_encounter_basis": "ordinary_opponent_presentation"' in cache_text
    assert '"combat_action"' not in cache_text
    assert '"seed": 0' in cache_text
    assert '"combat_visibility_edge_audit_prompt_version"' in cache_text
    assert '"combat_visibility_edge_strip_version"' in cache_text
    assert '"cinematic_letterbox_detection_version"' in cache_text
    assert '"candidate_annotation_relationship_repair_prompt_version"' in cache_text
    assert '"candidate_annotation_relationship_repair_schema_version"' in cache_text
    assert (
        '"candidate_annotation_relationship_repair_stage_contract_version"'
        in cache_text
    )
    assert '"candidate_annotation_relationship_repair_num_predict": 1024' in cache_text
    assert (
        '"candidate_annotation_relationship_repair_evidence_max_length": 160'
        in cache_text
    )


def test_batch_boundary_rejects_metadata_change_after_annotation(
    tmp_path: Path,
) -> None:
    """Annotation中のmetadata変更がbatch完了時に拒否されること。

    Arrange:
        - Annotation中にsource内容とmetadataを書き換えるruntimeが用意される
    Act:
        - 一つのCandidate Annotation batchが処理される
    Assert:
        - batch結果が返されずVideo Set snapshot変更として拒否されること
    """
    # Arrange
    video_set, configuration = _video_set_and_configuration(tmp_path)
    request = _requests()[0]
    source_path = video_set.sources[0].path

    def mutate_source() -> None:
        source_path.write_bytes(b"mutated-video-with-new-size")

    runtime = FakeStructuredVisionRuntime(
        _catalog(),
        (_annotations(_requests())[0],),
        on_candidate_annotation=mutate_source,
    )
    models = FakeModelRuntime("vision-model").resolve_models(configuration)

    # Act
    # Assert
    with pytest.raises(ValueError, match="Video Set snapshotが変更されました"):
        VideoSetVisionProcessor(runtime, RecordingRunObserver()).process(
            video_set=video_set,
            representatives=request.frame_candidates,
            representative_source_fingerprints=(StageFingerprint("c" * 64),),
            annotation_requests=(request,),
            configuration=configuration,
            resolved_models=models,
        )
    assert len(runtime.candidate_annotation_calls) == 1


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
    """先頭・途中・末尾Annotation失敗後も成功済みMomentが再利用されること。

    Arrange:
        - 3件のうち指定位置だけ失敗するfake VisionRuntimeが用意される
    Act:
        - 失敗runの後に同じ入力が再実行される
    Assert:
        - Scene Catalogと成功済みAnnotationは再実行されず失敗位置だけ生成されること
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

    # Act
    # Assert
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
    ] == [requests[failure_position].moment.identifier]
    assert result.annotations == annotations
    expected_cache_hits = [True] * len(requests)
    expected_cache_hits[failure_position] = False
    assert [
        item.cache_hit for item in result.annotation_diagnostics
    ] == expected_cache_hits


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
        - Candidate schedulerの外部処理開始が一度だけ通知されること
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


def test_warm_combat_fallback_reports_each_frame_reuse_without_external_work(
    tmp_path: Path,
) -> None:
    """warm fallbackの各frameが個別Stage reuseとして通知されること。

    Arrange:
        - Catalog、Primary、二つのfallback frameがCompleted Stageとして確定済みである
    Act:
        - 同じ入力がProgress Tracker付きで再実行される
    Assert:
        - 四つのCompleted Stageがhitとして通知されること
        - 外部Ollama処理が開始されないこと
    """
    # Arrange
    video_set, configuration = _video_set_and_configuration(tmp_path)
    request, annotations = _combat_fallback_fixture()
    models = FakeModelRuntime("vision-model").resolve_models(configuration)
    VideoSetVisionProcessor(
        FakeStructuredVisionRuntime(_catalog(), annotations),
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        representatives=(request.frame_candidates[0],),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=(request,),
        configuration=configuration,
        resolved_models=models,
    )
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer, clock=lambda: 10.0)
    progress.start_run()

    # Act
    result = VideoSetVisionProcessor(
        FakeStructuredVisionRuntime(
            _catalog(),
            annotations,
            reject_all_calls=True,
        ),
        observer,
        progress=progress,
    ).process(
        video_set=video_set,
        representatives=(request.frame_candidates[0],),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=(request,),
        configuration=configuration,
        resolved_models=models,
    )

    # Assert
    assert len(progress.completed_stage_events) == 4
    assert (
        tuple(
            (
                event.cache_hit_count,
                event.reuse_count,
                event.cache_miss_count,
                event.recompute_count,
            )
            for event in observer.progress_events
            if event.kind == "cache"
        )
        == ((1, 1, 0, 0),) * 4
    )
    assert all(item.cache_hit for item in result.annotation_diagnostics)
    assert not any(
        event.kind == "external_work_started" for event in observer.progress_events
    )


def test_vision_processor_records_annotation_duration_for_eta(
    tmp_path: Path,
) -> None:
    """Annotationごとの推論診断時間がETA sampleへ記録されること。

    Arrange:
        - 0.25秒の推論診断を返すAnnotationが5回再計算される
    Act:
        - 6件目のAnnotation Stageで残り1件のETAが通知される
    Assert:
        - 並列待ち時間でなく画像単位の診断時間によるETAが通知されること
    """
    # Arrange
    observer = RecordingRunObserver()
    current_time = [0.0]
    progress = RunProgressTracker(observer, clock=lambda: current_time[0])
    progress.start_run()
    requests = _requests()
    request = requests[0]
    annotation = _annotations(requests)[0]

    for run_index in range(5):
        run_folder = tmp_path / f"run-{run_index}"
        run_folder.mkdir()
        video_set, configuration = _video_set_and_configuration(run_folder)
        VideoSetVisionProcessor(
            FakeStructuredVisionRuntime(
                _catalog(),
                (annotation,),
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
    assert (event.estimation_state, event.eta_seconds) == ("available", 0.25)


def test_cache_reuse_duration_excludes_parallel_inference_wait(
    tmp_path: Path,
) -> None:
    """cache reuseの所要時間へ並列推論の待ち時間が記録されないこと。

    Arrange:
        - 先頭だけがcache hitで後続推論中にclockが100秒進む再開runが用意される
    Act:
        - 上限2でCandidate Annotationが再開される
    Assert:
        - cache hitの完了時間が0秒として記録されること
    """
    # Arrange
    video_set, base_configuration = _video_set_and_configuration(tmp_path)
    sequential_configuration = EffectiveConfiguration(
        video_input_folder=base_configuration.video_input_folder,
        output_folder=base_configuration.output_folder,
        ollama_max_parallel_requests=1,
    )
    parallel_configuration = replace(
        sequential_configuration,
        ollama_max_parallel_requests=2,
    )
    requests = _three_requests()
    annotations = _three_annotations(requests)
    models = FakeModelRuntime("vision-model").resolve_models(sequential_configuration)
    with pytest.raises(RuntimeError, match="fake raw response"):
        VideoSetVisionProcessor(
            FakeStructuredVisionRuntime(
                _catalog(),
                annotations,
                failure_moment_id=requests[1].moment.identifier,
            ),
            RecordingRunObserver(),
        ).process(
            video_set=video_set,
            representatives=tuple(request.frame_candidates[0] for request in requests),
            representative_source_fingerprints=(StageFingerprint("c" * 64),),
            annotation_requests=requests,
            configuration=sequential_configuration,
            resolved_models=models,
        )
    current_time = [0.0]
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer, clock=lambda: current_time[0])
    progress.start_run()

    def advance_clock(_call: CandidateAnnotationRequest) -> None:
        current_time[0] = 100.0

    # Act
    VideoSetVisionProcessor(
        FakeStructuredVisionRuntime(
            _catalog(),
            annotations,
            on_candidate_annotation_request=advance_clock,
        ),
        observer,
        progress=progress,
    ).process(
        video_set=video_set,
        representatives=tuple(request.frame_candidates[0] for request in requests),
        representative_source_fingerprints=(StageFingerprint("c" * 64),),
        annotation_requests=requests,
        configuration=parallel_configuration,
        resolved_models=models,
    )

    # Assert
    reused_annotation_events = tuple(
        event
        for event in progress.completed_stage_events
        if event.stage == ProcessingStage.ANNOTATE_CANDIDATE and event.reuse_count == 1
    )
    assert reused_annotation_events
    assert reused_annotation_events[0].elapsed_seconds == 0.0


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


def _combat_fallback_fixture(
    *,
    seed: str = "d",
    anchor_time: Fraction = Fraction(10),
) -> tuple[
    CandidateAnnotationRequest,
    tuple[CandidateAnnotation, ...],
]:
    """独立fallback評価用の一Momentとframe別Annotationを構築する。"""
    frame_prefix = "frame" if seed == "d" else f"frame-{seed}"
    primary = FrameCandidate(f"{frame_prefix}-primary", f"{seed}-primary".encode())
    first_fallback = FrameCandidate(
        f"{frame_prefix}-fallback-low",
        f"{seed}-fallback-low".encode(),
    )
    best_fallback = FrameCandidate(
        f"{frame_prefix}-fallback-high",
        f"{seed}-fallback-high".encode(),
    )
    moment = CandidateMoment(
        identifier="mom_" + seed * 64,
        source_pts=int(anchor_time),
        anchor_time=anchor_time,
        timeline_segment_id="seg_" + seed * 64,
        evidence=("scene",),
        proxy_quality_score=0.9,
        frame_candidate_ids=(
            primary.identifier,
            first_fallback.identifier,
            best_fallback.identifier,
        ),
    )
    request = CandidateAnnotationRequest(
        moment=moment,
        frame_candidates=(primary, first_fallback, best_fallback),
        context_cues=(),
        video_set_progress=Fraction(1, 2),
        selection_intent="ブログ本文を説明できる画像を選ぶ",
        cue_selection_policy_version="nearby-context-v1",
    )
    return (
        request,
        (
            CandidateAnnotation(
                candidate=primary,
                summary="攻撃effectで敵が見えない戦闘",
                candidate_moment_id=moment.identifier,
                scene_slug="exploration",
                blog_image_type="normal_gameplay",
                explanation_value="none",
                combat_encounter_kind="ordinary",
                combat_encounter_basis="ordinary_opponent_presentation",
            ),
            CandidateAnnotation(
                candidate=first_fallback,
                summary="敵が見える通常戦闘",
                candidate_moment_id=moment.identifier,
                scene_slug="exploration",
                blog_image_type="normal_gameplay",
                explanation_value="low",
                combat_encounter_kind="ordinary",
                combat_encounter_basis="ordinary_opponent_presentation",
            ),
            CandidateAnnotation(
                candidate=best_fallback,
                summary="敵と主人公が明瞭な通常戦闘",
                candidate_moment_id=moment.identifier,
                scene_slug="exploration",
                blog_image_type="normal_gameplay",
                explanation_value="high",
                combat_encounter_kind="ordinary",
                combat_encounter_basis="ordinary_opponent_presentation",
            ),
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
            SceneCatalogEntry(
                "exploration", "探索", "フィールド探索", "exploration", "ordinary"
            ),
            SceneCatalogEntry("climax", "終盤", "重要な対決", "event", "cinematic"),
            SceneCatalogEntry("other", "その他", "分類不能", "other", "ordinary"),
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
            combat_encounter_kind="ordinary",
            combat_encounter_basis="ordinary_opponent_presentation",
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
