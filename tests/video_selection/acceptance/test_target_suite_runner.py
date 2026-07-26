"""durable cold/warm target suite runnerのtest。"""

import hashlib
from dataclasses import fields, replace
from pathlib import Path

import pytest

from src.video_selection.acceptance.acceptance_profile import AcceptanceProfile
from src.video_selection.acceptance.atomic_json import (
    read_json_object,
    write_atomic_json,
)
from src.video_selection.acceptance.execute_acceptance_phase import (
    normalized_result_digest,
)
from src.video_selection.acceptance.target_suite_runner import (
    EnvironmentProbe,
    ModelResolver,
    OllamaDeploymentProbe,
    PhaseExecutor,
    TargetSuiteRunner,
)
from src.video_selection.configuration.resolve_effective_configuration import (
    resolve_effective_configuration,
)
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.model_update_status import ModelUpdateStatus
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.resolved_models import ResolvedModels
from src.video_selection.services.build_stage_fingerprint import (
    build_stage_fingerprint,
)
from src.video_selection.services.completed_stage_writer import CompletedStageWriter
from tests.video_selection.fakes.fake_model_runtime import FakeModelRuntime


def test_interrupt_after_cold_resumes_only_warm_then_waits_for_human_review(
    tmp_path: Path,
) -> None:
    """cold完了後の中断がwarmだけを再開しworksheet完了までpendingになること。

    Arrange:
        - warm初回だけinterruptするphase executorとrelease profileが用意される
    Act:
        - suiteが中断、resume、human review完了の3回実行される
    Assert:
        - coldは一度、warmは中断分を含む二度だけ呼ばれること
        - resume後はexit 3、review完了後はphase再実行なしでexit 0になること
        - acceptance record/baselineが生成されrelease workが削除されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []
    warm_interrupted = False

    def execute(
        phase: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        nonlocal warm_interrupted
        calls.append(phase)
        if phase == "warm" and not warm_interrupted:
            warm_interrupted = True
            return 130, _interrupted_phase(phase), None, None
        return _successful_phase(configuration, phase)

    runner = _runner(execute)

    # Act
    interrupted = runner.run(profile_path=profile_path, suite="release")
    pending = runner.run(profile_path=profile_path, suite="release")
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    worksheet_path = suite_root / "review-worksheet.json"
    worksheet = read_json_object(worksheet_path)
    assert worksheet is not None
    worksheet["reviewer"] = "reviewer"
    worksheet["completed_at"] = "2026-07-17T00:00:00+00:00"
    selected = worksheet["selected"]
    assert isinstance(selected, list)
    selected_item = selected[0]
    assert isinstance(selected_item, dict)
    selected_item.update(
        {
            "visual_quality": "pass",
            "blog_usable": "yes",
            "annotation_consistency": "consistent",
            "context_overrode_visual_invalidity": "no",
        }
    )
    checks = worksheet["suite_checks"]
    assert isinstance(checks, dict)
    checks["spoiler_monotonicity"] = "pass"
    write_atomic_json(worksheet_path, worksheet)
    passed = runner.run(
        profile_path=profile_path,
        suite="release",
        human_review_path=worksheet_path,
    )

    # Assert
    assert interrupted == 130
    assert pending == 3
    assert passed == 0
    assert calls == ["cold", "warm", "warm"]
    state = read_json_object(suite_root / "acceptance-state.json")
    assert state is not None
    phases = state["phases"]
    assert isinstance(phases, dict)
    warm = phases["warm"]
    assert isinstance(warm, dict)
    assert warm["duration_seconds"] == 14.0
    assert warm["attempt_count"] == 2
    assert warm["cache_hit_count"] == 1
    assert warm["cache_miss_count"] == 2
    assert warm["stage_durations_seconds"] == {"scan-video": 14.0}
    assert warm["completed_stage_counts"] == {"scan-video": 2}
    assert "phase_attempts" not in state
    record = read_json_object(suite_root / "acceptance.json")
    assert record is not None
    assert record["status"] == "passed"
    assert (suite_root / "baseline" / "baseline.json").is_file()
    assert (suite_root / "baseline" / "baseline.md").is_file()
    assert not (suite_root / "work").exists()


def test_reset_suite_discards_completed_state_and_runs_cold_again(
    tmp_path: Path,
) -> None:
    """明示--reset-suiteだけがcompleted stateを破棄してcoldから再実行すること。

    Arrange:
        - cold/warm完了済みでpending reviewのrelease suiteが用意される
    Act:
        - reset_suite=trueで同じsuiteが再実行される
    Assert:
        - 2回目もcold、warmの順で両phaseが呼ばれること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        phase: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(phase)
        return _successful_phase(configuration, phase)

    runner = _runner(execute)
    first = runner.run(profile_path=profile_path, suite="release")

    # Act
    second = runner.run(
        profile_path=profile_path,
        suite="release",
        reset_suite=True,
    )

    # Assert
    assert first == 3
    assert second == 3
    assert calls == ["cold", "warm", "cold", "warm"]


def test_reset_suite_fails_when_suite_root_survives_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """suite rootを削除できないresetが後続処理へ進まないこと。

    Arrange:
        - cold/warm完了済みsuiteと削除してもdirectoryを残すfilesystem境界が用意される
    Act:
        - reset_suite=trueで同じsuiteの再実行が試行される
    Assert:
        - reset失敗として拒否されphaseが再実行されないこと
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        phase: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(phase)
        return _successful_phase(configuration, phase)

    runner = _runner(execute)
    assert runner.run(profile_path=profile_path, suite="release") == 3
    monkeypatch.setattr(
        "src.video_selection.acceptance.target_suite_runner.shutil.rmtree",
        lambda _path: None,
    )

    # Act / Assert
    with pytest.raises(ValueError, match="完全に削除"):
        runner.run(
            profile_path=profile_path,
            suite="release",
            reset_suite=True,
        )
    assert calls == ["cold", "warm"]


def test_release_finalization_fails_when_private_work_survives_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """private workを削除できないrelease finalizationが不合格になること。

    Arrange:
        - cold/warmが完了するrelease suiteが用意される
        - 削除してもworkを残すfilesystem境界が用意される
    Act:
        - human review待ちまでrelease suiteが実行される
    Assert:
        - cleanup failureとしてexit 1になりpassing recordが生成されないこと
    """
    # Arrange
    profile_path = _profile(tmp_path)
    runner = _runner(
        lambda phase, configuration, _models, _suite_root: _successful_phase(
            configuration,
            phase,
        )
    )
    monkeypatch.setattr(
        "src.video_selection.acceptance.target_suite_runner.shutil.rmtree",
        lambda _path, *_args, **_kwargs: None,
    )

    # Act
    result = runner.run(profile_path=profile_path, suite="release")

    # Assert
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    state = read_json_object(suite_root / "acceptance-state.json")
    assert result == 1
    assert state is not None
    assert state["acceptance_status"] == "failed"
    assert state["last_failure"] == {
        "phase": "acceptance_cleanup",
        "exit_code": 1,
        "reason": "release_cleanup_failed",
    }
    assert (suite_root / "work").is_dir()
    assert not (suite_root / "acceptance.json").exists()


def test_privacy_failure_also_reports_release_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """privacy不合格時にもprivate work削除失敗が記録されること。

    Arrange:
        - privacy gateが不合格になるrelease suiteとworkを残すfilesystem境界が用意される
    Act:
        - release suiteのfinalizationが実行される
    Assert:
        - privacy不合格を先行理由に持つcleanup failureとしてexit 1になること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    runner = _runner(
        lambda phase, configuration, _models, _suite_root: _successful_phase(
            configuration,
            phase,
        )
    )

    def reject_privacy(*_args: object, **_kwargs: object) -> None:
        raise ValueError("privacy failure")

    monkeypatch.setattr(
        "src.video_selection.acceptance.target_suite_runner."
        "validate_acceptance_record_privacy",
        reject_privacy,
    )
    monkeypatch.setattr(
        "src.video_selection.acceptance.target_suite_runner.shutil.rmtree",
        lambda _path, *_args, **_kwargs: None,
    )

    # Act
    result = runner.run(profile_path=profile_path, suite="release")

    # Assert
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    state = read_json_object(suite_root / "acceptance-state.json")
    assert result == 1
    assert state is not None
    assert state["last_failure"] == {
        "phase": "acceptance_cleanup",
        "exit_code": 1,
        "reason": "release_cleanup_failed",
        "prior_reason": "privacy_gate_failed",
    }
    assert (suite_root / "work").is_dir()


def test_state_preserves_privacy_safe_performance_configuration(
    tmp_path: Path,
) -> None:
    """baseline元stateへ性能に影響する実効設定が保存されること。

    Arrange:
        - target acceptance用のprivate profileと設定が用意される
    Act:
        - release suiteのcold/warm phaseが完了する
    Assert:
        - 設定file digestとperformance設定がpathなしでstateへ保存されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    runner = _runner(
        lambda phase, configuration, _models, _suite_root: _successful_phase(
            configuration,
            phase,
        )
    )

    # Act
    assert runner.run(profile_path=profile_path, suite="release") == 3

    # Assert
    state = read_json_object(
        tmp_path
        / "artifacts"
        / "target-acceptance"
        / "release"
        / "acceptance-state.json"
    )
    assert state is not None
    configuration = state["configuration"]
    assert isinstance(configuration, dict)
    safe_configuration_fields = {
        item.name for item in fields(EffectiveConfiguration)
    } - {
        "video_input_folder",
        "output_folder",
        "scene_hint",
        "ollama_host",
        "provenance",
    }
    assert set(configuration) == safe_configuration_fields | {
        "configuration_digest",
        "scene_hint_identity",
        "ollama_endpoint_identity",
    }
    expected_digest = hashlib.sha256(
        (tmp_path / "video-selection.toml").read_bytes()
    ).hexdigest()
    assert configuration["configuration_digest"] == expected_digest
    assert configuration["scene_catalog_num_ctx"] == 32768
    assert configuration["candidate_annotation_num_ctx"] == 32768
    assert configuration["max_frame_candidates"] == 3
    assert configuration["ollama_max_parallel_requests"] == 1
    assert configuration["speech_to_text_beam_size"] == 5
    assert configuration["speech_chunk_seconds"] == 600.0
    assert configuration["speech_overlap_seconds"] == 5.0
    assert "ollama_host" not in configuration
    target = state["target"]
    assert isinstance(target, dict)
    assert target["ollama"] == {
        "deployment": "windows_native",
        "listener_process": "ollama.exe",
    }


def test_ollama_windows_binding_is_revalidated_after_model_resolution(
    tmp_path: Path,
) -> None:
    """model解決中に変わったWindows Ollama bindingが拒否されること。

    Arrange:
        - model解決前後で異なるdeployment証拠を返すprobeが用意される
    Act:
        - release suiteが開始される
    Assert:
        - phase開始前にbinding変更として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    probe_results: list[dict[str, object]] = [
        {
            "deployment": "windows_native",
            "listener_process": "ollama.exe",
        },
        {
            "deployment": "windows_native",
            "listener_process": "replacement.exe",
        },
    ]
    phase_calls: list[str] = []

    def execute(
        phase: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        phase_calls.append(phase)
        return _successful_phase(configuration, phase)

    runner = _runner(
        execute,
        ollama_deployment_probe=lambda _host: probe_results.pop(0),
    )

    # Act
    with pytest.raises(ValueError, match="Windows Ollama binding"):
        runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert phase_calls == []


def test_finalization_rejects_review_with_truncated_candidate_set(
    tmp_path: Path,
) -> None:
    """cold evidenceから候補を削除したhuman reviewではfinalizeされないこと。

    Arrange:
        - cold/warm完了後に生成されたprivate worksheetが用意される
        - immutableなrejected candidateがworksheetから削除される
    Act:
        - 変更されたworksheetでfinalizationが実行される
    Assert:
        - candidate集合不一致としてacceptance evidenceにならないこと
    """
    # Arrange
    profile_path = _profile(tmp_path)
    runner = _runner(
        lambda phase, configuration, _models, _suite_root: _successful_phase(
            configuration,
            phase,
        )
    )
    assert runner.run(profile_path=profile_path, suite="release") == 3
    worksheet_path = (
        tmp_path
        / "artifacts"
        / "target-acceptance"
        / "release"
        / "review-worksheet.json"
    )
    worksheet = read_json_object(worksheet_path)
    assert worksheet is not None
    worksheet["rejected"] = []
    write_atomic_json(worksheet_path, worksheet)

    # Act / Assert
    with pytest.raises(ValueError, match="candidate集合"):
        runner.run(
            profile_path=profile_path,
            suite="release",
            human_review_path=worksheet_path,
        )


def test_completed_phases_resume_worksheet_finalization_without_rerun(
    tmp_path: Path,
) -> None:
    """warm完了後の中断がphase再実行なしでworksheet生成から再開されること。

    Arrange:
        - cold/warm完了stateとdurable output/cacheを持つ未生成worksheet状態が用意される
    Act:
        - 同じrelease suiteが再開される
    Assert:
        - cold/warmを再実行せずworksheetが生成されexit 3になること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        phase: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(phase)
        return _successful_phase(configuration, phase)

    runner = _runner(execute)
    suite_root, _cold_configuration = _prepare_resume_without_worksheet(
        tmp_path,
        runner,
        profile_path,
    )

    # Act
    result = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert result == 3
    assert calls == ["cold", "warm"]
    assert (suite_root / "review-worksheet.json").is_file()


def test_resume_rejects_changed_completed_cold_report(tmp_path: Path) -> None:
    """完了phase後に変更されたcold reportからworksheetが生成されないこと。

    Arrange:
        - cold/warm完了後かつworksheet未生成のresume stateが用意される
        - cold reportのcandidate IDがphase確定後に変更される
    Act:
        - suiteがworksheet生成から再開される
    Assert:
        - phase確定時のcanonical report artifact不一致として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    runner = _runner(
        lambda phase, configuration, _models, _suite_root: _successful_phase(
            configuration,
            phase,
        )
    )
    _suite_root, cold_configuration = _prepare_resume_without_worksheet(
        tmp_path,
        runner,
        profile_path,
    )
    report_path = cold_configuration.output_folder / "report.json"
    report = read_json_object(report_path)
    assert report is not None
    selected = report["selected"]
    assert isinstance(selected, list)
    selected_item = selected[0]
    assert isinstance(selected_item, dict)
    selected_item["image_id"] = "frm_" + "3" * 64
    write_atomic_json(report_path, report)

    # Act / Assert
    with pytest.raises(ValueError, match="canonical report artifact"):
        runner.run(profile_path=profile_path, suite="release")


def test_review_finalization_rejects_changed_completed_cold_report(
    tmp_path: Path,
) -> None:
    """worksheet生成後に置換されたcold reportからreviewが確定されないこと。

    Arrange:
        - cold/warm完了後かつworksheet生成済みのresume stateが用意される
        - cold reportがphase確定後に同じnormalized resultの別内容へ置換される
    Act:
        - human review待ちのsuiteが再開される
    Assert:
        - phase確定時のcanonical report artifact不一致として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    runner = _runner(
        lambda phase, configuration, _models, _suite_root: _successful_phase(
            configuration,
            phase,
        )
    )
    assert runner.run(profile_path=profile_path, suite="release") == 3
    report_path = (
        tmp_path
        / "artifacts"
        / "target-acceptance"
        / "release"
        / "outputs"
        / "cold"
        / "report.json"
    )
    report = read_json_object(report_path)
    assert report is not None
    report["tampered"] = True
    write_atomic_json(report_path, report)

    # Act / Assert
    with pytest.raises(ValueError, match="canonical report artifact"):
        runner.run(profile_path=profile_path, suite="release")


def test_review_finalization_rejects_changed_selected_image(tmp_path: Path) -> None:
    """worksheet生成後に置換されたselected画像からreviewが確定されないこと。

    Arrange:
        - cold/warm完了後かつworksheet生成済みのresume stateが用意される
        - cold selected画像がphase確定後に別内容へ置換される
    Act:
        - human review待ちのsuiteが再開される
    Assert:
        - phase確定時のselected output artifact不一致として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    runner = _runner(
        lambda phase, configuration, _models, _suite_root: _successful_phase(
            configuration,
            phase,
        )
    )
    assert runner.run(profile_path=profile_path, suite="release") == 3
    selected_path = (
        tmp_path
        / "artifacts"
        / "target-acceptance"
        / "release"
        / "outputs"
        / "cold"
        / "images"
        / "0001_gameplay.webp"
    )
    selected_path.write_bytes(b"replaced-webp")

    # Act / Assert
    with pytest.raises(ValueError, match="selected output artifact"):
        runner.run(profile_path=profile_path, suite="release")


def test_resume_rejects_changed_completed_selection_artifact(
    tmp_path: Path,
) -> None:
    """完了phase後に変更されたselection artifactがworksheetへ使われないこと。

    Arrange:
        - cold/warm完了後かつworksheet未生成のresume stateが用意される
        - cold selection artifactがmanifest確定後に変更される
    Act:
        - suiteがworksheet生成から再開される
    Assert:
        - Completed Stage integrity不一致として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    runner = _runner(
        lambda phase, configuration, _models, _suite_root: _successful_phase(
            configuration,
            phase,
        )
    )
    suite_root, cold_configuration = _prepare_resume_without_worksheet(
        tmp_path,
        runner,
        profile_path,
    )
    state = read_json_object(suite_root / "acceptance-state.json")
    assert state is not None
    phases = state["phases"]
    assert isinstance(phases, dict)
    cold = phases["cold"]
    assert isinstance(cold, dict)
    selection_fingerprint = cold["selection_stage_fingerprint"]
    assert isinstance(selection_fingerprint, str)
    artifact_path = (
        cold_configuration.processing_cache_folder
        / "video-sets"
        / ("f" * 64)
        / ProcessingStage.SELECT_IMAGES.value
        / selection_fingerprint
        / "artifact.json"
    )
    artifact = read_json_object(artifact_path)
    assert artifact is not None
    artifact["rejected"] = []
    write_atomic_json(artifact_path, artifact)

    # Act / Assert
    with pytest.raises(ValueError, match="integrity"):
        runner.run(profile_path=profile_path, suite="release")


def test_completed_state_revalidates_current_suite_fingerprint(tmp_path: Path) -> None:
    """完了済みstateでも現在のsuite fingerprintが再検証されること。

    Arrange:
        - cold/warm完了後にmaterializerのsuite fingerprintが変化する
    Act:
        - human review待ちのsuiteが再開される
    Assert:
        - phaseを再実行せずidentity不一致として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        phase: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(phase)
        return _successful_phase(configuration, phase)

    assert _runner(execute).run(profile_path=profile_path, suite="release") == 3

    # Act
    with pytest.raises(ValueError) as error:
        _runner(execute, suite_fingerprint="c" * 64).run(
            profile_path=profile_path,
            suite="release",
        )

    # Assert
    assert "suite identity" in str(error.value)
    assert calls == ["cold", "warm"]


def test_completed_state_revalidates_current_model_identity(tmp_path: Path) -> None:
    """完了済みstateでも現在のResolved Model Identityが再検証されること。

    Arrange:
        - cold/warm完了後にmodel resolverのexecution identityが変化する
    Act:
        - human review待ちのsuiteが再開される
    Assert:
        - phaseを再実行せずidentity不一致として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        phase: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(phase)
        return _successful_phase(configuration, phase)

    assert _runner(execute).run(profile_path=profile_path, suite="release") == 3

    # Act
    with pytest.raises(ValueError) as error:
        _runner(execute, model_identity_seed="changed-model").run(
            profile_path=profile_path,
            suite="release",
        )

    # Assert
    assert "suite identity" in str(error.value)
    assert calls == ["cold", "warm"]


def test_completed_state_rejects_changed_environment_ollama_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """環境変数由来Ollama endpointが変わったstateは再利用されないこと。

    Arrange:
        - TOMLにhostを持たずOLLAMA_HOSTで完了したsuiteが用意される
    Act:
        - OLLAMA_HOSTを別endpointへ変えてsuiteが再開される
    Assert:
        - privacy-safeなsuite identity不一致として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path, include_ollama_host=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://first.example:11434")
    calls: list[str] = []

    def execute(
        phase: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(phase)
        return _successful_phase(configuration, phase)

    runner = _runner(execute)
    assert runner.run(profile_path=profile_path, suite="release") == 3
    state_path = (
        tmp_path
        / "artifacts"
        / "target-acceptance"
        / "release"
        / "acceptance-state.json"
    )
    state = read_json_object(state_path)
    assert state is not None
    assert state["ollama_endpoint_identity"] != "http://first.example:11434"
    monkeypatch.setenv("OLLAMA_HOST", "http://second.example:11434")

    # Act / Assert
    with pytest.raises(ValueError, match="suite identity"):
        runner.run(profile_path=profile_path, suite="release")
    assert calls == ["cold", "warm"]


def test_completed_state_ignores_model_update_diagnostic_change(
    tmp_path: Path,
) -> None:
    """同じ実行identityのmodel更新診断変更では完了stateが再利用されること。

    Arrange:
        - 初回はnot_requested、再開時はunchangedとなる同一Resolved Modelが用意される
    Act:
        - cold/warm完了後のhuman review待ちsuiteが再開される
    Assert:
        - phaseを再実行せずpending human reviewのまま再開されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []
    model_runtime = FakeModelRuntime("stable-model")
    resolution_count = 0

    def resolve(configuration: EffectiveConfiguration) -> ResolvedModels:
        nonlocal resolution_count
        resolution_count += 1
        resolved = model_runtime.resolve_models(configuration)
        if resolution_count == 1:
            return resolved
        return ResolvedModels(
            tuple(
                replace(item, update_status=ModelUpdateStatus.UNCHANGED)
                for item in resolved.items
            )
        )

    def execute(
        phase: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(phase)
        return _successful_phase(configuration, phase)

    runner = _runner(execute, model_resolver=resolve)
    assert runner.run(profile_path=profile_path, suite="release") == 3

    # Act
    resumed = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert resumed == 3
    assert resolution_count == 2
    assert calls == ["cold", "warm"]


def test_completed_state_rejects_changed_target_identity(tmp_path: Path) -> None:
    """target環境が変わった完了stateは再利用されないこと。

    Arrange:
        - cold/warm完了後にdriver identityが変わるtarget probeが用意される
    Act:
        - 同じprofileとsourceでsuiteが再開される
    Assert:
        - target identity不一致として既存stateの再利用が拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []
    target = {"os": "linux", "gpu_driver": "first"}

    def execute(
        phase: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(phase)
        return _successful_phase(configuration, phase)

    runner = _runner(execute, environment_probe=lambda: dict(target))
    assert runner.run(profile_path=profile_path, suite="release") == 3
    target["gpu_driver"] = "changed"

    # Act
    with pytest.raises(ValueError) as error:
        runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert "target identity" in str(error.value)
    assert calls == ["cold", "warm"]


def test_incomplete_phase_removes_uncommitted_output_before_rerun(
    tmp_path: Path,
) -> None:
    """phase state未確定の既存outputが再実行前に削除されること。

    Arrange:
        - cold phase recordなしでatomic publicationだけが残ったsuiteが用意される
    Act:
        - release suiteが実行される
    Assert:
        - executor呼出時にはstale cold outputがなくcold/warmが完了すること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    stale_output = suite_root / "outputs" / "cold"
    stale_output.mkdir(parents=True)
    (stale_output / "stale.json").write_text("{}", encoding="utf-8")
    calls: list[str] = []

    def execute(
        phase: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(phase)
        assert not (configuration.output_folder / "stale.json").exists()
        return _successful_phase(configuration, phase)

    runner = _runner(execute)

    # Act
    result = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert result == 3
    assert calls == ["cold", "warm"]


def test_interrupt_before_phase_record_can_retry_without_reset(
    tmp_path: Path,
) -> None:
    """phase recordより前のuser interruptでも明示resetなしで再開されること。

    Arrange:
        - phase executorが初回だけ記録を返す前にinterruptされるsuiteが用意される
    Act:
        - suiteが中断後に同じstateから再開される
    Assert:
        - 初回はexit 130となり、2回目はcoldから再開されること
        - 不完全なresource計測ではacceptanceが誤合格しないこと
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []
    interrupted_once = False

    def execute(
        phase: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        nonlocal interrupted_once
        calls.append(phase)
        if not interrupted_once:
            interrupted_once = True
            raise KeyboardInterrupt
        return _successful_phase(configuration, phase)

    runner = _runner(execute)

    # Act
    interrupted = runner.run(profile_path=profile_path, suite="release")
    resumed = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert interrupted == 130
    assert resumed == 1
    assert calls == ["cold", "cold", "warm"]
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    state = read_json_object(suite_root / "acceptance-state.json")
    assert state is not None
    phases = state["phases"]
    assert isinstance(phases, dict)
    cold = phases["cold"]
    assert isinstance(cold, dict)
    assert cold["attempt_count"] == 2
    assert cold["resource_sampling_complete"] is False


def test_pending_refinalization_removes_previously_passing_baseline(
    tmp_path: Path,
) -> None:
    """再評価がpendingなら以前のpassing baselineが削除されること。

    Arrange:
        - default worksheetはpendingのままexternal worksheetでpassされる
    Act:
        - external worksheetを指定せず同じsuiteが再評価される
    Assert:
        - statusがpendingへ戻り古いbaselineが残らないこと
    """
    # Arrange
    profile_path = _profile(tmp_path)
    runner = _runner(
        lambda phase, configuration, _models, _suite_root: _successful_phase(
            configuration,
            phase,
        )
    )
    assert runner.run(profile_path=profile_path, suite="release") == 3
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    default_worksheet = read_json_object(suite_root / "review-worksheet.json")
    assert default_worksheet is not None
    completed_worksheet = dict(default_worksheet)
    completed_worksheet["reviewer"] = "reviewer"
    completed_worksheet["completed_at"] = "2026-07-17T00:00:00+00:00"
    selected = completed_worksheet["selected"]
    assert isinstance(selected, list)
    selected_item = selected[0]
    assert isinstance(selected_item, dict)
    selected_item.update(
        {
            "visual_quality": "pass",
            "blog_usable": "yes",
            "annotation_consistency": "consistent",
            "context_overrode_visual_invalidity": "no",
        }
    )
    checks = completed_worksheet["suite_checks"]
    assert isinstance(checks, dict)
    checks["spoiler_monotonicity"] = "pass"
    external_path = tmp_path / "completed-review.json"
    write_atomic_json(external_path, completed_worksheet)
    assert (
        runner.run(
            profile_path=profile_path,
            suite="release",
            human_review_path=external_path,
        )
        == 0
    )
    assert (suite_root / "baseline" / "baseline.json").is_file()

    # Act
    result = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert result == 3
    assert not (suite_root / "baseline").exists()


def test_invalid_refinalization_preserves_previously_passing_baseline(
    tmp_path: Path,
) -> None:
    """不正なworksheetでの再評価前にはpassing baselineが保持されること。

    Arrange:
        - external worksheetで合格済みのsuiteとpassing baselineが用意される
        - candidate集合を欠落させたworksheetが用意される
    Act:
        - 不正なworksheetを指定して再finalizationが試行される
    Assert:
        - worksheet検証が拒否され既存baselineが変更されないこと
    """
    # Arrange
    profile_path = _profile(tmp_path)
    runner = _runner(
        lambda phase, configuration, _models, _suite_root: _successful_phase(
            configuration,
            phase,
        )
    )
    assert runner.run(profile_path=profile_path, suite="release") == 3
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    worksheet = read_json_object(suite_root / "review-worksheet.json")
    assert worksheet is not None
    worksheet["reviewer"] = "reviewer"
    worksheet["completed_at"] = "2026-07-17T00:00:00+00:00"
    selected = worksheet["selected"]
    assert isinstance(selected, list)
    selected_item = selected[0]
    assert isinstance(selected_item, dict)
    selected_item.update(
        {
            "visual_quality": "pass",
            "blog_usable": "yes",
            "annotation_consistency": "consistent",
            "context_overrode_visual_invalidity": "no",
        }
    )
    checks = worksheet["suite_checks"]
    assert isinstance(checks, dict)
    checks["spoiler_monotonicity"] = "pass"
    valid_path = tmp_path / "valid-review.json"
    write_atomic_json(valid_path, worksheet)
    assert (
        runner.run(
            profile_path=profile_path,
            suite="release",
            human_review_path=valid_path,
        )
        == 0
    )
    baseline_path = suite_root / "baseline" / "baseline.json"
    trusted_baseline = baseline_path.read_bytes()
    worksheet["rejected"] = []
    invalid_path = tmp_path / "invalid-review.json"
    write_atomic_json(invalid_path, worksheet)

    # Act / Assert
    with pytest.raises(ValueError, match="candidate集合"):
        runner.run(
            profile_path=profile_path,
            suite="release",
            human_review_path=invalid_path,
        )
    assert baseline_path.read_bytes() == trusted_baseline


def _runner(
    phase_executor: PhaseExecutor,
    *,
    suite_fingerprint: str = "d" * 64,
    model_identity_seed: str = "acceptance-runner",
    model_resolver: ModelResolver | None = None,
    environment_probe: EnvironmentProbe | None = None,
    ollama_deployment_probe: OllamaDeploymentProbe | None = None,
) -> TargetSuiteRunner:
    """target外でもstate machineを検証できるdependency構成を返す。"""
    model_runtime = FakeModelRuntime(model_identity_seed)

    def materialize(
        profile: AcceptanceProfile,
        suite_root: Path,
    ) -> tuple[Path, dict[str, object]]:
        del profile
        input_folder = suite_root / "work" / "input"
        input_folder.mkdir(parents=True, exist_ok=True)
        (input_folder / "scenario-001.mkv").write_bytes(b"anonymous")
        return input_folder, {"suite_fingerprint": suite_fingerprint}

    return TargetSuiteRunner(
        environment_probe=environment_probe
        or (
            lambda: {
                "host_os": "windows_11_pro",
                "environment": "wsl2",
                "gpu": "rtx_5090",
            }
        ),
        revision_probe=lambda _path: ("a" * 40, False),
        ollama_deployment_probe=ollama_deployment_probe
        or (
            lambda _host: {
                "deployment": "windows_native",
                "listener_process": "ollama.exe",
            }
        ),
        model_resolver=model_resolver or model_runtime.resolve_models,
        phase_executor=phase_executor,
        release_materializer=materialize,
        storage_preflight=lambda _profile, _input_folder: {
            "input_video_bytes": 9,
            "input_video_count": 1,
            "artifact_available_bytes": 200 * 1024**3,
            "required_artifact_capacity_bytes": 160 * 1024**3,
            "persistent_cache_budget_bytes": 64 * 1024**3,
            "peak_additional_budget_bytes": 96 * 1024**3,
        },
    )


def _successful_phase(
    configuration: EffectiveConfiguration,
    phase: str,
) -> tuple[
    int,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """durable output/cacheも作る成功phase evidenceを返す。"""
    candidate_id = "frm_" + "1" * 64
    image_bytes = b"selected-webp"
    image_relative_path = "images/0001_gameplay.webp"
    image_digest = hashlib.sha256(image_bytes).hexdigest()
    report: dict[str, object] = {
        "run": {
            "id": f"run_{phase}",
            "status": "completed",
            "started_at": "2026-07-26T00:00:00Z",
            "completed_at": "2026-07-26T00:00:01Z",
            "requested_image_count": 1,
            "selected_image_count": 1,
            "warnings": [],
        },
        "selected": [
            {
                "image_id": candidate_id,
                "output": {
                    "relative_path": image_relative_path,
                    "sha256": image_digest,
                    "width": 1920,
                    "height": 1080,
                    "bytes": len(image_bytes),
                },
            }
        ],
        "provenance": {"models": {}, "stages": []},
    }
    configuration.output_folder.mkdir(parents=True, exist_ok=True)
    image_path = configuration.output_folder / image_relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_bytes)
    report_path = configuration.output_folder / "report.json"
    write_atomic_json(report_path, report)
    with report_path.open("rb") as file:
        report_digest = hashlib.file_digest(file, "sha256").hexdigest()
    selection_semantic_input = {
        "requested_count": 1,
        "annotated_candidate_ids": [candidate_id, "frm_" + "2" * 64],
    }
    selection_fingerprint = build_stage_fingerprint(
        ProcessingStage.SELECT_IMAGES,
        (),
        selection_semantic_input,
    )
    artifact: dict[str, object] = {
        "rejected": [
            {
                "candidate_id": "frm_" + "2" * 64,
                "reason_code": "lower_marginal_utility",
            }
        ]
    }
    CompletedStageWriter(
        configuration.processing_cache_folder,
        subject_namespace="video-sets",
        subject_fingerprint="f" * 64,
    ).write(
        ProcessingStage.SELECT_IMAGES,
        selection_fingerprint,
        (),
        selection_semantic_input,
        artifact,
    )
    record: dict[str, object] = {
        "operation_status": "completed",
        "duration_seconds": 10.0,
        "cache_hit_count": 0,
        "cache_miss_count": 1,
        "reuse_count": 0,
        "unexpected_recompute_count": 0,
        "stage_durations_seconds": {"scan-video": 10.0},
        "completed_stage_counts": {"scan-video": 1},
        "persistent_cache_bytes": 1024,
        "peak_additional_bytes": 2048,
        "disk_sample_count": 1,
        "disk_sample_error_count": 0,
        "gpu_sample_count": 1,
        "gpu_sample_error_count": 0,
        "process_gpu_baseline_mib": 100,
        "system_gpu_baseline_mib": 200,
        "system_global_gpu_peak_mib": 1000,
        "ollama_global_gpu_peak_mib": 1000,
        "stt_non_ollama_gpu_peak_mib": 1000,
        "ollama_model_size_bytes": 512,
        "ollama_model_size_vram_bytes": 512,
        "ollama_model_observed": True,
        "ollama_model_fully_resident": True,
        "resource_sampling_complete": True,
        "speech_runtime_identity": "speech_" + "7" * 64,
        "canonical_report_sha256": report_digest,
        "normalized_result_digest": normalized_result_digest(report),
        "selection_stage_fingerprint": selection_fingerprint.value,
        "video_set": {
            "fingerprint": "f" * 64,
            "scenario_count": 1,
            "total_duration_seconds": "1",
        },
        "phase_marker": phase,
    }
    return 0, record, report, artifact


def _prepare_resume_without_worksheet(
    tmp_path: Path,
    runner: TargetSuiteRunner,
    profile_path: Path,
) -> tuple[Path, EffectiveConfiguration]:
    """完了phase evidenceを復元しworksheet直前のresume stateを返す。"""
    assert runner.run(profile_path=profile_path, suite="release") == 3
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    state_path = suite_root / "acceptance-state.json"
    state = read_json_object(state_path)
    assert state is not None
    state["worksheet_ready"] = False
    write_atomic_json(state_path, state)
    (suite_root / "review-worksheet.json").unlink()
    input_folder = suite_root / "work" / "input"
    input_folder.mkdir(parents=True)
    (input_folder / "scenario-001.mkv").write_bytes(b"anonymous")
    cold_configuration = resolve_effective_configuration(
        video_input_folder=input_folder,
        output_folder=suite_root / "outputs" / "cold",
        config_path=tmp_path / "video-selection.toml",
        environ={},
    )
    _successful_phase(cold_configuration, "cold")
    return suite_root, cold_configuration


def _interrupted_phase(phase: str) -> dict[str, object]:
    """resume時に累積される計測済みinterrupt evidenceを返す。"""
    return {
        "operation_status": "failed",
        "failure_reason": "user_interrupt",
        "failure_exit_code": 130,
        "duration_seconds": 4.0,
        "cache_hit_count": 1,
        "cache_miss_count": 1,
        "reuse_count": 0,
        "unexpected_recompute_count": 0,
        "stage_durations_seconds": {"scan-video": 4.0},
        "completed_stage_counts": {"scan-video": 1},
        "persistent_cache_bytes": 1024,
        "peak_additional_bytes": 2048,
        "disk_sample_count": 1,
        "disk_sample_error_count": 0,
        "gpu_sample_count": 1,
        "gpu_sample_error_count": 0,
        "process_gpu_baseline_mib": 100,
        "system_gpu_baseline_mib": 200,
        "system_global_gpu_peak_mib": 1000,
        "ollama_global_gpu_peak_mib": 1000,
        "stt_non_ollama_gpu_peak_mib": 1000,
        "ollama_model_size_bytes": 512,
        "ollama_model_size_vram_bytes": 512,
        "ollama_model_observed": True,
        "ollama_model_fully_resident": True,
        "resource_sampling_complete": True,
        "phase_marker": phase,
    }


def _profile(tmp_path: Path, *, include_ollama_host: bool = True) -> Path:
    """runner test用のprivate profile/config/sourceを作る。"""
    input_root = tmp_path / "private-input"
    input_root.mkdir()
    (input_root / "source.mkv").write_bytes(b"source")
    configuration = tmp_path / "video-selection.toml"
    configuration_text = 'config_version = "1.0.0"\n'
    if include_ollama_host:
        configuration_text += '\n[ollama]\nhost = "http://127.0.0.1:11434"\n'
    configuration.write_text(configuration_text, encoding="utf-8")
    path = tmp_path / "target.toml"
    path.write_text(
        f'''profile_version = "1.0.0"
input_root = "{input_root}"
configuration_path = "{configuration}"
artifact_root = "{tmp_path / "artifacts"}"

[release_suite]
expected_total_duration = "PT1S"
boundary_tolerance_seconds = 0

[[release_suite.intervals]]
relative_video_path = "source.mkv"
start = "PT0S"
end = "PT1S"
scenario_role = "test"

[full_scale_suite]
expected_video_count = 1
expected_total_duration = "PT1S"
duration_tolerance_seconds = 0
''',
        encoding="utf-8",
    )
    return path
