"""durable cold/warm target suite runnerのtest。"""

from pathlib import Path

import pytest

from src.video_selection.acceptance.acceptance_profile import AcceptanceProfile
from src.video_selection.acceptance.atomic_json import (
    read_json_object,
    write_atomic_json,
)
from src.video_selection.acceptance.target_suite_runner import (
    PhaseExecutor,
    TargetSuiteRunner,
)
from src.video_selection.configuration.resolve_effective_configuration import (
    resolve_effective_configuration,
)
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.resolved_models import ResolvedModels
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

    # Act
    result = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert result == 3
    assert calls == ["cold", "warm"]
    assert (suite_root / "review-worksheet.json").is_file()


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

    # Act / Assert
    with pytest.raises(ValueError, match="suite identity"):
        _runner(execute, suite_fingerprint="c" * 64).run(
            profile_path=profile_path,
            suite="release",
        )
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

    # Act / Assert
    with pytest.raises(ValueError, match="suite identity"):
        _runner(execute, model_identity_seed="changed-model").run(
            profile_path=profile_path,
            suite="release",
        )
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


def test_unmeasured_phase_interrupt_requires_reset_before_retry(
    tmp_path: Path,
) -> None:
    """計測記録を確定できないphase中断では再利用が拒否されること。

    Arrange:
        - phase executor自体が記録を返す前にinterruptされるsuiteが用意される
    Act:
        - suiteが中断後に同じstateから再開される
    Assert:
        - 初回はexit 130となり、再開には明示resetが要求されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        phase: str,
        _configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(phase)
        raise KeyboardInterrupt

    runner = _runner(execute)

    # Act
    interrupted = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert interrupted == 130
    with pytest.raises(ValueError, match="--reset-suite"):
        runner.run(profile_path=profile_path, suite="release")
    assert calls == ["cold"]


def _runner(
    phase_executor: PhaseExecutor,
    *,
    suite_fingerprint: str = "d" * 64,
    model_identity_seed: str = "acceptance-runner",
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
        environment_probe=lambda: {
            "host_os": "windows_11_pro",
            "environment": "wsl2",
            "gpu": "rtx_5090",
        },
        revision_probe=lambda _path: ("a" * 40, False),
        model_resolver=model_runtime.resolve_models,
        phase_executor=phase_executor,
        release_materializer=materialize,
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
    report: dict[str, object] = {
        "selected": [
            {
                "image_id": candidate_id,
                "output": {"relative_path": "images/0001_gameplay.webp"},
            }
        ]
    }
    configuration.output_folder.mkdir(parents=True, exist_ok=True)
    write_atomic_json(configuration.output_folder / "report.json", report)
    selection_fingerprint = "e" * 64
    artifact: dict[str, object] = {"rejected": []}
    artifact_path = (
        configuration.processing_cache_folder
        / "video-sets"
        / ("f" * 64)
        / "select-images"
        / selection_fingerprint
        / "artifact.json"
    )
    write_atomic_json(artifact_path, artifact)
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
        "gpu_sample_count": 1,
        "gpu_sample_error_count": 0,
        "process_gpu_baseline_mib": 100,
        "system_gpu_baseline_mib": 200,
        "system_global_gpu_peak_mib": 1000,
        "ollama_global_gpu_peak_mib": 1000,
        "stt_global_gpu_peak_mib": 1000,
        "ollama_model_size_vram_bytes": 512,
        "resource_sampling_complete": True,
        "normalized_result_digest": "9" * 64,
        "selection_stage_fingerprint": selection_fingerprint,
        "video_set": {
            "fingerprint": "8" * 64,
            "scenario_count": 1,
            "total_duration_seconds": "1",
        },
        "phase_marker": phase,
    }
    return 0, record, report, artifact


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
        "gpu_sample_count": 1,
        "gpu_sample_error_count": 0,
        "process_gpu_baseline_mib": 100,
        "system_gpu_baseline_mib": 200,
        "system_global_gpu_peak_mib": 1000,
        "ollama_global_gpu_peak_mib": 1000,
        "stt_global_gpu_peak_mib": 1000,
        "ollama_model_size_vram_bytes": 512,
        "resource_sampling_complete": True,
        "phase_marker": phase,
    }


def _profile(tmp_path: Path) -> Path:
    """runner test用のprivate profile/config/sourceを作る。"""
    input_root = tmp_path / "private-input"
    input_root.mkdir()
    (input_root / "source.mkv").write_bytes(b"source")
    configuration = tmp_path / "video-selection.toml"
    configuration.write_text(
        """config_version = "1.0.0"

[ollama]
host = "http://127.0.0.1:11434"
""",
        encoding="utf-8",
    )
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
