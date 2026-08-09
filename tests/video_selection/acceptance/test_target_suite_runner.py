"""durable cold/warm target suite runnerのtest。"""

import hashlib
import shutil
from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path

import pytest

from src.video_selection.acceptance.acceptance_attempt_journal import (
    AcceptanceAttemptJournal,
)
from src.video_selection.acceptance.acceptance_profile import AcceptanceProfile
from src.video_selection.acceptance.acceptance_run import (
    AcceptanceRunAttemptExecutionResult,
    normalized_result_digest,
)
from src.video_selection.acceptance.acceptance_run_reset import AcceptanceRunReset
from src.video_selection.acceptance.atomic_json import (
    read_json_object,
    write_atomic_json,
)
from src.video_selection.acceptance.source_snapshot_fingerprint import (
    acceptance_source_snapshot_fingerprint,
)
from src.video_selection.acceptance.target_suite_runner import (
    EnvironmentProbe,
    ModelResolver,
    OllamaDeploymentProbe,
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
from src.video_selection.services.canonical_output_publisher import (
    CanonicalOutputPublisher,
)
from src.video_selection.services.completed_stage_writer import CompletedStageWriter
from tests.video_selection.fakes.canonical_publication_factory import (
    build_canonical_publication_request,
)
from tests.video_selection.fakes.fake_model_runtime import FakeModelRuntime
from tests.video_selection.fakes.fake_video_stage_media_runtime import (
    FakeVideoStageMediaRuntime,
)


def test_interrupt_after_cold_resumes_only_warm_then_waits_for_human_review(
    tmp_path: Path,
) -> None:
    """cold完了後の中断がwarmだけを再開しworksheet完了までpendingになること。

    Arrange:
        - warm初回だけinterruptするRun Attempt executorとrelease profileが用意される
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
        run_name: str,
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
        calls.append(run_name)
        if run_name == "warm" and not warm_interrupted:
            warm_interrupted = True
            return 130, _interrupted_run_attempt(run_name), None, None
        return _successful_run_attempt(configuration, run_name)

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


def test_interrupted_cold_resumes_after_ollama_runtime_change_without_reset(
    tmp_path: Path,
) -> None:
    """Ollama runtime変更後も未完了coldが明示resetなしで再開されること。

    Arrange:
        - cold初回が中断され、同じmodel digestのOllama runtimeだけが更新される
    Act:
        - 同じsuiteが新しいruntime identityで再開される
    Assert:
        - suite全体のidentity不一致にせずcoldとwarmが実行されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []
    interrupted = False

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        nonlocal interrupted
        calls.append(run_name)
        if run_name == "cold" and not interrupted:
            interrupted = True
            return 130, _interrupted_run_attempt(run_name), None, None
        return _successful_run_attempt(configuration, run_name)

    first_models = FakeModelRuntime("stable-model").resolve_models

    def changed_runtime_models(
        configuration: EffectiveConfiguration,
    ) -> ResolvedModels:
        resolved = FakeModelRuntime("stable-model").resolve_models(configuration)
        return ResolvedModels(
            tuple(
                replace(
                    item,
                    runtime_identity=replace(
                        item.runtime_identity,
                        version="fake-2",
                    ),
                )
                for item in resolved.items
            )
        )

    assert (
        _runner(execute, model_resolver=first_models).run(
            profile_path=profile_path,
            suite="release",
        )
        == 130
    )

    # Act
    resumed = _runner(
        execute,
        model_resolver=changed_runtime_models,
    ).run(
        profile_path=profile_path,
        suite="release",
    )

    # Assert
    assert resumed == 3
    assert calls == ["cold", "cold", "warm"]
    state = read_json_object(
        tmp_path
        / "artifacts"
        / "target-acceptance"
        / "release"
        / "acceptance-state.json"
    )
    assert state is not None
    phases = state["phases"]
    assert isinstance(phases, dict)
    cold = phases["cold"]
    assert isinstance(cold, dict)
    attempts = cold["attempts"]
    assert isinstance(attempts, list)
    assert len(attempts) == 2
    runtime_identities: list[str] = []
    for attempt in attempts:
        assert isinstance(attempt, dict)
        context = attempt["execution_context"]
        assert isinstance(context, dict)
        models = context["models"]
        assert isinstance(models, dict)
        scene_catalog = models["scene_catalog"]
        assert isinstance(scene_catalog, dict)
        runtime_identity = scene_catalog["runtime_identity"]
        assert isinstance(runtime_identity, str)
        runtime_identities.append(runtime_identity)
    assert runtime_identities == ["ollama:fake-1", "ollama:fake-2"]


def test_resume_rechecks_storage_before_starting_remaining_workload(
    tmp_path: Path,
) -> None:
    """中断後の再開前に現在のartifact空き容量が再検査されること。

    Arrange:
        - 初回preflight後にcoldが中断されたrelease suiteが用意される
        - 再開時のpreflightだけが容量不足を返す
    Act:
        - 同じsuiteの再開が試行される
    Assert:
        - 保存済みpreflightを再利用せず容量不足で停止されること
        - 残りのrunが開始されないこと
    """
    # Arrange
    profile_path = _profile(tmp_path)
    run_calls: list[str] = []
    preflight_calls = 0

    def execute(
        run_name: str,
        _configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        run_calls.append(run_name)
        if len(run_calls) == 1:
            return 130, _interrupted_run_attempt(run_name), None, None
        pytest.fail("容量不足を検出した後にrunを開始してはならない")

    def storage_preflight(
        _profile: AcceptanceProfile,
        _input_folder: Path,
    ) -> dict[str, object]:
        nonlocal preflight_calls
        preflight_calls += 1
        if preflight_calls == 2:
            raise ValueError("Acceptance artifact容量が不足しています")
        return {
            "input_video_bytes": 9,
            "input_video_count": 1,
            "artifact_available_bytes": 200 * 1024**3,
            "required_artifact_capacity_bytes": 160 * 1024**3,
            "persistent_cache_budget_bytes": 64 * 1024**3,
            "peak_additional_budget_bytes": 96 * 1024**3,
        }

    runner = _runner(execute, storage_preflight=storage_preflight)
    assert runner.run(profile_path=profile_path, suite="release") == 130

    # Act
    with pytest.raises(ValueError) as error:
        runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert "容量が不足" in str(error.value)
    assert preflight_calls == 2
    assert run_calls == ["cold"]


def test_abandoned_active_phase_is_recovered_without_reset(
    tmp_path: Path,
) -> None:
    """process異常終了でactive phaseが残ってもsuiteが再開されること。

    Arrange:
        - cold中にprocess終了した単一active markerとjournalが永続化される
    Act:
        - `--reset-suite`なしで同じsuiteが再実行される
    Assert:
        - marker名と一致するcoldだけがabandoned attemptとして回復されること
        - coldとwarmがCompleted workから続行されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []
    interrupted = False

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        nonlocal interrupted
        calls.append(run_name)
        if not interrupted:
            interrupted = True
            return 130, _interrupted_run_attempt(run_name), None, None
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(execute)
    assert runner.run(profile_path=profile_path, suite="release") == 130
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    state_path = suite_root / "acceptance-state.json"
    _persist_active_attempt(
        suite_root,
        step_kind="phase",
        step_name="cold",
    )

    # Act
    resumed = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert resumed == 1
    assert calls == ["cold", "cold", "warm"]
    state = read_json_object(state_path)
    assert state is not None
    assert state.get("active_phase") is None
    phases = state["phases"]
    assert isinstance(phases, dict)
    cold = phases["cold"]
    assert isinstance(cold, dict)
    attempts = cold["attempts"]
    assert isinstance(attempts, list)
    abandoned = next(
        attempt
        for attempt in attempts
        if isinstance(attempt, dict)
        and attempt.get("failure_reason") == "process_abandoned"
    )
    assert abandoned["cache_hit_count"] == 2
    assert abandoned["cache_miss_count"] == 1
    assert abandoned["reuse_count"] == 2
    assert abandoned["unexpected_recompute_count"] == 1
    assert abandoned["stage_durations_seconds"] == {"scan-video": 3.0}


@pytest.mark.parametrize(
    (
        "suite",
        "step_kind",
        "step_name",
        "active_key",
        "attempts_key",
        "expected_calls",
    ),
    (
        pytest.param(
            "release",
            "phase",
            "warm",
            "active_phase",
            "phase_attempts",
            ("cold", "warm", "warm"),
            id="warm-phase",
        ),
        pytest.param(
            "full",
            "comparison",
            "fixed3",
            "active_comparison_run",
            "comparison_run_attempts",
            ("fixed3", "fixed3"),
            id="fixed3-comparison",
        ),
    ),
)
def test_abandoned_marker_recovers_only_matching_run(
    tmp_path: Path,
    suite: str,
    step_kind: str,
    step_name: str,
    active_key: str,
    attempts_key: str,
    expected_calls: tuple[str, ...],
) -> None:
    """process異常終了したrunだけがmarker名どおり再開されること。

    Arrange:
        - warmまたはfixed3の単一active markerとjournalが永続化される
    Act:
        - `--reset-suite`なしで同じsuiteが再実行される
    Assert:
        - 他のrunへ戻らず一致するrunだけがabandoned attemptから再開されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        if step_name == "warm" and run_name == "cold":
            return _successful_run_attempt(configuration, run_name)
        return 130, _interrupted_run_attempt(run_name), None, None

    runner = _runner(execute)
    assert runner.run(profile_path=profile_path, suite=suite) == 130
    suite_root = tmp_path / "artifacts" / "target-acceptance" / suite
    _persist_active_attempt(
        suite_root,
        step_kind=step_kind,
        step_name=step_name,
    )

    # Act
    resumed = runner.run(profile_path=profile_path, suite=suite)

    # Assert
    assert resumed == 130
    assert calls == list(expected_calls)
    state = read_json_object(suite_root / "acceptance-state.json")
    assert state is not None
    assert state.get(active_key) is None
    attempts_by_name = state[attempts_key]
    assert isinstance(attempts_by_name, dict)
    attempts = attempts_by_name[step_name]
    assert isinstance(attempts, list)
    assert any(
        isinstance(attempt, dict)
        and attempt.get("failure_reason") == "process_abandoned"
        for attempt in attempts
    )


def test_simultaneous_active_markers_are_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    """phaseとcomparisonの同時active stateが変更されず拒否されること。

    Arrange:
        - fixed3とcoldが同時にactiveな矛盾stateとjournalが永続化される
    Act:
        - `--reset-suite`なしで同じfull suiteが再実行される
    Assert:
        - 複数runの誤回復としてstateとjournalを変更せず拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        run_name: str,
        _configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        return 130, _interrupted_run_attempt(run_name), None, None

    runner = _runner(execute)
    assert runner.run(profile_path=profile_path, suite="full") == 130
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "full"
    state_path = suite_root / "acceptance-state.json"
    journal_path = suite_root / "work" / "active-attempt.json"
    _persist_active_attempt(
        suite_root,
        step_kind="comparison",
        step_name="fixed3",
    )
    state = read_json_object(state_path)
    assert state is not None
    execution_context = state["active_comparison_run_execution_context"]
    state.update(
        {
            "active_phase": "cold",
            "active_phase_started_at_epoch_seconds": 0.0,
            "active_phase_attempt_id": "abandoned-cold",
            "active_phase_execution_context": execution_context,
        }
    )
    write_atomic_json(state_path, state)
    state_before = state_path.read_bytes()
    journal_before = journal_path.read_bytes()

    # Act
    # Assert
    with pytest.raises(ValueError, match="複数のAcceptance Run"):
        runner.run(profile_path=profile_path, suite="full")
    assert calls == ["fixed3"]
    assert state_path.read_bytes() == state_before
    assert journal_path.read_bytes() == journal_before


def test_unknown_active_marker_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    """旧stateの未知active markerがmigrationされず拒否されること。

    Arrange:
        - source fingerprintがない旧stateへ未知phase markerとjournalが永続化される
    Act:
        - `--reset-suite`なしで同じrelease suiteが再実行される
    Assert:
        - stateとjournalを変更せずexecution plan不一致として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        run_name: str,
        _configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        return 130, _interrupted_run_attempt(run_name), None, None

    runner = _runner(execute)
    assert runner.run(profile_path=profile_path, suite="release") == 130
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    state_path = suite_root / "acceptance-state.json"
    journal_path = suite_root / "work" / "active-attempt.json"
    _persist_active_attempt(
        suite_root,
        step_kind="phase",
        step_name="cold",
    )
    state = read_json_object(state_path)
    assert state is not None
    state["active_phase"] = "unknown"
    state.pop("materialization_source_snapshot_fingerprint")
    write_atomic_json(state_path, state)
    state_before = state_path.read_bytes()
    journal_before = journal_path.read_bytes()

    # Act
    # Assert
    with pytest.raises(ValueError, match="execution planと一致しません"):
        runner.run(profile_path=profile_path, suite="release")
    assert calls == ["cold"]
    assert state_path.read_bytes() == state_before
    assert journal_path.read_bytes() == journal_before


def test_completed_release_rejects_out_of_plan_active_comparison_without_mutation(
    tmp_path: Path,
) -> None:
    """完了済みreleaseでも対象外comparison markerが変更されず拒否されること。

    Arrange:
        - coldとwarmが完了したrelease stateへfixed3 markerとjournalが永続化される
    Act:
        - `--reset-suite`なしで同じrelease suiteが再実行される
    Assert:
        - stateとjournalを変更せずexecution plan不一致として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(execute)
    assert runner.run(profile_path=profile_path, suite="release") == 3
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    state_path = suite_root / "acceptance-state.json"
    journal_path = suite_root / "work" / "active-attempt.json"
    state = read_json_object(state_path)
    assert state is not None
    state["active_comparison_run"] = "fixed3"
    write_atomic_json(state_path, state)
    AcceptanceAttemptJournal(journal_path).start(
        attempt_id="abandoned-fixed3",
        step_kind="comparison",
        step_name="fixed3",
        started_at_epoch_seconds=0.0,
        execution_context={},
    )
    state_before = state_path.read_bytes()
    journal_before = journal_path.read_bytes()

    # Act
    # Assert
    with pytest.raises(ValueError, match="execution planと一致しません"):
        runner.run(profile_path=profile_path, suite="release")
    assert calls == ["cold", "warm"]
    assert state_path.read_bytes() == state_before
    assert journal_path.read_bytes() == journal_before


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
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(run_name)
        assert configuration.durable_video_identity_cache_folder == (
            tmp_path / "artifacts" / "target-acceptance" / "video-identities"
        )
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(execute)
    first = runner.run(profile_path=profile_path, suite="release")
    identity_marker = (
        tmp_path
        / "artifacts"
        / "target-acceptance"
        / "video-identities"
        / "identity-marker"
    )
    identity_marker.parent.mkdir(parents=True, exist_ok=True)
    identity_marker.write_text("durable", encoding="utf-8")

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
    assert identity_marker.read_text(encoding="utf-8") == "durable"


@pytest.mark.parametrize(
    ("reset_run", "expected_calls", "processing_cache_preserved"),
    (
        pytest.param(
            "parallelism-baseline",
            ["fixed3", "cold", "warm"],
            False,
            id="parallelism-baseline",
        ),
        pytest.param(
            "fresh-processing",
            ["cold", "warm"],
            False,
            id="fresh-processing",
        ),
        pytest.param(
            "cache-reuse",
            ["warm"],
            True,
            id="cache-reuse",
        ),
    ),
)
def test_reset_run_reexecutes_only_the_requested_dependency_suffix(
    tmp_path: Path,
    reset_run: AcceptanceRunReset,
    expected_calls: list[str],
    processing_cache_preserved: bool,
) -> None:
    """指定runと依存する後続runだけが安全に再実行されること。

    Arrange:
        - 基準測定、本処理、cache再利用が完了したfull suiteが用意される
        - materialized入力、processing cache、Video Identity cacheへ印が置かれる
    Act:
        - 利用者向け名称で一つのrun resetが実行される
    Assert:
        - 依存suffixだけが再実行され、入力とVideo Identityが保持されること
        - cache再利用だけのresetではprocessing cacheも保持されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        result = _successful_run_attempt(configuration, run_name)
        record = result[1]
        fixed_three = configuration.video_scan_workers == 3
        record["video_scan_parallelism"] = {
            "mode": "fixed" if fixed_three else "auto",
            "configured_workers": 3 if fixed_three else "auto",
            "initial_workers": 3,
            "peak_workers": 3 if fixed_three else 6,
            "scan_wall_seconds": 120.0 if fixed_three else 80.0,
        }
        record["stage_artifact_content_digest"] = "9" * 64
        return result

    runner = _runner(execute)
    assert runner.run(profile_path=profile_path, suite="full") == 3
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "full"
    input_marker = suite_root / "work" / "input" / "scenario-001.mkv"
    processing_marker = (
        suite_root
        / "work"
        / "input"
        / ".game-screen-pick"
        / "cache"
        / "processing-marker"
    )
    processing_marker.parent.mkdir(parents=True, exist_ok=True)
    processing_marker.write_text("processing", encoding="utf-8")
    identity_marker = suite_root.parent / "video-identities" / "identity-marker"
    identity_marker.parent.mkdir(parents=True, exist_ok=True)
    identity_marker.write_text("identity", encoding="utf-8")
    calls.clear()

    # Act
    result = runner.run(
        profile_path=profile_path,
        suite="full",
        reset_run=reset_run,
    )

    # Assert
    assert result == 3
    assert calls == expected_calls
    assert input_marker.read_bytes() == b"anonymous"
    assert identity_marker.read_text(encoding="utf-8") == "identity"
    assert processing_marker.exists() is processing_cache_preserved


def test_release_cache_reuse_reset_preserves_fresh_processing_cache(
    tmp_path: Path,
) -> None:
    """Review待ちreleaseで本処理cacheだけを使って再利用測定されること。

    Arrange:
        - 本処理がprocessing cacheへ印を残すrelease suiteが用意される
        - 本処理とcache再利用が完了してhuman review待ちになる
    Act:
        - cache-reuseだけがresetされて同じsuiteが再実行される
    Assert:
        - 本処理を再実行せず、保持されたprocessing cacheで再利用測定されること
        - 合格確定前のprivate workが安全な再開のため保持されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []
    processing_marker: Path | None = None

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        nonlocal processing_marker
        calls.append(run_name)
        marker = configuration.processing_cache_folder / "fresh-processing-marker"
        if run_name == "cold":
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("fresh-processing", encoding="utf-8")
            processing_marker = marker
        else:
            assert marker.read_text(encoding="utf-8") == "fresh-processing"
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(execute)
    assert runner.run(profile_path=profile_path, suite="release") == 3
    assert processing_marker is not None

    # Act
    result = runner.run(
        profile_path=profile_path,
        suite="release",
        reset_run="cache-reuse",
    )

    # Assert
    assert result == 3
    assert calls == ["cold", "warm", "warm"]
    assert processing_marker.read_text(encoding="utf-8") == "fresh-processing"


def test_cache_reuse_reset_rejects_missing_fresh_processing_cache(
    tmp_path: Path,
) -> None:
    """本処理cacheがない再利用resetがrun開始前に拒否されること。

    Arrange:
        - 合格確定時のcleanupまで完了したrelease suiteが用意される
    Act:
        - cache-reuseだけのresetが試行される
    Assert:
        - materializationとrunを再開せずfresh-processing resetが案内されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    materialization_calls: list[str] = []
    runner = _runner(execute, materialization_calls=materialization_calls)
    assert runner.run(profile_path=profile_path, suite="release") == 3
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    worksheet_path = suite_root / "review-worksheet.json"
    _complete_review_worksheet(worksheet_path)
    assert (
        runner.run(
            profile_path=profile_path,
            suite="release",
            human_review_path=worksheet_path,
        )
        == 0
    )
    assert not (suite_root / "work").exists()
    calls.clear()
    materialization_calls.clear()

    # Act
    with pytest.raises(ValueError, match="fresh-processing"):
        runner.run(
            profile_path=profile_path,
            suite="release",
            reset_run="cache-reuse",
        )

    # Assert
    assert calls == []
    assert materialization_calls == []


def test_reset_run_stops_before_state_change_when_suffix_deletion_fails(
    tmp_path: Path,
) -> None:
    """依存suffixを完全に削除できないresetがstateを変更しないこと。

    Arrange:
        - 全run完了済みfull suiteのcache再利用outputが外部symlinkへ置換される
    Act:
        - 本処理からのresetが試行される
    Assert:
        - 外部directoryを削除せず、state変更と追加runを開始しないこと
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        result = _successful_run_attempt(configuration, run_name)
        fixed_three = configuration.video_scan_workers == 3
        result[1]["video_scan_parallelism"] = {
            "mode": "fixed" if fixed_three else "auto",
            "configured_workers": 3 if fixed_three else "auto",
            "initial_workers": 3,
            "peak_workers": 3 if fixed_three else 6,
            "scan_wall_seconds": 120.0 if fixed_three else 80.0,
        }
        result[1]["stage_artifact_content_digest"] = "9" * 64
        return result

    runner = _runner(execute)
    assert runner.run(profile_path=profile_path, suite="full") == 3
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "full"
    state_path = suite_root / "acceptance-state.json"
    state_before = read_json_object(state_path)
    assert state_before is not None
    warm_output = suite_root / "outputs" / "warm"
    shutil.rmtree(warm_output)
    external_output = tmp_path / "external-output"
    external_output.mkdir()
    external_marker = external_output / "keep"
    external_marker.write_text("external", encoding="utf-8")
    warm_output.symlink_to(external_output, target_is_directory=True)
    calls.clear()

    # Act
    with pytest.raises(ValueError, match="symbolic link"):
        runner.run(
            profile_path=profile_path,
            suite="full",
            reset_run="fresh-processing",
        )

    # Assert
    assert read_json_object(state_path) == state_before
    assert external_marker.read_text(encoding="utf-8") == "external"
    assert calls == []


def test_release_rejects_parallelism_baseline_reset(tmp_path: Path) -> None:
    """release suiteに存在しない並列基準resetが拒否されること。

    Arrange:
        - release profileと未呼出のrun executorが用意される
    Act:
        - parallelism baseline reset付きでrelease suiteが実行される
    Assert:
        - run開始前に対象不在として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(execute)

    # Act
    with pytest.raises(ValueError, match="full suite"):
        runner.run(
            profile_path=profile_path,
            suite="release",
            reset_run="parallelism-baseline",
        )

    # Assert
    assert calls == []


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
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(execute)
    assert runner.run(profile_path=profile_path, suite="release") == 3
    monkeypatch.setattr(
        "src.video_selection.acceptance.target_suite_runner.shutil.rmtree",
        lambda _path: None,
    )

    # Act
    # Assert
    with pytest.raises(ValueError, match="完全に削除"):
        runner.run(
            profile_path=profile_path,
            suite="release",
            reset_suite=True,
        )
    assert calls == ["cold", "warm"]


@pytest.mark.parametrize(
    "protected_source",
    ("input_root", "configuration_path", "profile_path"),
)
def test_suite_sources_inside_reset_root_are_rejected_before_deletion(
    tmp_path: Path,
    protected_source: str,
) -> None:
    """suite削除対象内の入力・設定sourceが削除前に拒否されること。

    Arrange:
        - input root、通常設定、private profileのいずれかがsuite root内に置かれる
    Act:
        - reset_suite=trueでrelease suiteが実行される
    Assert:
        - sourceが残ったまま削除対象との重複として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    suite_root.mkdir(parents=True)
    profile_text = profile_path.read_text(encoding="utf-8")
    if protected_source == "input_root":
        protected_path = suite_root / "user-input"
        protected_path.mkdir()
        (protected_path / "source.mkv").write_bytes(b"source")
        old_line = next(
            line for line in profile_text.splitlines() if line.startswith("input_root")
        )
        profile_path.write_text(
            profile_text.replace(
                old_line,
                f'input_root = "{protected_path}"',
                1,
            ),
            encoding="utf-8",
        )
    elif protected_source == "configuration_path":
        protected_path = suite_root / "video-selection.toml"
        protected_path.write_bytes((tmp_path / "video-selection.toml").read_bytes())
        old_line = next(
            line
            for line in profile_text.splitlines()
            if line.startswith("configuration_path")
        )
        profile_path.write_text(
            profile_text.replace(
                old_line,
                f'configuration_path = "{protected_path}"',
                1,
            ),
            encoding="utf-8",
        )
    else:
        protected_path = suite_root / "target.toml"
        protected_path.write_text(profile_text, encoding="utf-8")
        profile_path = protected_path
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    # Act
    # Assert
    with pytest.raises(ValueError, match="suite削除対象"):
        _runner(execute).run(
            profile_path=profile_path,
            suite="release",
            reset_suite=True,
        )
    assert protected_path.exists()
    assert calls == []


def test_release_finalization_fails_when_private_work_survives_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """private workを削除できないrelease finalizationが不合格になること。

    Arrange:
        - cold/warmとhuman reviewが完了するrelease suiteが用意される
        - 削除してもworkを残すfilesystem境界が用意される
    Act:
        - passing recordのfinalizationが実行される
    Assert:
        - cleanup failureとしてexit 1になりpassing recordが生成されないこと
    """
    # Arrange
    profile_path = _profile(tmp_path)
    runner = _runner(
        lambda run_name, configuration, _models, _suite_root: _successful_run_attempt(
            configuration,
            run_name,
        )
    )
    assert runner.run(profile_path=profile_path, suite="release") == 3
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    worksheet_path = suite_root / "review-worksheet.json"
    _complete_review_worksheet(worksheet_path)
    monkeypatch.setattr(
        "src.video_selection.acceptance.target_suite_runner.shutil.rmtree",
        lambda _path, *_args, **_kwargs: None,
    )

    # Act
    result = runner.run(
        profile_path=profile_path,
        suite="release",
        human_review_path=worksheet_path,
    )

    # Assert
    state = read_json_object(suite_root / "acceptance-state.json")
    record = read_json_object(suite_root / "acceptance.json")
    assert result == 1
    assert state is not None
    assert state["acceptance_status"] == "failed"
    assert state["last_failure"] == {
        "phase": "acceptance_cleanup",
        "exit_code": 1,
        "reason": "release_cleanup_failed",
    }
    assert (suite_root / "work").is_dir()
    assert record is not None
    assert record["status"] == "pending_human_review"


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
        lambda run_name, configuration, _models, _suite_root: _successful_run_attempt(
            configuration,
            run_name,
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
        lambda run_name, configuration, _models, _suite_root: _successful_run_attempt(
            configuration,
            run_name,
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
        "video_identity_cache_folder",
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
    assert configuration["video_scan_workers"] == "auto"
    assert configuration["video_scan_auto_max_workers"] == 6
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


def test_full_suite_rejects_auto_cap_that_cannot_exceed_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3以下のauto worker上限がfull run開始前に拒否されること。

    Arrange:
        - auto worker上限が3のfull suite設定が用意される
    Act:
        - full target suiteが実行される
    Assert:
        - fixed3比較を超えられない設定としてrun開始前に拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    monkeypatch.setenv("GAME_SCREEN_PICK_VIDEO_SCAN_AUTO_MAX_WORKERS", "3")
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    # Act
    with pytest.raises(ValueError) as exc_info:
        _runner(execute).run(profile_path=profile_path, suite="full")

    # Assert
    assert "4 worker以上" in str(exc_info.value)
    assert calls == []


def test_full_suite_rejects_cpu_backend_with_three_worker_capacity(
    tmp_path: Path,
) -> None:
    """24 logical CPUで3 workerまでのCPU decode構成が事前拒否されること。

    Arrange:
        - auto上限6でもCPU decodeにより最大3 workerとなるfull suiteが用意される
    Act:
        - full target suiteが実行される
    Assert:
        - 4 workerへ到達不能としてrun開始前に拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    configuration_path = tmp_path / "video-selection.toml"
    configuration_path.write_text(
        configuration_path.read_text(encoding="utf-8").replace(
            'decode_backend = "nvdec"',
            'decode_backend = "cpu"',
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    # Act
    with pytest.raises(ValueError) as exc_info:
        _runner(execute).run(profile_path=profile_path, suite="full")

    # Assert
    assert "4 worker以上" in str(exc_info.value)
    assert calls == []


def test_full_suite_rejects_scenario_count_below_four(
    tmp_path: Path,
) -> None:
    """4未満のVideo数で4 workerへ到達不能なfull構成が事前拒否されること。

    Arrange:
        - NVDECとauto上限6でもVideo数が3のfull suiteが用意される
    Act:
        - full target suiteが実行される
    Assert:
        - 4 workerへ到達不能としてrun開始前に拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    profile_path.write_text(
        profile_path.read_text(encoding="utf-8").replace(
            "expected_video_count = 12",
            "expected_video_count = 3",
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    # Act
    with pytest.raises(ValueError) as exc_info:
        _runner(execute).run(profile_path=profile_path, suite="full")

    # Assert
    assert "4 worker以上" in str(exc_info.value)
    assert calls == []


def test_full_suite_rejects_scenario_count_before_first_reachable_growth(
    tmp_path: Path,
) -> None:
    """初回growthに必要なVideo数を満たさないfull構成が事前拒否されること。

    Arrange:
        - 最大6 workerでもrolling判断後に4 workerを満たせない7動画が用意される
    Act:
        - full target suiteが実行される
    Assert:
        - 長時間のrun開始前に4 workerへ到達不能として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    profile_path.write_text(
        profile_path.read_text(encoding="utf-8").replace(
            "expected_video_count = 12",
            "expected_video_count = 7",
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    # Act
    with pytest.raises(ValueError) as exc_info:
        _runner(execute).run(profile_path=profile_path, suite="full")

    # Assert
    assert "4 worker以上" in str(exc_info.value)
    assert calls == []


def test_full_suite_compares_fixed_three_with_auto_before_warm_run(
    tmp_path: Path,
) -> None:
    """full suiteで固定3とautoのtarget比較が自動判定されること。

    Arrange:
        - 固定3よりautoのVideo Scanが速く同一成果物を返すtargetが用意される
    Act:
        - full target suiteがhuman review待ちまで実行される
    Assert:
        - fixed3、auto cold、warmの順で実行されること
        - fixed3 cacheがauto cold前に削除されること
        - comparisonの全gateがacceptance recordで合格すること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[tuple[str, str | int]] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ]:
        calls.append((run_name, configuration.video_scan_workers))
        fixed_three = configuration.video_scan_workers == 3
        marker = configuration.processing_cache_folder / "fixed3-marker"
        identity_marker = (
            configuration.durable_video_identity_cache_folder / "identity-marker"
        )
        if run_name == "cold" and not fixed_three:
            assert not marker.exists()
            assert identity_marker.read_text(encoding="utf-8") == "durable"
        result = _successful_run_attempt(configuration, run_name)
        record = result[1]
        workers = 3 if fixed_three else 6
        record["video_scan_parallelism"] = {
            "mode": "fixed" if fixed_three else "auto",
            "configured_workers": 3 if fixed_three else "auto",
            "initial_workers": 3,
            "peak_workers": workers,
            "scan_wall_seconds": 120.0 if fixed_three else 80.0,
        }
        record["stage_artifact_content_digest"] = "9" * 64
        if fixed_three:
            marker.write_text("fixed", encoding="utf-8")
            identity_marker.parent.mkdir(parents=True, exist_ok=True)
            identity_marker.write_text("durable", encoding="utf-8")
        return result

    runner = _runner(execute)

    # Act
    result = runner.run(profile_path=profile_path, suite="full")

    # Assert
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "full"
    state = read_json_object(suite_root / "acceptance-state.json")
    record = read_json_object(suite_root / "acceptance.json")
    assert result == 3
    assert calls == [
        ("fixed3", 3),
        ("cold", "auto"),
        ("warm", "auto"),
    ]
    assert state is not None
    assert state["fixed3_cache_released"] is True
    phases = state["phases"]
    assert isinstance(phases, dict)
    assert set(phases) == {"cold", "warm"}
    comparison_runs = state["comparison_runs"]
    assert isinstance(comparison_runs, dict)
    assert set(comparison_runs) == {"fixed3"}
    assert record is not None
    comparison = record["video_scan_parallelism_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["passed"] is True
    gates = record["automatic_gates"]
    assert isinstance(gates, dict)
    assert gates["video_scan_fixed_three_workers"] is True
    assert gates["video_scan_auto_exceeded_three_workers"] is True
    assert gates["video_scan_stage_artifacts_equal"] is True
    assert gates["video_scan_resource_budget"] is True
    assert gates["video_scan_wall_time_improved"] is True


def test_interrupted_auto_cold_preserves_cache_after_fixed_three_release(
    tmp_path: Path,
) -> None:
    """auto cold中断後に比較cache削除が再実行されないこと。

    Arrange:
        - fixed3完了後のauto cold初回だけinterruptするfull suiteが用意される
    Act:
        - full suiteが中断後に再開される
    Assert:
        - auto coldが残したcache markerを再開時に利用できること
        - fixed3は再実行されずcold試行だけが累積されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[tuple[str, str | int]] = []
    cold_interrupted = False

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        nonlocal cold_interrupted
        calls.append((run_name, configuration.video_scan_workers))
        fixed_three = configuration.video_scan_workers == 3
        resume_marker = configuration.processing_cache_folder / "auto-resume-marker"
        if run_name == "cold" and not fixed_three and not cold_interrupted:
            cold_interrupted = True
            resume_marker.parent.mkdir(parents=True, exist_ok=True)
            resume_marker.write_text("preserve", encoding="utf-8")
            interrupted_record = _interrupted_run_attempt(run_name)
            interrupted_record["video_scan_parallelism"] = {
                "mode": "auto",
                "configured_workers": "auto",
                "decode_backend": "nvdec",
                "auto_max_workers": 6,
                "initial_workers": 3,
                "final_workers": 4,
                "peak_workers": 4,
                "completed_scans": 1,
                "scan_wall_seconds": 20.0,
                "changes": [],
            }
            return 130, interrupted_record, None, None
        if run_name == "cold" and not fixed_three:
            assert resume_marker.read_text(encoding="utf-8") == "preserve"
        exit_code, record, report, artifact = _successful_run_attempt(
            configuration,
            run_name,
        )
        workers = 3 if fixed_three else 6
        record["video_scan_parallelism"] = {
            "mode": "fixed" if fixed_three else "auto",
            "configured_workers": 3 if fixed_three else "auto",
            "decode_backend": "nvdec",
            "auto_max_workers": 6,
            "initial_workers": 3,
            "final_workers": workers,
            "peak_workers": workers,
            "completed_scans": 1,
            "scan_wall_seconds": 120.0 if fixed_three else 40.0,
            "changes": [],
        }
        record["stage_artifact_content_digest"] = "8" * 64
        return exit_code, record, report, artifact

    runner = _runner(execute)

    # Act
    interrupted = runner.run(profile_path=profile_path, suite="full")
    resumed = runner.run(profile_path=profile_path, suite="full")

    # Assert
    state = read_json_object(
        tmp_path / "artifacts" / "target-acceptance" / "full" / "acceptance-state.json"
    )
    assert interrupted == 130
    assert resumed == 3
    assert calls == [
        ("fixed3", 3),
        ("cold", "auto"),
        ("cold", "auto"),
        ("warm", "auto"),
    ]
    assert state is not None
    phases = state["phases"]
    assert isinstance(phases, dict)
    cold = phases["cold"]
    assert isinstance(cold, dict)
    assert cold["attempt_count"] == 2
    parallelism = cold["video_scan_parallelism"]
    assert isinstance(parallelism, dict)
    assert parallelism["scan_wall_seconds"] == 60.0
    assert parallelism["attempt_count"] == 2
    assert parallelism["measurement_complete"] is True
    assert state["fixed3_cache_released"] is True


def test_full_suite_remeasures_fixed_three_after_context_change_before_auto_cold(
    tmp_path: Path,
) -> None:
    """auto cold開始前のcontext変更ではfixed3だけが再測定されること。

    Arrange:
        - fixed3だけが完了し、Video Identity cacheが残るfull suiteが用意される
        - auto cold開始前にtarget CPU identityが変更される
    Act:
        - full suiteが現在contextから再開される
    Assert:
        - fixed3だけが現在contextで再測定されてからcold/warmが実行されること
        - durable Video Identity cacheが保持されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[tuple[str, str | int]] = []
    target = {
        "host_os": "windows_11_pro",
        "environment": "wsl2",
        "cpu": "initial",
        "gpu": "rtx_5090",
        "logical_cpu_count": 24,
        "visible_ram_bytes": 32 * 1024**3,
    }

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append((run_name, configuration.video_scan_workers))
        fixed_three = configuration.video_scan_workers == 3
        identity_marker = (
            configuration.durable_video_identity_cache_folder / "identity-marker"
        )
        if fixed_three and calls.count(("fixed3", 3)) == 2:
            assert identity_marker.read_text(encoding="utf-8") == "durable"
        result = _successful_run_attempt(configuration, run_name)
        record = result[1]
        workers = 3 if fixed_three else 6
        record["video_scan_parallelism"] = {
            "mode": "fixed" if fixed_three else "auto",
            "configured_workers": 3 if fixed_three else "auto",
            "initial_workers": 3,
            "peak_workers": workers,
            "scan_wall_seconds": 120.0 if fixed_three else 80.0,
        }
        record["stage_artifact_content_digest"] = "9" * 64
        if fixed_three:
            identity_marker.parent.mkdir(parents=True, exist_ok=True)
            identity_marker.write_text("durable", encoding="utf-8")
        return result

    runner = _runner(execute, environment_probe=lambda: dict(target))
    assert runner.run(profile_path=profile_path, suite="full") == 3
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "full"
    state_path = suite_root / "acceptance-state.json"
    state = read_json_object(state_path)
    assert state is not None
    state["phases"] = {}
    state["worksheet_ready"] = False
    state.pop("review_candidate_digest", None)
    write_atomic_json(state_path, state)
    (suite_root / "review-worksheet.json").unlink()
    for phase in ("cold", "warm"):
        shutil.rmtree(suite_root / "outputs" / phase)
    calls.clear()
    target["cpu"] = "changed"

    # Act
    result = runner.run(profile_path=profile_path, suite="full")

    # Assert
    assert result == 3
    assert calls == [
        ("fixed3", 3),
        ("cold", "auto"),
        ("warm", "auto"),
    ]
    resumed_state = read_json_object(state_path)
    assert resumed_state is not None
    comparison_runs = resumed_state["comparison_runs"]
    assert isinstance(comparison_runs, dict)
    fixed_three = comparison_runs["fixed3"]
    assert isinstance(fixed_three, dict)
    execution_context = fixed_three["execution_context"]
    assert isinstance(execution_context, dict)
    recorded_target = execution_context["target"]
    assert isinstance(recorded_target, dict)
    assert recorded_target["cpu"] == "changed"


def test_full_suite_requires_baseline_reset_after_fresh_processing_started(
    tmp_path: Path,
) -> None:
    """本処理開始後のcontext変更では並列基準からのresetが要求されること。

    Arrange:
        - 並列基準完了後に本処理が中断されたfull suiteが用意される
        - 再開前にtarget CPU identityが変更される
    Act:
        - full suiteの再開が試行される
    Assert:
        - 追加runを開始せずparallelism-baseline resetが要求されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[tuple[str, str | int]] = []
    target = {
        "host_os": "windows_11_pro",
        "environment": "wsl2",
        "cpu": "initial",
        "gpu": "rtx_5090",
        "logical_cpu_count": 24,
        "visible_ram_bytes": 32 * 1024**3,
    }
    cold_interrupted = False

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        nonlocal cold_interrupted
        calls.append((run_name, configuration.video_scan_workers))
        fixed_three = configuration.video_scan_workers == 3
        if run_name == "cold" and not fixed_three and not cold_interrupted:
            cold_interrupted = True
            interrupted = _interrupted_run_attempt(run_name)
            interrupted["video_scan_parallelism"] = {
                "mode": "auto",
                "configured_workers": "auto",
                "decode_backend": "nvdec",
                "auto_max_workers": 6,
                "initial_workers": 3,
                "final_workers": 4,
                "peak_workers": 4,
                "completed_scans": 1,
                "scan_wall_seconds": 20.0,
                "changes": [],
            }
            return 130, interrupted, None, None
        result = _successful_run_attempt(configuration, run_name)
        record = result[1]
        workers = 3 if fixed_three else 6
        record["video_scan_parallelism"] = {
            "mode": "fixed" if fixed_three else "auto",
            "configured_workers": 3 if fixed_three else "auto",
            "decode_backend": "nvdec",
            "auto_max_workers": 6,
            "initial_workers": 3,
            "final_workers": workers,
            "peak_workers": workers,
            "completed_scans": 1,
            "scan_wall_seconds": 120.0 if fixed_three else 80.0,
            "changes": [],
        }
        record["stage_artifact_content_digest"] = "8" * 64
        return result

    runner = _runner(execute, environment_probe=lambda: dict(target))
    interrupted = runner.run(profile_path=profile_path, suite="full")
    target["cpu"] = "changed"

    # Act
    with pytest.raises(ValueError) as error:
        runner.run(profile_path=profile_path, suite="full")

    # Assert
    assert interrupted == 130
    assert "--reset-run parallelism-baseline" in str(error.value)
    assert calls == [("fixed3", 3), ("cold", "auto")]


def test_release_remeasures_fresh_processing_after_context_change_before_reuse(
    tmp_path: Path,
) -> None:
    """cache再利用開始前のcontext変更で本処理だけが再測定されること。

    Arrange:
        - 本処理だけが旧contextで完了したrelease suiteが用意される
        - cache再利用開始前にtarget CPU identityが変更される
    Act:
        - release suiteが現在contextから再開される
    Assert:
        - 旧processing cacheを破棄して本処理、cache再利用の順で実行されること
        - materialized入力とVideo Identity cacheが保持されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []
    target = {
        "host_os": "windows_11_pro",
        "environment": "wsl2",
        "cpu": "initial",
        "gpu": "rtx_5090",
        "logical_cpu_count": 24,
        "visible_ram_bytes": 32 * 1024**3,
    }
    old_cache_marker: Path | None = None
    identity_marker: Path | None = None
    verify_reset = False

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        if verify_reset and calls == ["cold"]:
            assert old_cache_marker is not None
            assert not old_cache_marker.exists()
            assert identity_marker is not None
            assert identity_marker.read_text(encoding="utf-8") == "identity"
            assert (
                configuration.video_input_folder / "scenario-001.mkv"
            ).read_bytes() == b"anonymous"
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(execute, environment_probe=lambda: dict(target))
    assert runner.run(profile_path=profile_path, suite="release") == 3
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    state_path = suite_root / "acceptance-state.json"
    state = read_json_object(state_path)
    assert state is not None
    phases = state["phases"]
    assert isinstance(phases, dict)
    phases.pop("warm")
    state["worksheet_ready"] = False
    state.pop("review_candidate_digest", None)
    write_atomic_json(state_path, state)
    shutil.rmtree(suite_root / "outputs" / "warm")
    (suite_root / "review-worksheet.json").unlink()
    (suite_root / "acceptance.json").unlink()
    input_marker = suite_root / "work" / "input" / "scenario-001.mkv"
    input_marker.parent.mkdir(parents=True, exist_ok=True)
    input_marker.write_bytes(b"anonymous")
    old_cache_marker = (
        input_marker.parent / ".game-screen-pick" / "cache" / "old-context"
    )
    old_cache_marker.parent.mkdir(parents=True, exist_ok=True)
    old_cache_marker.write_text("old", encoding="utf-8")
    identity_marker = suite_root.parent / "video-identities" / "identity-marker"
    identity_marker.parent.mkdir(parents=True, exist_ok=True)
    identity_marker.write_text("identity", encoding="utf-8")
    calls.clear()
    target["cpu"] = "changed"
    verify_reset = True

    # Act
    result = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert result == 3
    assert calls == ["cold", "warm"]
    assert identity_marker.read_text(encoding="utf-8") == "identity"


def test_release_requires_fresh_reset_after_context_change_once_reuse_started(
    tmp_path: Path,
) -> None:
    """cache再利用開始後のcontext変更で追加runが拒否されること。

    Arrange:
        - 本処理完了後にcache再利用attemptが始まったrelease suiteが用意される
        - 再開前にtarget CPU identityが変更される
    Act:
        - release suiteの再開が試行される
    Assert:
        - 追加runを開始せずfresh processing resetが要求されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []
    target = {
        "host_os": "windows_11_pro",
        "environment": "wsl2",
        "cpu": "initial",
        "gpu": "rtx_5090",
        "logical_cpu_count": 24,
        "visible_ram_bytes": 32 * 1024**3,
    }

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(execute, environment_probe=lambda: dict(target))
    assert runner.run(profile_path=profile_path, suite="release") == 3
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    state_path = suite_root / "acceptance-state.json"
    state = read_json_object(state_path)
    assert state is not None
    phases = state["phases"]
    assert isinstance(phases, dict)
    warm = phases.pop("warm")
    assert isinstance(warm, dict)
    warm_attempts = warm["attempts"]
    assert isinstance(warm_attempts, list)
    state["phase_attempts"] = {"warm": warm_attempts}
    state["worksheet_ready"] = False
    state.pop("review_candidate_digest", None)
    write_atomic_json(state_path, state)
    shutil.rmtree(suite_root / "outputs" / "warm")
    (suite_root / "review-worksheet.json").unlink()
    (suite_root / "acceptance.json").unlink()
    calls.clear()
    target["cpu"] = "changed"

    # Act
    with pytest.raises(ValueError) as error:
        runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert "--reset-run fresh-processing" in str(error.value)
    assert calls == []


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
    run_calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        run_calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(
        execute,
        ollama_deployment_probe=lambda _host: probe_results.pop(0),
    )

    # Act
    with pytest.raises(ValueError, match="Windows Ollama binding"):
        runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert run_calls == []


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
        lambda run_name, configuration, _models, _suite_root: _successful_run_attempt(
            configuration,
            run_name,
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

    # Act
    # Assert
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
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

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
        lambda run_name, configuration, _models, _suite_root: _successful_run_attempt(
            configuration,
            run_name,
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

    # Act
    # Assert
    with pytest.raises(ValueError, match="canonical report artifact"):
        runner.run(profile_path=profile_path, suite="release")


def test_resume_rejects_changed_completed_warm_report(tmp_path: Path) -> None:
    """完了warm成果物が再検証されずworksheet生成へ進まないこと。

    Arrange:
        - cold/warm完了後かつworksheet未生成のresume stateが用意される
        - warm reportがphase確定後に変更される
    Act:
        - suiteがworksheet生成から再開される
    Assert:
        - warmのcanonical report artifact不一致として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    runner = _runner(
        lambda run_name, configuration, _models, _suite_root: _successful_run_attempt(
            configuration,
            run_name,
        )
    )
    suite_root, _cold_configuration = _prepare_resume_without_worksheet(
        tmp_path,
        runner,
        profile_path,
    )
    report_path = suite_root / "outputs" / "warm" / "report.json"
    report = read_json_object(report_path)
    assert report is not None
    report["tampered"] = True
    write_atomic_json(report_path, report)

    # Act
    # Assert
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
        lambda run_name, configuration, _models, _suite_root: _successful_run_attempt(
            configuration,
            run_name,
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

    # Act
    # Assert
    with pytest.raises(ValueError, match="canonical report artifact"):
        runner.run(profile_path=profile_path, suite="release")


def test_review_finalization_rejects_changed_completed_cold_markdown(
    tmp_path: Path,
) -> None:
    """worksheet生成後に変更されたMarkdownからreviewが確定されないこと。

    Arrange:
        - cold/warm完了後かつworksheet生成済みのresume stateが用意される
        - cold report.mdがphase確定後に変更される
    Act:
        - human review待ちのsuiteが再開される
    Assert:
        - phase確定時のMarkdown artifact不一致として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    runner = _runner(
        lambda run_name, configuration, _models, _suite_root: _successful_run_attempt(
            configuration,
            run_name,
        )
    )
    assert runner.run(profile_path=profile_path, suite="release") == 3
    markdown_path = (
        tmp_path
        / "artifacts"
        / "target-acceptance"
        / "release"
        / "outputs"
        / "cold"
        / "report.md"
    )
    markdown_path.write_text("tampered report\n", encoding="utf-8")

    # Act
    with pytest.raises(ValueError) as error:
        runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert "Markdown artifact" in str(error.value)


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
        lambda run_name, configuration, _models, _suite_root: _successful_run_attempt(
            configuration,
            run_name,
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

    # Act
    # Assert
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
        lambda run_name, configuration, _models, _suite_root: _successful_run_attempt(
            configuration,
            run_name,
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

    # Act
    # Assert
    with pytest.raises(ValueError, match="integrity"):
        runner.run(profile_path=profile_path, suite="release")


def test_completed_state_revalidates_current_source_snapshot(tmp_path: Path) -> None:
    """完了済みstateでも現在のsource snapshotが軽量に再検証されること。

    Arrange:
        - cold/warm完了後にprivate sourceのsize・mtimeが変化する
    Act:
        - human review待ちのsuiteが再開される
    Assert:
        - materializationとphaseを再実行せずsnapshot不一致として拒否されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    assert _runner(execute).run(profile_path=profile_path, suite="release") == 3
    (tmp_path / "private-input" / "source.mkv").write_bytes(b"changed-source")

    # Act
    # Assert
    with pytest.raises(ValueError) as error:
        _runner(execute).run(
            profile_path=profile_path,
            suite="release",
        )

    assert "source snapshot" in str(error.value)
    assert calls == ["cold", "warm"]


def test_completed_state_keeps_recorded_model_identity_after_runtime_update(
    tmp_path: Path,
) -> None:
    """完了済みstateが現在のmodel更新で無効化されないこと。

    Arrange:
        - cold/warm完了後にmodel resolverのexecution identityが変化する
    Act:
        - human review待ちのsuiteが再開される
    Assert:
        - phaseを再実行せず記録済み成果のhuman review待ちが維持されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    assert _runner(execute).run(profile_path=profile_path, suite="release") == 3

    # Act
    resumed = _runner(execute, model_identity_seed="changed-model").run(
        profile_path=profile_path,
        suite="release",
    )

    # Assert
    assert resumed == 3
    assert calls == ["cold", "warm"]


def test_completed_state_ignores_current_ollama_endpoint_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完了後のOllama endpoint変更が記録済み成果を無効化しないこと。

    Arrange:
        - TOMLにhostを持たずOLLAMA_HOSTで完了したsuiteが用意される
    Act:
        - OLLAMA_HOSTを別endpointへ変えてsuiteが再開される
    Assert:
        - phaseを再実行せずhuman review待ちが維持されること
    """
    # Arrange
    profile_path = _profile(tmp_path, include_ollama_host=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://first.example:11434")
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

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

    # Act
    resumed = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert resumed == 3
    assert calls == ["cold", "warm"]


def test_completed_state_ignores_current_scan_configuration_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完了後の実効scan設定変更が記録済み成果を無効化しないこと。

    Arrange:
        - TOMLにscan上限を持たず環境変数値6で完了したsuiteが用意される
    Act:
        - auto worker上限を5へ変えてsuiteが再開される
    Assert:
        - phaseを再実行せず記録済み設定と成果が維持されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    monkeypatch.setenv("GAME_SCREEN_PICK_VIDEO_SCAN_AUTO_MAX_WORKERS", "6")
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(execute)
    assert runner.run(profile_path=profile_path, suite="release") == 3
    state = read_json_object(
        tmp_path
        / "artifacts"
        / "target-acceptance"
        / "release"
        / "acceptance-state.json"
    )
    monkeypatch.setenv("GAME_SCREEN_PICK_VIDEO_SCAN_AUTO_MAX_WORKERS", "5")

    # Act
    resumed = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert resumed == 3
    assert state is not None
    effective_digest = state["effective_configuration_digest"]
    assert isinstance(effective_digest, str)
    assert len(effective_digest) == 64
    assert calls == ["cold", "warm"]


def test_completed_state_finalization_does_not_resolve_models_again(
    tmp_path: Path,
) -> None:
    """完了済みstateのfinalizationで現在のmodelが再解決されないこと。

    Arrange:
        - 解決回数を記録するModel Resolverでcold/warmが完了される
    Act:
        - cold/warm完了後のhuman review待ちsuiteが再開される
    Assert:
        - model解決とphaseを再実行せずhuman review待ちが維持されること
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
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(execute, model_resolver=resolve)
    assert runner.run(profile_path=profile_path, suite="release") == 3

    # Act
    resumed = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert resumed == 3
    assert resolution_count == 1
    assert calls == ["cold", "warm"]


def test_completed_release_state_does_not_rematerialize_retained_private_work(
    tmp_path: Path,
) -> None:
    """Review待ちrelease stateの再確認でprivate workが再生成されないこと。

    Arrange:
        - cold/warm完了とworksheet生成後にprivate workを保持するsuiteが用意される
    Act:
        - human review待ちの同じrelease suiteが再実行される
    Assert:
        - 完了stateと公開成果物からfinalizationだけが再開されること
        - 保持済みrelease materializationが再実行されないこと
    """
    # Arrange
    profile_path = _profile(tmp_path)
    phase_calls: list[str] = []
    materialization_calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        phase_calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(
        execute,
        materialization_calls=materialization_calls,
    )
    assert runner.run(profile_path=profile_path, suite="release") == 3
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    assert (suite_root / "work").is_dir()

    # Act
    resumed = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert resumed == 3
    assert phase_calls == ["cold", "warm"]
    assert materialization_calls == ["release"]
    assert (suite_root / "work").is_dir()


def test_completed_state_keeps_recorded_target_after_environment_change(
    tmp_path: Path,
) -> None:
    """完了後のtarget環境変更が記録済み成果を無効化しないこと。

    Arrange:
        - cold/warm完了後にdriver identityが変わるtarget probeが用意される
    Act:
        - 同じprofileとsourceでsuiteが再開される
    Assert:
        - phaseを再実行せず記録済みtargetのhuman review待ちが維持されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []
    target = {
        "os": "linux",
        "gpu_driver": "first",
        "visible_ram_bytes": 32 * 1024**3,
    }

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(execute, environment_probe=lambda: dict(target))
    assert runner.run(profile_path=profile_path, suite="release") == 3
    target["gpu_driver"] = "changed"

    # Act
    resumed = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert resumed == 3
    assert calls == ["cold", "warm"]


@pytest.mark.parametrize(
    "delta_bytes",
    (-12 * 1024, -4 * 1024, -(1024**2), 1024**2),
)
def test_completed_state_accepts_boot_level_visible_ram_variation(
    tmp_path: Path,
    delta_bytes: int,
) -> None:
    """起動単位の微小なvisible RAM差で完了stateが再利用されること。

    Arrange:
        - 実測visible RAMを持つcold/warm完了stateが用意される
    Act:
        - 現在値だけが1 MiB以内で変動したtargetからsuiteが再開される
    Assert:
        - 初回実測値がstateに保持され、phase再実行なしで再開されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []
    initial_ram_bytes = 32 * 1024**3
    target = {
        "os": "linux",
        "gpu_driver": "stable",
        "visible_ram_bytes": initial_ram_bytes,
    }

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(execute, environment_probe=lambda: dict(target))
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
    stored_target = state["target"]
    assert isinstance(stored_target, dict)
    assert stored_target["visible_ram_bytes"] == initial_ram_bytes
    target["visible_ram_bytes"] = initial_ram_bytes + delta_bytes

    # Act
    resumed = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert resumed == 3
    assert calls == ["cold", "warm"]
    resumed_state = read_json_object(state_path)
    assert resumed_state is not None
    resumed_target = resumed_state["target"]
    assert isinstance(resumed_target, dict)
    assert resumed_target["visible_ram_bytes"] == initial_ram_bytes


@pytest.mark.parametrize(
    "delta_bytes",
    (-(1024**2) - 1, 1024**2 + 1),
)
def test_completed_state_keeps_recorded_visible_ram_after_large_change(
    tmp_path: Path,
    delta_bytes: int,
) -> None:
    """完了後の大きなvisible RAM差が記録済み成果を無効化しないこと。

    Arrange:
        - 実測visible RAMを持つcold/warm完了stateが用意される
    Act:
        - 現在値が1 MiBを超えて変わったtargetからsuiteが再開される
    Assert:
        - phaseを再実行せず記録済みtargetが維持されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []
    initial_ram_bytes = 32 * 1024**3
    target = {
        "os": "linux",
        "gpu_driver": "stable",
        "visible_ram_bytes": initial_ram_bytes,
    }

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(execute, environment_probe=lambda: dict(target))
    assert runner.run(profile_path=profile_path, suite="release") == 3
    target["visible_ram_bytes"] = initial_ram_bytes + delta_bytes

    # Act
    resumed = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert resumed == 3
    assert calls == ["cold", "warm"]


@pytest.mark.parametrize(
    "invalid_value",
    (None, True, "34359738368", "missing"),
)
def test_completed_state_ignores_invalid_current_visible_ram(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    """完了後の現在RAM probe異常が記録済み成果を無効化しないこと。

    Arrange:
        - 正の整数のvisible RAMを持つcold/warm完了stateが用意される
    Act:
        - 現在targetのRAM fieldが欠落または不正型へ変更される
    Assert:
        - phaseを再実行せず記録済みtargetが維持されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    calls: list[str] = []
    target: dict[str, object] = {
        "os": "linux",
        "gpu_driver": "stable",
        "visible_ram_bytes": 32 * 1024**3,
    }

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(execute, environment_probe=lambda: dict(target))
    assert runner.run(profile_path=profile_path, suite="release") == 3
    if invalid_value == "missing":
        target.pop("visible_ram_bytes")
    else:
        target["visible_ram_bytes"] = invalid_value

    # Act
    resumed = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert resumed == 3
    assert calls == ["cold", "warm"]


def test_incomplete_phase_removes_invalid_output_before_rerun(
    tmp_path: Path,
) -> None:
    """phase state未確定の不正outputが再実行前に削除されること。

    Arrange:
        - cold phase recordなしで不完全なoutputだけが残ったsuiteが用意される
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
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> tuple[
        int,
        dict[str, object],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        calls.append(run_name)
        assert not (configuration.output_folder / "stale.json").exists()
        return _successful_run_attempt(configuration, run_name)

    runner = _runner(execute)

    # Act
    result = runner.run(profile_path=profile_path, suite="release")

    # Assert
    assert result == 3
    assert calls == ["cold", "warm"]


def test_completed_publication_survives_missing_phase_record(
    tmp_path: Path,
) -> None:
    """phase記録前に終了しても検証済みCanonical outputが保持されること。

    Arrange:
        - cold phase stateを持たずatomic publicationだけが完了したsuiteが用意される
    Act:
        - release suiteが同じcold Output Folderから再開される
    Assert:
        - executor開始時に完成済みreportとMarkdownが保持されていること
        - coldとwarmがそのまま完了されること
    """
    # Arrange
    profile_path = _profile(tmp_path)
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    output_folder = suite_root / "outputs" / "cold"
    publication_root = tmp_path / "completed-publication"
    publication_root.mkdir()
    request = build_canonical_publication_request(publication_root)
    request = replace(
        request,
        configuration=replace(
            request.configuration,
            output_folder=output_folder,
        ),
    )
    CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(request)
    calls: list[str] = []

    def execute(
        run_name: str,
        configuration: EffectiveConfiguration,
        _models: ResolvedModels,
        _suite_root: Path,
    ) -> AcceptanceRunAttemptExecutionResult:
        calls.append(run_name)
        if run_name == "cold":
            assert (configuration.output_folder / "report.json").is_file()
            assert (configuration.output_folder / "report.md").is_file()
        return _successful_run_attempt(configuration, run_name)

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
        - Run Attempt executorが初回だけ記録を返す前にinterruptされるsuiteが用意される
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
        run_name: str,
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
        calls.append(run_name)
        if not interrupted_once:
            interrupted_once = True
            raise KeyboardInterrupt
        return _successful_run_attempt(configuration, run_name)

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
        lambda run_name, configuration, _models, _suite_root: _successful_run_attempt(
            configuration,
            run_name,
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


@pytest.mark.parametrize(
    "failure",
    (OSError("baseline disk failure"), KeyboardInterrupt()),
)
def test_baseline_failure_cannot_commit_passed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    """baseline生成失敗時にpassed stateが先行確定されないこと。

    Arrange:
        - cold/warmとpassing human reviewが完了したrelease suiteが用意される
        - baseline生成境界でIO失敗またはuser interruptが発生する
    Act:
        - passing recordのfinalizationが実行される
    Assert:
        - stateがfailedとなりpassed recordとbaselineが公開されないこと
    """
    # Arrange
    profile_path = _profile(tmp_path)
    runner = _runner(
        lambda run_name, configuration, _models, _suite_root: _successful_run_attempt(
            configuration, run_name
        )
    )
    assert runner.run(profile_path=profile_path, suite="release") == 3
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

    def fail_baseline(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(
        "src.video_selection.acceptance.target_suite_runner.write_normalized_baseline",
        fail_baseline,
    )

    # Act
    # Assert
    with pytest.raises(type(failure)):
        runner.run(profile_path=profile_path, suite="release")
    state = read_json_object(suite_root / "acceptance-state.json")
    record = read_json_object(suite_root / "acceptance.json")
    assert state is not None
    assert state["acceptance_status"] == "failed"
    last_failure = state["last_failure"]
    assert isinstance(last_failure, dict)
    assert last_failure["reason"] == "acceptance_finalization_failed"
    assert record is not None
    assert record["status"] == "pending_human_review"
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
        lambda run_name, configuration, _models, _suite_root: _successful_run_attempt(
            configuration,
            run_name,
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

    # Act
    # Assert
    with pytest.raises(ValueError, match="candidate集合"):
        runner.run(
            profile_path=profile_path,
            suite="release",
            human_review_path=invalid_path,
        )
    assert baseline_path.read_bytes() == trusted_baseline


def _runner(
    labeled_run_attempt_executor: Callable[
        [str, EffectiveConfiguration, ResolvedModels, Path],
        AcceptanceRunAttemptExecutionResult,
    ],
    *,
    suite_fingerprint: str = "d" * 64,
    model_identity_seed: str = "acceptance-runner",
    model_resolver: ModelResolver | None = None,
    environment_probe: EnvironmentProbe | None = None,
    ollama_deployment_probe: OllamaDeploymentProbe | None = None,
    materialization_calls: list[str] | None = None,
    storage_preflight: Callable[
        [AcceptanceProfile, Path],
        dict[str, object],
    ]
    | None = None,
) -> TargetSuiteRunner:
    """target外でもstate machineを検証できるdependency構成を返す。"""
    model_runtime = FakeModelRuntime(model_identity_seed)

    def materialize(
        profile: AcceptanceProfile,
        suite_root: Path,
    ) -> tuple[Path, dict[str, object]]:
        if materialization_calls is not None:
            materialization_calls.append(suite_root.name)
        input_folder = suite_root / "work" / "input"
        input_folder.mkdir(parents=True, exist_ok=True)
        (input_folder / "scenario-001.mkv").write_bytes(b"anonymous")
        return input_folder, {
            "suite_fingerprint": suite_fingerprint,
            "source_snapshot_fingerprint": (
                acceptance_source_snapshot_fingerprint(
                    profile,
                    suite_root.name,
                )
            ),
            "scenario_count": profile.full_expected_video_count,
        }

    return TargetSuiteRunner(
        environment_probe=environment_probe
        or (
            lambda: {
                "host_os": "windows_11_pro",
                "environment": "wsl2",
                "gpu": "rtx_5090",
                "logical_cpu_count": 24,
                "visible_ram_bytes": 32 * 1024**3,
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
        run_attempt_executor=(
            lambda configuration, models, suite_root: labeled_run_attempt_executor(
                configuration.output_folder.name,
                configuration,
                models,
                suite_root,
            )
        ),
        release_materializer=materialize,
        full_materializer=materialize,
        storage_preflight=storage_preflight
        or (
            lambda _profile, _input_folder: {
                "input_video_bytes": 9,
                "input_video_count": 1,
                "artifact_available_bytes": 200 * 1024**3,
                "required_artifact_capacity_bytes": 160 * 1024**3,
                "persistent_cache_budget_bytes": 64 * 1024**3,
                "peak_additional_budget_bytes": 96 * 1024**3,
            }
        ),
    )


def _persist_active_attempt(
    suite_root: Path,
    *,
    step_kind: str,
    step_name: str,
) -> None:
    """計測済みattemptに続くprocess異常終了状態を永続化する。"""
    state_path = suite_root / "acceptance-state.json"
    state = read_json_object(state_path)
    assert state is not None
    attempts_key = (
        "phase_attempts" if step_kind == "phase" else "comparison_run_attempts"
    )
    attempts_by_name = state[attempts_key]
    assert isinstance(attempts_by_name, dict)
    attempts = attempts_by_name[step_name]
    assert isinstance(attempts, list)
    prior_attempt = attempts[-1]
    assert isinstance(prior_attempt, dict)
    execution_context = prior_attempt["execution_context"]
    assert isinstance(execution_context, dict)
    active_key = "active_phase" if step_kind == "phase" else "active_comparison_run"
    attempt_id = f"abandoned-{step_name}"
    state.update(
        {
            active_key: step_name,
            f"{active_key}_started_at_epoch_seconds": 0.0,
            f"{active_key}_attempt_id": attempt_id,
            f"{active_key}_execution_context": execution_context,
        }
    )
    write_atomic_json(state_path, state)
    journal = AcceptanceAttemptJournal(suite_root / "work" / "active-attempt.json")
    journal.start(
        attempt_id=attempt_id,
        step_kind=step_kind,
        step_name=step_name,
        started_at_epoch_seconds=0.0,
        execution_context=execution_context,
    )
    journal.record_snapshot(
        {
            "cache_hit_count": 2,
            "cache_miss_count": 1,
            "reuse_count": 2,
            "unexpected_recompute_count": 1,
            "stage_durations_seconds": {"scan-video": 3.0},
            "completed_stage_counts": {"scan-video": 1},
        },
        {"1" * 64: "recomputed"},
    )


def _successful_run_attempt(
    configuration: EffectiveConfiguration,
    run_name: str,
) -> tuple[
    int,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """durable output/cacheも作る成功Run Attempt evidenceを返す。"""
    candidate_id = "frm_" + "1" * 64
    image_bytes = b"selected-webp"
    image_relative_path = "images/0001_gameplay.webp"
    image_digest = hashlib.sha256(image_bytes).hexdigest()
    report: dict[str, object] = {
        "run": {
            "id": f"run_{run_name}",
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
    markdown_path = configuration.output_folder / "report.md"
    markdown_path.write_text("canonical report\n", encoding="utf-8")
    with markdown_path.open("rb") as file:
        markdown_digest = hashlib.file_digest(file, "sha256").hexdigest()
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
        "canonical_markdown_sha256": markdown_digest,
        "normalized_result_digest": normalized_result_digest(report),
        "selection_stage_fingerprint": selection_fingerprint.value,
        "video_set": {
            "fingerprint": "f" * 64,
            "scenario_count": 1,
            "total_duration_seconds": "1",
        },
        "run_name_marker": run_name,
    }
    return 0, record, report, artifact


def _prepare_resume_without_worksheet(
    tmp_path: Path,
    runner: TargetSuiteRunner,
    profile_path: Path,
) -> tuple[Path, EffectiveConfiguration]:
    """完了Phase evidenceを復元しworksheet直前のresume stateを返す。"""
    assert runner.run(profile_path=profile_path, suite="release") == 3
    suite_root = tmp_path / "artifacts" / "target-acceptance" / "release"
    state_path = suite_root / "acceptance-state.json"
    state = read_json_object(state_path)
    assert state is not None
    state["worksheet_ready"] = False
    write_atomic_json(state_path, state)
    (suite_root / "review-worksheet.json").unlink()
    input_folder = suite_root / "work" / "input"
    input_folder.mkdir(parents=True, exist_ok=True)
    (input_folder / "scenario-001.mkv").write_bytes(b"anonymous")
    cold_configuration = resolve_effective_configuration(
        video_input_folder=input_folder,
        output_folder=suite_root / "outputs" / "cold",
        config_path=tmp_path / "video-selection.toml",
        environ={},
    )
    _successful_run_attempt(cold_configuration, "cold")
    return suite_root, cold_configuration


def _complete_review_worksheet(worksheet_path: Path) -> None:
    """Test用worksheetを全gate合格として完了する。"""
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


def _interrupted_run_attempt(run_name: str) -> dict[str, object]:
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
        "run_name_marker": run_name,
    }


def _profile(tmp_path: Path, *, include_ollama_host: bool = True) -> Path:
    """runner test用のprivate profile/config/sourceを作る。"""
    input_root = tmp_path / "private-input"
    input_root.mkdir()
    (input_root / "source.mkv").write_bytes(b"source")
    configuration = tmp_path / "video-selection.toml"
    configuration_text = (
        'config_version = "1.0.0"\n\n[frame_extraction]\ndecode_backend = "nvdec"\n'
    )
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
expected_video_count = 12
expected_total_duration = "PT1S"
duration_tolerance_seconds = 0
''',
        encoding="utf-8",
    )
    return path
