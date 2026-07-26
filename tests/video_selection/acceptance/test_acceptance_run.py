"""Acceptance Run Attempt実行境界のtest。"""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.video_selection.acceptance.acceptance_run import (
    execute_acceptance_run_attempt,
    normalized_result_digest,
    video_scan_parallelism_diagnostics,
)
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.run_failure import RunFailure
from tests.video_selection.fakes.fake_model_runtime import FakeModelRuntime


def test_normalized_result_digest_includes_published_image_identity() -> None:
    """公開画像のhashまたは寸法が異なるcold/warm digestが一致しないこと。

    Arrange:
        - 同じ選定結果と異なる画像hashまたは寸法を持つcanonical reportが用意される
    Act:
        - 各reportのnormalized result digestが生成される
    Assert:
        - 公開画像bytesまたは寸法が異なるdigestはcold digestと一致しないこと
    """
    # Arrange
    cold_report = _canonical_report(
        sha256="a" * 64,
        width=1920,
        height=1080,
        size_bytes=1000,
    )
    changed_outputs = (
        _canonical_report(
            sha256="b" * 64,
            width=1920,
            height=1080,
            size_bytes=1000,
        ),
        _canonical_report(
            sha256="a" * 64,
            width=1280,
            height=720,
            size_bytes=900,
        ),
    )

    # Act
    cold_digest = normalized_result_digest(cold_report)
    changed_digests = tuple(
        normalized_result_digest(report) for report in changed_outputs
    )

    # Assert
    assert all(digest != cold_digest for digest in changed_digests)


def test_normalized_result_digest_includes_all_semantic_report_results() -> None:
    """利用者に見えるcanonical report結果の差がdigestへ含まれること。

    Arrange:
        - 同じ画像と、意味的sectionだけが異なる複数reportが用意される
    Act:
        - 各reportのnormalized result digestが生成される
    Assert:
        - run結果、選定集計、棄却、near miss、Cue、modelの差が検出されること
    """
    # Arrange
    report = _canonical_report(
        sha256="a" * 64,
        width=1920,
        height=1080,
        size_bytes=1000,
    )
    changed_reports: list[dict[str, object]] = []
    replacements: tuple[tuple[str, object], ...] = (
        (
            "run",
            {
                "id": "run_changed",
                "status": "completed_with_warnings",
                "started_at": "2026-07-26T00:00:00Z",
                "completed_at": "2026-07-26T00:01:00Z",
                "requested_image_count": 1,
                "selected_image_count": 1,
                "warnings": [{"code": "selection_shortfall"}],
            },
        ),
        ("selection_summary", {"selected": 2}),
        ("rejection_summary", {"total": 1, "by_reason": {"low_quality": 1}}),
        ("near_misses", [{"image_id": "frm_" + "2" * 64}]),
        ("context_cues", [{"id": "cue_" + "3" * 64}]),
    )
    for section, replacement in replacements:
        changed = copy.deepcopy(report)
        changed[section] = replacement
        changed_reports.append(changed)
    changed_provenance = copy.deepcopy(report)
    provenance = changed_provenance["provenance"]
    assert isinstance(provenance, dict)
    provenance["models"] = {"scene_catalog": {"execution_identity": "changed"}}
    changed_reports.append(changed_provenance)

    # Act
    digest = normalized_result_digest(report)
    changed_digests = tuple(
        normalized_result_digest(changed) for changed in changed_reports
    )

    # Assert
    assert all(changed_digest != digest for changed_digest in changed_digests)


def test_normalized_result_digest_excludes_run_specific_diagnostics() -> None:
    """run固有identityと性能診断の差がwarm結果差にされないこと。

    Arrange:
        - semantic resultが同じでrun ID、timestamp、Stage診断だけが異なる
          reportが用意される
    Act:
        - cold/warm相当のnormalized result digestが生成される
    Assert:
        - 両digestが一致すること
    """
    # Arrange
    cold = _canonical_report(
        sha256="a" * 64,
        width=1920,
        height=1080,
        size_bytes=1000,
    )
    warm = copy.deepcopy(cold)
    run = warm["run"]
    assert isinstance(run, dict)
    run["id"] = "run_warm"
    run["started_at"] = "2026-07-26T01:00:00Z"
    run["completed_at"] = "2026-07-26T01:00:01Z"
    provenance = warm["provenance"]
    assert isinstance(provenance, dict)
    cold_provenance = cold["provenance"]
    assert isinstance(cold_provenance, dict)
    cold_runtime = cold_provenance["runtime"]
    warm_runtime = provenance["runtime"]
    assert isinstance(cold_runtime, dict)
    assert isinstance(warm_runtime, dict)
    cold_runtime["video_scan_parallelism"] = {
        "initial_workers": 6,
        "final_workers": 5,
        "changes": [{"reason": "cpu_pressure"}],
    }
    warm_runtime["video_scan_parallelism"] = {
        "initial_workers": 6,
        "final_workers": 6,
        "changes": [],
    }
    models = provenance["models"]
    assert isinstance(models, dict)
    scene_catalog = models["scene_catalog"]
    assert isinstance(scene_catalog, dict)
    scene_catalog["local_identity_before_update"] = "ollama:sha256:" + "a" * 64
    scene_catalog["update_status"] = "unchanged"
    stages = provenance["stages"]
    assert isinstance(stages, list)
    stage = stages[0]
    assert isinstance(stage, dict)
    stage.update(
        {
            "attempt_count": 2,
            "cache_hits": 1,
            "cache_misses": 0,
            "duration_ms": 1,
            "eval_tokens": 0,
            "prompt_eval_tokens": 0,
            "recomputed_items": 0,
            "validation_failures": 0,
        }
    )

    # Act
    cold_digest = normalized_result_digest(cold)
    warm_digest = normalized_result_digest(warm)

    # Assert
    assert warm_digest == cold_digest


def test_parallelism_evidence_is_extracted_from_run_metrics() -> None:
    """worker診断がcanonical reportから抽出されること。

    Arrange:
        - Video Scan parallelism診断を持つreportが用意される
    Act:
        - parallelism診断が抽出される
    Assert:
        - 診断値が保持されること
    """
    # Arrange
    report = _canonical_report(
        sha256="a" * 64,
        width=1920,
        height=1080,
        size_bytes=1000,
    )
    provenance = report["provenance"]
    assert isinstance(provenance, dict)
    runtime = provenance["runtime"]
    assert isinstance(runtime, dict)
    runtime["video_scan_parallelism"] = {
        "mode": "auto",
        "configured_workers": "auto",
        "initial_workers": 6,
        "peak_workers": 6,
        "scan_wall_seconds": 80.0,
    }
    # Act
    diagnostics = video_scan_parallelism_diagnostics(report)

    # Assert
    assert diagnostics["initial_workers"] == 6
    assert diagnostics["scan_wall_seconds"] == 80.0


@pytest.mark.parametrize("identity_key", ("execution_identity", "runtime_identity"))
def test_normalized_result_digest_retains_model_execution_and_runtime_identity(
    identity_key: str,
) -> None:
    """model executionまたはruntime identityの差がdigestへ保持されること。

    Arrange:
        - 同じ結果と一つだけ異なるmodel execution/runtime identityを持つ
          reportが用意される
    Act:
        - cold/changed reportのnormalized result digestが生成される
    Assert:
        - cache意味を変えるmodel identity差が一致扱いされないこと
    """
    # Arrange
    cold = _canonical_report(
        sha256="a" * 64,
        width=1920,
        height=1080,
        size_bytes=1000,
    )
    changed = copy.deepcopy(cold)
    provenance = changed["provenance"]
    assert isinstance(provenance, dict)
    models = provenance["models"]
    assert isinstance(models, dict)
    scene_catalog = models["scene_catalog"]
    assert isinstance(scene_catalog, dict)
    scene_catalog[identity_key] = "changed"

    # Act
    cold_digest = normalized_result_digest(cold)
    changed_digest = normalized_result_digest(changed)

    # Assert
    assert changed_digest != cold_digest


def test_run_attempt_duration_excludes_resource_monitor_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run durationがresource monitor停止時間を含まず記録されること。

    Arrange:
        - pipeline完了時刻を20、monitor停止後時刻を100とするclockが用意される
    Act:
        - operation failureになるAcceptance Run Attemptが実行される
    Assert:
        - 開始10からpipeline完了20までの10秒だけが記録されること
    """
    # Arrange
    now = [10.0]
    configuration = EffectiveConfiguration(
        video_input_folder=tmp_path / "input",
        output_folder=tmp_path / "output",
    )
    resolved_models = FakeModelRuntime("run-attempt-timing").resolve_models(
        configuration
    )
    failure = RunFailure(
        reason_code="test_failure",
        exit_code=1,
        remediation_code="retry",
        resume_guidance="completed_stages_reusable",
    )

    def execute(_self: object, _operation: object) -> tuple[int, RunFailure]:
        now[0] = 20.0
        return 1, failure

    def stop_disk_monitor() -> dict[str, object]:
        now[0] = 100.0
        return {
            "disk_sampling_complete": True,
            "persistent_cache_bytes": 0,
            "peak_additional_bytes": 0,
            "disk_sample_count": 1,
            "disk_sample_error_count": 0,
        }

    monkeypatch.setattr(
        "src.video_selection.acceptance.acceptance_run.time.monotonic",
        lambda: now[0],
    )
    monkeypatch.setattr(
        "src.video_selection.acceptance.acceptance_run.build_real_application",
        lambda *_args, **_kwargs: SimpleNamespace(
            run=lambda _configuration: None,
            video_scan_parallelism_diagnostics={
                "mode": "auto",
                "configured_workers": "auto",
                "initial_workers": 3,
                "final_workers": 4,
                "peak_workers": 4,
                "completed_scans": 2,
                "scan_wall_seconds": 8.0,
                "changes": [],
            },
        ),
    )
    monkeypatch.setattr(
        "src.video_selection.acceptance.acceptance_run.InternalRunController.execute",
        execute,
    )
    monkeypatch.setattr(
        "src.video_selection.acceptance.acceptance_run.DiskUsageMonitor",
        lambda **_kwargs: SimpleNamespace(
            start=lambda: None,
            stop=stop_disk_monitor,
        ),
    )
    monkeypatch.setattr(
        "src.video_selection.acceptance.acceptance_run.GpuResourceMonitor",
        lambda **_kwargs: SimpleNamespace(
            start=lambda: None,
            stop=lambda: {
                "resource_sampling_complete": True,
                "gpu_sample_count": 1,
                "gpu_sample_error_count": 0,
                "process_gpu_baseline_mib": 0,
                "system_gpu_baseline_mib": 0,
                "system_global_gpu_peak_mib": 0,
                "ollama_global_gpu_peak_mib": 0,
                "stt_non_ollama_gpu_peak_mib": 0,
                "ollama_model_size_bytes": 0,
                "ollama_model_size_vram_bytes": 0,
                "ollama_model_observed": False,
                "ollama_model_fully_resident": False,
            },
        ),
    )

    # Act
    exit_code, attempt_record, report, selection_artifact = (
        execute_acceptance_run_attempt(
            configuration=configuration,
            resolved_models=resolved_models,
            suite_root=tmp_path / "suite",
        )
    )

    # Assert
    assert exit_code == 1
    assert attempt_record["duration_seconds"] == 10.0
    assert attempt_record["resource_sampling_complete"] is True
    assert attempt_record["video_scan_parallelism"] == {
        "mode": "auto",
        "configured_workers": "auto",
        "initial_workers": 3,
        "final_workers": 4,
        "peak_workers": 4,
        "completed_scans": 2,
        "scan_wall_seconds": 8.0,
        "changes": [],
    }
    assert report is None
    assert selection_artifact is None
    assert now[0] == 100.0


def _canonical_report(
    *,
    sha256: str,
    width: int,
    height: int,
    size_bytes: int,
) -> dict[str, object]:
    """semantic digest境界を持つcanonical reportを返す。"""
    return {
        "schema": {"name": "game-screen-pick/report", "version": "1.0.0"},
        "run": {
            "id": "run_cold",
            "status": "completed",
            "started_at": "2026-07-26T00:00:00Z",
            "completed_at": "2026-07-26T00:01:00Z",
            "requested_image_count": 1,
            "selected_image_count": 1,
            "warnings": [],
        },
        "artifacts": {"report_json": "report.json"},
        "video_set": {"id": "vset_test"},
        "selection_summary": {"selected": 1},
        "rejection_summary": {"total": 0, "by_reason": {}},
        "selected": [
            {
                "image_id": "frm_" + "1" * 64,
                "selection_index": 1,
                "classification": {"blog_image_type": "normal_gameplay"},
                "annotation": {"summary": "探索"},
                "selection": {"marginal_utility": 0.9},
                "output": {
                    "relative_path": "images/0001_exploration.webp",
                    "sha256": sha256,
                    "width": width,
                    "height": height,
                    "bytes": size_bytes,
                },
            }
        ],
        "near_miss_publication": {"json_limit_for_this_run": 0},
        "near_misses": [],
        "context_cues": [],
        "provenance": {
            "selection": {"policy_version": "video-set-selection-v2"},
            "runtime": {"environment": "wsl2"},
            "tools": {"ffmpeg": "test"},
            "models": {
                "scene_catalog": {
                    "store": "ollama",
                    "configured_name": "vision:latest",
                    "canonical_name": "vision:latest",
                    "local_identity_before_update": None,
                    "update_status": "bootstrapped",
                    "execution_identity": "ollama:sha256:" + "a" * 64,
                    "runtime_identity": "ollama:0.31.2",
                }
            },
            "contracts": {"report_schema": "1.0.0"},
            "stages": [
                {
                    "name": "final_selection",
                    "status": "completed",
                    "fingerprint": "stg_" + "4" * 64,
                    "upstream_fingerprints": [],
                    "cache_hits": 0,
                    "cache_misses": 1,
                    "recomputed_items": 1,
                    "attempt_count": 1,
                    "validation_failures": 0,
                    "effective_settings": {"requested_image_count": 1},
                    "tool_refs": [],
                    "model_refs": [],
                    "contract_refs": [],
                    "duration_ms": 100,
                    "prompt_eval_tokens": 10,
                    "eval_tokens": 5,
                }
            ],
        },
        "privacy": {"absolute_paths": "omitted"},
    }
