"""Acceptance phase実行境界のtest。"""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.video_selection.acceptance.execute_acceptance_phase import (
    execute_acceptance_phase,
    normalized_result_digest,
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


def test_phase_duration_excludes_resource_monitor_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """phase durationがresource monitor停止時間を含まず記録されること。

    Arrange:
        - pipeline完了時刻を20、monitor停止後時刻を100とするclockが用意される
    Act:
        - operation failureになるAcceptance phaseが実行される
    Assert:
        - 開始10からpipeline完了20までの10秒だけが記録されること
    """
    # Arrange
    now = [10.0]
    configuration = EffectiveConfiguration(
        video_input_folder=tmp_path / "input",
        output_folder=tmp_path / "output",
    )
    resolved_models = FakeModelRuntime("phase-timing").resolve_models(configuration)
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
        "src.video_selection.acceptance.execute_acceptance_phase.time.monotonic",
        lambda: now[0],
    )
    monkeypatch.setattr(
        "src.video_selection.acceptance.execute_acceptance_phase."
        "build_real_application",
        lambda *_args, **_kwargs: SimpleNamespace(run=lambda _configuration: None),
    )
    monkeypatch.setattr(
        "src.video_selection.acceptance.execute_acceptance_phase."
        "InternalRunController.execute",
        execute,
    )
    monkeypatch.setattr(
        "src.video_selection.acceptance.execute_acceptance_phase.DiskUsageMonitor",
        lambda **_kwargs: SimpleNamespace(
            start=lambda: None,
            stop=stop_disk_monitor,
        ),
    )
    monkeypatch.setattr(
        "src.video_selection.acceptance.execute_acceptance_phase.GpuResourceMonitor",
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
                "stt_global_gpu_peak_mib": 0,
                "ollama_model_size_bytes": 0,
                "ollama_model_size_vram_bytes": 0,
                "ollama_model_observed": False,
                "ollama_model_fully_resident": False,
            },
        ),
    )

    # Act
    exit_code, phase_record, report, selection_artifact = execute_acceptance_phase(
        phase="cold",
        configuration=configuration,
        resolved_models=resolved_models,
        suite_root=tmp_path / "suite",
    )

    # Assert
    assert exit_code == 1
    assert phase_record["duration_seconds"] == 10.0
    assert phase_record["resource_sampling_complete"] is True
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
            "models": {},
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
