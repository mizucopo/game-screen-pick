"""Reject complete, self-digested run manifests with incorrect nested contracts."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.models.video_selection_request import VideoSelectionRequest
from src.services.video_selector import VideoSelector
from src.utils.video_selection_files import file_sha256, json_digest
from tests.migration.support.pipeline_fixture import (
    FIXTURE_ROOT,
    FixtureHttp,
    RecordingExtractor,
    assert_contract,
    cache_root,
    load_json,
    pipeline_contract,
    request_for,
)


@pytest.fixture(scope="module", params=("sampled_frames", "semantic_video"))
def completed_pipeline(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> tuple[VideoSelectionRequest, str]:
    """Capture real Python output once per method, with fixed HTTP and native FFmpeg."""
    method = str(request.param)
    selection_request = request_for(tmp_path_factory.mktemp(method), method)
    with pytest.MonkeyPatch.context() as patch:
        FixtureHttp(method).install(patch)
        VideoSelector(selection_request, frame_extractor=RecordingExtractor()).run()
    assert_contract(selection_request, method)
    return selection_request, method


@pytest.fixture
def manifest_output(
    completed_pipeline: tuple[VideoSelectionRequest, str],
) -> Iterator[tuple[VideoSelectionRequest, str]]:
    """Restore the three mutated temporary artifacts after every independent case."""
    request, _ = completed_pipeline
    run = next(cache_root(request).glob("runs/*/run-manifest.json")).parent
    paths = (
        run / "run-manifest.json",
        next(run.glob("completion-*.json")),
        Path(request.output_dir) / "report.json",
    )
    originals = {path: path.read_bytes() for path in paths}
    try:
        yield completed_pipeline
    finally:
        for path, content in originals.items():
            path.write_bytes(content)


@pytest.mark.parametrize(
    "mutation",
    (
        "batch_sizes",
        "candidate_multipliers",
        "context_offset",
        "model_options",
        "models",
        "run_identity",
        "missing_nested",
        "extra_nested",
        "nested_number_type",
        "nested_boolean_type",
        "nested_list_type",
    ),
)
def test_run_manifest_rejects_self_consistent_mutation(
    manifest_output: tuple[VideoSelectionRequest, str], mutation: str
) -> None:
    """Valid digests and unchanged report decisions cannot hide wrong run settings."""
    request, method = manifest_output
    manifest_path = next(cache_root(request).glob("runs/*/run-manifest.json"))
    manifest = load_json(manifest_path)
    if mutation == "batch_sizes":
        manifest["batch_sizes"]["primary"] = 1
    elif mutation == "candidate_multipliers":
        manifest["candidate_multipliers"]["secondary"] = 2
    elif mutation == "context_offset":
        manifest["context_offset_seconds"] = 0.5
    elif mutation == "model_options":
        manifest["model_options"]["seed"] = 272
    elif mutation == "models":
        manifest["models"]["primary"]["capabilities"].append("audio")
    elif mutation == "run_identity":
        manifest["run_identity"]["game_context"] = "different run context"
    elif mutation == "missing_nested":
        del manifest["models"]["secondary"]["resolved_name"]
    elif mutation == "extra_nested":
        manifest["model_options"]["unexpected"] = {"enabled": True}
    elif mutation == "nested_number_type":
        manifest["batch_sizes"]["primary"] = 12.0
    elif mutation == "nested_boolean_type":
        manifest["run_identity"]["output_count"] = True
    else:
        assert mutation == "nested_list_type"
        manifest["models"]["primary"]["capabilities"][0] = {"name": "vision"}
    manifest["manifest_digest"] = json_digest(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report_path = Path(request.output_dir) / "report.json"
    report = load_json(report_path)
    report["manifest_digest"] = manifest["manifest_digest"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    completion_path = next(manifest_path.parent.glob("completion-*.json"))
    completion = load_json(completion_path)
    completion["manifest_digest"] = manifest["manifest_digest"]
    for item in completion["artifacts"]:
        if item["path"] == report_path.name:
            item.update(
                size=report_path.stat().st_size, sha256=file_sha256(report_path)
            )
    completion_path.write_text(json.dumps(completion), encoding="utf-8")

    # This passes every existing self-integrity check. The old manifest subset
    # and the normalized report also remain identical, isolating the new guard.
    actual = pipeline_contract(request)
    expected = load_json(FIXTURE_ROOT / "expected" / f"{method}.json")
    for field in (
        "run_key",
        "run_manifest_schema_version",
        "algorithm_version",
        "prompt_version",
        "phase_versions",
        "inputs",
        "report",
    ):
        assert actual[field] == expected[field]
    error = (
        "schema"
        if mutation
        in {
            "missing_nested",
            "extra_nested",
            "nested_number_type",
            "nested_boolean_type",
            "nested_list_type",
        }
        else "values"
    )
    with pytest.raises(AssertionError, match=f"run manifest {error} changed"):
        assert_contract(request, method)
