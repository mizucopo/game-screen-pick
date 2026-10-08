"""Protect both video methods, durable old caches, and existing output artifacts."""

import io
from pathlib import Path
from urllib.request import Request

import pytest

from src.models.video_selection import FrameCandidate
from src.services.video_selector import VideoSelector, measure_candidate
from src.utils.video_selection_files import file_sha256
from tests.migration.support.pipeline_fixture import (
    FIXTURE_ROOT,
    FixtureHttp,
    RecordingExtractor,
    assert_cold_inference_calls,
    assert_contract,
    cache_root,
    load_json,
    request_for,
    restore_stored_cache,
)

METHODS = ("sampled_frames", "semantic_video")


def test_pipeline_fixture_inventory_is_fixed() -> None:
    """Verify that regression tests consume reviewed media and cache bytes."""
    provenance = load_json(FIXTURE_ROOT / "baseline-provenance.json")
    assert (
        file_sha256(FIXTURE_ROOT / "synthetic-game.mkv") == provenance["input_sha256"]
    )
    for relative, expected_hash in provenance["files"].items():
        assert file_sha256(FIXTURE_ROOT / relative) == expected_hash


@pytest.mark.parametrize("method", METHODS)
def test_pipeline_fixture_cold_warm_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """A real decode and fixed HTTP replies select the recorded frames on both runs."""
    request = request_for(tmp_path, method)
    http = FixtureHttp(method)
    http.install(monkeypatch)
    cold_extractor = RecordingExtractor()
    VideoSelector(request, frame_extractor=cold_extractor).run()
    assert cold_extractor.probes == 1
    assert_cold_inference_calls(http)
    assert_contract(request, method)
    old_artifacts = {
        path.name: file_sha256(path) for path in Path(request.output_dir).iterdir()
    }

    http.forbid_calls = True
    warm_extractor = RecordingExtractor()
    VideoSelector(request, frame_extractor=warm_extractor).run()

    assert warm_extractor.probes == 0
    assert warm_extractor.extractions == []
    assert_contract(request, method)
    assert {
        path.name: file_sha256(path) for path in Path(request.output_dir).iterdir()
    } == old_artifacts


@pytest.mark.parametrize(
    ("method", "duplicate_kind"),
    (
        ("sampled_frames", "primary"),
        ("sampled_frames", "secondary"),
        ("semantic_video", "video"),
        ("semantic_video", "primary"),
        ("semantic_video", "secondary"),
    ),
)
def test_pipeline_fixture_rejects_duplicate_cold_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    duplicate_kind: str,
) -> None:
    """Identical extra HTTP inference must fail even if report and media still match."""

    captured: dict[str, Request] = {}

    class CapturingHttp(FixtureHttp):
        def __call__(self, request: Request, *, timeout: float) -> io.BytesIO:
            response = super().__call__(request, timeout=timeout)
            if self.calls[-1] == duplicate_kind:
                captured[duplicate_kind] = request
            return response

    request = request_for(tmp_path, method)
    http = CapturingHttp(method)
    http.install(monkeypatch)
    VideoSelector(request, frame_extractor=RecordingExtractor()).run()
    # Replay the exact emitted request independently. Each media check performs
    # real FFmpeg work; nesting both in one HTTP call would consume its deadline.
    with http(captured[duplicate_kind], timeout=1):
        pass
    assert http.calls.count(duplicate_kind) == 2
    assert_contract(request, method)
    with pytest.raises(AssertionError, match="cold inference sequence"):
        assert_cold_inference_calls(http)


@pytest.mark.parametrize("method", METHODS)
def test_pipeline_fixture_resumes_after_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Interrupt after a durable primary response and resume only the missing stage."""
    request = request_for(tmp_path, method)
    interrupted_http = FixtureHttp(method, interrupt_secondary=True)
    interrupted_http.install(monkeypatch)
    with pytest.raises(KeyboardInterrupt, match="durable primary batch"):
        VideoSelector(request, frame_extractor=RecordingExtractor()).run()
    assert interrupted_http.calls.count("primary") == 1
    assert list(cache_root(request).glob("videos/*/assessments/primary/*.json"))
    assert not list(cache_root(request).glob("runs/*/completion-*.json"))
    assert not (Path(request.output_dir) / "report.json").exists()

    resumed_http = FixtureHttp(method)
    resumed_http.install(monkeypatch)
    resumed_extractor = RecordingExtractor()
    VideoSelector(request, frame_extractor=resumed_extractor).run()

    assert "primary" not in resumed_http.calls
    assert "video" not in resumed_http.calls
    assert resumed_http.calls.count("secondary") == 1
    assert resumed_extractor.probes == 0
    assert len(resumed_extractor.extractions) == 2
    assert all(width is None for _, width in resumed_extractor.extractions)
    assert_contract(request, method)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("state", ("normal", "partial", "corrupt", "corrupt-digest"))
def test_pipeline_fixture_replays_stored_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, state: str
) -> None:
    """Replay committed old bytes rather than building a new cache during the test."""
    request = request_for(tmp_path, method)
    restore_stored_cache(request, method, state)
    if state == "corrupt-digest":
        for path in cache_root(request).glob("videos/*/mechanical-analysis/*.json"):
            original = load_json(
                FIXTURE_ROOT
                / "stored-cache"
                / method
                / path.relative_to(cache_root(request))
            )
            corrupt = load_json(path)
            assert corrupt["data"]["payload_digest"] == "0" * 64
            assert (
                corrupt["data"]["payload_digest"] != original["data"]["payload_digest"]
            )
            corrupt["data"]["payload_digest"] = original["data"]["payload_digest"]
            assert corrupt == original, "key and complete payload must stay valid"
    http = FixtureHttp(method)
    http.forbid_calls = state != "partial"
    http.install(monkeypatch)
    measured_ids: list[str] = []

    def record_measurement(candidate: FrameCandidate) -> FrameCandidate | None:
        measured_ids.append(candidate.frame_id)
        return measure_candidate(candidate)

    monkeypatch.setattr(
        "src.services.video_selector.measure_candidate", record_measurement
    )
    extractor = RecordingExtractor()
    VideoSelector(request, frame_extractor=extractor).run()

    assert extractor.probes == 0
    assert len(extractor.extractions) == 2
    assert all(width is None for _, width in extractor.extractions)
    assert "primary" not in http.calls
    assert "video" not in http.calls
    if state == "partial":
        assert http.calls.count("secondary") == 1
    if state in {"corrupt", "corrupt-digest"}:
        assert len(measured_ids) == (6 if method == "sampled_frames" else 3)
    else:
        assert measured_ids == []
    assert_contract(request, method, exact_assessment_keys=True)


@pytest.mark.parametrize("method", METHODS)
def test_pipeline_fixture_preserves_modified_completed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """A user-edited completed image is rejected without overwriting any artifact."""
    request = request_for(tmp_path, method)
    http = FixtureHttp(method)
    http.install(monkeypatch)
    VideoSelector(request, frame_extractor=RecordingExtractor()).run()
    output = Path(request.output_dir)
    selected = output / "selected-01.jpg"
    selected.write_bytes(b"fixture user edit: must survive rerun")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    http.forbid_calls = True

    with pytest.raises(RuntimeError, match="完了済み成果物が変更されています"):
        VideoSelector(request, frame_extractor=RecordingExtractor()).run()

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
