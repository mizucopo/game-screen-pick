"""Reject self-consistent records whose receipts, links or provenance are false."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Never

import pytest
from PIL import Image

from src.models.video_selection import FrameCandidate
from src.models.video_selection_request import VideoSelectionRequest
from src.services.video_selector import VideoSelector, measure_candidate
from src.utils.video_selection_files import file_sha256, json_digest
from tests.migration.support.pipeline_fixture import (
    FixtureHttp,
    RecordingExtractor,
    assert_contract,
    cache_root,
    load_json,
    request_for,
    restore_stored_cache,
)


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _refresh_payload(value: dict[str, object]) -> None:
    value["payload_digest"] = json_digest(
        {key: item for key, item in value.items() if key != "payload_digest"}
    )


def _phase(request: VideoSelectionRequest, phase: str) -> Path:
    paths = list(cache_root(request).glob(f"videos/*/{phase}/**/*.json"))
    assert len(paths) == 1
    return paths[0]


def _mutate_context_records(records: list[dict[str, str]], mutation: str) -> None:
    """Reuse the same false receipt recipe in rejection and recovery controls."""
    if mutation in {"wrong_sha", "wrong_last_sha"}:
        index = -1 if mutation == "wrong_last_sha" else 0
        records[index]["sha256"] = "0" * 64
    else:
        suffix = "-before.jpg" if mutation == "missing_before" else "-after.jpg"
        records.remove(
            next(record for record in records if record["name"].endswith(suffix))
        )


def _link_candidate_manifest(request: VideoSelectionRequest) -> None:
    """Keep every payload's own digest valid; test the receipt rather than a link."""
    candidate_path = _phase(request, "candidate-extraction")
    candidate = load_json(candidate_path)
    _refresh_payload(candidate["data"])
    _write(candidate_path, candidate)
    mechanical_path = _phase(request, "mechanical-analysis")
    mechanical = load_json(mechanical_path)
    mechanical["data"]["source_frames_digest"] = candidate["data"]["payload_digest"]
    _refresh_payload(mechanical["data"])
    _write(mechanical_path, mechanical)


@pytest.fixture(scope="module", params=("sampled_frames", "semantic_video"))
def completed_pipeline(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> tuple[VideoSelectionRequest, str]:
    """Run real Python output once per method; never regenerate its goldens."""
    method = str(request.param)
    selection_request = request_for(tmp_path_factory.mktemp(method), method)
    with pytest.MonkeyPatch.context() as patch:
        FixtureHttp(method).install(patch)
        VideoSelector(selection_request, frame_extractor=RecordingExtractor()).run()
    assert_contract(selection_request, method)
    return selection_request, method


@pytest.fixture
def durable_output(
    completed_pipeline: tuple[VideoSelectionRequest, str],
) -> Iterator[tuple[VideoSelectionRequest, str]]:
    """Each mutation starts from identical cache and output bytes."""
    request, _ = completed_pipeline
    roots = (cache_root(request), Path(request.output_dir))
    saved = {
        path: path.read_bytes()
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }
    try:
        yield completed_pipeline
    finally:
        for root in roots:
            for path in root.rglob("*"):
                if path.is_file() and path not in saved:
                    path.unlink()
        for path, content in saved.items():
            path.write_bytes(content)


@pytest.mark.parametrize("record_index", (0, -1))
def test_candidate_receipt_rejects_fabricated_hash(
    durable_output: tuple[VideoSelectionRequest, str],
    record_index: int,
) -> None:
    """Correct candidate/mechanical self-digests cannot bless a false JPEG hash."""
    request, method = durable_output
    path = _phase(request, "candidate-extraction")
    candidate = load_json(path)
    candidate["data"]["source_frames"][record_index]["image_sha256"] = "0" * 64
    _write(path, candidate)
    _link_candidate_manifest(request)
    with pytest.raises(AssertionError, match="candidate JPEG receipt"):
        assert_contract(request, method)


@pytest.mark.parametrize(
    "mutation", ("wrong_sha", "wrong_last_sha", "missing_before", "missing_after")
)
def test_context_receipt_rejects_false_or_missing_record(
    durable_output: tuple[VideoSelectionRequest, str], mutation: str
) -> None:
    """A schema-compatible context record must match its file and required set."""
    request, method = durable_output
    path = _phase(request, "secondary-context")
    context = load_json(path)
    records = context["data"]["frames"]
    _mutate_context_records(records, mutation)
    _write(path, context)
    with pytest.raises(AssertionError, match="context (JPEG receipt|record names)"):
        assert_contract(request, method)


def test_mechanical_contract_rejects_wrong_candidate_link(
    durable_output: tuple[VideoSelectionRequest, str],
) -> None:
    """Only the source link is wrong; both phase payloads remain self-consistent."""
    request, method = durable_output
    path = _phase(request, "mechanical-analysis")
    mechanical = load_json(path)
    mechanical["data"]["source_frames_digest"] = "0" * 64
    _refresh_payload(mechanical["data"])
    _write(path, mechanical)
    with pytest.raises(AssertionError, match="mechanical candidate-manifest link"):
        assert_contract(request, method)


@pytest.mark.parametrize(
    ("collection", "field"), (("videos", "path"), ("selected", "video"))
)
@pytest.mark.parametrize("mutation", ("basename", "foreign_root"))
def test_report_contract_rejects_false_absolute_path(
    durable_output: tuple[VideoSelectionRequest, str],
    collection: str,
    field: str,
    mutation: str,
) -> None:
    """Recompute report integrity so normalization alone cannot hide false roots."""
    request, method = durable_output
    report_path = Path(request.output_dir) / "report.json"
    report = load_json(report_path)
    filename = Path(request.input_videos[0]).name
    report[collection][0][field] = (
        filename if mutation == "basename" else f"/wrong/root/{filename}"
    )
    _write(report_path, report)
    completion_path = next(cache_root(request).glob("runs/*/completion-*.json"))
    completion = load_json(completion_path)
    for record in completion["artifacts"]:
        if record["path"] == "report.json":
            record.update(
                size=report_path.stat().st_size, sha256=file_sha256(report_path)
            )
    _write(completion_path, completion)
    with pytest.raises(AssertionError, match="report absolute input path"):
        assert_contract(request, method)


def test_completion_contract_rejects_valid_duplicate(
    durable_output: tuple[VideoSelectionRequest, str],
) -> None:
    """Every duplicated field and file is correct; cardinality alone must fail."""
    request, method = durable_output
    path = next(cache_root(request).glob("runs/*/completion-*.json"))
    completion = load_json(path)
    completion["artifacts"].append(dict(completion["artifacts"][0]))
    _write(path, completion)
    with pytest.raises(AssertionError, match="completion artifact count"):
        assert_contract(request, method)


@pytest.mark.parametrize("phase", ("candidate-extraction", "secondary-context"))
def test_receipt_contract_accepts_own_reencoded_jpeg(
    durable_output: tuple[VideoSelectionRequest, str], phase: str
) -> None:
    """JPEG hashes differ across encoders but must be truthful within each output."""
    request, method = durable_output
    path = _phase(request, phase)
    envelope = load_json(path)
    candidate = phase == "candidate-extraction"
    record = envelope["data"]["source_frames" if candidate else "frames"][0]
    image_path = (
        path.parent
        / "frames"
        / (f"{record['frame_id']}.jpg" if candidate else record["name"])
    )
    original = image_path.read_bytes()
    with Image.open(image_path) as image:
        image.convert("RGB").save(image_path, format="JPEG", quality=100, subsampling=0)
    assert image_path.read_bytes() != original
    record["image_sha256" if candidate else "sha256"] = file_sha256(image_path)
    if candidate:
        record["file_size"] = image_path.stat().st_size
    _write(path, envelope)
    if candidate:
        _link_candidate_manifest(request)
    assert_contract(request, method)


@pytest.mark.parametrize("method", ("sampled_frames", "semantic_video"))
@pytest.mark.parametrize(
    "mutation", ("wrong_sha", "wrong_last_sha", "missing_before", "missing_after")
)
def test_context_receipt_recovers_only_needed_frame_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, mutation: str
) -> None:
    """Requested context miss reextracts one frame; prior inference remains reusable."""
    request = request_for(tmp_path, method)
    FixtureHttp(method, interrupt_secondary=True).install(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        VideoSelector(request, frame_extractor=RecordingExtractor()).run()
    path = _phase(request, "secondary-context")
    context = load_json(path)
    records = context["data"]["frames"]
    _mutate_context_records(records, mutation)
    _write(path, context)
    http = FixtureHttp(method)
    http.install(monkeypatch)
    extractor = RecordingExtractor()
    VideoSelector(request, frame_extractor=extractor).run()
    assert "primary" not in http.calls and "video" not in http.calls
    assert http.calls.count("secondary") == 1
    assert extractor.probes == 0
    assert sum(width == 960 for _, width in extractor.extractions) == 1
    assert sum(width is None for _, width in extractor.extractions) == 2
    assert_contract(request, method)


@pytest.mark.parametrize("method", ("sampled_frames", "semantic_video"))
def test_context_receipt_allows_prior_unused_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """A prior unrequested receipt needs no JPEG or re-extraction in this run."""
    request = request_for(tmp_path, method)
    FixtureHttp(method, interrupt_secondary=True).install(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        VideoSelector(request, frame_extractor=RecordingExtractor()).run()
    path = _phase(request, "secondary-context")
    context = load_json(path)
    unused = {"name": "prior-unused-before.jpg", "sha256": "a" * 64}
    context["data"]["frames"].append(unused)
    _write(path, context)
    assert not (path.parent / "frames" / unused["name"]).exists()
    http = FixtureHttp(method)
    http.install(monkeypatch)
    extractor = RecordingExtractor()
    VideoSelector(request, frame_extractor=extractor).run()
    assert "primary" not in http.calls and "video" not in http.calls
    assert http.calls.count("secondary") == 1
    assert extractor.probes == 0
    assert len(extractor.extractions) == 2
    assert all(width is None for _, width in extractor.extractions)
    assert unused in load_json(path)["data"]["frames"]
    assert_contract(request, method)


def test_completed_shortcut_does_not_read_unneeded_context(
    durable_output: tuple[VideoSelectionRequest, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completed artifacts reuse despite broken intermediate context JSON."""
    request, method = durable_output
    path = _phase(request, "secondary-context")
    path.write_bytes(b"{invalid unused context")
    output = Path(request.output_dir)
    before = {artifact.name: artifact.read_bytes() for artifact in output.iterdir()}
    http = FixtureHttp(method)
    http.forbid_calls = True
    http.install(monkeypatch)

    def unexpected_read(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("completed run must not read context")

    monkeypatch.setattr(VideoSelector, "_read_context_frame_records", unexpected_read)
    extractor = RecordingExtractor()
    VideoSelector(request, frame_extractor=extractor).run()
    assert http.calls == []
    assert extractor.probes == 0 and extractor.extractions == []
    assert {
        artifact.name: artifact.read_bytes() for artifact in output.iterdir()
    } == before
    assert path.read_bytes() == b"{invalid unused context"


@pytest.mark.parametrize("method", ("sampled_frames", "semantic_video"))
def test_wrong_mechanical_link_replays_only_affected_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """A self-consistent wrong dependency misses mechanics, preserving old AI keys."""
    request = request_for(tmp_path, method)
    restore_stored_cache(request, method, "normal")
    path = _phase(request, "mechanical-analysis")
    mechanical = load_json(path)
    mechanical["data"]["source_frames_digest"] = "0" * 64
    _refresh_payload(mechanical["data"])
    _write(path, mechanical)
    candidate = load_json(_phase(request, "candidate-extraction"))
    expected_ids = {record["frame_id"] for record in candidate["data"]["source_frames"]}
    measured: list[str] = []

    def recording_measure(candidate: FrameCandidate) -> FrameCandidate | None:
        measured.append(candidate.frame_id)
        return measure_candidate(candidate)

    monkeypatch.setattr(
        "src.services.video_selector.measure_candidate", recording_measure
    )
    http = FixtureHttp(method)
    http.forbid_calls = True
    http.install(monkeypatch)
    extractor = RecordingExtractor()
    VideoSelector(request, frame_extractor=extractor).run()
    assert set(measured) == expected_ids and len(measured) == len(expected_ids)
    assert http.calls == [] and extractor.probes == 0
    assert len(extractor.extractions) == 2
    assert all(width is None for _, width in extractor.extractions)
    assert_contract(request, method, exact_assessment_keys=True)


@pytest.mark.parametrize("mutation", ("invalid_name", "duplicate", "invalid_sha"))
def test_unused_context_record_still_requires_valid_structure(
    durable_output: tuple[VideoSelectionRequest, str], mutation: str
) -> None:
    """Ignoring unused JPEG truth must retain every record's wire validation."""
    request, method = durable_output
    path = _phase(request, "secondary-context")
    context = load_json(path)
    records = context["data"]["frames"]
    extra = {"name": "prior-unused-before.jpg", "sha256": "a" * 64}
    if mutation == "invalid_name":
        extra["name"] = "prior-unused.jpg"
    elif mutation == "invalid_sha":
        extra["sha256"] = "not-a-sha256"
    else:
        extra = dict(records[0])
    records.append(extra)
    _write(path, context)
    with pytest.raises(AssertionError, match="context JPEG receipt"):
        assert_contract(request, method)
