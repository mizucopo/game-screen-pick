"""Reviewed recording may update temporary references; normal comparison may not."""

import importlib.util
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from src.models.video_selection import FrameCandidate
from src.services.video_selector import VideoSelector, measure_candidate
from src.utils.video_selection_files import file_sha256, json_digest
from tests.migration.support import pipeline_fixture as fixture


def _fixed_bytes(root: Path) -> dict[str, bytes]:
    """Protect all checked-in media/JSON, while unrelated tool code may be edited."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jpg", ".png", ".mkv", ".mp4"}
    }


@pytest.fixture
def recorder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Import the real CLI against a disposable, complete fixture-directory copy."""
    original_root = fixture.FIXTURE_ROOT
    before = _fixed_bytes(original_root)
    copied = Path(shutil.copytree(original_root, tmp_path / "references"))
    monkeypatch.setattr(fixture, "FIXTURE_ROOT", copied)
    spec = importlib.util.spec_from_file_location(
        "isolated_migration_baseline_recorder", original_root / "record_baseline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "FIXTURE_ROOT", copied)
    assert fixture.FIXTURE_ROOT == module.FIXTURE_ROOT == copied
    assert module.request_for is fixture.request_for
    assert module.request_for.__globals__["FIXTURE_ROOT"] == copied
    try:
        yield module
    finally:
        assert _fixed_bytes(original_root) == before, (
            "repository references were written"
        )


def _mutate_rejected_extraction(
    recorder: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Let production write truthful receipts/keys from a still-rejected new JPEG."""
    changed: list[str] = []

    class StillRejectedExtractor(fixture.RecordingExtractor):
        def extract_frame(
            self,
            video: Path,
            timestamp_seconds: float,
            output_path: Path,
            *,
            max_width: int | None,
            video_stream_index: int = 0,
        ) -> None:
            super().extract_frame(
                video,
                timestamp_seconds,
                output_path,
                max_width=max_width,
                video_stream_index=video_stream_index,
            )
            if (
                "candidate-extraction" not in output_path.parts
                or timestamp_seconds != 2.5
            ):
                return
            assert output_path.stem == "f0814901973064699110200003"
            with (
                Image.open(output_path) as original,
                Image.new("RGB", original.size, (137, 137, 137)) as replacement,
            ):
                replacement.save(output_path, format="JPEG", quality=100, subsampling=0)
            assert (
                measure_candidate(
                    FrameCandidate(
                        output_path.stem, timestamp_seconds, str(output_path)
                    )
                )
                is None
            )
            changed.append(output_path.stem)

    monkeypatch.setattr(recorder, "RecordingExtractor", StillRejectedExtractor)
    return changed


def test_reviewed_full_recording_updates_changed_candidate_pixels(
    recorder: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full reviewed update must capture intentional pixels before old comparison."""
    root = Path(recorder.FIXTURE_ROOT)
    before = fixture.load_json(root / "expected" / "sampled_frames.json")
    previous_candidate = next(
        (root / "stored-cache" / "sampled_frames").glob(
            "videos/*/candidate-extraction/**/frames/f0814901973064699110200003.jpg"
        )
    ).read_bytes()
    changed = _mutate_rejected_extraction(recorder, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["record_baseline.py", "--reviewed-update"])

    recorder.main()

    assert changed == ["f0814901973064699110200003"]
    provenance = fixture.load_json(root / "baseline-provenance.json")
    for name, digest in provenance["files"].items():
        assert file_sha256(root / name) == digest
    assert provenance["input_sha256"] == file_sha256(root / "synthetic-game.mkv")
    candidate_path = next(
        (root / "stored-cache" / "sampled_frames").glob(
            "videos/*/candidate-extraction/**/manifest.json"
        )
    )
    candidate = fixture.load_json(candidate_path)["data"]
    record = candidate["source_frames"][2]
    image = candidate_path.parent / "frames" / f"{record['frame_id']}.jpg"
    assert image.read_bytes() != previous_candidate
    assert record["image_sha256"] == file_sha256(image)
    assert record["file_size"] == image.stat().st_size
    assert candidate["payload_digest"] == json_digest(
        {key: value for key, value in candidate.items() if key != "payload_digest"}
    )
    with Image.open(image) as decoded:
        assert decoded.format == "JPEG" and decoded.size == (160, 96)
        assert decoded.getextrema() == ((137, 137),) * 3
    mechanical_path = next(
        (root / "stored-cache" / "sampled_frames").glob(
            "videos/*/mechanical-analysis/*.json"
        )
    )
    assert (
        fixture.load_json(mechanical_path)["data"]["source_frames_digest"]
        == (candidate["payload_digest"])
    )
    after = fixture.load_json(root / "expected" / "sampled_frames.json")
    for field in ("mechanical", "assessments", "report"):
        assert after[field] == before[field], "decisions and AI replies must not change"
    old_keys = {
        item["phase"]: item["cache_key"] for item in before["assessment_envelopes"]
    }
    new_keys = {
        item["phase"]: item["cache_key"] for item in after["assessment_envelopes"]
    }
    assert old_keys.keys() == new_keys.keys() == {"primary", "secondary"}
    assert all(new_keys[stage] != old_keys[stage] for stage in old_keys)

    # Recorded caches contain no completion/output shortcut. Real native readers
    # must reuse both updated assessment keys without HTTP or candidate extraction.
    for method in ("sampled_frames", "semantic_video"):
        request = fixture.request_for(tmp_path / f"replay-{method}", method)
        fixture.restore_stored_cache(request, method, "normal")
        assert not list(fixture.cache_root(request).glob("runs/*/completion-*.json"))
        http = fixture.FixtureHttp(method)
        http.forbid_calls = True
        http.install(monkeypatch)
        extractor = fixture.RecordingExtractor()
        VideoSelector(request, frame_extractor=extractor).run()
        assert http.calls == []
        assert extractor.probes == 0
        assert len(extractor.extractions) == 2
        assert all(width is None for _, width in extractor.extractions)
        fixture.assert_contract(request, method, exact_assessment_keys=True)


def test_default_comparison_still_rejects_changed_candidate(
    recorder: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recording's explicit capture mode must not weaken normal qualification."""
    before = _fixed_bytes(recorder.FIXTURE_ROOT)
    changed = _mutate_rejected_extraction(recorder, monkeypatch)
    request = fixture.request_for(tmp_path / "normal-comparison", "sampled_frames")
    fixture.FixtureHttp("sampled_frames").install(monkeypatch)
    VideoSelector(request, frame_extractor=recorder.RecordingExtractor()).run()
    assert changed == ["f0814901973064699110200003"]
    with pytest.raises(AssertionError, match="candidate pixels.*channel MAE"):
        fixture.assert_contract(request, "sampled_frames")
    assert _fixed_bytes(recorder.FIXTURE_ROOT) == before


def test_manifest_only_recording_keeps_strict_candidate_comparison(
    recorder: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest-only update cannot silently promote new candidate pixels."""
    before = _fixed_bytes(recorder.FIXTURE_ROOT)
    changed = _mutate_rejected_extraction(recorder, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["record_baseline.py", "--reviewed-update", "--run-manifest-only"],
    )
    with pytest.raises(AssertionError, match="candidate pixels.*channel MAE"):
        recorder.main()
    assert changed == ["f0814901973064699110200003"]
    assert _fixed_bytes(recorder.FIXTURE_ROOT) == before


@pytest.mark.parametrize(
    "flag", ("--inference-media-only", "--selected-contact-sheet-only")
)
def test_partial_media_recording_preserves_unrelated_baselines(
    recorder: ModuleType, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    """Existing narrow capture paths never rewrite stored cache/expected/provenance."""
    root = Path(recorder.FIXTURE_ROOT)
    targets = {
        (
            f"inference-media/{method}/primary.png"
            if flag == "--inference-media-only"
            else f"reference-images/{method}/selected-contact-sheet.png"
        )
        for method in ("sampled_frames", "semantic_video")
    }
    for name in targets:
        (root / name).write_bytes(b"temporary outdated media must be recorded")
    before = _fixed_bytes(recorder.FIXTURE_ROOT)
    changed = _mutate_rejected_extraction(recorder, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["record_baseline.py", "--reviewed-update", flag])

    recorder.main()

    assert changed == ["f0814901973064699110200003"]
    after = _fixed_bytes(recorder.FIXTURE_ROOT)
    assert before.keys() == after.keys()
    updates = {name for name in before if before[name] != after[name]}
    assert targets <= updates, "the explicitly selected media must be recorded"
    for name in targets:
        with Image.open(root / name) as recorded:
            recorded.verify()
    if flag == "--inference-media-only":
        assert all(name.startswith("inference-media/") for name in updates)
    else:
        assert updates == {
            f"reference-images/{method}/selected-contact-sheet.png"
            for method in ("sampled_frames", "semantic_video")
        }


def test_recording_requires_explicit_reviewed_update(
    recorder: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing review intent refuses before pipeline work or any reference write."""
    before = _fixed_bytes(recorder.FIXTURE_ROOT)
    changed = _mutate_rejected_extraction(recorder, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["record_baseline.py"])
    with pytest.raises(SystemExit) as rejected:
        recorder.main()
    assert rejected.value.code == 2
    assert changed == []
    assert _fixed_bytes(recorder.FIXTURE_ROOT) == before


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("false_sha", "candidate JPEG receipt: SHA-256"),
        ("missing_receipt", "candidate record names/order/time changed"),
        ("wrong_link", "mechanical candidate-manifest link"),
    ),
)
def test_full_capture_keeps_intrinsic_candidate_guards(
    recorder: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    reason: str,
) -> None:
    """Skipping old reference pixels must retain truthful receipts/vector/links."""
    before = _fixed_bytes(recorder.FIXTURE_ROOT)
    request = fixture.request_for(tmp_path / "intrinsic-guards", "sampled_frames")
    fixture.FixtureHttp("sampled_frames").install(monkeypatch)
    VideoSelector(request, frame_extractor=fixture.RecordingExtractor()).run()
    fixture.assert_contract(request, "sampled_frames")
    root = fixture.cache_root(request)
    candidate_path = next(root.glob("videos/*/candidate-extraction/**/manifest.json"))
    mechanical_path = next(root.glob("videos/*/mechanical-analysis/*.json"))
    candidate = fixture.load_json(candidate_path)
    mechanical = fixture.load_json(mechanical_path)
    if mutation == "false_sha":
        record = candidate["data"]["source_frames"][0]
        image = candidate_path.parent / "frames" / f"{record['frame_id']}.jpg"
        record["image_sha256"] = "0" * 64
        assert record["image_sha256"] != file_sha256(image)
    elif mutation == "missing_receipt":
        omitted = candidate["data"]["source_frames"].pop()
        assert (
            candidate_path.parent / "frames" / f"{omitted['frame_id']}.jpg"
        ).is_file()
    candidate["data"]["payload_digest"] = json_digest(
        {
            key: value
            for key, value in candidate["data"].items()
            if key != "payload_digest"
        }
    )
    mechanical["data"]["source_frames_digest"] = (
        "0" * 64 if mutation == "wrong_link" else candidate["data"]["payload_digest"]
    )
    mechanical["data"]["payload_digest"] = json_digest(
        {
            key: value
            for key, value in mechanical["data"].items()
            if key != "payload_digest"
        }
    )
    recorder.write_json(candidate_path, candidate)
    recorder.write_json(mechanical_path, mechanical)

    # Both self-digests and all unrelated links are valid. Failure must name the
    # receipt/vector/link guard, rather than a generic payload-digest mismatch.
    with pytest.raises(AssertionError, match=reason):
        fixture.pipeline_contract(request, compare_candidate_pixels=False)
    assert _fixed_bytes(recorder.FIXTURE_ROOT) == before
