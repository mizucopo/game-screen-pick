"""Every extracted candidate must match pixels, even outside inference/output."""

import io
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from src.models.video_selection import FrameCandidate
from src.models.video_selection_request import VideoSelectionRequest
from src.services.video_selector import (
    PRIMARY_BATCH_SIZE,
    SECONDARY_BATCH_SIZE,
    VideoSelector,
    VideoSource,
    image_difference_hash,
    measure_candidate,
)
from src.utils.video_selection_files import file_sha256, json_digest
from tests.migration.support.pipeline_fixture import (
    FixtureHttp,
    RecordingExtractor,
    assert_contract,
    cache_root,
    load_json,
    request_for,
)


@dataclass(frozen=True)
class AssessmentPlan:
    """Keep the actual model/source and ordered candidates used by production."""

    model: str
    stage: str
    source: VideoSource
    candidates: tuple[FrameCandidate, ...]
    cache_key: str


@dataclass(frozen=True)
class CompletedPipeline:
    request: VideoSelectionRequest
    method: str
    selector: VideoSelector
    http: FixtureHttp
    plans: dict[str, AssessmentPlan]
    extractor: RecordingExtractor


def _completed_pipeline(
    root: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> CompletedPipeline:
    """Use isolated real outputs; neither record nor rewrite stored references."""
    request = request_for(root, method)
    http = FixtureHttp(method)
    http.install(monkeypatch)
    extractor = RecordingExtractor()
    selector = VideoSelector(request, frame_extractor=extractor)
    original_key = selector._assessment_cache_key
    plans: dict[str, AssessmentPlan] = {}

    def capture_key(
        model: str,
        stage: str,
        source: VideoSource,
        candidates: Sequence[FrameCandidate] | None = None,
    ) -> str:
        key = original_key(model, stage, source, candidates)
        if candidates is not None:
            plans[stage] = AssessmentPlan(model, stage, source, tuple(candidates), key)
        return key

    with monkeypatch.context() as capture_patch:
        capture_patch.setattr(selector, "_assessment_cache_key", capture_key)
        selector.run()
    assert set(plans) == {"primary", "secondary"}
    assert_contract(request, method)
    http.forbid_calls = True
    return CompletedPipeline(request, method, selector, http, plans, extractor)


def _phase(request: VideoSelectionRequest, phase: str) -> Path:
    paths = list(cache_root(request).glob(f"videos/*/{phase}/**/*.json"))
    assert len(paths) == 1
    return paths[0]


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _refresh_payload(data: dict[str, Any]) -> None:
    data["payload_digest"] = json_digest(
        {key: value for key, value in data.items() if key != "payload_digest"}
    )


def _refresh_candidate_dependencies(pipeline: CompletedPipeline) -> None:
    """Rebuild real cache dependencies so an unrelated stale key cannot reject."""
    candidate_path = _phase(pipeline.request, "candidate-extraction")
    candidate = load_json(candidate_path)
    _refresh_payload(candidate["data"])
    _write(candidate_path, candidate)
    mechanical_path = _phase(pipeline.request, "mechanical-analysis")
    mechanical = load_json(mechanical_path)
    mechanical["data"]["source_frames_digest"] = candidate["data"]["payload_digest"]
    _refresh_payload(mechanical["data"])
    _write(mechanical_path, mechanical)
    selector = pipeline.selector
    source = selector.sources[0]
    candidates = [
        FrameCandidate(
            frame_id=record["frame_id"],
            timestamp_seconds=record["timestamp_seconds"],
            path=str(candidate_path.parent / "frames" / f"{record['frame_id']}.jpg"),
            video_index=source.index,
        )
        for record in candidate["data"]["source_frames"]
    ]
    native_manifest = selector._load_candidate_manifest(source, candidates)
    assert native_manifest is not None, (
        "the native candidate receipt reader must accept"
    )
    selector._candidate_manifest_digests[source.index] = native_manifest[1]
    measured = selector._load_mechanical_candidates(source, candidates)
    assert measured is not None, "the native mechanical reader must accept"
    assert [item.frame_id for item in measured] == [
        item["frame_id"] for item in mechanical["data"]["candidates"]
    ]

    for plan in tuple(pipeline.plans.values()):
        old_path = (
            plan.source.cache_dir
            / "assessments"
            / plan.stage
            / f"{plan.cache_key}.json"
        )
        assessment = load_json(old_path)
        new_key = selector._assessment_cache_key(
            plan.model, plan.stage, plan.source, plan.candidates
        )
        assert new_key != plan.cache_key
        assessment["cache_key"] = new_key
        assessment["payload_digest"] = json_digest(
            {"cache_key": new_key, "assessments": assessment["assessments"]}
        )
        new_path = old_path.with_name(f"{new_key}.json")
        _write(new_path, assessment)
        old_path.unlink()
        batch_size = (
            PRIMARY_BATCH_SIZE if plan.stage == "primary" else SECONDARY_BATCH_SIZE
        )
        state = selector._load_assessment_state_or_miss(
            new_path, new_key, plan.candidates, batch_size
        )
        assert set(state) == set(assessment["assessments"]), (
            "the native assessment reader must accept every unchanged reply"
        )
        pipeline.plans[plan.stage] = replace(plan, cache_key=new_key)


def _replace_candidate_bytes(
    pipeline: CompletedPipeline, frame_id: str, replacement: bytes
) -> FrameCandidate:
    path = _phase(pipeline.request, "candidate-extraction")
    envelope = load_json(path)
    record = next(
        item
        for item in envelope["data"]["source_frames"]
        if item["frame_id"] == frame_id
    )
    image_path = path.parent / "frames" / f"{frame_id}.jpg"
    image_path.write_bytes(replacement)
    record["image_sha256"] = file_sha256(image_path)
    record["file_size"] = image_path.stat().st_size
    _write(path, envelope)
    _refresh_candidate_dependencies(pipeline)
    return FrameCandidate(
        frame_id=frame_id,
        timestamp_seconds=record["timestamp_seconds"],
        path=str(image_path),
        video_index=pipeline.selector.sources[0].index,
    )


def _replace_candidate(
    pipeline: CompletedPipeline,
    frame_id: str,
    replacement: Image.Image,
    *,
    identity_orientation: bool = False,
) -> FrameCandidate:
    encoded = io.BytesIO()
    replacement.save(encoded, format="JPEG", quality=100, subsampling=0)
    value = encoded.getvalue()
    if identity_orientation:
        value = _with_orientation(value, 1)
    return _replace_candidate_bytes(pipeline, frame_id, value)


def _with_orientation(encoded: bytes, orientation: int) -> bytes:
    """Add only an APP1 tag, preserving the entire original encoded image scan."""
    exif = Image.Exif()
    exif[274] = orientation
    metadata = exif.tobytes()
    segment = b"\xff\xe1" + (len(metadata) + 2).to_bytes(2, "big") + metadata
    return encoded[:2] + segment + encoded[2:]


def _candidate_path(pipeline: CompletedPipeline, index: int) -> Path:
    path = _phase(pipeline.request, "candidate-extraction")
    record = load_json(path)["data"]["source_frames"][index]
    return path.parent / "frames" / f"{record['frame_id']}.jpg"


def _assert_pixel_rejection(pipeline: CompletedPipeline, *, reason: str = "") -> None:
    """The comparison is read-only and reaches the pixel/metadata-specific guard."""
    roots = (cache_root(pipeline.request), Path(pipeline.request.output_dir))

    def snapshot() -> dict[Path, bytes]:
        return {
            path: path.read_bytes()
            for root in roots
            for path in root.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    calls = list(pipeline.http.calls)
    probes = pipeline.extractor.probes
    extractions = list(pipeline.extractor.extractions)
    with pytest.raises(AssertionError, match=f"candidate pixels.*{reason}"):
        assert_contract(pipeline.request, pipeline.method)
    assert snapshot() == before
    assert pipeline.http.calls == calls
    assert pipeline.extractor.probes == probes
    assert pipeline.extractor.extractions == extractions


def test_candidate_pixels_reject_changed_still_rejected_jpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Truthful hashes/current keys cannot hide a wrongly extracted rejected frame."""
    pipeline = _completed_pipeline(tmp_path, monkeypatch, "sampled_frames")
    frame_id = "f0814901973064699110200003"
    mechanical = load_json(_phase(pipeline.request, "mechanical-analysis"))["data"]
    assert frame_id in mechanical["rejected_frame_ids"]
    output = Path(pipeline.request.output_dir)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    image_path = (
        _phase(pipeline.request, "candidate-extraction").parent
        / "frames"
        / f"{frame_id}.jpg"
    )
    with (
        Image.open(image_path) as original,
        Image.new("RGB", original.size, (137, 137, 137)) as replacement,
    ):
        candidate = _replace_candidate(pipeline, frame_id, replacement)
    assert measure_candidate(candidate) is None, "the replacement must stay rejected"
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before

    _assert_pixel_rejection(pipeline, reason="channel MAE")
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


@pytest.mark.parametrize(
    ("method", "index"),
    [("sampled_frames", index) for index in (0, 1, 3, 4, 5)]
    + [("semantic_video", index) for index in (0, 1, 2)],
)
def test_candidate_pixels_reject_change_at_every_other_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, index: int
) -> None:
    """First/last and unselected or secondary-unsent frames share the same gate."""
    pipeline = _completed_pipeline(tmp_path, monkeypatch, method)
    path = _candidate_path(pipeline, index)
    frame_id = path.stem
    if method == "sampled_frames" and index == 3:
        assert frame_id in {
            item.frame_id for item in pipeline.plans["primary"].candidates
        }
        assert frame_id not in {
            item.frame_id for item in pipeline.plans["secondary"].candidates
        }
        assert frame_id not in {
            item["frame_id"]
            for item in load_json(Path(pipeline.request.output_dir) / "report.json")[
                "selected"
            ]
        }
    with (
        Image.open(path) as original,
        Image.new("RGB", original.size, (137, 137, 137)) as replacement,
    ):
        _replace_candidate(pipeline, frame_id, replacement)
    _assert_pixel_rejection(pipeline)


@pytest.mark.parametrize(
    "mutation", ("dimensions", "rotation", "dhash", "mae", "max", "psnr")
)
def test_candidate_pixels_enforce_each_declared_media_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Numeric negatives isolate one threshold; dimensions/rotation are strict."""
    pipeline = _completed_pipeline(tmp_path, monkeypatch, "sampled_frames")
    path = _candidate_path(pipeline, 2 if mutation in {"dhash", "mae"} else 0)
    with Image.open(path) as original:
        reference = original.convert("RGB")
    with reference:
        if mutation == "dimensions":
            replacement = reference.resize((reference.width - 1, reference.height))
        elif mutation == "rotation":
            replacement = reference.transpose(Image.Transpose.ROTATE_90)
        else:
            pixels = np.asarray(reference, dtype=np.int16).copy()
            if mutation == "dhash":
                pixels[:, reference.width // 2 :] += 1
            elif mutation == "mae":
                pixels += 2
            elif mutation == "max":
                pixels[64, 64] += 24
            else:
                pixels[:32, :16] += 16
            replacement = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
        with replacement:
            _replace_candidate(pipeline, path.stem, replacement)
        if mutation in {"dimensions", "rotation"}:
            reason = "dimensions changed"
        else:
            with Image.open(path) as actual:
                difference = np.abs(
                    np.asarray(actual.convert("RGB"), dtype=np.int16)
                    - np.asarray(reference, dtype=np.int16)
                )
                same_hash = image_difference_hash(actual) == image_difference_hash(
                    reference
                )
            mae = float(difference.mean(axis=(0, 1)).max())
            maximum = int(difference.max())
            mse = float(np.mean(difference.astype(np.float64) ** 2))
            psnr = 10 * math.log10(255**2 / mse)
            assert same_hash == (mutation != "dhash")
            assert (mae <= 1) == (mutation != "mae")
            assert (maximum <= 16) == (mutation != "max")
            assert (psnr >= 40) == (mutation != "psnr")
            reason = {
                "dhash": "dHash changed",
                "mae": "channel MAE",
                "max": "maximum pixel difference",
                "psnr": "PSNR",
            }[mutation]
    _assert_pixel_rejection(pipeline, reason=reason)


@pytest.mark.parametrize("mutation", ("orientation_tag", "png"))
def test_candidate_pixels_reject_wrong_metadata_with_identical_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """EXIF-only orientation and PNG bytes cannot hide behind identical raw pixels."""
    pipeline = _completed_pipeline(tmp_path, monkeypatch, "sampled_frames")
    path = _candidate_path(pipeline, 3)
    original = path.read_bytes()
    with Image.open(io.BytesIO(original)) as reference:
        if mutation == "orientation_tag":
            replacement = _with_orientation(original, 6)
        else:
            encoded = io.BytesIO()
            reference.save(encoded, format="PNG")
            replacement = encoded.getvalue()
        with Image.open(io.BytesIO(replacement)) as actual:
            assert actual.size == reference.size
            assert actual.convert("RGB").tobytes() == reference.convert("RGB").tobytes()
            assert image_difference_hash(actual) == image_difference_hash(reference)
            if mutation == "orientation_tag":
                assert actual.getexif()[274] == 6
            else:
                assert actual.format == "PNG"
    _replace_candidate_bytes(pipeline, path.stem, replacement)
    _assert_pixel_rejection(pipeline)


@pytest.mark.parametrize("method", ("sampled_frames", "semantic_video"))
def test_candidate_pixels_accept_alternate_jpeg_for_all_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Every candidate can use alternate bytes with truthful receipts/current keys."""
    pipeline = _completed_pipeline(tmp_path, monkeypatch, method)
    records = load_json(_phase(pipeline.request, "candidate-extraction"))["data"][
        "source_frames"
    ]
    assert len(records) == (6 if method == "sampled_frames" else 3)
    for index in range(len(records)):
        path = _candidate_path(pipeline, index)
        before = path.read_bytes()
        with Image.open(path) as image:
            _replace_candidate(
                pipeline, path.stem, image, identity_orientation=index == 0
            )
        assert path.read_bytes() != before
        with Image.open(path) as actual:
            assert actual.getexif().get(274, 1) == 1
    assert_contract(pipeline.request, method)
