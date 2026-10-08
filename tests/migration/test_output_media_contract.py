"""Final contact sheets must preserve selected images, rank and timestamp labels."""

import json
from pathlib import Path

import pytest
from PIL import Image

from src.models.video_selection_request import VideoSelectionRequest
from src.services.video_selector import VideoSelector
from src.utils.video_selection_files import file_sha256
from tests.migration.support.pipeline_fixture import (
    FixtureHttp,
    RecordingExtractor,
    assert_contract,
    cache_root,
    load_json,
    request_for,
)


@pytest.fixture(params=("sampled_frames", "semantic_video"))
def completed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> tuple[VideoSelectionRequest, str]:
    """Generate actual production artifacts with the fixed real-video pipeline."""
    method = str(request.param)
    selection_request = request_for(tmp_path, method)
    FixtureHttp(method).install(monkeypatch)
    VideoSelector(selection_request, frame_extractor=RecordingExtractor()).run()
    assert_contract(selection_request, method)
    return selection_request, method


def _refresh_sheet_integrity(request: VideoSelectionRequest) -> None:
    """Keep each artifact's own integrity valid so pixels are the only rejection."""
    output = Path(request.output_dir)
    sheet = output / "selected-contact-sheet.jpg"
    report_path = output / "report.json"
    report = load_json(report_path)
    for item in report["artifact_integrity"]:
        if item["path"] == sheet.name:
            item.update(size=sheet.stat().st_size, sha256=file_sha256(sheet))
    report_path.write_text(json.dumps(report), encoding="utf-8")
    completion_path = next(cache_root(request).glob("runs/*/completion-*.json"))
    completion = load_json(completion_path)
    for item in completion["artifacts"]:
        artifact = output / item["path"]
        item.update(size=artifact.stat().st_size, sha256=file_sha256(artifact))
    completion_path.write_text(json.dumps(completion), encoding="utf-8")


def test_output_contract_accepts_reencoded_contact_sheet(
    completed_output: tuple[VideoSelectionRequest, str],
) -> None:
    """An actual output sheet may use different JPEG bytes within the RGB gate."""
    request, method = completed_output
    sheet = Path(request.output_dir) / "selected-contact-sheet.jpg"
    original_bytes = sheet.read_bytes()
    with Image.open(sheet) as original:
        image = original.convert("RGB")
    with image:
        image.save(sheet, format="JPEG", quality=100, subsampling=0)
    assert sheet.read_bytes() != original_bytes
    _refresh_sheet_integrity(request)
    assert_contract(request, method)


@pytest.mark.parametrize("mutation", ("blank", "reordered", "mislabel"))
def test_output_contract_rejects_changed_contact_sheet(
    completed_output: tuple[VideoSelectionRequest, str], mutation: str
) -> None:
    """Self-consistent hash, dimensions and report cannot hide wrong sheet content."""
    request, method = completed_output
    sheet = Path(request.output_dir) / "selected-contact-sheet.jpg"
    with Image.open(sheet) as original:
        image = original.convert("RGB")
    with image:
        if mutation == "blank":
            image.paste("black", (0, 0, image.width, image.height))
        elif mutation == "mislabel":
            # Copy rank 02 / its time over rank 01, preserving both thumbnails.
            with image.crop((480, 0, 960, 32)) as wrong_label:
                image.paste(wrong_label, (0, 0))
        else:
            # Keep labels and padding positions while swapping only thumbnails.
            first, second = (0, 32, 480, 302), (480, 32, 960, 302)
            with image.crop(first) as left, image.crop(second) as right:
                image.paste(right, first)
                image.paste(left, second)
        image.save(sheet, format="JPEG", quality=100, subsampling=0)
    _refresh_sheet_integrity(request)
    with pytest.raises(AssertionError, match="selected-contact-sheet output"):
        assert_contract(request, method)
