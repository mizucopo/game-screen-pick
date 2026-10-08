"""Final contact sheets must preserve selected images, rank and timestamp labels."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageFont

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
from tests.migration.support.sheet_contract import redraw_sheet_labels, sheet_contract


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


@pytest.mark.parametrize("quality", (91, 100))
def test_output_contract_accepts_verified_alternate_font(
    completed_output: tuple[VideoSelectionRequest, str], quality: int
) -> None:
    """A second bundled font preserves rank/time/source and actual thumbnail pixels."""
    request, method = completed_output
    sheet = Path(request.output_dir) / "selected-contact-sheet.jpg"
    contract = sheet_contract(method, "selected")
    with Image.open(sheet) as original:
        image = original.convert("RGB")
    with image, image.copy() as reference:
        redraw_sheet_labels(image, contract, ImageFont.load_default_imagefont())
        with (
            image.crop(contract.label_box(0)) as actual_label,
            reference.crop(contract.label_box(0)) as original_label,
        ):
            assert actual_label.tobytes() != original_label.tobytes()
        for index in range(len(contract.labels)):
            x, _, right, bottom = contract.label_box(index)
            thumbnail = (x, bottom, right, bottom + contract.image_height)
            with (
                image.crop(thumbnail) as actual_thumbnail,
                reference.crop(thumbnail) as original_thumbnail,
            ):
                assert actual_thumbnail.tobytes() == original_thumbnail.tobytes()
        image.save(sheet, format="JPEG", quality=quality, subsampling=0)
    _refresh_sheet_integrity(request)
    assert_contract(request, method)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_label",
        "wrong_rank",
        "wrong_time",
        "wrong_source",
        "reordered_labels",
        "offset",
        "unreviewed_font",
        "label_padding",
        "thumbnail_edge",
        "unused_cell",
    ),
)
def test_output_contract_rejects_wrong_alternate_font_bitmap(
    completed_output: tuple[VideoSelectionRequest, str], mutation: str
) -> None:
    """Correct integrity cannot hide wrong rank/time/source/position pixels."""
    request, method = completed_output
    sheet = Path(request.output_dir) / "selected-contact-sheet.jpg"
    contract = sheet_contract(method, "selected")
    labels = list(contract.labels)
    if mutation == "wrong_rank":
        labels[0] = "09" + labels[0][2:]
    elif mutation == "wrong_time":
        labels[0] = labels[0].replace("00:", "09:", 1)
    elif mutation == "wrong_source":
        labels[0] = labels[0].replace("synthetic-game", "synthetic-gamo", 1)
    elif mutation == "reordered_labels":
        labels[0], labels[1] = labels[1], labels[0]
    font = (
        ImageFont.load_default(size=18)
        if mutation == "unreviewed_font"
        else ImageFont.load_default_imagefont()
    )
    with Image.open(sheet) as original:
        image = original.convert("RGB")
    with image:
        redraw_sheet_labels(image, replace(contract, labels=tuple(labels)), font)
        x, y, right, bottom = contract.label_box(0)
        if mutation == "missing_label":
            image.paste("black", contract.label_box(0))
        elif mutation == "offset":
            with image.crop(contract.label_box(0)) as shifted:
                image.paste("black", contract.label_box(0))
                image.paste(shifted, (x + 1, y))
        elif mutation == "label_padding":
            image.putpixel((right - 10, y + 8), (255, 255, 255))
        elif mutation == "thumbnail_edge":
            image.putpixel((x + 1, bottom), (255, 255, 255))
        elif mutation == "unused_cell":
            with image.crop(contract.label_box(0)) as extra:
                image.paste(extra, (2 * contract.cell_width, 0))
        image.save(sheet, format="JPEG", quality=100, subsampling=0)
    _refresh_sheet_integrity(request)
    with pytest.raises(AssertionError, match="selected-contact-sheet output"):
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
