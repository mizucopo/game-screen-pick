"""Reviewed sheet labels and portable font profiles, independent of output JSON."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class SheetContract:
    """Fix cell geometry and text observed in the six committed sheet bitmaps."""

    columns: int
    cell_width: int
    label_height: int
    image_height: int
    labels: tuple[str, ...]

    def label_box(self, index: int) -> tuple[int, int, int, int]:
        """Return only the known label strip, never a thumbnail pixel."""
        x = index % self.columns * self.cell_width
        y = index // self.columns * (self.label_height + self.image_height)
        return x, y, x + self.cell_width, y + self.label_height


# These expectations were transcribed from the reviewed PNGs, not obtained from
# the submitted request/report or the production contact-sheet renderer.
_TIMES = {
    ("sampled_frames", "primary"): ("00:00", "00:02", "00:04", "00:04", "00:06"),
    ("sampled_frames", "secondary"): ("00:00", "00:02", "00:04", "00:06"),
    ("semantic_video", "primary"): ("00:02", "00:04", "00:06"),
    ("semantic_video", "secondary"): ("00:02", "00:04", "00:06"),
    ("sampled_frames", "selected"): ("00:02", "00:04"),
    ("semantic_video", "selected"): ("00:02", "00:04"),
}


def sheet_contract(method: str, kind: str) -> SheetContract:
    """Return the immutable meaning and geometry of a reviewed fixture sheet."""
    contextual = kind == "secondary"
    suffix = "  [before | selected | after]" if contextual else ""
    prefix = "" if kind == "selected" else "A"
    labels = tuple(
        f"{prefix}{index:02d}  {timestamp}  synthetic-game.mkv{suffix}"
        for index, timestamp in enumerate(_TIMES[method, kind], start=1)
    )
    return SheetContract(
        columns=2 if contextual else 3,
        cell_width=960 if contextual else 480,
        label_height=42 if contextual else 32,
        image_height=180 if contextual else 270,
        labels=labels,
    )


def redraw_sheet_labels(
    image: Image.Image,
    contract: SheetContract,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    """Draw one audited font at the required position without touching thumbnails."""
    draw = ImageDraw.Draw(image)
    for index, text in enumerate(contract.labels):
        x, y, right, bottom = contract.label_box(index)
        draw.rectangle((x, y, right - 1, bottom - 1), fill="black")
        draw.text((x + 10, y + 8), text, font=font, fill="white")


def _labels_match(
    image: Image.Image, template: Image.Image, contract: SheetContract
) -> bool:
    """Check full text/position/background against a trusted bitmap font profile."""
    for index in range(len(contract.labels)):
        with (
            image.crop(contract.label_box(index)) as actual,
            template.crop(contract.label_box(index)) as expected,
        ):
            difference = np.abs(
                np.asarray(actual, dtype=np.int16)
                - np.asarray(expected, dtype=np.int16)
            )
        # These are the existing RGB limits, not looser OCR/confidence scores.
        # dHash is applied to the remaining sheet after meaning is established;
        # comparing glyph dHashes would require the very font identity we waive.
        if (
            int(difference.max()) > 16
            or float(difference.mean(axis=(0, 1)).max()) > 1.0
            or float(np.mean(np.square(difference.astype(np.float64)))) > 255**2 / 10**4
        ):
            return False
    return True


def assert_sheet_pixels(
    actual: Image.Image,
    reference: Image.Image,
    label: str,
    contract: SheetContract,
    assert_pixels: Callable[[Image.Image, Image.Image, str], None],
) -> None:
    """Verify label meaning before excluding glyphs from the unchanged pixel gate.

    The approved profiles are Pillow's bundled Aileron 10 and bitmap default.
    Unregistered fonts fail closed until separately reviewed and qualified.
    No submitted JSON, font name or renderer claim establishes label meaning.
    """
    rows = (len(contract.labels) + contract.columns - 1) // contract.columns
    dimensions = (
        contract.columns * contract.cell_width,
        rows * (contract.label_height + contract.image_height),
    )
    assert actual.size == reference.size == dimensions, f"{label}: dimensions changed"
    with actual.convert("RGB") as actual_rgb, reference.convert("RGB") as reference_rgb:
        verified = False
        for font in (
            ImageFont.load_default(size=10),
            ImageFont.load_default_imagefont(),
        ):
            with Image.new("RGB", dimensions, "black") as template:
                redraw_sheet_labels(template, contract, font)
                if _labels_match(actual_rgb, template, contract):
                    verified = True
                    break
        assert verified, (
            f"{label}: label meaning/position or unreviewed font profile changed"
        )
        # Each complete occupied label strip has just passed its text, placement
        # and black-padding gate. Stop exactly at 32/42, before the first image row.
        # Empty cells remain unmasked, so added labels or layout content still fail.
        for index in range(len(contract.labels)):
            box = contract.label_box(index)
            actual_rgb.paste("black", box)
            reference_rgb.paste("black", box)
        assert_pixels(actual_rgb, reference_rgb, label)
