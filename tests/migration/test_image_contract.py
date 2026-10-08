"""Rust移植に使う固定画像の数値・境界採否・dHash契約を検証する."""

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from src.models.video_selection import FrameAssessment, FrameCandidate
from src.services.video_selector import (
    image_difference_hash,
    image_entropy,
    measure_candidate,
    select_final_frames,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "rust_migration"
EXPECTATIONS = json.loads((FIXTURE_ROOT / "image_expectations.json").read_text())


def _recipe_pixels(recipe: dict[str, Any]) -> np.ndarray[Any, Any]:
    """言語非依存のpixel recipeを復元する（期待値の再計算はしない）."""
    if recipe["kind"] == "columns":
        return np.tile(
            np.array(recipe["values"], dtype=np.uint8), (recipe["height"], 1)
        )
    y, x = np.indices((recipe["height"], recipe["width"]))
    if recipe["kind"] == "rgb_grid":
        return np.stack(
            [
                (x * x_factor + y * y_factor) % recipe["modulus"]
                for x_factor, y_factor in recipe["channels"]
            ],
            axis=2,
        ).astype(np.uint8)
    if recipe["kind"] == "half":
        low_positions = x < recipe["width"] // 2
    else:
        low_positions = (x // int(recipe["kind"][-1])) % 2 == 0
    return np.asarray(
        np.where(low_positions, recipe["low"], recipe["high"]), dtype=np.uint8
    )


@pytest.mark.parametrize(
    "expected", EXPECTATIONS["images"], ids=lambda row: row["name"]
)
def test_fixed_image_measurement_contract(expected: dict[str, Any]) -> None:
    """encoderに依存しない固定pixelと閾値の等号を含め、採否は厳密比較する."""
    path = FIXTURE_ROOT / expected["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
    with Image.open(path) as image:
        assert list(image.size) == expected["dimensions"]
        np.testing.assert_array_equal(
            np.asarray(image), _recipe_pixels(expected["recipe"])
        )
        assert f"{image_difference_hash(image):016x}" == expected["difference_hash"]

    image_bgr = cv2.imread(str(path))
    assert image_bgr is not None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    actual = {
        "brightness": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "entropy": image_entropy(gray),
    }
    for field, value in actual.items():
        assert value == pytest.approx(
            expected[field],
            abs=EXPECTATIONS["numeric_absolute_tolerance"],
            rel=EXPECTATIONS["numeric_relative_tolerance"],
        ), field
    candidate = measure_candidate(FrameCandidate("f00001", 0.5, str(path)))
    assert (candidate is not None) == expected["accepted"]
    if candidate is not None:
        assert candidate.quality_score == pytest.approx(
            expected["quality_score"],
            abs=EXPECTATIONS["numeric_absolute_tolerance"],
            rel=EXPECTATIONS["numeric_relative_tolerance"],
        )
        assert f"{candidate.difference_hash:016x}" == expected["difference_hash"]


@pytest.mark.parametrize(
    ("frame_order", "expected_id"),
    [(["f00002", "f00001"], "f00002"), (["f00001", "f00002"], "f00001")],
)
def test_complete_selection_tie_preserves_candidate_order(
    frame_order: list[str], expected_id: str
) -> None:
    """完全同点ではPython maxの先頭候補が勝つことも移植契約に残す."""
    candidates = [FrameCandidate(frame_id, 1.0, "unused") for frame_id in frame_order]
    assessments = {
        frame_id: FrameAssessment(frame_id, 80.0, False, "探索", "fixed response")
        for frame_id in frame_order
    }
    selected = select_final_frames(candidates, assessments, assessments, 1)
    assert selected[0].candidate.frame_id == expected_id
    assert selected[0].aggregate_score == 80.0
