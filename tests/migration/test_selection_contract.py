"""固定AI評価からUnicode sceneの集計が最終選択に及ぼす契約を検証する."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.models.video_selection import FrameAssessment, FrameCandidate
from src.services import video_selector

ROOT = Path(__file__).parents[1] / "fixtures" / "rust_migration"
EXPECTED = json.loads(
    (ROOT / "selection_expectations.json").read_text(encoding="utf-8")
)


def _select(case: dict[str, Any]) -> list[dict[str, str | float]]:
    """保存した候補・両AI応答をproductionの最終選定へ渡す."""
    candidates = [
        FrameCandidate(**{**row, "difference_hash": int(row["difference_hash"], 16)})
        for row in case["candidates"]
    ]
    primary = {
        row["frame_id"]: FrameAssessment(**row) for row in case["primary_response"]
    }
    secondary = {
        row["frame_id"]: FrameAssessment(**row) for row in case["secondary_response"]
    }
    selected = video_selector.select_final_frames(
        candidates, primary, secondary, case["count"]
    )
    return [
        {
            "frame_id": frame.candidate.frame_id,
            "aggregate_score": frame.aggregate_score,
        }
        for frame in selected
    ]


@pytest.mark.parametrize("case", EXPECTED["cases"], ids=lambda case: case["name"])
def test_unicode_scene_selection_contract(case: dict[str, Any]) -> None:
    """scene名の正規化差を順位・選択IDという公開契約で検出する."""
    assert _select(case) == case["expected_selected"]


@pytest.mark.parametrize("case", EXPECTED["cases"], ids=lambda case: case["name"])
def test_fixture_rejects_simplified_scene_normalization(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """lower置換・ASCII限定の誤移植では保存済み選択に一致しない."""
    control = case["negative_control"]

    def simplified(assessment: FrameAssessment) -> str:
        if control["normalization"] == "lower":
            normalized = "".join(
                character
                for character in assessment.scene.lower()
                if character.isalnum()
            )
        else:
            normalized = "".join(
                character
                for character in assessment.scene.casefold()
                if character.isascii() and character.isalnum()
            )
        return normalized[:48] or "その他"

    monkeypatch.setattr(video_selector, "normalize_scene", simplified)
    actual = _select(case)
    assert actual == control["expected_selected"]
    assert actual != case["expected_selected"]
