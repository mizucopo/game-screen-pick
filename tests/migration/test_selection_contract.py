"""固定AI評価からUnicode sceneの集計が二次・最終選択に及ぼす契約を検証する."""

import json
from collections.abc import Callable
from pathlib import Path
from types import FunctionType
from typing import Any

import pytest

from src.models.video_selection import FrameAssessment, FrameCandidate
from src.services import video_selector

ROOT = Path(__file__).parents[1] / "fixtures" / "rust_migration"
EXPECTED = json.loads(
    (ROOT / "selection_expectations.json").read_text(encoding="utf-8")
)


def _candidates_and_primary(
    case: dict[str, Any],
) -> tuple[list[FrameCandidate], dict[str, FrameAssessment]]:
    """二つの選定経路へ同じ保存済み候補と一次応答を渡す."""
    candidates = [
        FrameCandidate(**{**row, "difference_hash": int(row["difference_hash"], 16)})
        for row in case["candidates"]
    ]
    primary = {
        row["frame_id"]: FrameAssessment(**row) for row in case["primary_response"]
    }
    return candidates, primary


def _select_secondary(case: dict[str, Any]) -> list[str]:
    """保存した一次応答から二次評価へ送る候補IDを順序つきで返す."""
    candidates, primary = _candidates_and_primary(case)
    return [
        candidate.frame_id
        for candidate in video_selector.select_diverse_candidates(
            candidates, primary, case["count"]
        )
    ]


def _select(case: dict[str, Any]) -> list[dict[str, str | float]]:
    """保存した候補・両AI応答をproductionの最終選定へ渡す."""
    candidates, primary = _candidates_and_primary(case)
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
def test_unicode_scene_secondary_selection_contract(case: dict[str, Any]) -> None:
    """Unicodeの同値sceneが二次候補poolで競合するID/順序を固定する."""
    assert _select_secondary(case) == case["expected_secondary_candidates"]


@pytest.mark.parametrize("case", EXPECTED["cases"], ids=lambda case: case["name"])
def test_unicode_scene_selection_contract(case: dict[str, Any]) -> None:
    """scene名の正規化差を順位・選択IDという公開契約で検出する."""
    assert _select(case) == case["expected_selected"]


def _simplified_scene_normalizer(
    normalization: str,
) -> Callable[[FrameAssessment], str]:
    def simplified(assessment: FrameAssessment) -> str:
        if normalization == "lower":
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

    return simplified


@pytest.mark.parametrize("case", EXPECTED["cases"], ids=lambda case: case["name"])
def test_fixture_rejects_simplified_scene_normalization(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """lower置換・ASCII限定の誤移植では保存済み選択に一致しない."""
    control = case["negative_control"]
    monkeypatch.setattr(
        video_selector,
        "normalize_scene",
        _simplified_scene_normalizer(control["normalization"]),
    )
    actual = _select(case)
    assert actual == control["expected_selected"]
    assert actual != case["expected_selected"]


@pytest.mark.parametrize("case", EXPECTED["cases"], ids=lambda case: case["name"])
def test_fixture_rejects_secondary_only_simplified_scene_normalization(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """最終選定が正しくても二次poolだけの誤正規化を検出する."""
    control = case["negative_control"]
    secondary_selection = video_selector.select_diverse_candidates
    # 関数のglobalsだけを差し替え、最終選定側のnormalize_sceneは維持する。
    monkeypatch.setattr(
        video_selector,
        "select_diverse_candidates",
        FunctionType(
            secondary_selection.__code__,
            {
                **secondary_selection.__globals__,
                "normalize_scene": _simplified_scene_normalizer(
                    control["normalization"]
                ),
            },
        ),
    )
    assert _select(case) == case["expected_selected"]
    actual = _select_secondary(case)
    assert actual == control["expected_secondary_candidates"]
    assert actual != case["expected_secondary_candidates"]
