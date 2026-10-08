"""固定AI評価からUnicode sceneの集計が二次・最終選択に及ぼす契約を検証する."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from types import FunctionType
from typing import Any

import pytest

from src.models.video_selection import FrameAssessment, FrameCandidate, VideoMetadata
from src.models.video_selection_request import VideoSelectionRequest
from src.services import video_selector
from src.services.video_phase_cache import VideoCacheIdentity

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


class _SecondaryPoolObserved(Exception):
    """本物のcallerが二次評価へ渡すpoolを記録した時点で実行を止める."""


def _select_secondary(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[str]:
    """productionの件数計算と時刻sortを通る二次評価候補IDを返す."""
    candidates, primary = _candidates_and_primary(case)
    selector = video_selector.VideoSelector(
        VideoSelectionRequest(
            input_video="synthetic.mp4",
            output_dir=str(tmp_path),
            output_count=case["count"],
            game_title=None,
            game_context="fixed synthetic context",
            primary_model="primary",
            secondary_model="secondary",
            ollama_host="unused",
            ollama_timeout=1.0,
            allow_cpu=True,
            ffmpeg_workers=1,
            sample_interval_seconds=None,
            debug=False,
        )
    )
    selector._usable_candidates = tuple(candidates)
    selector.sources = (
        video_selector.VideoSource(
            index=0,
            path=Path("synthetic.mp4"),
            metadata=VideoMetadata(60.0, 160, 96, "ffv1", "4/1"),
            end_margin_seconds=0.0,
            timestamps=tuple(candidate.timestamp_seconds for candidate in candidates),
            identity=VideoCacheIdentity("synthetic.mp4", 1, "unused"),
            cache_dir=tmp_path,
            candidate_cache_key="unused",
        ),
    )
    secondary_ids: list[str] = []

    def observe_secondary(
        _primary_candidates: list[FrameCandidate],
        _primary_assessments: dict[str, FrameAssessment],
        initial_candidates: Sequence[FrameCandidate],
    ) -> None:
        secondary_ids.extend(candidate.frame_id for candidate in initial_candidates)
        raise _SecondaryPoolObserved

    with monkeypatch.context() as boundary:
        # 候補・一次応答と副作用境界だけを固定し、poolの選定・件数・sortは変更しない。
        boundary.setattr(selector, "_prepare_run", lambda: None)
        boundary.setattr(selector, "_register_output", lambda: None)
        boundary.setattr(selector, "_verify_completion", lambda: False)
        boundary.setattr(selector, "_extract_candidates", lambda: candidates)
        boundary.setattr(selector, "_preselect_candidates", lambda _rows: candidates)
        boundary.setattr(
            selector,
            "_assess_with_source_backfill",
            lambda *_args, **_kwargs: (candidates, primary),
        )
        boundary.setattr(
            selector, "_assess_secondary_with_primary_backfill", observe_secondary
        )
        with pytest.raises(_SecondaryPoolObserved):
            selector._run_locked()
    return secondary_ids


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


@pytest.mark.parametrize(
    "case", EXPECTED["secondary_cases"], ids=lambda case: case["name"]
)
def test_unicode_scene_secondary_selection_contract(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unicodeの同値sceneが二次候補poolで競合するID/順序を固定する."""
    assert case["count"] == 2
    assert case["expected_pool_count"] == 6
    assert (
        sum(not row["is_transition"] for row in case["primary_response"])
        > case["expected_pool_count"]
    )
    actual = _select_secondary(case, monkeypatch, tmp_path)
    assert len(actual) == case["expected_pool_count"]
    assert actual == case["expected_secondary_candidates"]


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


@pytest.mark.parametrize(
    "case", EXPECTED["secondary_cases"], ids=lambda case: case["name"]
)
def test_fixture_rejects_secondary_only_simplified_scene_normalization(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
    for final_case in EXPECTED["cases"]:
        assert _select(final_case) == final_case["expected_selected"]
    actual = _select_secondary(case, monkeypatch, tmp_path)
    assert len(actual) == case["expected_pool_count"]
    assert actual == control["expected_secondary_candidates"]
    assert actual != case["expected_secondary_candidates"]
