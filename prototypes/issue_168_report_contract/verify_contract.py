"""Issue #168の公開report contract sampleを検証する。"""

from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

FRAME_ID_PATTERN = re.compile(r"frm_([0-9a-f]{64})\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
STAGE_FINGERPRINT_PATTERN = re.compile(r"stg_[0-9a-f]{64}\Z")


def load_json(path: Path) -> dict[str, Any]:
    """JSON objectを読み込む。"""
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def assert_relative_posix_path(value: str) -> None:
    """root相対の正規化済みPOSIX pathであることを確認する。"""
    path = PurePosixPath(value)
    assert not path.is_absolute(), value
    assert ".." not in path.parts, value
    assert "\\" not in value, value


def display_time(value: Fraction) -> str:
    """非負の正確な秒数をhalf-upでミリ秒表示する。"""
    milliseconds = math.floor(value * 1000 + Fraction(1, 2))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def assert_video_time(value: dict[str, Any]) -> None:
    """PTSと公開有理数が同じVideo Timeを表すことを確認する。"""
    time_base = value["time_base"]
    offset = value["offset_seconds"]
    actual = Fraction(
        (value["source_pts"] - value["origin_pts"])
        * time_base["numerator"],
        time_base["denominator"],
    )
    expected = Fraction(offset["numerator"], offset["denominator"])
    assert actual == expected
    assert math.gcd(offset["numerator"], offset["denominator"]) == 1
    assert value["display"] == display_time(expected)


def assert_video_set(report: dict[str, Any]) -> None:
    """source path、fingerprint、durationの整合性を確認する。"""
    sources = report["video_set"]["sources"]
    duration_sum = Fraction()
    for order, source in enumerate(sources, start=1):
        assert source["order"] == order
        assert_relative_posix_path(source["relative_path"])
        assert source["fingerprint"]["algorithm"] == "sha256-whole-file"
        assert SHA256_PATTERN.fullmatch(source["fingerprint"]["value"])

        duration = source["duration"]
        time_base = duration["time_base"]
        actual = Fraction(
            (duration["source_pts"] - duration["origin_pts"])
            * time_base["numerator"],
            time_base["denominator"],
        )
        expected = Fraction(duration["exact_seconds"])
        assert actual == expected
        assert duration["display"] == display_time(expected)
        duration_sum += expected

    video_set_duration = report["video_set"]["duration"]
    assert duration_sum == Fraction(video_set_duration["exact_seconds"])
    assert video_set_duration["display"] == display_time(duration_sum)


def assert_selected(report: dict[str, Any], markdown: str) -> None:
    """stable ID、output name、説明、時刻、Markdown projectionを確認する。"""
    selected = report["selected"]
    assert len(selected) == report["run"]["selected_image_count"]
    assert len(selected) == report["selection_summary"]["selected"]
    source_ids = {source["id"] for source in report["video_set"]["sources"]}

    digest_prefixes: dict[str, int] = {}
    digests: dict[str, str] = {}
    for item in selected:
        match = FRAME_ID_PATTERN.fullmatch(item["image_id"])
        assert match is not None
        digest = match.group(1)
        digests[item["image_id"]] = digest
        digest_prefixes[digest[:12]] = digest_prefixes.get(digest[:12], 0) + 1

    for expected_index, item in enumerate(selected, start=1):
        assert item["selection_index"] == expected_index
        digest = digests[item["image_id"]]
        digest_part = digest if digest_prefixes[digest[:12]] > 1 else digest[:12]
        scene_slug = item["classification"]["scene_slug"]
        expected_name = f"{expected_index:04d}_{scene_slug}_{digest_part}.webp"
        expected_path = f"images/{expected_name}"
        output = item["output"]
        assert output["relative_path"] == expected_path
        assert_relative_posix_path(output["relative_path"])
        assert SHA256_PATTERN.fullmatch(output["sha256"])
        assert output["width"] > 0
        assert output["height"] > 0
        assert output["bytes"] > 0

        source = item["source"]
        assert source["video_id"] in source_ids
        assert_video_time(source["video_time"])

        annotation = item["annotation"]
        assert annotation["summary"]
        assert annotation["representative_frame_reason"]
        selection = item["selection"]
        assert "reason" not in selection
        assert selection["reason_codes"]
        assert selection["decision_explanation"]["renderer"]
        assert selection["decision_explanation"]["text"]

        classification = item["classification"]
        evidence = classification["spoiler_evidence"]
        if classification["spoiler_risk"] == "none":
            assert evidence is None
        else:
            assert evidence["source"] == "candidate_annotation"
            assert evidence["summary"]

        alt = f"{expected_index:02d} — {classification['scene_display_name']}"
        image_embed = f"[![{alt}]({expected_path})]({expected_path})"
        assert image_embed in markdown


def assert_rejections(report: dict[str, Any]) -> None:
    """全件集計とbounded near miss集合の整合性を確認する。"""
    summary = report["rejection_summary"]
    assert summary["total"] == sum(summary["by_reason"].values())
    assert summary["total"] == report["selection_summary"]["not_selected"]

    requested = report["run"]["requested_image_count"]
    limit = min(summary["total"], 100, max(20, requested * 2))
    policy = report["near_miss_publication"]
    assert policy["json_limit_for_this_run"] == limit
    near_misses = report["near_misses"]
    assert len(near_misses) <= limit
    assert set(summary["by_reason"]) <= {
        item["rejection"]["reason_code"] for item in near_misses
    }

    selected_ids = {item["image_id"] for item in report["selected"]}
    source_ids = {source["id"] for source in report["video_set"]["sources"]}
    for item in near_misses:
        assert FRAME_ID_PATTERN.fullmatch(item["image_id"])
        assert item["source"]["video_id"] in source_ids
        assert_video_time(item["source"]["video_time"])
        counterfactual = item["counterfactual_selection"]
        expected_utility = (
            counterfactual["base_utility"]
            - counterfactual["spoiler_penalty"]
            + counterfactual["coverage_bonus"]
            - counterfactual["temporal_diversity_penalty"]
        )
        assert math.isclose(
            expected_utility,
            counterfactual["marginal_utility"],
            rel_tol=0,
            abs_tol=1e-12,
        )
        rejection = item["rejection"]
        for reference_name in (
            "blocked_by_image_id",
            "nearest_selected_image_id",
        ):
            if reference_name in rejection:
                assert rejection[reference_name] in selected_ids


def assert_provenance(report: dict[str, Any]) -> None:
    """Stage fingerprintとregistry参照の整合性を確認する。"""
    provenance = report["provenance"]
    stages = provenance["stages"]
    fingerprints = {stage["fingerprint"] for stage in stages}
    assert all(STAGE_FINGERPRINT_PATTERN.fullmatch(value) for value in fingerprints)
    for stage in stages:
        assert set(stage["upstream_fingerprints"]) <= fingerprints
        assert set(stage.get("tool_refs", ())) <= set(provenance["tools"])
        assert set(stage.get("model_refs", ())) <= set(provenance["models"])
        assert set(stage.get("contract_refs", ())) <= set(
            provenance["contracts"]
        )


def assert_context_cues(report: dict[str, Any]) -> None:
    """公開Cue metadataとCandidate Annotation参照の整合性を確認する。"""
    cues = report["context_cues"]
    cue_ids = {cue["id"] for cue in cues}
    assert len(cue_ids) == len(cues)
    source_ids = {source["id"] for source in report["video_set"]["sources"]}
    for cue in cues:
        assert cue["video_id"] in source_ids
        assert cue["source_kind"] in {"embedded_subtitle", "speech_to_text"}
        assert cue["reliability"]["policy"] in {"usable", "low"}
        if cue["timestamp_basis"] == "source_pts":
            assert_video_time(cue["start"])
            assert_video_time(cue["end"])
            start = Fraction(
                cue["start"]["offset_seconds"]["numerator"],
                cue["start"]["offset_seconds"]["denominator"],
            )
            end = Fraction(
                cue["end"]["offset_seconds"]["numerator"],
                cue["end"]["offset_seconds"]["denominator"],
            )
        else:
            assert cue["timestamp_basis"] == "asr_sample_grid_estimate"
            values = []
            for endpoint in (cue["start"], cue["end"]):
                exact = Fraction(endpoint["sample_index"], endpoint["sample_rate_hz"])
                assert exact == Fraction(endpoint["exact_seconds"])
                assert endpoint["display"] == display_time(exact)
                values.append(exact)
            start, end = values
        assert start < end

    for item in report["selected"]:
        annotation = item["annotation"]
        supporting_ids = set(annotation["supporting_context_cue_ids"])
        assert supporting_ids <= cue_ids
        if annotation["context_cue_relevance"] in {"unavailable", "none"}:
            assert not supporting_ids


def assert_privacy(report: dict[str, Any]) -> None:
    """公開境界からraw textとsecret-bearing値が除外されることを確認する。"""
    privacy = report["privacy"]
    assert privacy["absolute_paths"] == "omitted"
    assert privacy["model_reasoning_trace"] == "omitted"
    assert privacy["raw_model_responses"] == "omitted"
    assert privacy["raw_context_cue_text"] == "processing_cache_only"
    assert privacy["environment_variables"] == "omitted"
    assert privacy["credentials"] == "omitted"
    assert privacy["prompt_bodies"] == "omitted"
    assert privacy["stack_traces"] == "omitted"
    for cue in report["context_cues"]:
        assert cue["text_included"] is False
        assert "text" not in cue


def verify(report_path: Path, markdown_path: Path) -> None:
    """prototype artifacts全体の公開contractを検証する。"""
    report = load_json(report_path)
    markdown = markdown_path.read_text()
    assert report["schema"] == {
        "name": "game-screen-pick/report",
        "version": "1.0.0",
    }
    assert report["artifacts"]["image_contract"]["format"] == "webp"
    assert report["artifacts"]["image_contract"]["quality"] == 95
    assert report["artifacts"]["image_contract"]["configurable"] is False
    assert report["artifacts"]["publication_contract"]["non_atomic_fallback"] is False
    assert report["artifacts"]["projection_contract"]["cache_or_model_reread"] is False
    assert_video_set(report)
    assert_selected(report, markdown)
    assert_rejections(report)
    assert_provenance(report)
    assert_context_cues(report)
    assert_privacy(report)


if __name__ == "__main__":
    root = Path(__file__).parent
    verify(root / "report.sample.json", root / "variant-a-gallery.md")
    print("Issue #168 report contract sample: OK")
