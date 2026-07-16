"""Canonical Selection Reportからgallery-first Markdownを描画する。"""

from typing import Any, cast


def render_human_selection_report(report: dict[str, object]) -> str:
    """検証対象のreport objectだけから決定的なMarkdownを返す。"""
    run = _mapping(report["run"])
    video_set = _mapping(report["video_set"])
    selection_summary = _mapping(report["selection_summary"])
    selected = _mapping_list(report["selected"])
    near_misses = _mapping_list(report["near_misses"])
    provenance = _mapping(report["provenance"])
    lines = [
        "# 画像選定レポート",
        "",
        f"`{run['id']}` · {run['started_at']}",
        "",
    ]
    warnings = _mapping_list(run["warnings"])
    for warning in warnings:
        lines.extend(
            (
                "> [!WARNING]",
                f"> {warning['message']}",
                "",
            )
        )
    lines.extend(
        (
            "## Summary",
            "",
            "| Requested | Selected | Videos | Duration | Candidate Moments |",
            "|---:|---:|---:|---:|---:|",
            (
                f"| {run['requested_image_count']} | {run['selected_image_count']} | "
                f"{len(cast(list[object], video_set['sources']))} | "
                f"{_mapping(video_set['duration'])['display']} | "
                f"{selection_summary['candidate_moments']} |"
            ),
            "",
            "| Blog Image Type | Target | Actual |",
            "|---|---:|---:|",
        )
    )
    image_types = _mapping(selection_summary["blog_image_type"])
    for name, counts_value in image_types.items():
        counts = _mapping(counts_value)
        lines.append(f"| {name} | {counts['target']} | {counts['actual']} |")
    lines.extend(("", "## Selected images", ""))
    sources = {item["id"]: item for item in _mapping_list(video_set["sources"])}
    for item in selected:
        _append_selected(lines, item, sources)
    lines.extend(("## Near misses", ""))
    if near_misses:
        lines.extend(
            (
                "| Candidate | Counterfactual utility | Not selected because |",
                "|---|---:|---|",
            )
        )
        for item in near_misses[:10]:
            annotation = _mapping(item["annotation"])
            counterfactual = _mapping(item["counterfactual_selection"])
            rejection = _mapping(item["rejection"])
            identifier = str(item["image_id"])
            detail = str(rejection["reason_code"])
            if "similarity" in rejection:
                detail += f" ({float(rejection['similarity']):.3f})"
            lines.append(
                f"| `{_abbreviate(identifier, 8)}` {annotation['summary']} | "
                f"{float(counterfactual['marginal_utility']):.6f} | `{detail}` |"
            )
    else:
        lines.append("該当なし。")
    lines.extend(("", "## Reproduction appendix", "", "### Selection funnel", ""))
    shortfall = _mapping(selection_summary["shortfall"])
    refined_count = int(selection_summary["candidate_moments"]) - int(
        selection_summary["moments_without_valid_frame"]
    )
    requested = shortfall["requested"]
    selected_count = shortfall["selected"]
    lines.extend(
        (
            "| Stage | Input | Kept |",
            "|---|---:|---:|",
            (
                f"| Candidate Moment discovery | "
                f"{selection_summary['candidate_moments']} | "
                f"{selection_summary['candidate_moments']} |"
            ),
            (
                f"| Frame refinement | {selection_summary['candidate_moments']} | "
                f"{refined_count} |"
            ),
            (
                f"| Candidate Annotation | "
                f"{selection_summary['candidate_annotations']} | "
                f"{selection_summary['candidate_annotations']} |"
            ),
            (
                f"| Final selection | {selection_summary['candidate_annotations']} | "
                f"{selection_summary['selected']} |"
            ),
            "",
            (
                f"Requested {requested} / selected {selected_count}。"
                f" all_candidate_moments_exhausted="
                f"`{str(shortfall['all_candidate_moments_exhausted']).lower()}`。"
            ),
            "",
            "### Decision ledger",
            "",
            (
                "| Frame ID | Decision | Type | Base | Coverage | Spoiler | "
                "Temporal | Marginal | Reasons |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        )
    )
    for item in selected:
        classification = _mapping(item["classification"])
        selection = _mapping(item["selection"])
        reasons = ", ".join(cast(list[str], selection["reason_codes"]))
        lines.append(
            f"| `{_abbreviate(str(item['image_id']), 12)}` | "
            f"selected {int(item['selection_index']):02d} | "
            f"{classification['blog_image_type']} | "
            f"{float(selection['base_utility']):.6f} | "
            f"{float(selection['coverage_bonus']):.6f} | "
            f"{float(selection['spoiler_penalty']):.6f} | "
            f"{float(selection['temporal_diversity_penalty']):.6f} | "
            f"**{float(selection['marginal_utility']):.6f}** | `{reasons}` |"
        )
    for item in near_misses[:10]:
        classification = _mapping(item["classification"])
        score = _mapping(item["counterfactual_selection"])
        rejection = _mapping(item["rejection"])
        lines.append(
            f"| `{_abbreviate(str(item['image_id']), 12)}` | "
            f"`{rejection['reason_code']}` | {classification['blog_image_type']} | "
            f"{float(score['base_utility']):.6f} | "
            f"{float(score['coverage_bonus']):.6f} | "
            f"{float(score['spoiler_penalty']):.6f} | "
            f"{float(score['temporal_diversity_penalty']):.6f} | "
            f"{float(score['marginal_utility']):.6f} | — |"
        )
    lines.extend(("", "### Stage provenance", ""))
    lines.extend(
        (
            "| Stage | Fingerprint | Cache | Duration | Contracts |",
            "|---|---|---|---:|---|",
        )
    )
    for stage in _mapping_list(provenance["stages"]):
        contracts = ", ".join(cast(list[str], stage["contract_refs"])) or "—"
        lines.append(
            f"| `{stage['name']}` | `{_abbreviate(str(stage['fingerprint']), 8)}` | "
            f"{stage['cache_hits']} hit / {stage['cache_misses']} miss | "
            f"{int(stage['duration_ms']) / 1000:.3f}s | `{contracts}` |"
        )
    lines.extend(("", "### Model and tool provenance", ""))
    schema = _mapping(report["schema"])
    lines.append(f"- Report schema: `{schema['name']}@{schema['version']}`")
    models = _mapping(provenance["models"])
    for role, model_value in models.items():
        model = _mapping(model_value)
        lines.append(
            f"- {role}: `{model['configured_name']}` @ "
            f"`{_abbreviate(str(model['execution_identity']), 18)}` "
            f"({model['update_status']})"
        )
    tools = _mapping(provenance["tools"])
    for name, version in tools.items():
        lines.append(f"- {name}: `{version}`")
    lines.extend(
        (
            "",
            "### Deliberately omitted",
            "",
            "- absolute input/cache/output paths",
            "- environment variables and credentials",
            (
                "- prompt bodies, model reasoning traces, raw model responses, "
                "and stack traces"
            ),
            "- generated screen-text quotations",
            "- raw Context Cue text（processing cacheだけに保持）",
            "",
            "完全なscore内訳、PTS/time base、Stage provenanceは"
            "[`report.json`](report.json)を参照する。",
            "",
        )
    )
    return "\n".join(lines)


def _append_selected(
    lines: list[str],
    item: dict[str, Any],
    sources: dict[object, dict[str, Any]],
) -> None:
    index = int(item["selection_index"])
    output = _mapping(item["output"])
    source = _mapping(item["source"])
    classification = _mapping(item["classification"])
    annotation = _mapping(item["annotation"])
    selection = _mapping(item["selection"])
    decision = _mapping(selection["decision_explanation"])
    source_video = sources[source["video_id"]]
    video_time = _mapping(source["video_time"])
    relative_path = str(output["relative_path"])
    lines.extend(
        (
            f"### {index:02d} — {annotation['summary']}",
            "",
            (
                f"[![{index:02d} — {classification['scene_display_name']}]"
                f"({relative_path})]({relative_path})"
            ),
            "",
            f"`{_abbreviate(str(item['image_id']), 12)}` · `{relative_path}`",
            "",
            f"- **画像の説明（model）**: {annotation['summary']}",
            (
                "- **Representative Frameの理由（model）**: "
                f"{annotation['representative_frame_reason']}"
            ),
            f"- **採用理由（selector）**: {decision['text']}",
            (
                f"- **Reason codes**: `"
                f"{', '.join(cast(list[str], selection['reason_codes']))}`"
            ),
            (
                f"- **Source**: Video {source_video['order']} "
                f"`{source_video['relative_path']}` (`{source['video_id']}`) · "
                f"`{video_time['display']}`"
            ),
            (
                f"- **Classification**: `{classification['scene_slug']}` · "
                f"`{classification['scene_selection_role']}` · "
                f"`{classification['blog_image_type']}` · spoiler "
                f"`{classification['spoiler_risk']}`"
            ),
            _context_line(annotation),
            f"- **Utility**: {float(selection['marginal_utility']):.6f}",
            "",
        )
    )
    evidence = classification["spoiler_evidence"]
    if evidence is not None:
        evidence_value = _mapping(evidence)
        lines.extend(
            (
                "<details>",
                "<summary>Spoiler evidence（model）</summary>",
                "",
                str(evidence_value["summary"]),
                "",
                "</details>",
                "",
            )
        )


def _context_line(annotation: dict[str, Any]) -> str:
    relevance = str(annotation["context_cue_relevance"])
    cue_ids = cast(list[str], annotation["supporting_context_cue_ids"])
    suffix = "" if not cue_ids else " · " + " / ".join(f"`{item}`" for item in cue_ids)
    return f"- **Context**: `{relevance}`{suffix}"


def _abbreviate(value: str, count: int) -> str:
    if value.startswith("frm_"):
        return f"frm_{value[4 : 4 + count]}…"
    if value.startswith("stg_"):
        return f"stg_{value[4 : 4 + count]}…"
    if len(value) <= count:
        return value
    return value[:count] + "…"


def _mapping(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], value)


def _mapping_list(value: object) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], value)
