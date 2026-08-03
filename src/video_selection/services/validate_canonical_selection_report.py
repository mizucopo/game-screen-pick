"""Canonical report schemaとstaging artifactのcross-validation。"""

import hashlib
import json
import math
import stat
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from PIL import Image, UnidentifiedImageError

from ..models.candidate_annotation import (
    SELECTION_COVERAGE_FACETS,
    candidate_annotation_free_text_is_safe,
)
from ..models.canonical_publication_request import CanonicalPublicationRequest
from ..models.report_value import string_looks_private
from .build_canonical_selection_report import REPORT_SCHEMA_VERSION
from .render_human_selection_report import render_human_selection_report
from .report_time import display_report_time, exact_seconds_string
from .serialize_canonical_selection_report import (
    serialize_canonical_selection_report,
)
from .validate_report_schema_compatibility import (
    validate_report_schema_compatibility,
)


def validate_canonical_selection_report(
    report: dict[str, object],
    staging_folder: Path,
    request: CanonicalPublicationRequest,
    *,
    allow_model_update_diagnostic_mismatch: bool = False,
) -> None:
    """schema、画像、JSON、Markdown、privacyの公開前整合を検証する。"""
    try:
        report_from_disk = _read_json_artifact(staging_folder)
        _validate_schema(report_from_disk)
        _validate_schema(report)
        _validate_json_artifact(report, report_from_disk, staging_folder)
        _validate_intrinsic_report_relationships(report)
        _validate_report_relationships(
            report,
            request,
            allow_model_update_diagnostic_mismatch=(
                allow_model_update_diagnostic_mismatch
            ),
        )
        _validate_images(report, staging_folder)
        _validate_markdown(report, staging_folder)
        _validate_privacy(report, staging_folder, request)
        validate_canonical_selection_output_layout(report, staging_folder)
    except PermissionError:
        raise
    except (
        AssertionError,
        FileNotFoundError,
        IsADirectoryError,
        KeyError,
        NotADirectoryError,
        UnidentifiedImageError,
        TypeError,
        ValueError,
    ) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            "Canonical Selection Report"
        ):
            raise
        msg = f"Canonical Selection Reportの検証に失敗しました: {error}"
        raise ValueError(msg) from error
    except OSError:
        raise


def load_validated_canonical_selection_report(
    output_folder: Path,
) -> dict[str, object]:
    """公開済みfolderを自己完結したCanonical outputとして再検証する。"""
    try:
        report = _read_json_artifact(output_folder)
        report_schema_major = validate_report_schema_compatibility(report)
        _validate_reader_schema(report, report_schema_major)
        _validate_json_artifact(report, report, output_folder)
        _validate_intrinsic_report_relationships(
            report,
            allow_unknown_run_status=True,
        )
        _validate_images(report, output_folder)
        _validate_markdown(report, output_folder)
        _validate_published_strings_are_private_safe(report)
        validate_canonical_selection_output_layout(report, output_folder)
        return report
    except PermissionError:
        raise
    except (
        AssertionError,
        FileNotFoundError,
        IsADirectoryError,
        KeyError,
        NotADirectoryError,
        UnidentifiedImageError,
        TypeError,
        ValueError,
    ) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            "Canonical Selection Report"
        ):
            raise
        msg = f"Canonical Selection Reportの検証に失敗しました: {error}"
        raise ValueError(msg) from error
    except OSError:
        raise


def _validate_schema(report: dict[str, object]) -> None:
    errors = sorted(
        _schema_validator().iter_errors(report),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path) or "root"
        raise ValueError(
            f"Canonical Selection Report schema不一致: {location}: {error.message}"
        )


def _validate_reader_schema(
    report: dict[str, object],
    report_schema_major: int,
) -> None:
    """対応majorの既知構造を検証し、minor追加fieldとenum値を許可する。"""
    errors = sorted(
        _reader_schema_validator(report_schema_major).iter_errors(report),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path) or "root"
        raise ValueError(
            f"Canonical Selection Report schema不一致: {location}: {error.message}"
        )


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    schema = _load_schema(REPORT_SCHEMA_VERSION)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@lru_cache(maxsize=2)
def _reader_schema_validator(major: int) -> Draft202012Validator:
    schema = _reader_compatible_schema(_load_schema(f"{major}.0.0"))
    schema_properties = cast(dict[str, Any], schema["properties"])
    schema_identity = cast(dict[str, Any], schema_properties["schema"])
    identity_properties = cast(dict[str, Any], schema_identity["properties"])
    version = cast(dict[str, Any], identity_properties["version"])
    version.pop("const")
    version["type"] = "string"
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _load_schema(version: str) -> dict[str, Any]:
    schema_path = (
        Path(__file__).parent.parent / "schemas" / f"report-{version}.schema.json"
    )
    schema_value: object = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(schema_value, dict):
        raise ValueError("Canonical Selection Report schemaがJSON objectではありません")
    return cast(dict[str, Any], schema_value)


def _reader_compatible_schema(value: object) -> Any:
    """既知構造を残し、対応major内の追加fieldと文字列enum追加を許可する。"""
    if isinstance(value, dict):
        schema = {key: _reader_compatible_schema(child) for key, child in value.items()}
        if schema.get("additionalProperties") is False:
            schema["additionalProperties"] = True
        one_of = schema.pop("oneOf", None)
        if isinstance(one_of, list):
            schema["anyOf"] = one_of
        enum = schema.get("enum")
        if (
            isinstance(enum, list)
            and enum
            and all(isinstance(item, str) for item in enum)
        ):
            schema.pop("enum")
            schema["type"] = "string"
        return schema
    if isinstance(value, list):
        return [_reader_compatible_schema(item) for item in value]
    return value


def _validate_json_artifact(
    report: dict[str, object],
    report_from_disk: dict[str, object],
    staging_folder: Path,
) -> None:
    report_path = staging_folder / "report.json"
    content = report_path.read_text(encoding="utf-8")
    if content != serialize_canonical_selection_report(report):
        raise ValueError("Canonical Selection Report JSON serializationが一致しません")
    if report_from_disk != report:
        raise ValueError("Canonical Selection Report JSON objectが一致しません")


def _read_json_artifact(staging_folder: Path) -> dict[str, object]:
    content = (staging_folder / "report.json").read_text(encoding="utf-8")
    parsed: object = json.loads(content, parse_constant=_reject_json_constant)
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise ValueError("Canonical Selection Report JSONはobjectである必要があります")
    return cast(dict[str, object], parsed)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"非有限JSON numberは使用できません: {value}")


def _validate_report_relationships(
    report: dict[str, object],
    request: CanonicalPublicationRequest,
    *,
    allow_model_update_diagnostic_mismatch: bool,
) -> None:
    """reportと今回のPublication Requestとの関係を検証する。"""
    run = _mapping(report["run"])
    warnings = _mapping_list(run["warnings"])
    selected = _mapping_list(report["selected"])
    rejection_summary = _mapping(report["rejection_summary"])
    warning_codes = [str(item["code"]) for item in warnings]
    selection = request.selection_result
    conditional_coverage = _mapping(
        _mapping(report["selection_summary"])["conditional_coverage"]
    )
    conditional_facets = _mapping(conditional_coverage["facets"])
    expected_conditional_facets = {
        facet: {
            "eligible": selection.selection_coverage_eligible_counts[facet],
            "minimum": selection.selection_coverage_minimums[facet],
            "actual": selection.selection_coverage_actuals[facet],
            "reallocated": selection.selection_coverage_reallocated[facet],
        }
        for facet in selection.selection_coverage_eligible_counts
    }
    if (
        ("selection_shortfall" in warning_codes) != selection.shortfall
        or run["requested_image_count"] != selection.requested_count
        or rejection_summary["by_reason"] != selection.rejection_counts
        or conditional_facets != expected_conditional_facets
        or [str(item["image_id"]) for item in selected]
        != [item.candidate.identifier for item in selection.selected]
    ):
        raise ValueError("Canonical Selection Reportのselection countが一致しません")
    _validate_warning_relationships(
        warnings,
        request,
        allow_model_update_diagnostic_mismatch=(allow_model_update_diagnostic_mismatch),
    )


def _validate_intrinsic_report_relationships(
    report: dict[str, object],
    *,
    allow_unknown_run_status: bool = False,
) -> None:
    """外部requestなしでCanonical report内部の参照と件数を検証する。"""
    run = _mapping(report["run"])
    warnings = _mapping_list(run["warnings"])
    selected = _mapping_list(report["selected"])
    near_misses = _mapping_list(report["near_misses"])
    selection_summary = _mapping(report["selection_summary"])
    rejection_summary = _mapping(report["rejection_summary"])
    warning_codes = [str(item["code"]) for item in warnings]
    expected_status = "completed_with_warnings" if warnings else "completed"
    run_status = str(run["status"])
    status_is_valid = run_status == expected_status or (
        allow_unknown_run_status
        and run_status not in {"completed", "completed_with_warnings"}
    )
    if (
        not status_is_valid
        or len(warning_codes) != len(set(warning_codes))
        or len(selected) != run["selected_image_count"]
        or len(selected) != selection_summary["selected"]
        or _mapping(selection_summary["shortfall"])["requested"]
        != run["requested_image_count"]
        or _mapping(selection_summary["shortfall"])["selected"] != len(selected)
        or len(near_misses)
        > _mapping(report["near_miss_publication"])["json_limit_for_this_run"]
        or rejection_summary["total"] != selection_summary["not_selected"]
        or rejection_summary["total"]
        != sum(cast(dict[str, int], rejection_summary["by_reason"]).values())
    ):
        raise ValueError("Canonical Selection Reportのselection countが一致しません")
    if "conditional_coverage" in selection_summary:
        conditional_coverage = _mapping(selection_summary["conditional_coverage"])
        conditional_facets = _mapping(conditional_coverage["facets"])
        applies = bool(conditional_coverage["applies"])
        if applies != (int(run["requested_image_count"]) >= 10):
            raise ValueError("Canonical Selection Reportの条件付きcoverageが不正です")
        for facet_name in SELECTION_COVERAGE_FACETS:
            facet = _mapping(conditional_facets[facet_name])
            eligible = int(facet["eligible"])
            minimum = int(facet["minimum"])
            actual = int(facet["actual"])
            reallocated = bool(facet["reallocated"])
            if (
                minimum != int(applies and eligible > 0)
                or actual > eligible
                or reallocated != (applies and actual == 0)
            ):
                raise ValueError(
                    "Canonical Selection Reportの条件付きcoverageが不正です"
                )
    selected_ids = {str(item["image_id"]) for item in selected}
    if len(selected_ids) != len(selected):
        raise ValueError(
            "Canonical Selection ReportのSelected Image IDが重複しています"
        )
    sources = _mapping_list(_mapping(report["video_set"])["sources"])
    source_ids = {str(item["id"]) for item in sources}
    if len(source_ids) != len(sources):
        raise ValueError("Canonical Selection ReportのVideo Source IDが重複しています")
    context_cues = _mapping_list(report["context_cues"])
    cue_ids = {str(item["id"]) for item in context_cues}
    if len(cue_ids) != len(context_cues):
        raise ValueError("Canonical Selection ReportのContext Cue IDが重複しています")
    rejection_reasons = set(cast(dict[str, int], rejection_summary["by_reason"]))
    published_reasons = {
        str(_mapping(item["rejection"])["reason_code"]) for item in near_misses
    }
    if not rejection_reasons <= published_reasons:
        raise ValueError(
            "Canonical Selection ReportのNear Miss reason coverageが不足しています"
        )
    for expected_index, item in enumerate(selected, start=1):
        if item["selection_index"] != expected_index:
            raise ValueError("Canonical Selection Reportのselection indexが不正です")
        _validate_candidate_record(item, source_ids, cue_ids)
        score = _mapping(item["selection"])
        _validate_score(score)
    for item in near_misses:
        _validate_candidate_record(item, source_ids, cue_ids)
        _validate_score(_mapping(item["counterfactual_selection"]))
        rejection = _mapping(item["rejection"])
        for field in ("blocked_by_image_id", "nearest_selected_image_id"):
            if field in rejection and rejection[field] not in selected_ids:
                raise ValueError(
                    "Canonical Selection Reportのrejection参照が解決できません"
                )
    _validate_semantic_group_relationships(selected, near_misses)
    source_durations = {
        str(item["id"]): Fraction(str(_mapping(item["duration"])["exact_seconds"]))
        for item in _mapping_list(_mapping(report["video_set"])["sources"])
    }
    for cue in context_cues:
        video_id = str(cue["video_id"])
        if video_id not in source_ids:
            raise ValueError("Canonical Selection ReportのContext Cue sourceが不正です")
        _validate_context_cue_time(cue, source_durations[video_id])
    stages = _mapping_list(_mapping(report["provenance"])["stages"])
    stage_fingerprints = {str(item["fingerprint"]) for item in stages}
    tools = set(_mapping(_mapping(report["provenance"])["tools"]))
    models = set(_mapping(_mapping(report["provenance"])["models"]))
    contracts = set(_mapping(_mapping(report["provenance"])["contracts"]))
    for stage in stages:
        if (
            not set(cast(list[str], stage["upstream_fingerprints"]))
            <= stage_fingerprints
            or not set(cast(list[str], stage["tool_refs"])) <= tools
            or not set(cast(list[str], stage["model_refs"])) <= models
            or not set(cast(list[str], stage["contract_refs"])) <= contracts
        ):
            raise ValueError(
                "Canonical Selection ReportのStage provenance参照が解決できません"
            )


def _validate_warning_relationships(
    warnings: list[dict[str, Any]],
    request: CanonicalPublicationRequest,
    *,
    allow_model_update_diagnostic_mismatch: bool,
) -> None:
    """warning codeとdomain resultの対応を検証する。"""
    by_code = {str(item["code"]): item for item in warnings}
    shortfall = by_code.get("selection_shortfall")
    if (
        shortfall is not None
        and _mapping(shortfall["details"]) != request.selection_result.rejection_counts
    ):
        raise ValueError(
            "Canonical Selection ReportのSelection Shortfall warningが不正です"
        )
    unavailable = by_code.get("model_update_unavailable")
    if allow_model_update_diagnostic_mismatch:
        return
    expected_roles = [
        role.value for role in request.resolved_models.unavailable_roles()
    ]
    if (unavailable is None) != (not expected_roles) or (
        unavailable is not None
        and _mapping(unavailable["details"])["roles"] != expected_roles
    ):
        raise ValueError("Canonical Selection Reportのmodel update warningが不正です")


def _validate_semantic_group_relationships(
    selected: list[dict[str, Any]],
    near_misses: list[dict[str, Any]],
) -> None:
    """Semantic Duplicate Groupの代表とblocking参照を検証する。"""
    selected_group_by_image: dict[str, tuple[str, str]] = {}
    selected_group_ids: set[str] = set()
    for item in selected:
        selection = _mapping(item["selection"])
        if "semantic_group" not in selection:
            continue
        selected_semantic_group = _mapping(selection["semantic_group"])
        group_id = str(selected_semantic_group["id"])
        if group_id in selected_group_ids:
            message = (
                "Canonical Selection Reportの"
                "Semantic Duplicate Group代表が重複しています"
            )
            raise ValueError(message)
        selected_group_ids.add(group_id)
        selected_group_by_image[str(item["image_id"])] = (
            group_id,
            str(selected_semantic_group["basis"]),
        )

    for item in near_misses:
        rejection = _mapping(item["rejection"])
        if rejection["reason_code"] != "semantic_duplicate":
            continue
        blocker_id = rejection.get("blocked_by_image_id")
        rejected_semantic_group = rejection.get("semantic_group")
        if blocker_id is None or rejected_semantic_group is None:
            raise ValueError(
                "Canonical Selection ReportのSemantic Duplicate Group参照がありません"
            )
        group = _mapping(rejected_semantic_group)
        expected_group = (str(group["id"]), str(group["basis"]))
        if selected_group_by_image.get(str(blocker_id)) != expected_group:
            raise ValueError(
                "Canonical Selection ReportのSemantic Duplicate Group参照が一致しません"
            )


def _validate_context_cue_time(
    cue: dict[str, Any],
    source_duration: Fraction,
) -> None:
    """Context Cueのbasis固有時刻と表示値を検証する。"""
    basis = str(cue["timestamp_basis"])
    start_value = _mapping(cue["start"])
    end_value = _mapping(cue["end"])
    start = _context_time_offset(start_value, basis)
    end = _context_time_offset(end_value, basis)
    if start < 0 or end <= start or end > source_duration:
        raise ValueError("Canonical Selection ReportのContext Cue timeが不正です")
    for value, offset in ((start_value, start), (end_value, end)):
        if value["display"] != display_report_time(offset) or (
            "exact_seconds" in value
            and value["exact_seconds"] != exact_seconds_string(offset)
        ):
            raise ValueError(
                "Canonical Selection ReportのContext Cue表示時刻が不正です"
            )


def _context_time_offset(value: dict[str, Any], basis: str) -> Fraction:
    if basis == "source_pts":
        if "source_pts" not in value:
            if "offset_seconds" not in value:
                raise ValueError(
                    "Canonical Selection ReportのContext Cue時刻がありません"
                )
            return _fraction(_mapping(value["offset_seconds"]))
        time_base = _fraction(_mapping(value["time_base"]))
        offset = _fraction(_mapping(value["offset_seconds"]))
        if (int(value["source_pts"]) - int(value["origin_pts"])) * time_base != offset:
            raise ValueError("Canonical Selection ReportのContext Cue PTSが不正です")
        return offset
    if basis == "container_text_ms":
        if "offset_seconds" not in value or "sample_index" in value:
            raise ValueError(
                "Canonical Selection Reportのcontainer text timeが不正です"
            )
        return _fraction(_mapping(value["offset_seconds"]))
    if "sample_index" in value:
        return Fraction(int(value["sample_index"]), int(value["sample_rate_hz"]))
    if "offset_seconds" in value:
        return _fraction(_mapping(value["offset_seconds"]))
    raise ValueError("Canonical Selection ReportのASR sample timeが不正です")


def _validate_candidate_record(
    item: dict[str, Any],
    source_ids: set[str],
    cue_ids: set[str],
) -> None:
    source = _mapping(item["source"])
    annotation = _mapping(item["annotation"])
    if source["video_id"] not in source_ids:
        raise ValueError("Canonical Selection ReportのVideo Source参照が不正です")
    if not set(cast(list[str], annotation["supporting_context_cue_ids"])) <= cue_ids:
        raise ValueError("Canonical Selection ReportのContext Cue参照が不正です")
    video_time = _mapping(source["video_time"])
    time_base = _fraction(_mapping(video_time["time_base"]))
    offset = _fraction(_mapping(video_time["offset_seconds"]))
    if (
        int(video_time["source_pts"]) - int(video_time["origin_pts"])
    ) * time_base != offset:
        raise ValueError("Canonical Selection Reportのsource PTSが不正です")


def _validate_score(score: dict[str, Any]) -> None:
    expected = (
        float(score["base_utility"])
        + float(score["coverage_bonus"])
        - float(score["spoiler_penalty"])
        - float(score["temporal_diversity_penalty"])
    )
    if not math.isclose(
        expected,
        float(score["marginal_utility"]),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Canonical Selection ReportのMarginal Selection Utilityが不正です"
        )


def _validate_images(report: dict[str, object], staging_folder: Path) -> None:
    selected = _mapping_list(report["selected"])
    prefixes: dict[str, int] = {}
    for item in selected:
        digest = str(item["image_id"])[4:]
        prefixes[digest[:12]] = prefixes.get(digest[:12], 0) + 1
    width = max(4, len(str(len(selected))))
    for expected_index, item in enumerate(selected, start=1):
        output = _mapping(item["output"])
        classification = _mapping(item["classification"])
        digest = str(item["image_id"])[4:]
        digest_part = digest if prefixes[digest[:12]] > 1 else digest[:12]
        expected_path = (
            f"images/{expected_index:0{width}d}_{classification['scene_slug']}_"
            f"{digest_part}.webp"
        )
        if output["relative_path"] != expected_path:
            raise ValueError(
                "Canonical Selection ReportのSelected Image filenameが不正です"
            )
        image_path = staging_folder / expected_path
        if not _is_regular_file(image_path):
            raise ValueError(
                "Canonical Selection ReportのSelected Imageが見つかりません"
            )
        content = image_path.read_bytes()
        if (
            hashlib.sha256(content).hexdigest() != output["sha256"]
            or len(content) != output["bytes"]
        ):
            raise ValueError(
                "Canonical Selection ReportのSelected Image hashが不正です"
            )
        with Image.open(image_path) as image:
            if (
                image.format != "WEBP"
                or image.size != (output["width"], output["height"])
                or image.getexif()
                or {"exif", "icc_profile", "xmp"} & image.info.keys()
            ):
                raise ValueError(
                    "Canonical Selection ReportのSelected Image encodingが不正です"
                )


def _validate_markdown(report: dict[str, object], staging_folder: Path) -> None:
    expected = render_human_selection_report(report)
    actual = (staging_folder / "report.md").read_text(encoding="utf-8")
    if actual != expected:
        raise ValueError(
            "Canonical Selection ReportのMarkdown projectionが一致しません"
        )


def _validate_privacy(
    report: dict[str, object],
    staging_folder: Path,
    request: CanonicalPublicationRequest,
) -> None:
    serialized = serialize_canonical_selection_report(report) + (
        staging_folder / "report.md"
    ).read_text(encoding="utf-8")
    private_paths = {
        str(request.video_set.input_folder.resolve(strict=False)),
        str(request.configuration.output_folder.resolve(strict=False)),
        str(request.configuration.processing_cache_folder.resolve(strict=False)),
        str(staging_folder.resolve(strict=False)),
    }
    raw_context_texts = tuple(
        cue.text
        for stage in request.video_stage_results
        for cue in stage.context.cues
        if cue.text
    )
    free_text_is_safe = candidate_annotation_free_text_is_safe(
        tuple(_published_free_text(report)),
        raw_context_texts,
    )
    if any(value in serialized for value in private_paths) or not free_text_is_safe:
        raise ValueError(
            "Canonical Selection Reportに非公開pathまたはContext Cueがあります"
        )
    _validate_published_strings_are_private_safe(report)


def _validate_published_strings_are_private_safe(
    report: dict[str, object],
) -> None:
    """公開reportに絶対pathやendpointらしい文字列がないことを検証する。"""
    for value in _all_strings(report):
        if string_looks_private(value):
            raise ValueError(
                "Canonical Selection Reportに絶対pathまたはendpointがあります"
            )


def _published_free_text(report: dict[str, object]) -> list[str]:
    """Context Cue由来になり得る公開自由文だけを返す。"""
    values: list[str] = []
    for item in (
        *_mapping_list(report["selected"]),
        *_mapping_list(report["near_misses"]),
    ):
        annotation = _mapping(item["annotation"])
        values.extend(
            (
                str(annotation["summary"]),
                str(annotation["representative_frame_reason"]),
            )
        )
        spoiler_evidence = _mapping(item["classification"])["spoiler_evidence"]
        if spoiler_evidence is not None:
            values.append(str(_mapping(spoiler_evidence)["summary"]))
    return values


def validate_canonical_selection_output_layout(
    report: dict[str, object],
    staging_folder: Path,
) -> None:
    """reportが参照するartifactだけでcanonical outputが構成されることを検証する。"""
    expected = {
        "images",
        "report.json",
        "report.md",
        *(
            str(_mapping(item["output"])["relative_path"])
            for item in _mapping_list(report["selected"])
        ),
    }
    actual = {
        path.relative_to(staging_folder).as_posix()
        for path in staging_folder.rglob("*")
    }
    if actual != expected or any(
        path.is_symlink() for path in staging_folder.rglob("*")
    ):
        raise ValueError("Canonical Selection Reportのstaging layoutが不正です")


def _all_strings(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            item for key, child in value.items() for item in (key, *_all_strings(child))
        ]
    if isinstance(value, list):
        return [item for child in value for item in _all_strings(child)]
    return [value] if isinstance(value, str) else []


def _fraction(value: dict[str, Any]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _mapping(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], value)


def _mapping_list(value: object) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], value)


def _is_regular_file(path: Path) -> bool:
    """欠損だけをFalseとし、access failureをcorruptionへ変換しない。"""
    try:
        mode = path.lstat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return False
    return stat.S_ISREG(mode)
