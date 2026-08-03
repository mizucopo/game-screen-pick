"""Canonical Selection Reportとatomic publicationのtest。"""

import json
import os
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from src.video_selection.models.model_role import ModelRole
from src.video_selection.models.selection_rejection_reason import (
    SelectionRejectionReason,
)
from src.video_selection.services.canonical_output_publisher import (
    CanonicalOutputPublisher,
)
from src.video_selection.services.render_human_selection_report import (
    render_human_selection_report,
)
from src.video_selection.services.serialize_canonical_selection_report import (
    serialize_canonical_selection_report,
)
from src.video_selection.services.validate_canonical_selection_report import (
    load_validated_canonical_selection_report,
    validate_canonical_selection_report,
)
from tests.video_selection.fakes.canonical_publication_factory import (
    build_canonical_publication_request,
)
from tests.video_selection.fakes.fake_model_runtime import FakeModelRuntime
from tests.video_selection.fakes.fake_video_stage_media_runtime import (
    FakeVideoStageMediaRuntime,
)


def test_selected_webp_and_reports_are_published_from_one_canonical_object(
    tmp_path: Path,
) -> None:
    """WebP、JSON、Markdownが同じCanonical Selection Reportから公開されること。

    Arrange:
        - 2枚要求に対して1枚を選択したVideo Set選定結果が用意される
        - raw Context Cue本文と絶対pathを持つ内部入力が用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - 元解像度、quality 95、metadataなしのWebPが公開されること
        - JSONのID、path、count、hash、reasonとMarkdown投影が一致すること
        - Selection Shortfallだけがwarning付き正常成果物として示されること
        - raw Context Cue本文と絶対pathが成果物へ含まれないこと
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    runtime = FakeVideoStageMediaRuntime()

    # Act
    report = cast(dict[str, Any], CanonicalOutputPublisher(runtime).publish(request))

    # Assert
    output_folder = request.configuration.output_folder
    assert sorted(
        path.relative_to(output_folder).as_posix() for path in output_folder.rglob("*")
    ) == [
        "images",
        "images/0001_test-scene_aaaaaaaaaaaa.webp",
        "report.json",
        "report.md",
    ]
    report_from_disk = json.loads(
        (output_folder / "report.json").read_text(encoding="utf-8")
    )
    assert report_from_disk == report
    assert report["schema"] == {
        "name": "game-screen-pick/report",
        "version": "2.1.0",
    }
    assert report["run"]["status"] == "completed_with_warnings"
    assert report["run"]["warnings"] == [
        {
            "code": "selection_shortfall",
            "message": "要求2枚に対して1枚を選択しました。",
            "details": {"similarity_ceiling": 1},
        }
    ]
    context_start = report["context_cues"][0]["start"]
    assert context_start["source_pts"] == 1000
    assert context_start["time_base"] == {"numerator": 1, "denominator": 1000}
    selected = report["selected"][0]
    image_path = output_folder / selected["output"]["relative_path"]
    image_bytes = image_path.read_bytes()
    assert selected["image_id"] == "frm_" + "a" * 64
    assert selected["output"]["bytes"] == len(image_bytes)
    assert selected["output"]["sha256"]
    assert selected["selection"]["reason_codes"] == [
        "high_quality",
        "high_explanation_value",
        "normal_gameplay_coverage",
    ]
    assert runtime.extracted_original_frame_calls == [
        (request.video_set.sources[0].path, 0, 15)
    ]
    with Image.open(image_path) as image:
        assert image.format == "WEBP"
        assert image.size == (64, 48)
        assert not image.getexif()
        assert not ({"exif", "icc_profile", "xmp"} & image.info.keys())
    markdown = (output_folder / "report.md").read_text(encoding="utf-8")
    assert (
        "[![01 — 探索](images/0001_test-scene_aaaaaaaaaaaa.webp)]"
        "(images/0001_test-scene_aaaaaaaaaaaa.webp)"
    ) in markdown
    assert "`similarity_ceiling`" in markdown
    serialized = (output_folder / "report.json").read_text(encoding="utf-8") + markdown
    assert "公開してはいけない秘密の台詞" not in serialized
    assert str(tmp_path) not in serialized
    golden_folder = Path(__file__).parents[1] / "fixtures" / "canonical_publication"
    normalized = deepcopy(report)
    for item in cast(list[dict[str, Any]], normalized["selected"]):
        output = cast(dict[str, Any], item["output"])
        output["sha256"] = "<sha256>"
        output["bytes"] = "<bytes>"
    assert normalized == json.loads(
        (golden_folder / "report.normalized.json").read_text(encoding="utf-8")
    )
    assert markdown == (golden_folder / "report.md").read_text(encoding="utf-8")


def test_semantic_duplicate_group_evidence_is_published(tmp_path: Path) -> None:
    """Semantic Duplicate Groupのprivacy-safeな比較根拠が公開されること。

    Arrange:
        - 同じSemantic Duplicate Groupの代表と未採用候補を持つ選定結果が用意される
    Act:
        - Canonical Selection Reportが公開される
    Assert:
        - report schema minorが更新され代表と除外へ同じGroup IDと根拠が記録されること
        - 除外候補へsemantic duplicate reasonとblocking selected IDが記録されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    selection = request.selection_result
    selected = selection.selected[0]
    rejected = selection.rejected[0]
    semantic_group_id = "semantic_" + "3" * 64
    request = replace(
        request,
        selection_result=replace(
            selection,
            selected=(
                replace(
                    selected,
                    reason_codes=(
                        *selected.reason_codes,
                        "semantic_group_representative",
                    ),
                    semantic_group_id=semantic_group_id,
                    semantic_group_basis="combat_encounter_sequence",
                ),
            ),
            rejected=(
                replace(
                    rejected,
                    reason_code=SelectionRejectionReason.SEMANTIC_DUPLICATE,
                    blocked_by_image_id=selected.candidate.identifier,
                    nearest_selected_image_id=None,
                    similarity=None,
                    semantic_group_id=semantic_group_id,
                    semantic_group_basis="combat_encounter_sequence",
                ),
            ),
        ),
    )

    # Act
    report = cast(
        dict[str, Any],
        CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(request),
    )

    # Assert
    assert report["schema"]["version"] == "2.1.0"
    selected_record = report["selected"][0]
    near_miss = report["near_misses"][0]
    assert selected_record["selection"]["semantic_group"] == {
        "id": semantic_group_id,
        "basis": "combat_encounter_sequence",
    }
    assert near_miss["rejection"] == {
        "reason_code": "semantic_duplicate",
        "blocked_by_image_id": selected.candidate.identifier,
        "semantic_group": {
            "id": semantic_group_id,
            "basis": "combat_encounter_sequence",
        },
    }
    markdown = (request.configuration.output_folder / "report.md").read_text(
        encoding="utf-8"
    )
    assert "combat_encounter_sequence" in markdown
    assert "semantic_duplicate" in markdown


def test_colliding_digest_prefixes_expand_only_affected_output_names(
    tmp_path: Path,
) -> None:
    """同じ12文字prefixの選択IDだけが完全digest filenameへ拡張されること。

    Arrange:
        - Frame Candidate IDの先頭12文字が衝突する2枚の完全選定結果が用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - 両方のfilenameだけが完全64文字digestへ拡張されること
        - warningなしのcompleted reportが公開されること
    """
    # Arrange
    request = build_canonical_publication_request(
        tmp_path,
        shortfall=False,
        colliding_digest_prefixes=True,
    )

    # Act
    report = cast(
        dict[str, Any],
        CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(request),
    )

    # Assert
    paths = [item["output"]["relative_path"] for item in report["selected"]]
    assert paths == [
        "images/0001_test-scene_" + "123456789abc" + "a" * 52 + ".webp",
        "images/0002_test-scene_" + "123456789abc" + "b" * 52 + ".webp",
    ]
    assert report["run"]["status"] == "completed"
    assert report["run"]["warnings"] == []


def test_conditional_coverage_counts_and_reallocation_are_published(
    tmp_path: Path,
) -> None:
    """条件付きcoverageの候補数、最低数、実績、再配分が公開されること。

    Arrange:
        - 通常戦闘が選択済みでイベントが未採用の10枚要求結果が用意される
    Act:
        - Canonical Selection Reportが公開される
    Assert:
        - 通常戦闘の最低枠達成とイベント枠の再配分がJSONへ記録されること
        - 同じ値がMarkdownへ投影されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    selection = request.selection_result
    selected = selection.selected[0]
    rejected = selection.rejected[0]
    selected_candidate = replace(
        selected.candidate,
        annotation=replace(
            selected.candidate.annotation,
            combat_encounter_kind="ordinary",
            combat_encounter_basis="ordinary_opponent_presentation",
        ),
    )
    event_candidate = replace(
        rejected.candidate,
        annotation=replace(
            rejected.candidate.annotation,
            blog_image_type="event",
        ),
    )
    request = replace(
        request,
        configuration=replace(request.configuration, image_count=10),
        selection_result=replace(
            selection,
            selected=(replace(selected, candidate=selected_candidate),),
            rejected=(replace(rejected, candidate=event_candidate),),
            requested_count=10,
        ),
    )

    # Act
    report = cast(
        dict[str, Any],
        CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(request),
    )

    # Assert
    coverage = cast(
        dict[str, Any],
        cast(dict[str, Any], report["selection_summary"])["conditional_coverage"],
    )
    assert coverage == {
        "applies": True,
        "minimum_requested_image_count": 10,
        "facets": {
            "ordinary_combat": {
                "eligible": 1,
                "minimum": 1,
                "actual": 1,
                "reallocated": False,
            },
            "event": {
                "eligible": 1,
                "minimum": 1,
                "actual": 0,
                "reallocated": True,
            },
        },
    }
    markdown = (request.configuration.output_folder / "report.md").read_text(
        encoding="utf-8"
    )
    assert "| ordinary_combat | 1 | 1 | 1 | `false` |" in markdown
    assert "| event | 1 | 1 | 0 | `true` |" in markdown


def test_duplicate_video_source_ids_are_rejected_before_publication(
    tmp_path: Path,
) -> None:
    """重複したVideo Source IDを持つCanonical reportが拒否されること。

    Arrange:
        - 正常公開済みreportへ同じIDのVideo Source recordが追加される
    Act:
        - Canonical Selection Reportの公開前検証が実行される
    Assert:
        - Video Source ID重複として拒否されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    output_folder = request.configuration.output_folder
    report = cast(
        dict[str, Any],
        CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(request),
    )
    video_set = cast(dict[str, Any], report["video_set"])
    sources = cast(list[dict[str, Any]], video_set["sources"])
    sources.append(deepcopy(sources[0]))
    (output_folder / "report.json").write_text(
        serialize_canonical_selection_report(report),
        encoding="utf-8",
    )

    # Act
    # Assert
    with pytest.raises(ValueError, match="Video Source IDが重複"):
        validate_canonical_selection_report(report, output_folder, request)


def test_short_context_cue_text_does_not_trigger_privacy_false_positive(
    tmp_path: Path,
) -> None:
    """短いContext Cue本文が構造化値との偶然一致で拒否されないこと。

    Arrange:
        - 本文が1文字だけのContext Cueを持つpublication requestが用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - 公開free textへ逐語転載されていない成果物が正常公開されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    stage = request.video_stage_results[0]
    cue = replace(stage.context.cues[0], text="1")
    request = replace(
        request,
        video_stage_results=(
            replace(stage, context=replace(stage.context, cues=(cue,))),
        ),
    )

    # Act
    report = cast(
        dict[str, Any],
        CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(request),
    )

    # Assert
    assert report["run"]["status"] == "completed_with_warnings"
    assert request.configuration.output_folder.is_dir()


def test_near_miss_free_text_is_escaped_in_markdown_table(tmp_path: Path) -> None:
    """Near Missのmodel自由文に含まれる表区切り文字がescapeされること。

    Arrange:
        - summaryに縦棒を含むNear Missが用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - Markdown表へescape済みsummaryが描画されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    rejected = request.selection_result.rejected[0]
    annotation = replace(
        rejected.candidate.annotation,
        summary="HP | MPを比較する画面。",
    )
    candidate = replace(rejected.candidate, annotation=annotation)
    request = replace(
        request,
        selection_result=replace(
            request.selection_result,
            rejected=(replace(rejected, candidate=candidate),),
        ),
    )

    # Act
    CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(request)

    # Assert
    markdown = (request.configuration.output_folder / "report.md").read_text(
        encoding="utf-8"
    )
    assert "HP \\| MPを比較する画面。" in markdown
    assert "HP | MPを比較する画面。" not in markdown


def test_non_aligned_source_pts_context_cue_uses_exact_offset_fallback(
    tmp_path: Path,
) -> None:
    """source PTS gridに非整列のCue時刻がexact offsetで公開されること。

    Arrange:
        - source PTSから整数origin PTSを復元できないContext Cueが用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - timestamp basisを保ったままlosslessなoffset時刻が公開されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    stage = request.video_stage_results[0]
    original_cue = stage.context.cues[0]
    provenance = original_cue.provenance
    if provenance is None:
        raise AssertionError("fixtureにContext Cue provenanceが必要です")
    cue = replace(
        original_cue,
        start=Fraction(29, 30),
        end=Fraction(59, 30),
        provenance=replace(
            provenance,
            source_pts=1000,
            source_time_base=Fraction(1, 1000),
        ),
    )
    request = replace(
        request,
        video_stage_results=(
            replace(stage, context=replace(stage.context, cues=(cue,))),
        ),
    )

    # Act
    report = cast(
        dict[str, Any],
        CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(request),
    )

    # Assert
    context = report["context_cues"][0]
    assert context["timestamp_basis"] == "source_pts"
    assert context["start"] == {
        "offset_seconds": {"numerator": 29, "denominator": 30},
        "display": "00:00:00.967",
    }
    assert context["end"] == {
        "offset_seconds": {"numerator": 59, "denominator": 30},
        "display": "00:00:01.967",
    }


@pytest.mark.parametrize(
    ("checkpoint", "error"),
    [
        ("before-image-write", PermissionError("permission denied")),
        ("before-report-json-write", OSError("disk full")),
        ("before-markdown-render", RuntimeError("renderer failed")),
        ("before-flush", OSError("flush failed")),
        ("before-rename", OSError("rename failed")),
    ],
)
def test_publication_faults_leave_no_output_folder(
    tmp_path: Path,
    checkpoint: str,
    error: Exception,
) -> None:
    """書込み、権限、rename faultでfinal Output Folderが残されないこと。

    Arrange:
        - 公開途中の指定checkpointで失敗するfault injectorが用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - faultが呼出元へ返されること
        - final Output Folderとstaging Folderが残されないこと
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)

    def inject(actual: str, _staging_folder: Path) -> None:
        if actual == checkpoint:
            raise error

    # Act
    # Assert
    with pytest.raises(type(error), match=str(error)):
        CanonicalOutputPublisher(
            FakeVideoStageMediaRuntime(),
            fault_injector=inject,
        ).publish(request)
    assert not request.configuration.output_folder.exists()
    assert not tuple(tmp_path.glob(".output.*.staging"))


def test_fault_after_final_rename_preserves_reusable_completed_output(
    tmp_path: Path,
) -> None:
    """final rename後のflush失敗でも完成済みoutputが次回再利用されること。

    Arrange:
        - final rename直後に親directory flush失敗を注入するpublisherが用意される
    Act:
        - 初回失敗後に同じPublication Requestが再実行される
    Assert:
        - rename済みの検証済みoutputが削除されないこと
        - 次回runで全artifactがbyte変更なしに再利用されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)

    def fail_after_rename(checkpoint: str, _folder: Path) -> None:
        if checkpoint == "after-rename-before-parent-flush":
            raise OSError("parent directory flush failed")

    with pytest.raises(OSError, match="parent directory flush failed"):
        CanonicalOutputPublisher(
            FakeVideoStageMediaRuntime(),
            fault_injector=fail_after_rename,
        ).publish(request)
    output_folder = request.configuration.output_folder
    before = {
        path.relative_to(output_folder): path.read_bytes()
        for path in output_folder.rglob("*")
        if path.is_file()
    }
    retry_runtime = FakeVideoStageMediaRuntime()

    # Act
    publisher = CanonicalOutputPublisher(retry_runtime)
    publisher.publish(request)

    # Assert
    after = {
        path.relative_to(output_folder): path.read_bytes()
        for path in output_folder.rglob("*")
        if path.is_file()
    }
    assert publisher.reused_completed_publication is True
    assert after == before
    assert retry_runtime.extracted_original_frame_calls == []


def test_reader_revalidates_historical_report_with_its_major_schema(
    tmp_path: Path,
) -> None:
    """履歴reportが対応major固有のschemaとprojectionで再検証されること。

    Arrange:
        - report@2の完成outputから追加fieldを除いた履歴report@1が用意される
    Act:
        - 完成Canonical Outputのreader検証が実行される
    Assert:
        - report@1として再検証され同じobjectが返されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    report = cast(
        dict[str, Any],
        CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(request),
    )
    output_folder = request.configuration.output_folder
    cast(dict[str, Any], report["schema"])["version"] = "1.0.0"
    cast(dict[str, Any], report["selection_summary"]).pop("conditional_coverage")
    report_value = cast(dict[str, object], report)
    (output_folder / "report.json").write_text(
        serialize_canonical_selection_report(report_value),
        encoding="utf-8",
    )
    markdown_path = output_folder / "report.md"
    current_markdown = markdown_path.read_text(encoding="utf-8")
    summary, coverage_and_later = current_markdown.split(
        "\nConditional coverage:",
        maxsplit=1,
    )
    _coverage, selected_and_later = coverage_and_later.split(
        "\n## Selected images",
        maxsplit=1,
    )
    markdown_path.write_text(
        (summary + "\n## Selected images" + selected_and_later).replace(
            "game-screen-pick/report@2.1.0",
            "game-screen-pick/report@1.0.0",
        ),
        encoding="utf-8",
    )

    # Act
    loaded = load_validated_canonical_selection_report(output_folder)

    # Assert
    assert loaded == report


def test_reader_accepts_future_minor_unknown_fields_and_enum_values(
    tmp_path: Path,
) -> None:
    """対応majorの将来minorにある未知fieldとenum値が保持されること。

    Arrange:
        - report@2の完成outputへ将来minor、未知field、未知enum値が追加される
    Act:
        - 完成Canonical Outputのreader検証が実行される
    Assert:
        - 未知値を失わず同じobjectが返されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    report = cast(
        dict[str, Any],
        CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(request),
    )
    output_folder = request.configuration.output_folder
    cast(dict[str, Any], report["schema"])["version"] = "2.8.0"
    cast(dict[str, Any], report["run"])["status"] = "completed_with_diagnostics"
    report["future_field"] = {"future_enum": "new_value"}
    selected = cast(list[dict[str, Any]], report["selected"])
    cast(dict[str, Any], selected[0]["classification"])["blog_image_type"] = (
        "future_gameplay"
    )
    report_value = cast(dict[str, object], report)
    (output_folder / "report.md").write_text(
        render_human_selection_report(report_value),
        encoding="utf-8",
    )
    conditional_coverage = cast(
        dict[str, Any],
        cast(dict[str, Any], report["selection_summary"])["conditional_coverage"],
    )
    cast(dict[str, Any], conditional_coverage["facets"])["future_facet"] = {
        "future_metadata": "new_value"
    }
    (output_folder / "report.json").write_text(
        serialize_canonical_selection_report(report_value),
        encoding="utf-8",
    )

    # Act
    loaded = load_validated_canonical_selection_report(output_folder)

    # Assert
    assert loaded == report


def test_completed_output_permission_failure_never_deletes_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完成Outputのaccess障害が破損修復へ変換されないこと。

    Arrange:
        - 検証済みCanonical Outputとreportだけを拒否するfilesystem障害が用意される
    Act:
        - 完成Outputが再検証される
    Assert:
        - PermissionErrorが返され全artifact bytesが保持されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(request)
    output_folder = request.configuration.output_folder
    before = {
        path.relative_to(output_folder): path.read_bytes()
        for path in output_folder.rglob("*")
        if path.is_file()
    }
    report_path = output_folder / "report.json"
    original_read_text = Path.read_text

    def deny_report_read(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if path == report_path:
            raise PermissionError("injected output permission failure")
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", deny_report_read)

    # Act
    # Assert
    with pytest.raises(PermissionError, match="injected output permission failure"):
        load_validated_canonical_selection_report(output_folder)
    after = {
        path.relative_to(output_folder): path.read_bytes()
        for path in output_folder.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_completed_image_type_permission_failure_never_deletes_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完成画像のtype検査access障害でOutputが削除されないこと。

    Arrange:
        - 検証済みCanonical Outputと画像lstatだけを拒否する障害が用意される
    Act:
        - 完成Outputが再検証される
    Assert:
        - PermissionErrorが返され全artifact bytesが保持されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(request)
    output_folder = request.configuration.output_folder
    before = {
        path.relative_to(output_folder): path.read_bytes()
        for path in output_folder.rglob("*")
        if path.is_file()
    }
    image_path = next((output_folder / "images").glob("*.webp"))
    original_lstat = Path.lstat

    def deny_image_lstat(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        if path == image_path:
            raise PermissionError("injected image type permission failure")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", deny_image_lstat)

    # Act
    # Assert
    with pytest.raises(
        PermissionError,
        match="injected image type permission failure",
    ):
        load_validated_canonical_selection_report(output_folder)
    after = {
        path.relative_to(output_folder): path.read_bytes()
        for path in output_folder.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("artifact", ["report.json", "report.md"])
def test_schema_or_renderer_mismatch_leaves_no_output_folder(
    tmp_path: Path,
    artifact: str,
) -> None:
    """schema不正またはMarkdown projection不一致が公開前に拒否されること。

    Arrange:
        - 最終validation直前にJSONまたはMarkdownを破損するfaultが用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - cross-artifact validationが失敗すること
        - final Output Folderとstaging Folderが残されないこと
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)

    def corrupt(actual: str, staging_folder: Path) -> None:
        if actual != "before-validation":
            return
        path = staging_folder / artifact
        path.write_text("{}\n" if artifact.endswith("json") else "broken\n")

    # Act
    # Assert
    with pytest.raises(ValueError, match="Canonical Selection Report"):
        CanonicalOutputPublisher(
            FakeVideoStageMediaRuntime(),
            fault_injector=corrupt,
        ).publish(request)
    assert not request.configuration.output_folder.exists()
    assert not tuple(tmp_path.glob(".output.*.staging"))


def test_unknown_nested_field_is_rejected_by_exact_producer_schema(
    tmp_path: Path,
) -> None:
    """report@2.1.0の既知objectへ追加された未知fieldが拒否されること。

    Arrange:
        - validation直前にVideo Time契約へ未知fieldを追加するfaultが用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - exact producer schema不一致として失敗すること
        - final Output Folderが残されないこと
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)

    def add_unknown_field(actual: str, staging_folder: Path) -> None:
        if actual != "before-validation":
            return
        path = staging_folder / "report.json"
        report = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        video_set = cast(dict[str, Any], report["video_set"])
        time_contract = cast(dict[str, Any], video_set["time_contract"])
        time_contract["future_field"] = "not-in-report-2.1.0"
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # Act
    # Assert
    with pytest.raises(ValueError, match="schema不一致"):
        CanonicalOutputPublisher(
            FakeVideoStageMediaRuntime(),
            fault_injector=add_unknown_field,
        ).publish(request)
    assert not request.configuration.output_folder.exists()


def test_publication_uses_one_final_directory_rename(tmp_path: Path) -> None:
    """検証済みstaging Folderが一回のdirectory renameだけで公開されること。

    Arrange:
        - rename呼出しを記録して実際のrenameを行う境界が用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - hidden sibling stagingからfinal Output Folderへのrenameが一回だけ行われること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    calls: list[tuple[Path, Path]] = []

    def rename(source: Path, destination: Path) -> None:
        calls.append((source, destination))
        source.rename(destination)

    # Act
    CanonicalOutputPublisher(
        FakeVideoStageMediaRuntime(),
        directory_renamer=rename,
    ).publish(request)

    # Assert
    assert len(calls) == 1
    source, destination = calls[0]
    assert source.parent == destination.parent
    assert source.name.startswith(".output.")
    assert source.name.endswith(".staging")
    assert destination == request.configuration.output_folder


@pytest.mark.parametrize(
    ("move_before_failure", "output_is_preserved"),
    [(False, False), (True, True)],
)
def test_directory_rename_failure_preserves_only_completed_moved_output(
    tmp_path: Path,
    move_before_failure: bool,
    output_is_preserved: bool,
) -> None:
    """実rename境界の失敗で移動済み完成成果物だけが保持されること。

    Arrange:
        - staging移動前または移動直後に失敗するdirectory renamerが用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - rename errorが呼出元へ返されること
        - 移動前はoutputがなく、移動後は検証済みoutputが保持されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)

    def fail_rename(source: Path, destination: Path) -> None:
        if move_before_failure:
            source.rename(destination)
        raise OSError("directory rename failed")

    # Act
    # Assert
    with pytest.raises(OSError, match="directory rename failed"):
        CanonicalOutputPublisher(
            FakeVideoStageMediaRuntime(),
            directory_renamer=fail_rename,
        ).publish(request)
    assert request.configuration.output_folder.exists() is output_is_preserved
    assert not tuple(tmp_path.glob(".output.*.staging"))


def test_nonempty_output_is_rejected_before_frame_extraction(tmp_path: Path) -> None:
    """非空Output Folderがframe再抽出より前に拒否されること。

    Arrange:
        - 既存fileを持つOutput Folderが用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - Output Folder契約違反が返されること
        - selected frameが再抽出されないこと
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    output_folder = request.configuration.output_folder
    output_folder.mkdir()
    (output_folder / "existing.txt").write_text("existing", encoding="utf-8")
    runtime = FakeVideoStageMediaRuntime()

    # Act
    # Assert
    with pytest.raises(ValueError, match="存在しないか空"):
        CanonicalOutputPublisher(runtime).publish(request)
    assert runtime.extracted_original_frame_calls == []


def test_empty_output_is_removed_and_replaced_by_atomic_publication(
    tmp_path: Path,
) -> None:
    """既存の空Output Folderが処理前に除かれ正常成果物へ置換されること。

    Arrange:
        - 既存する空Output Folderと完全なpublication requestが用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - 空Folderが検証後に除かれ完全な成果物だけが公開されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    output_folder = request.configuration.output_folder
    output_folder.mkdir()

    # Act
    CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(request)

    # Assert
    assert (output_folder / "report.json").is_file()
    assert (output_folder / "report.md").is_file()
    assert len(tuple((output_folder / "images").glob("*.webp"))) == 1


def test_completed_publication_is_reused_without_changing_output(
    tmp_path: Path,
) -> None:
    """公開後の完了記録前に中断されても同じoutputがそのまま再利用されること。

    Arrange:
        - atomic publicationまで完了したCanonical outputが用意される
        - 呼出元だけが完了を記録できず同じrequestが再実行される
    Act:
        - 同じOutput FolderへCanonical Output Publisherが再実行される
    Assert:
        - 既存reportが正常結果として返されること
        - 全artifact byteが変更されずframeも再抽出されないこと
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    first_report = CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(
        request
    )
    output_folder = request.configuration.output_folder
    before = {
        path.relative_to(output_folder): path.read_bytes()
        for path in output_folder.rglob("*")
        if path.is_file()
    }
    retry_runtime = FakeVideoStageMediaRuntime()

    # Act
    resumed_publisher = CanonicalOutputPublisher(retry_runtime)
    resumed_report = resumed_publisher.publish(request)

    # Assert
    after = {
        path.relative_to(output_folder): path.read_bytes()
        for path in output_folder.rglob("*")
        if path.is_file()
    }
    assert resumed_report == first_report
    assert resumed_publisher.reused_completed_publication is True
    assert after == before
    assert retry_runtime.extracted_original_frame_calls == []
    assert not tuple(tmp_path.glob(".output.*.staging"))


@pytest.mark.parametrize("first_update_is_unavailable", [False, True])
def test_completed_publication_is_reused_across_model_update_diagnostics(
    tmp_path: Path,
    first_update_is_unavailable: bool,
) -> None:
    """model更新確認結果だけが変わっても完成済みoutputが再利用されること。

    Arrange:
        - 同じ実行model identityで更新確認の成否だけが異なるrequestが用意される
        - 一方のrequestで完成したCanonical outputが用意される
    Act:
        - もう一方の更新確認結果を持つrequestでpublicationが再実行される
    Assert:
        - 完成済みreportと全artifact byteがそのまま返されること
        - selected frameが再抽出されないこと
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path, shortfall=False)
    unavailable = frozenset({ModelRole.CANDIDATE_ANNOTATION})
    first_models = FakeModelRuntime(
        "canonical-publication",
        unavailable_roles=unavailable if first_update_is_unavailable else frozenset(),
    ).resolve_models(request.configuration)
    first_request = replace(request, resolved_models=first_models)
    first_report = CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(
        first_request
    )
    output_folder = request.configuration.output_folder
    before = {
        path.relative_to(output_folder): path.read_bytes()
        for path in output_folder.rglob("*")
        if path.is_file()
    }
    retry_models = FakeModelRuntime(
        "canonical-publication",
        unavailable_roles=frozenset() if first_update_is_unavailable else unavailable,
    ).resolve_models(request.configuration)
    retry_request = replace(request, resolved_models=retry_models)
    retry_runtime = FakeVideoStageMediaRuntime()
    publisher = CanonicalOutputPublisher(retry_runtime)

    # Act
    resumed_report = publisher.publish(retry_request)

    # Assert
    after = {
        path.relative_to(output_folder): path.read_bytes()
        for path in output_folder.rglob("*")
        if path.is_file()
    }
    assert resumed_report == first_report
    assert publisher.reused_completed_publication is True
    assert after == before
    assert retry_runtime.extracted_original_frame_calls == []
    assert not tuple(tmp_path.glob(".output.*.staging"))


def test_completed_publication_with_different_semantics_is_preserved_and_rejected(
    tmp_path: Path,
) -> None:
    """既存outputと異なる意味結果が上書きされず明示的に拒否されること。

    Arrange:
        - similarity設定0.72で完成したCanonical outputが用意される
        - 同じOutput Folderへsimilarity設定0.80のrequestが用意される
    Act:
        - 変更後requestでCanonical Output Publisherが実行される
    Assert:
        - 意味結果の不一致が返されること
        - 既存artifactが一byteも変更されないこと
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(request)
    output_folder = request.configuration.output_folder
    before = {
        path.relative_to(output_folder): path.read_bytes()
        for path in output_folder.rglob("*")
        if path.is_file()
    }
    changed_request = replace(
        request,
        configuration=replace(
            request.configuration,
            similarity_threshold=0.80,
        ),
    )

    # Act
    # Assert
    with pytest.raises(ValueError, match="意味結果と一致しません"):
        CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(changed_request)
    after = {
        path.relative_to(output_folder): path.read_bytes()
        for path in output_folder.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not tuple(tmp_path.glob(".output.*.staging"))


def test_completed_selected_image_survives_later_image_failure(
    tmp_path: Path,
) -> None:
    """後続画像の公開失敗後もencode済みSelected Imageが再利用されること。

    Arrange:
        - 2枚選定され、2枚目の処理開始前だけ失敗するpublisherが用意される
    Act:
        - 初回失敗後に同じPublication Requestが再実行される
    Assert:
        - retryでは2枚目だけが元動画から再抽出されること
        - 最終Output Folderへ2枚とも完全に公開されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path, shortfall=False)
    fault_count = 0

    def fail_second_image(checkpoint: str, _staging: Path) -> None:
        nonlocal fault_count
        if checkpoint != "before-image-write":
            return
        fault_count += 1
        if fault_count == 2:
            raise OSError("injected second image failure")

    first_runtime = FakeVideoStageMediaRuntime()
    with pytest.raises(OSError, match="injected second image failure"):
        CanonicalOutputPublisher(
            first_runtime,
            fault_injector=fail_second_image,
        ).publish(request)
    retry_runtime = FakeVideoStageMediaRuntime()

    # Act
    CanonicalOutputPublisher(retry_runtime).publish(request)

    # Assert
    assert first_runtime.extracted_original_frame_calls == [
        (request.video_set.sources[0].path, 0, 15)
    ]
    assert retry_runtime.extracted_original_frame_calls == [
        (request.video_set.sources[0].path, 0, 30)
    ]
    published_images = tuple(
        (request.configuration.output_folder / "images").glob("*.webp")
    )
    assert len(published_images) == 2


def test_resumed_publication_matches_uninterrupted_semantic_output(
    tmp_path: Path,
) -> None:
    """画像途中から再開しても中断なしと同じ公開outputになること。

    Arrange:
        - 同じ意味入力を持つ中断なし用と再開用のPublication Requestが用意される
        - 再開用publisherは2枚目の開始前に一度だけ失敗する
    Act:
        - 中断なしrunと、失敗後に再開したrunがそれぞれ公開される
    Assert:
        - report全体、選択ID・順序、全WebP bytesが一致すること
        - 再開時に確定済み1枚目を元動画から再抽出しないこと
    """
    # Arrange
    clean_root = tmp_path / "clean"
    resumed_root = tmp_path / "resumed"
    clean_root.mkdir()
    resumed_root.mkdir()
    clean_request = build_canonical_publication_request(
        clean_root,
        shortfall=False,
    )
    resumed_request = build_canonical_publication_request(
        resumed_root,
        shortfall=False,
    )
    CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(clean_request)
    image_attempt = 0

    def fail_second_image(checkpoint: str, _staging: Path) -> None:
        nonlocal image_attempt
        if checkpoint != "before-image-write":
            return
        image_attempt += 1
        if image_attempt == 2:
            raise OSError("injected publication interruption")

    with pytest.raises(OSError, match="injected publication interruption"):
        CanonicalOutputPublisher(
            FakeVideoStageMediaRuntime(),
            fault_injector=fail_second_image,
        ).publish(resumed_request)
    retry_runtime = FakeVideoStageMediaRuntime()

    # Act
    CanonicalOutputPublisher(retry_runtime).publish(resumed_request)

    # Assert
    clean_report = json.loads(
        (clean_request.configuration.output_folder / "report.json").read_text(
            encoding="utf-8"
        )
    )
    resumed_report = json.loads(
        (resumed_request.configuration.output_folder / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert resumed_report == clean_report
    assert [item["image_id"] for item in resumed_report["selected"]] == [
        item["image_id"] for item in clean_report["selected"]
    ]
    clean_images = {
        path.relative_to(clean_request.configuration.output_folder): path.read_bytes()
        for path in (clean_request.configuration.output_folder / "images").glob(
            "*.webp"
        )
    }
    resumed_images = {
        path.relative_to(resumed_request.configuration.output_folder): path.read_bytes()
        for path in (resumed_request.configuration.output_folder / "images").glob(
            "*.webp"
        )
    }
    assert resumed_images == clean_images
    assert retry_runtime.extracted_original_frame_calls == [
        (resumed_request.video_set.sources[0].path, 0, 30)
    ]


def test_atomic_rename_unavailable_fails_before_frame_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """directory rename不能filesystemがartifact生成前に拒否されること。

    Arrange:
        - atomic rename probeだけが失敗するfilesystem faultが用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - atomic rename不能として失敗すること
        - selected frameが再抽出されずOutput Folderが残らないこと
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    runtime = FakeVideoStageMediaRuntime()

    def fail_rename(_source: Path, _destination: Path) -> None:
        raise OSError("unsupported")

    monkeypatch.setattr(
        "src.video_selection.services.canonical_output_publisher.os.rename",
        fail_rename,
    )

    # Act
    # Assert
    with pytest.raises(OSError, match="atomic directory rename"):
        CanonicalOutputPublisher(runtime).publish(request)
    assert runtime.extracted_original_frame_calls == []
    assert not request.configuration.output_folder.exists()


def test_same_size_and_mtime_video_set_change_is_accepted_by_metadata_contract(
    tmp_path: Path,
) -> None:
    """frame抽出後もsizeとmtimeが同じsourceは同一snapshotと扱われること。

    Arrange:
        - validation checkpointでsizeとmtimeを保って書き換えるfaultが用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - metadata-only snapshot契約に従ってpublicationが完了されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    source_path = request.video_set.sources[0].path
    original_stat = source_path.stat()

    def mutate_source(actual: str, _staging_folder: Path) -> None:
        if actual == "before-validation":
            source_path.write_bytes(b"x" * original_stat.st_size)
            os.utime(
                source_path,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )

    # Act
    CanonicalOutputPublisher(
        FakeVideoStageMediaRuntime(),
        fault_injector=mutate_source,
    ).publish(request)

    # Assert
    assert request.configuration.output_folder.is_dir()
    assert not tuple(tmp_path.glob(".output.*.staging"))


def test_verified_local_model_after_update_failure_is_reported_as_warning(
    tmp_path: Path,
) -> None:
    """更新不能でも検証済みlocal modelを使ったroleがwarningへ記録されること。

    Arrange:
        - 全枚数を選択しCandidate Annotation model更新だけが不能なrunが用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - model_update_unavailableが対象role付きで公開されること
        - 選定自体はcompleted_with_warningsとしてatomicに公開されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path, shortfall=False)
    models = FakeModelRuntime(
        "canonical-publication",
        unavailable_roles=frozenset({ModelRole.CANDIDATE_ANNOTATION}),
    ).resolve_models(request.configuration)
    request = replace(request, resolved_models=models)

    # Act
    report = cast(
        dict[str, Any],
        CanonicalOutputPublisher(FakeVideoStageMediaRuntime()).publish(request),
    )

    # Assert
    assert report["run"]["status"] == "completed_with_warnings"
    assert report["run"]["warnings"] == [
        {
            "code": "model_update_unavailable",
            "message": (
                "model更新確認を完了できず検証済みlocal artifactを使用しました。"
            ),
            "details": {"roles": ["candidate_annotation"]},
        }
    ]
