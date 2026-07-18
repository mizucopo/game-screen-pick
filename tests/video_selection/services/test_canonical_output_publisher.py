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
from src.video_selection.services.canonical_output_publisher import (
    CanonicalOutputPublisher,
)
from src.video_selection.services.serialize_canonical_selection_report import (
    serialize_canonical_selection_report,
)
from src.video_selection.services.validate_canonical_selection_report import (
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
        "version": "1.0.0",
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

    # Act / Assert
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
        (
            "after-rename-before-parent-flush",
            OSError("parent directory flush failed"),
        ),
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

    # Act / Assert
    with pytest.raises(type(error), match=str(error)):
        CanonicalOutputPublisher(
            FakeVideoStageMediaRuntime(),
            fault_injector=inject,
        ).publish(request)
    assert not request.configuration.output_folder.exists()
    assert not tuple(tmp_path.glob(".output.*.staging"))


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

    # Act / Assert
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
    """report@1.0.0の既知objectへ追加された未知fieldが拒否されること。

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
        time_contract["future_field"] = "not-in-report-1.0.0"
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # Act / Assert
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


@pytest.mark.parametrize("move_before_failure", [False, True])
def test_directory_rename_failure_leaves_no_output_folder(
    tmp_path: Path,
    move_before_failure: bool,
) -> None:
    """実rename境界が移動前後に失敗してもfinal成果物が除去されること。

    Arrange:
        - staging移動前または移動直後に失敗するdirectory renamerが用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - rename errorが呼出元へ返されること
        - final Output Folderとstaging Folderが残されないこと
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)

    def fail_rename(source: Path, destination: Path) -> None:
        if move_before_failure:
            source.rename(destination)
        raise OSError("directory rename failed")

    # Act / Assert
    with pytest.raises(OSError, match="directory rename failed"):
        CanonicalOutputPublisher(
            FakeVideoStageMediaRuntime(),
            directory_renamer=fail_rename,
        ).publish(request)
    assert not request.configuration.output_folder.exists()
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

    # Act / Assert
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

    # Act / Assert
    with pytest.raises(OSError, match="atomic directory rename"):
        CanonicalOutputPublisher(runtime).publish(request)
    assert runtime.extracted_original_frame_calls == []
    assert not request.configuration.output_folder.exists()


def test_same_stat_video_set_change_before_rename_discards_staging(
    tmp_path: Path,
) -> None:
    """frame抽出後の同一stat内容変更がfinal rename前に検知されること。

    Arrange:
        - validation checkpointでsize、inode、mtimeを保って書き換えるfaultが用意される
    Act:
        - Canonical Output Publisherが実行される
    Assert:
        - Video Set snapshot変更として失敗すること
        - final Output Folderとstaging Folderが残らないこと
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

    # Act / Assert
    with pytest.raises(ValueError, match="Video Set snapshotが変更"):
        CanonicalOutputPublisher(
            FakeVideoStageMediaRuntime(),
            fault_injector=mutate_source,
        ).publish(request)
    assert not request.configuration.output_folder.exists()
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
