"""release suite stream-copy materializerのtest。"""

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from src.video_selection.acceptance import (
    release_suite_materializer as release_materializer_module,
)
from src.video_selection.acceptance.acceptance_profile import AcceptanceProfile
from src.video_selection.acceptance.release_interval import ReleaseInterval
from src.video_selection.acceptance.release_suite_materializer import (
    ReleaseSuiteMaterializer,
)
from src.video_selection.models.media_runtime_identity import MediaRuntimeIdentity


def test_anonymous_clips_preserve_all_streams_and_record_actual_boundaries(
    tmp_path: Path,
) -> None:
    """匿名clipが全streamをcopyし実測境界からsuite identityを作ること。

    Arrange:
        - 2 streamのsource、keyframe差を含むprobe、fake FFmpegが用意される
    Act:
        - release suiteがmaterializeされる
    Assert:
        - map 0/copytsのstream-copy commandが匿名filenameへ実行されること
        - 実測境界、duration、contentからdescriptorが生成されること
        - manifestへsource pathや動画名が記録されないこと
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")
    commands: list[list[str]] = []

    def run(command: list[str]) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(b"anonymous-clip")

    def probe(path: Path) -> dict[str, object]:
        return {
            "start": Fraction(0 if path == source else 8),
            "duration": Fraction(100 if path == source else 1802),
            "end": Fraction(100 if path == source else 1810),
            "streams": (("audio", "aac"), ("video", "h264")),
        }

    materializer = ReleaseSuiteMaterializer(
        command_runner=run,
        media_probe=probe,
        media_runtime_probe=_media_runtime,
    )

    # Act
    input_folder, descriptor = materializer.materialize(
        profile,
        profile.artifact_root / "release",
    )

    # Assert
    assert Path(commands[0][-1]).name == "scenario-001.mkv"
    assert commands[0][commands[0].index("-map") + 1] == "0"
    assert commands[0][commands[0].index("-c") + 1] == "copy"
    assert "-copyts" in commands[0]
    assert commands[0][commands[0].index("-map_metadata") + 1] == "-1"
    assert commands[0][commands[0].index("-map_chapters") + 1] == "-1"
    assert commands[0][commands[0].index("-fflags") + 1] == "+bitexact"
    assert commands[0][commands[0].index("-to") + 1] == "1810.000000"
    assert descriptor["scenario_count"] == 1
    assert descriptor["total_duration"] == {"numerator": 1802, "denominator": 1}
    manifest = (input_folder.parent / "release-materialization.json").read_text(
        encoding="utf-8"
    )
    assert "private-video" not in manifest
    assert str(profile.input_root) not in manifest


def test_boundary_outside_tolerance_removes_partial_clips(tmp_path: Path) -> None:
    """実測境界がtoleranceを超えるとpartial clipが残らないこと。

    Arrange:
        - 期待startから10秒ずれたclip probeが用意される
    Act:
        - release suiteのmaterializeが試行される
    Assert:
        - preflight failureとなり未確定clipが公開されないこと
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")

    def run(command: list[str]) -> None:
        Path(command[-1]).write_bytes(b"clip")

    def probe(path: Path) -> dict[str, object]:
        return {
            "start": Fraction(0 if path == source else 20),
            "duration": Fraction(100 if path == source else 1800),
            "end": Fraction(100 if path == source else 1820),
            "streams": (("video", "h264"),),
        }

    suite_root = profile.artifact_root / "release"

    # Act
    # Assert
    with pytest.raises(ValueError, match="実測境界"):
        ReleaseSuiteMaterializer(
            command_runner=run,
            media_probe=probe,
            media_runtime_probe=_media_runtime,
        ).materialize(profile, suite_root)
    assert not (suite_root / "work" / "input" / "scenario-001.mkv").exists()


def test_aggregate_measured_duration_must_remain_within_suite_tolerance(
    tmp_path: Path,
) -> None:
    """実測合計違反でも確定済みintervalが保持されること。

    Arrange:
        - 各終了境界が4秒ずつ延びた2区間と5秒toleranceが用意される
    Act:
        - 合計1808秒のrelease suiteがmaterializeされる
    Assert:
        - 期待1800秒との差がaggregate toleranceを超えるため拒否されること
        - 再開可能な確定済みclipとinterval checkpointが保持されること
    """
    # Arrange
    base_profile = _profile(tmp_path)
    profile = replace(
        base_profile,
        release_intervals=(
            ReleaseInterval(
                "private-video.mkv",
                Fraction(0),
                Fraction(900),
                "opening",
            ),
            ReleaseInterval(
                "private-video.mkv",
                Fraction(900),
                Fraction(1800),
                "combat",
            ),
        ),
    )
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")

    def run(command: list[str]) -> None:
        Path(command[-1]).write_bytes(b"clip")

    def probe(path: Path) -> dict[str, object]:
        if path == source:
            start, end = Fraction(0), Fraction(2000)
        elif path.name == "scenario-001.mkv":
            start, end = Fraction(0), Fraction(904)
        else:
            start, end = Fraction(900), Fraction(1804)
        return {
            "start": start,
            "duration": end - start,
            "end": end,
            "streams": (("video", "h264"),),
        }

    suite_root = profile.artifact_root / "release"

    # Act
    # Assert
    with pytest.raises(ValueError, match="実測合計duration"):
        ReleaseSuiteMaterializer(
            command_runner=run,
            media_probe=probe,
            media_runtime_probe=_media_runtime,
        ).materialize(profile, suite_root)
    assert sorted(path.name for path in (suite_root / "work" / "input").iterdir()) == [
        "scenario-001.mkv",
        "scenario-002.mkv",
    ]
    assert (
        len(tuple((suite_root / "work" / "interval-checkpoints").glob("*.json"))) == 2
    )


def test_nonzero_source_start_offsets_ffmpeg_stop_timestamp(tmp_path: Path) -> None:
    """非0 source startを加えた停止timestampでclipが作成されること。

    Arrange:
        - container start_timeが5秒のsourceと10〜1810秒のintervalが用意される
    Act:
        - release suiteがmaterializeされる
    Assert:
        - FFmpegの停止timestampにsource startを加えた1815秒が指定されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")
    commands: list[list[str]] = []

    def run(command: list[str]) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(b"clip")

    def probe(path: Path) -> dict[str, object]:
        return {
            "start": Fraction(5 if path == source else 13),
            "duration": Fraction(95 if path == source else 1802),
            "end": Fraction(100 if path == source else 1815),
            "streams": (("video", "h264"),),
        }

    # Act
    ReleaseSuiteMaterializer(
        command_runner=run,
        media_probe=probe,
        media_runtime_probe=_media_runtime,
    ).materialize(profile, profile.artifact_root / "release")

    # Assert
    command = commands[0]
    assert command[command.index("-to") + 1] == "1815.000000"


def test_completed_materialization_is_reused_without_ffmpeg(tmp_path: Path) -> None:
    """同じprofileとcontentの確定済みclipがresume時に再利用されること。

    Arrange:
        - 一度materialize済みのrelease suiteが用意される
    Act:
        - 同じprofileでmaterializeが再度呼ばれる
    Assert:
        - FFmpegを再実行せず同じdescriptorが返されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")
    call_count = 0

    def run(command: list[str]) -> None:
        nonlocal call_count
        call_count += 1
        Path(command[-1]).write_bytes(b"clip")

    def probe(path: Path) -> dict[str, object]:
        return {
            "start": Fraction(0 if path == source else 8),
            "duration": Fraction(100 if path == source else 1802),
            "end": Fraction(100 if path == source else 1810),
            "streams": (("video", "h264"),),
        }

    materializer = ReleaseSuiteMaterializer(
        command_runner=run,
        media_probe=probe,
        media_runtime_probe=_media_runtime,
    )
    suite_root = profile.artifact_root / "release"
    _, first = materializer.materialize(profile, suite_root)

    # Act
    _, second = materializer.materialize(profile, suite_root)

    # Assert
    assert call_count == 1
    assert second == first


@pytest.mark.parametrize("linked_component", ["suite", "work", "input"])
def test_completed_materialization_rejects_symlinked_suite_owned_ancestor(
    tmp_path: Path,
    linked_component: str,
) -> None:
    """確定済みrelease inputのsuite-owned ancestorがsymlinkなら拒否されること。

    Arrange:
        - 確定済みrelease materializationが用意される
        - suite、work、inputのいずれかが外部directoryへのsymlinkへ置換される
    Act:
        - 同じmaterializationの復元が試行される
    Assert:
        - resetを要求して外部directoryを変更せず拒否されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")

    def run(command: list[str]) -> None:
        Path(command[-1]).write_bytes(b"clip")

    def probe(path: Path) -> dict[str, object]:
        return {
            "start": Fraction(0 if path == source else 8),
            "duration": Fraction(100 if path == source else 1802),
            "end": Fraction(100 if path == source else 1810),
            "streams": (("video", "h264"),),
        }

    materializer = ReleaseSuiteMaterializer(
        command_runner=run,
        media_probe=probe,
        media_runtime_probe=_media_runtime,
    )
    suite_root = profile.artifact_root / "release"
    input_folder, _descriptor = materializer.materialize(profile, suite_root)
    linked_path = {
        "suite": suite_root,
        "work": suite_root / "work",
        "input": input_folder,
    }[linked_component]
    external_path = tmp_path / f"external-{linked_component}"
    linked_path.rename(external_path)
    linked_path.symlink_to(external_path, target_is_directory=True)
    protected_clip = {
        "suite": external_path / "work" / "input" / "scenario-001.mkv",
        "work": external_path / "input" / "scenario-001.mkv",
        "input": external_path / "scenario-001.mkv",
    }[linked_component]

    # Act
    with pytest.raises(ValueError) as error:
        materializer.materialize(profile, suite_root)

    # Assert
    assert "--reset-suite" in str(error.value)
    assert protected_clip.read_bytes() == b"clip"


def test_completed_interval_survives_later_interval_failure(
    tmp_path: Path,
) -> None:
    """後続clip失敗後も確定済みintervalだけが再利用されること。

    Arrange:
        - 2区間のうち2本目だけ初回に失敗するstream-copy runnerが用意される
    Act:
        - 初回失敗後に同じrelease materializationが再実行される
    Assert:
        - 1本目は再作成されず2本目だけが再実行されること
        - 最終descriptorが2区間の実測結果から確定されること
    """
    # Arrange
    profile = replace(
        _profile(tmp_path),
        release_intervals=(
            ReleaseInterval(
                "private-video.mkv",
                Fraction(0),
                Fraction(900),
                "opening",
            ),
            ReleaseInterval(
                "private-video.mkv",
                Fraction(900),
                Fraction(1800),
                "combat",
            ),
        ),
    )
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")
    calls: list[str] = []
    second_failed = False

    def run(command: list[str]) -> None:
        nonlocal second_failed
        output = Path(command[-1])
        calls.append(output.name)
        if output.name == "scenario-002.mkv" and not second_failed:
            second_failed = True
            raise OSError("simulated second interval failure")
        output.write_bytes(output.name.encode())

    def probe(path: Path) -> dict[str, object]:
        if path == source:
            start, end = Fraction(0), Fraction(2000)
        elif path.name == "scenario-001.mkv":
            start, end = Fraction(0), Fraction(900)
        else:
            start, end = Fraction(900), Fraction(1800)
        return {
            "start": start,
            "duration": end - start,
            "end": end,
            "streams": (("video", "h264"),),
        }

    materializer = ReleaseSuiteMaterializer(
        command_runner=run,
        media_probe=probe,
        media_runtime_probe=_media_runtime,
    )
    suite_root = profile.artifact_root / "release"
    with pytest.raises(OSError, match="simulated second interval failure"):
        materializer.materialize(profile, suite_root)

    # Act
    input_folder, descriptor = materializer.materialize(profile, suite_root)

    # Assert
    assert calls == [
        "scenario-001.mkv",
        "scenario-002.mkv",
        "scenario-002.mkv",
    ]
    assert descriptor["scenario_count"] == 2
    assert descriptor["total_duration"] == {
        "numerator": 1800,
        "denominator": 1,
    }
    assert (input_folder / "scenario-001.mkv").read_bytes() == (b"scenario-001.mkv")


def test_media_runtime_change_restarts_only_incomplete_release_materialization(
    tmp_path: Path,
) -> None:
    """未完成release materializationが異なるMedia Runtimeを混在させないこと。

    Arrange:
        - 旧Media Runtimeで1区間目が確定し2区間目だけ失敗する
    Act:
        - 新Media Runtimeで同じrelease materializationが再開される
    Assert:
        - 旧runtimeの1区間目を含むmaterializationだけが作り直されること
        - 新runtimeで2区間とも揃ったdescriptorが確定されること
    """
    # Arrange
    profile = replace(
        _profile(tmp_path),
        release_intervals=(
            ReleaseInterval(
                "private-video.mkv",
                Fraction(0),
                Fraction(900),
                "opening",
            ),
            ReleaseInterval(
                "private-video.mkv",
                Fraction(900),
                Fraction(1800),
                "combat",
            ),
        ),
    )
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")
    calls: list[str] = []
    second_failed = False

    def run(command: list[str]) -> None:
        nonlocal second_failed
        output = Path(command[-1])
        calls.append(output.name)
        if output.name == "scenario-002.mkv" and not second_failed:
            second_failed = True
            raise OSError("simulated second interval failure")
        output.write_bytes(output.name.encode())

    def probe(path: Path) -> dict[str, object]:
        if path == source:
            start, end = Fraction(0), Fraction(2000)
        elif path.name == "scenario-001.mkv":
            start, end = Fraction(0), Fraction(900)
        else:
            start, end = Fraction(900), Fraction(1800)
        return {
            "start": start,
            "duration": end - start,
            "end": end,
            "streams": (("video", "h264"),),
        }

    suite_root = profile.artifact_root / "release"
    with pytest.raises(OSError, match="simulated second interval failure"):
        ReleaseSuiteMaterializer(
            command_runner=run,
            media_probe=probe,
            media_runtime_probe=_media_runtime,
        ).materialize(profile, suite_root)

    # Act
    _input_folder, descriptor = ReleaseSuiteMaterializer(
        command_runner=run,
        media_probe=probe,
        media_runtime_probe=_changed_media_runtime,
    ).materialize(profile, suite_root)

    # Assert
    assert calls == [
        "scenario-001.mkv",
        "scenario-002.mkv",
        "scenario-001.mkv",
        "scenario-002.mkv",
    ]
    assert descriptor["scenario_count"] == 2


def test_completed_materialization_reuses_clips_after_media_runtime_change(
    tmp_path: Path,
) -> None:
    """Media Runtime変更後も確定済みrelease clipが再利用されること。

    Arrange:
        - 固定Media Runtimeで確定したrelease materializationが用意される
    Act:
        - 異なるFFmpeg/ffprobe build identityでresumeが試行される
    Assert:
        - clipを作り直さず同じdescriptorが返されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")

    def run(command: list[str]) -> None:
        Path(command[-1]).write_bytes(b"clip")

    def probe(path: Path) -> dict[str, object]:
        return {
            "start": Fraction(0 if path == source else 8),
            "duration": Fraction(100 if path == source else 1802),
            "end": Fraction(100 if path == source else 1810),
            "streams": (("video", "h264"),),
        }

    suite_root = profile.artifact_root / "release"
    _input_folder, first = ReleaseSuiteMaterializer(
        command_runner=run,
        media_probe=probe,
        media_runtime_probe=_media_runtime,
    ).materialize(profile, suite_root)
    changed_runtime_materializer = ReleaseSuiteMaterializer(
        command_runner=run,
        media_probe=probe,
        media_runtime_probe=_changed_media_runtime,
    )

    # Act
    _input_folder, second = changed_runtime_materializer.materialize(
        profile,
        suite_root,
    )

    # Assert
    assert second == first


def test_stray_supported_clip_requires_reset(tmp_path: Path) -> None:
    """manifest外の対応videoがrelease inputにあるとresumeが拒否されること。

    Arrange:
        - 確定済みrelease inputへ余分なscenario videoが追加される
    Act:
        - 同じprofileでresume materializeが試行される
    Assert:
        - manifestの完全一致違反としてresetが必要になること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")

    def run(command: list[str]) -> None:
        Path(command[-1]).write_bytes(b"clip")

    def probe(path: Path) -> dict[str, object]:
        return {
            "start": Fraction(0 if path == source else 8),
            "duration": Fraction(100 if path == source else 1802),
            "end": Fraction(100 if path == source else 1810),
            "streams": (("video", "h264"),),
        }

    materializer = ReleaseSuiteMaterializer(
        command_runner=run,
        media_probe=probe,
        media_runtime_probe=_media_runtime,
    )
    suite_root = profile.artifact_root / "release"
    input_folder, _ = materializer.materialize(profile, suite_root)
    (input_folder / "scenario-999.mkv").write_bytes(b"stray")

    # Act
    # Assert
    with pytest.raises(ValueError, match="匿名input"):
        materializer.materialize(profile, suite_root)


def test_corrupted_terminal_manifest_is_rebuilt_from_interval_checkpoint(
    tmp_path: Path,
) -> None:
    """破損した終端manifestが確定intervalを再作成せず修復されること。

    Arrange:
        - 確定済みrelease clipとscenario countだけを壊したmanifestが用意される
    Act:
        - 同じprofileでmaterializationが再開される
    Assert:
        - FFmpegを再実行せず同じdescriptorとclip bytesが復元されること
        - 終端manifestの導出値が正しい値へ書き直されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")
    calls: list[list[str]] = []

    def run(command: list[str]) -> None:
        calls.append(command)
        Path(command[-1]).write_bytes(b"clip")

    def probe(path: Path) -> dict[str, object]:
        return {
            "start": Fraction(0 if path == source else 8),
            "duration": Fraction(100 if path == source else 1802),
            "end": Fraction(100 if path == source else 1810),
            "streams": (("video", "h264"),),
        }

    suite_root = profile.artifact_root / "release"
    materializer = ReleaseSuiteMaterializer(
        command_runner=run,
        media_probe=probe,
        media_runtime_probe=_media_runtime,
    )
    input_folder, expected = materializer.materialize(profile, suite_root)
    manifest_path = input_folder.parent / "release-materialization.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["descriptor"]["scenario_count"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    clip_bytes = (input_folder / "scenario-001.mkv").read_bytes()

    def unexpected_run(_command: list[str]) -> None:
        raise AssertionError("FFmpeg must not run during terminal recovery")

    def unexpected_probe(_path: Path) -> dict[str, object]:
        raise AssertionError("media must not be probed during terminal recovery")

    def unexpected_runtime_probe() -> MediaRuntimeIdentity:
        raise AssertionError("current runtime must not be probed during recovery")

    # Act
    _input_folder, resumed = ReleaseSuiteMaterializer(
        command_runner=unexpected_run,
        media_probe=unexpected_probe,
        media_runtime_probe=unexpected_runtime_probe,
    ).materialize(profile, suite_root)

    # Assert
    assert len(calls) == 1
    assert resumed == expected
    assert (input_folder / "scenario-001.mkv").read_bytes() == clip_bytes
    repaired = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert repaired["descriptor"]["scenario_count"] == 1


def test_pending_interval_checkpoint_recovers_after_commit_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """clip置換後のcommit中断からFFmpegなしで復元されること。

    Arrange:
        - 検証済みclipの固定名置換後にcheckpoint昇格だけが失敗される
        - outputとdurable pending checkpointが残される
    Act:
        - 同じmaterializationが新しいprocessとして再開される
    Assert:
        - FFmpeg、media probe、現在runtime probeなしでpendingが確定されること
        - 同じclip bytesとdescriptorから終端manifestが生成されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")

    def run(command: list[str]) -> None:
        Path(command[-1]).write_bytes(b"completed-before-interruption")

    def probe(path: Path) -> dict[str, object]:
        return {
            "start": Fraction(0 if path == source else 8),
            "duration": Fraction(100 if path == source else 1802),
            "end": Fraction(100 if path == source else 1810),
            "streams": (("video", "h264"),),
        }

    original_promoter = release_materializer_module._promote_checkpoint

    def interrupt_promotion(_pending: Path, _checkpoint: Path) -> None:
        raise OSError("injected checkpoint commit interruption")

    monkeypatch.setattr(
        release_materializer_module,
        "_promote_checkpoint",
        interrupt_promotion,
    )
    suite_root = profile.artifact_root / "release"
    with pytest.raises(OSError, match="checkpoint commit interruption"):
        ReleaseSuiteMaterializer(
            command_runner=run,
            media_probe=probe,
            media_runtime_probe=_media_runtime,
        ).materialize(profile, suite_root)
    work_root = suite_root / "work"
    output_path = work_root / "input" / "scenario-001.mkv"
    pending_path = work_root / "interval-checkpoints" / ".scenario-001.pending.json"
    checkpoint_path = (
        work_root / "interval-checkpoints" / ".scenario-001.checkpoint.json"
    )
    assert output_path.read_bytes() == b"completed-before-interruption"
    assert pending_path.is_file()
    assert not checkpoint_path.exists()
    monkeypatch.setattr(
        release_materializer_module,
        "_promote_checkpoint",
        original_promoter,
    )

    def unexpected_run(_command: list[str]) -> None:
        raise AssertionError("FFmpeg must not run during pending recovery")

    def unexpected_probe(_path: Path) -> dict[str, object]:
        raise AssertionError("media must not be probed during pending recovery")

    def unexpected_runtime_probe() -> MediaRuntimeIdentity:
        raise AssertionError("current runtime must not be probed during recovery")

    # Act
    _input_folder, descriptor = ReleaseSuiteMaterializer(
        command_runner=unexpected_run,
        media_probe=unexpected_probe,
        media_runtime_probe=unexpected_runtime_probe,
    ).materialize(profile, suite_root)

    # Assert
    assert descriptor["scenario_count"] == 1
    assert output_path.read_bytes() == b"completed-before-interruption"
    assert checkpoint_path.is_file()
    assert not pending_path.exists()
    assert (work_root / "release-materialization.json").is_file()


def test_corrupted_partial_context_restarts_only_release_materialization(
    tmp_path: Path,
) -> None:
    """破損したpartial contextだけが再構築されること。

    Arrange:
        - 1区間目の確定後に2区間目で失敗したrelease materializationが用意される
        - Media Runtime context JSONが破損される
    Act:
        - 同じprofileでmaterializationが再開される
    Assert:
        - checkpoint自身がidentityを持つ1区間目が再利用されること
        - 未完了の2区間目だけが再作成されること
        - suite resetなしで終端descriptorが確定されること
    """
    # Arrange
    profile = replace(
        _profile(tmp_path),
        release_intervals=(
            ReleaseInterval(
                "private-video.mkv",
                Fraction(0),
                Fraction(900),
                "opening",
            ),
            ReleaseInterval(
                "private-video.mkv",
                Fraction(900),
                Fraction(1800),
                "combat",
            ),
        ),
    )
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")
    calls: list[str] = []
    second_failed = False

    def run(command: list[str]) -> None:
        nonlocal second_failed
        output = Path(command[-1])
        calls.append(output.name)
        if output.name == "scenario-002.mkv" and not second_failed:
            second_failed = True
            raise OSError("simulated second interval failure")
        output.write_bytes(output.name.encode())

    def probe(path: Path) -> dict[str, object]:
        if path == source:
            start, end = Fraction(0), Fraction(2000)
        elif path.name == "scenario-001.mkv":
            start, end = Fraction(0), Fraction(900)
        else:
            start, end = Fraction(900), Fraction(1800)
        return {
            "start": start,
            "duration": end - start,
            "end": end,
            "streams": (("video", "h264"),),
        }

    suite_root = profile.artifact_root / "release"
    materializer = ReleaseSuiteMaterializer(
        command_runner=run,
        media_probe=probe,
        media_runtime_probe=_media_runtime,
    )
    with pytest.raises(OSError, match="simulated second interval failure"):
        materializer.materialize(profile, suite_root)
    context_path = suite_root / "work" / "release-materialization-context.json"
    context_path.write_text("{broken", encoding="utf-8")

    # Act
    _input_folder, descriptor = materializer.materialize(profile, suite_root)

    # Assert
    assert calls == [
        "scenario-001.mkv",
        "scenario-002.mkv",
        "scenario-002.mkv",
    ]
    assert descriptor["scenario_count"] == 2


def test_failed_runtime_replacement_preserves_previous_interval_checkpoint(
    tmp_path: Path,
) -> None:
    """新runtimeでの置換失敗時に旧interval checkpointが保持されること。

    Arrange:
        - 旧Media Runtimeで1区間だけ確定した未完了materializationが用意される
        - 新Media Runtimeのstream copyが一時clip作成後に失敗する
    Act:
        - 新runtimeでrelease materializationが再開される
    Assert:
        - 旧clipとcheckpoint bytesが変更されないこと
        - 新runtimeの未確定一時clipだけが破棄されること
    """
    # Arrange
    profile = replace(
        _profile(tmp_path),
        release_intervals=(
            ReleaseInterval(
                "private-video.mkv",
                Fraction(0),
                Fraction(900),
                "opening",
            ),
            ReleaseInterval(
                "private-video.mkv",
                Fraction(900),
                Fraction(1800),
                "combat",
            ),
        ),
    )
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")

    def probe(path: Path) -> dict[str, object]:
        if path == source:
            start, end = Fraction(0), Fraction(2000)
        elif path.name == "scenario-001.mkv":
            start, end = Fraction(0), Fraction(900)
        else:
            start, end = Fraction(900), Fraction(1800)
        return {
            "start": start,
            "duration": end - start,
            "end": end,
            "streams": (("video", "h264"),),
        }

    def write_old(command: list[str]) -> None:
        if Path(command[-1]).name == "scenario-002.mkv":
            raise OSError("injected old incomplete materialization")
        Path(command[-1]).write_bytes(b"old-complete")

    suite_root = profile.artifact_root / "release"
    with pytest.raises(OSError, match="injected old incomplete materialization"):
        ReleaseSuiteMaterializer(
            command_runner=write_old,
            media_probe=probe,
            media_runtime_probe=_media_runtime,
        ).materialize(profile, suite_root)
    work_root = suite_root / "work"
    output_path = work_root / "input" / "scenario-001.mkv"
    checkpoint_path = (
        work_root / "interval-checkpoints" / ".scenario-001.checkpoint.json"
    )
    old_output = output_path.read_bytes()
    old_checkpoint = checkpoint_path.read_bytes()

    def fail_new(command: list[str]) -> None:
        Path(command[-1]).write_bytes(b"new-partial")
        raise OSError("injected replacement failure")

    # Act
    # Assert
    with pytest.raises(OSError, match="injected replacement failure"):
        ReleaseSuiteMaterializer(
            command_runner=fail_new,
            media_probe=probe,
            media_runtime_probe=lambda: MediaRuntimeIdentity(
                "8.0",
                "8.0",
                "b" * 64,
            ),
        ).materialize(profile, suite_root)
    assert output_path.read_bytes() == old_output
    assert checkpoint_path.read_bytes() == old_checkpoint
    assert not tuple((work_root / "interval-checkpoints").glob(".scenario-001.*.tmp"))


def _profile(tmp_path: Path) -> AcceptanceProfile:
    """一つの30分intervalを持つprofileを返す。"""
    return AcceptanceProfile(
        profile_version="1.0.0",
        input_root=tmp_path / "input",
        configuration_path=tmp_path / "config.toml",
        artifact_root=tmp_path / "artifacts",
        release_expected_total_duration=Fraction(1800),
        release_boundary_tolerance_seconds=Fraction(5),
        release_intervals=(
            ReleaseInterval(
                relative_video_path="private-video.mkv",
                start=Fraction(10),
                end=Fraction(1810),
                scenario_role="representative-gameplay",
            ),
        ),
        full_expected_video_count=12,
        full_expected_total_duration=Fraction(182400),
        full_duration_tolerance_seconds=Fraction(60),
        profile_digest="a" * 64,
    )


def _media_runtime() -> MediaRuntimeIdentity:
    """test用の固定Media Runtime Identityを返す。"""
    return MediaRuntimeIdentity("7.1", "7.1", "a" * 64)


def _changed_media_runtime() -> MediaRuntimeIdentity:
    """test用の変更後Media Runtime Identityを返す。"""
    return MediaRuntimeIdentity("7.2", "7.2", "c" * 64)
