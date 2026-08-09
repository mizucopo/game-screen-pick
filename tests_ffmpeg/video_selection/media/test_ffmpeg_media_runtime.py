"""system FFmpeg MediaRuntimeのintegration test。"""

import signal
import stat
import struct
import sys
import threading
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.video_selection.media.ffmpeg_media_runtime import FfmpegMediaRuntime
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.empty_video_scan_partition import (
    EmptyVideoScanPartition,
)
from src.video_selection.models.media_probe import MediaProbe
from src.video_selection.models.media_runtime_error import MediaRuntimeError
from src.video_selection.models.media_runtime_failure_reason import (
    MediaRuntimeFailureReason,
)
from src.video_selection.models.media_stream import MediaStream
from src.video_selection.models.native_video_scan import NativeVideoScan
from src.video_selection.services.discover_video_set import discover_video_set
from src.video_selection.services.select_scene_signal_frames import (
    select_scene_signal_frames,
)
from src.video_selection.services.video_stage_processor import VideoStageProcessor
from tests.video_selection.fakes.fake_speech_runtime import FakeSpeechRuntime
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver
from tests_ffmpeg.support.ffmpeg_fixture_factory import (
    generate_av1_aac_video,
    generate_cfr_video,
    generate_corrupt_video,
    generate_delayed_video_with_audio,
    generate_discontinuous_audio,
    generate_nonzero_start_video,
    generate_odd_dimension_video,
    generate_quantized_audio,
    generate_scene_change_video,
    generate_stream_matrix_video,
    generate_vfr_video,
)


def _write_version_tool(path: Path, tool: str, version: str) -> None:
    path.write_text(
        f"#!{sys.executable}\nprint({f'{tool} version {version}'!r})\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_failing_tool(path: Path) -> None:
    path.write_text(
        f"#!{sys.executable}\nraise SystemExit(7)\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_capable_tool(
    path: Path,
    tool: str,
    build_marker: str,
    probe_document: str = '{"program_version": {}}',
    omitted_capability_option: str | None = None,
) -> None:
    script = f"""#!{sys.executable}
import sys

arguments = sys.argv[1:]
if "-version" in arguments:
    print({f"{tool} version 6.1.1"!r})
    print({f"built with compiler-{build_marker}"!r})
    print({f"configuration: --build-marker={build_marker}"!r})
    print("libavutil 58.0.0 / 58.0.0")
elif "-demuxers" in arguments:
    print(" D mov")
    print(" D matroska")
elif "-decoders" in arguments:
    print(" A....D aac")
    print(" V..... libdav1d")
    print(" S..... subrip")
elif "-encoders" in arguments:
    if {omitted_capability_option!r} != "-encoders":
        print(" V....D mjpeg")
        print(" V....D ppm")
        print(" A....D pcm_s16le")
        print(" S..... srt")
elif "-muxers" in arguments:
    if {omitted_capability_option!r} != "-muxers":
        print(" E image2")
        print(" E image2pipe")
        print(" E s16le")
        print(" E srt")
elif "-filters" in arguments:
    print(" ... aformat")
    print(" ... aresample")
    print(" ... asettb")
    print("T.C asetnsamples")
    print(" ... asetpts")
    print(" ... ashowinfo")
    print(" ... atrim")
    print(" ... concat")
    print(" ... format")
    print(" ... nullsink")
    print("..C scale")
    print(" ... select")
    print(" ... setpts")
    print(" ... showinfo")
    print(" ... split")
else:
    print({probe_document!r})
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_preflight_accepts_supported_same_build_runtime() -> None:
    """同一buildの対応FFmpegとffprobeがpreflightで受理されること。

    Arrange:
        - PATH上のsystem FFmpegとffprobeを使うMediaRuntimeが用意される
    Act:
        - runtime preflightが実行される
    Assert:
        - 両toolの同じbuild versionが返されること
    """
    # Arrange
    runtime = FfmpegMediaRuntime()

    # Act
    identity = runtime.preflight()

    # Assert
    assert identity.ffmpeg_version == identity.ffprobe_version
    assert identity.ffmpeg_version


def test_preflight_accepts_ffmpeg_6_1_capability_flags(tmp_path: Path) -> None:
    """FFmpeg 6.1形式のfilter flagを持つ対応tool pairが受理されること。

    Arrange:
        - T.Cと..Cのfilter flagを返す同一buildのfake tool pairが用意される
    Act:
        - runtime preflightが実行される
    Assert:
        - 両toolの同じbuild versionが返されること
    """
    # Arrange
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffprobe = tmp_path / "ffprobe"
    _write_capable_tool(fake_ffmpeg, "ffmpeg", "same")
    _write_capable_tool(fake_ffprobe, "ffprobe", "same")
    runtime = FfmpegMediaRuntime(
        ffmpeg_executable=str(fake_ffmpeg),
        ffprobe_executable=str(fake_ffprobe),
    )

    # Act
    identity = runtime.preflight()

    # Assert
    assert identity.ffmpeg_version == "6.1.1"
    assert identity.ffprobe_version == "6.1.1"


def test_preflight_distinguishes_same_version_runtime_builds(tmp_path: Path) -> None:
    """同じversionの異なるbuildが別Media Runtime Identityになること。

    Arrange:
        - versionが同じでbuild markerが異なる2組の対応tool pairが用意される
    Act:
        - 各runtimeのpreflightが実行される
    Assert:
        - buildとcapabilityから導出されたidentityが異なること
    """
    # Arrange
    first_folder = tmp_path / "first"
    second_folder = tmp_path / "second"
    first_folder.mkdir()
    second_folder.mkdir()
    first_ffmpeg = first_folder / "ffmpeg"
    first_ffprobe = first_folder / "ffprobe"
    second_ffmpeg = second_folder / "ffmpeg"
    second_ffprobe = second_folder / "ffprobe"
    _write_capable_tool(first_ffmpeg, "ffmpeg", "first")
    _write_capable_tool(first_ffprobe, "ffprobe", "first")
    _write_capable_tool(second_ffmpeg, "ffmpeg", "second")
    _write_capable_tool(second_ffprobe, "ffprobe", "second")
    first_runtime = FfmpegMediaRuntime(
        ffmpeg_executable=str(first_ffmpeg),
        ffprobe_executable=str(first_ffprobe),
    )
    second_runtime = FfmpegMediaRuntime(
        ffmpeg_executable=str(second_ffmpeg),
        ffprobe_executable=str(second_ffprobe),
    )

    # Act
    first_identity = first_runtime.preflight()
    second_identity = second_runtime.preflight()

    # Assert
    assert first_identity.ffmpeg_version == second_identity.ffmpeg_version
    assert first_identity.ffprobe_version == second_identity.ffprobe_version
    assert first_identity != second_identity


@pytest.mark.parametrize(
    "omitted_capability_option",
    ["-encoders", "-muxers"],
)
def test_preflight_rejects_missing_required_output_capability(
    tmp_path: Path,
    omitted_capability_option: str,
) -> None:
    """必要なencoderまたはmuxerがないtool pairが拒否されること。

    Arrange:
        - 必要なencoderまたはmuxerだけを欠く同一buildのfake tool pairが用意される
    Act:
        - runtime preflightが実行される
    Assert:
        - missing_required_demuxer_or_decoderとして失敗すること
    """
    # Arrange
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffprobe = tmp_path / "ffprobe"
    _write_capable_tool(
        fake_ffmpeg,
        "ffmpeg",
        "same",
        omitted_capability_option=omitted_capability_option,
    )
    _write_capable_tool(fake_ffprobe, "ffprobe", "same")
    runtime = FfmpegMediaRuntime(
        ffmpeg_executable=str(fake_ffmpeg),
        ffprobe_executable=str(fake_ffprobe),
    )

    # Act
    with pytest.raises(MediaRuntimeError) as captured:
        runtime.preflight()

    # Assert
    assert (
        captured.value.reason
        is MediaRuntimeFailureReason.MISSING_REQUIRED_DEMUXER_OR_DECODER
    )


def test_preflight_reports_missing_ffmpeg_with_stable_reason() -> None:
    """FFmpeg不在がstable reasonへ変換されること。

    Arrange:
        - 存在しないFFmpeg executableを指定したMediaRuntimeが用意される
    Act:
        - runtime preflightが実行される
    Assert:
        - ffmpeg_not_foundとして失敗すること
    """
    # Arrange
    runtime = FfmpegMediaRuntime(
        ffmpeg_executable="definitely-missing-game-screen-pick-ffmpeg",
    )

    # Act
    with pytest.raises(MediaRuntimeError) as captured:
        runtime.preflight()

    # Assert
    assert captured.value.reason is MediaRuntimeFailureReason.FFMPEG_NOT_FOUND


def test_preflight_reports_unexecutable_ffmpeg_with_stable_reason(
    tmp_path: Path,
) -> None:
    """version probeを実行できないFFmpegがstable reasonへ変換されること。

    Arrange:
        - version commandが非zero終了するfake FFmpegが用意される
    Act:
        - runtime preflightが実行される
    Assert:
        - ffmpeg_not_foundとして失敗すること
    """
    # Arrange
    fake_ffmpeg = tmp_path / "ffmpeg"
    _write_failing_tool(fake_ffmpeg)
    runtime = FfmpegMediaRuntime(ffmpeg_executable=str(fake_ffmpeg))

    # Act
    with pytest.raises(MediaRuntimeError) as captured:
        runtime.preflight()

    # Assert
    assert captured.value.reason is MediaRuntimeFailureReason.FFMPEG_NOT_FOUND


def test_preflight_rejects_ffmpeg_below_version_floor(tmp_path: Path) -> None:
    """6.1.1未満のFFmpegがstable reasonで拒否されること。

    Arrange:
        - version 6.0.0を返すfake FFmpegが用意される
    Act:
        - runtime preflightが実行される
    Assert:
        - unsupported_ffmpeg_versionとして失敗すること
    """
    # Arrange
    fake_ffmpeg = tmp_path / "ffmpeg"
    _write_version_tool(fake_ffmpeg, "ffmpeg", "6.0.0")
    runtime = FfmpegMediaRuntime(ffmpeg_executable=str(fake_ffmpeg))

    # Act
    with pytest.raises(MediaRuntimeError) as captured:
        runtime.preflight()

    # Assert
    assert captured.value.reason is MediaRuntimeFailureReason.UNSUPPORTED_FFMPEG_VERSION


def test_preflight_reports_malformed_version_with_stable_reason(
    tmp_path: Path,
) -> None:
    """解釈不能なversion outputがstable reasonへ変換されること。

    Arrange:
        - FFmpeg形式ではないversion行を返すfake toolが用意される
    Act:
        - runtime preflightが実行される
    Assert:
        - unsupported_ffmpeg_versionとして失敗すること
    """
    # Arrange
    fake_ffmpeg = tmp_path / "ffmpeg"
    _write_version_tool(fake_ffmpeg, "unexpected-tool", "6.1.1")
    runtime = FfmpegMediaRuntime(ffmpeg_executable=str(fake_ffmpeg))

    # Act
    with pytest.raises(MediaRuntimeError) as captured:
        runtime.preflight()

    # Assert
    assert captured.value.reason is MediaRuntimeFailureReason.UNSUPPORTED_FFMPEG_VERSION


def test_preflight_rejects_ffmpeg_and_ffprobe_build_mismatch(
    tmp_path: Path,
) -> None:
    """FFmpegとffprobeのbuild不一致がstable reasonで拒否されること。

    Arrange:
        - 異なる対応versionを返すfake FFmpegとffprobeが用意される
    Act:
        - runtime preflightが実行される
    Assert:
        - ffmpeg_ffprobe_version_mismatchとして失敗すること
    """
    # Arrange
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffprobe = tmp_path / "ffprobe"
    _write_version_tool(fake_ffmpeg, "ffmpeg", "6.1.1-build-a")
    _write_version_tool(fake_ffprobe, "ffprobe", "6.1.1-build-b")
    runtime = FfmpegMediaRuntime(
        ffmpeg_executable=str(fake_ffmpeg),
        ffprobe_executable=str(fake_ffprobe),
    )

    # Act
    with pytest.raises(MediaRuntimeError) as captured:
        runtime.preflight()

    # Assert
    assert (
        captured.value.reason
        is MediaRuntimeFailureReason.FFMPEG_FFPROBE_VERSION_MISMATCH
    )


def test_preflight_rejects_same_version_from_different_builds(
    tmp_path: Path,
) -> None:
    """同一versionでもbuild signatureが異なるtool pairが拒否されること。

    Arrange:
        - versionと能力は同じでbuild設定だけが異なるfake pairが用意される
    Act:
        - runtime preflightが実行される
    Assert:
        - ffmpeg_ffprobe_version_mismatchとして失敗すること
    """
    # Arrange
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffprobe = tmp_path / "ffprobe"
    _write_capable_tool(fake_ffmpeg, "ffmpeg", "a")
    _write_capable_tool(fake_ffprobe, "ffprobe", "b")
    runtime = FfmpegMediaRuntime(
        ffmpeg_executable=str(fake_ffmpeg),
        ffprobe_executable=str(fake_ffprobe),
    )

    # Act
    with pytest.raises(MediaRuntimeError) as captured:
        runtime.preflight()

    # Assert
    assert (
        captured.value.reason
        is MediaRuntimeFailureReason.FFMPEG_FFPROBE_VERSION_MISMATCH
    )


def test_preflight_rejects_missing_required_media_capability(
    tmp_path: Path,
) -> None:
    """必要なdemuxer・decoder・filter不在がstable reasonで拒否されること。

    Arrange:
        - 対応versionだけを返しmedia能力を持たないfake toolsが用意される
    Act:
        - runtime preflightが実行される
    Assert:
        - missing_required_demuxer_or_decoderとして失敗すること
    """
    # Arrange
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffprobe = tmp_path / "ffprobe"
    _write_version_tool(fake_ffmpeg, "ffmpeg", "6.1.1")
    _write_version_tool(fake_ffprobe, "ffprobe", "6.1.1")
    runtime = FfmpegMediaRuntime(
        ffmpeg_executable=str(fake_ffmpeg),
        ffprobe_executable=str(fake_ffprobe),
    )

    # Act
    with pytest.raises(MediaRuntimeError) as captured:
        runtime.preflight()

    # Assert
    assert (
        captured.value.reason
        is MediaRuntimeFailureReason.MISSING_REQUIRED_DEMUXER_OR_DECODER
    )


def test_preflight_reports_invalid_probe_capability_with_stable_reason(
    tmp_path: Path,
) -> None:
    """不正なffprobe JSON能力応答がstable reasonへ変換されること。

    Arrange:
        - 必要FFmpeg能力とobjectでないffprobe JSONを返すfake pairが用意される
    Act:
        - runtime preflightが実行される
    Assert:
        - missing_required_demuxer_or_decoderとして失敗すること
    """
    # Arrange
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffprobe = tmp_path / "ffprobe"
    _write_capable_tool(fake_ffmpeg, "ffmpeg", "same")
    _write_capable_tool(fake_ffprobe, "ffprobe", "same", probe_document="[]")
    runtime = FfmpegMediaRuntime(
        ffmpeg_executable=str(fake_ffmpeg),
        ffprobe_executable=str(fake_ffprobe),
    )

    # Act
    with pytest.raises(MediaRuntimeError) as captured:
        runtime.preflight()

    # Assert
    assert (
        captured.value.reason
        is MediaRuntimeFailureReason.MISSING_REQUIRED_DEMUXER_OR_DECODER
    )


def test_probe_reports_cfr_stream_semantics(tmp_path: Path) -> None:
    """CFR fixtureのcontainerとvideo stream意味が返されること。

    Arrange:
        - lavfiから生成された2fpsのCFR Matroskaが用意される
    Act:
        - MediaRuntimeでfixtureがprobeされる
    Assert:
        - container、stream種別、codec、time base、解像度が返されること
    """
    # Arrange
    video_path = generate_cfr_video(tmp_path / "cfr.mkv")
    runtime = FfmpegMediaRuntime()

    # Act
    probe = runtime.probe(video_path)

    # Assert
    assert "matroska" in probe.format_names
    assert len(probe.streams) == 1
    stream = probe.streams[0]
    assert stream.index == 0
    assert stream.kind == "video"
    assert stream.codec_name == "ffv1"
    assert stream.time_base is not None
    assert stream.time_base > 0
    assert (stream.width, stream.height) == (64, 48)


def test_scan_video_frames_preserves_cfr_pts_and_pixels(tmp_path: Path) -> None:
    """CFR scanでsource PTSとraw pixel artifactが保持されること。

    Arrange:
        - 0.5秒間隔で4frameを持つCFR fixtureが用意される
    Act:
        - MediaRuntimeで全video frameがscanされる
    Assert:
        - exact時刻が0、0.5、1.0、1.5秒で返されること
        - 各artifactが64x48 RGB24のraw pixelを持つこと
    """
    # Arrange
    video_path = generate_cfr_video(tmp_path / "cfr.mkv")
    runtime = FfmpegMediaRuntime()

    # Act
    frames = tuple(
        runtime.scan_video_frames(
            video_path,
            stream_index=0,
            max_dimension=64,
        )
    )

    # Assert
    assert [Fraction(frame.pts) * frame.time_base for frame in frames] == [
        Fraction(0),
        Fraction(1, 2),
        Fraction(1),
        Fraction(3, 2),
    ]
    assert all((frame.width, frame.height) == (64, 48) for frame in frames)
    assert all(frame.pixel_format == "rgb24" for frame in frames)
    assert all(len(frame.pixels) == 64 * 48 * 3 for frame in frames)


def test_scan_video_frames_preserves_vfr_source_timing(tmp_path: Path) -> None:
    """VFR scanで存在するsource frameの不均一な時刻が保持されること。

    Arrange:
        - 0、0.25、0.75、1.0秒にframeを持つVFR fixtureが用意される
    Act:
        - MediaRuntimeで全video frameがscanされる
    Assert:
        - 固定fps slotへ変換されずsource時刻がexactに返されること
    """
    # Arrange
    video_path = generate_vfr_video(tmp_path / "vfr.mkv")
    runtime = FfmpegMediaRuntime()

    # Act
    frames = tuple(
        runtime.scan_video_frames(
            video_path,
            stream_index=0,
            max_dimension=64,
        )
    )

    # Assert
    assert [Fraction(frame.pts) * frame.time_base for frame in frames] == [
        Fraction(0),
        Fraction(1, 4),
        Fraction(3, 4),
        Fraction(1),
    ]


def test_scan_video_records_observed_vfr_timing_hint(tmp_path: Path) -> None:
    """VFRの実測最小PTS差がresource hintへ記録されること。

    Arrange:
        - 0、0.25、0.75、1.0秒にframeを持つVFR fixtureが用意される
    Act:
        - composite Video Scanが実行される
    Assert:
        - 最小0.25秒のPTS差と同一PTS最大1frameが記録されること
    """
    # Arrange
    video_path = generate_vfr_video(tmp_path / "vfr-hint.mkv")
    runtime = FfmpegMediaRuntime()
    stream = runtime.probe(video_path).streams[0]

    # Act
    scan = runtime.scan_video(
        video_path,
        stream,
        tmp_path / "vfr-hint-artifacts",
        heartbeat_interval_seconds=0.25,
        scene_change_threshold=0.25,
        scene_min_interval_seconds=0.25,
        decode_backend="cpu",
    )

    # Assert
    assert scan.minimum_frame_delta_ts is not None
    assert Fraction(scan.minimum_frame_delta_ts) * scan.time_base == Fraction(1, 4)
    assert scan.maximum_frame_count_per_pts == 1
    assert (scan.maximum_frame_width, scan.maximum_frame_height) == (64, 48)


def test_scan_video_records_maximum_dimensions_across_timeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """途中で変化するframe寸法の最大値がresource hintへ記録されること。

    Arrange:
        - probe寸法320x180から1920x1080へ変化するtimeline出力が用意される
    Act:
        - composite Video Scanが実行される
    Assert:
        - 全timeline frameで観測された最大幅と最大高さが記録されること
    """
    # Arrange
    artifact_folder = tmp_path / "resolution-change-artifacts"
    process = MagicMock()
    process.pid = 123
    process.returncode = None
    process.stderr.__iter__.return_value = iter(
        (
            "[showinfo@timeline] n: 0 pts: 0 duration: 1 s:320x180\n",
            "[showinfo@heartbeat] n: 0 pts: 0 duration: 1 s:320x180\n",
            "[showinfo@timeline] n: 1 pts: 1 duration: 1 s:1920x1080\n",
            "[showinfo@timeline] n: 2 pts: 2 duration: 1 s:1280x720\n",
        )
    )

    def start_process(*_args: object, **_kwargs: object) -> MagicMock:
        heartbeat_folder = artifact_folder / "heartbeats"
        scene_folder = artifact_folder / ".scene-proxies"
        (heartbeat_folder / "000000000000.jpg").write_bytes(b"sentinel")
        (heartbeat_folder / "000000000001.jpg").write_bytes(b"heartbeat")
        (scene_folder / "000000000000.jpg").write_bytes(b"sentinel")
        return process

    def reap(_process: object) -> tuple[int, float]:
        process.returncode = 0
        return (0, 0.01)

    monkeypatch.setattr(
        "src.video_selection.media.ffmpeg_media_runtime.subprocess.Popen",
        start_process,
    )
    monkeypatch.setattr(
        "src.video_selection.media.ffmpeg_media_runtime.wait_for_process",
        reap,
    )
    runtime = FfmpegMediaRuntime()
    stream = MediaStream(
        index=0,
        kind="video",
        codec_name="h264",
        time_base=Fraction(1, 10),
        start_pts=0,
        duration_ts=3,
        width=320,
        height=180,
        sample_rate=None,
        channels=None,
        language=None,
        is_default=True,
        is_forced=False,
        is_attached_picture=False,
    )

    # Act
    scan = runtime.scan_video(
        tmp_path / "source.mkv",
        stream,
        artifact_folder,
        heartbeat_interval_seconds=1.0,
        scene_change_threshold=1.0,
        scene_min_interval_seconds=1.0,
        decode_backend="cpu",
    )

    # Assert
    assert scan.maximum_frame_width == 1920
    assert scan.maximum_frame_height == 1080


def test_scan_video_emits_heartbeat_and_scene_signals_from_one_decode(
    tmp_path: Path,
) -> None:
    """一回のdecodeからheartbeat proxyとscene signalが生成されること。

    Arrange:
        - 1秒ごとに内容が変わる3秒の実FFmpeg fixtureが用意される
    Act:
        - composite Video Scanが実行される
    Assert:
        - exactなorigin、最終frame終端、1秒heartbeatが返されること
        - scene signalが320px以下の一時画像とともに返されること
        - Heartbeat Proxyが960px以下のmetadataなしMJPEGとして保存されること
        - native frameの最小PTS差と同一PTS最大frame数が記録されること
        - decode passが1回として記録されること
    """
    # Arrange
    video_path = generate_scene_change_video(tmp_path / "scenes.mkv")
    runtime = FfmpegMediaRuntime()
    stream = runtime.probe(video_path).streams[0]
    artifact_folder = tmp_path / "scan-artifacts"

    # Act
    scan = runtime.scan_video(
        video_path,
        stream,
        artifact_folder,
        heartbeat_interval_seconds=1.0,
        scene_change_threshold=0.25,
        scene_min_interval_seconds=0.5,
        decode_backend="cpu",
    )

    # Assert
    assert scan.decode_pass_count == 1
    assert scan.cpu_seconds > 0
    assert scan.minimum_frame_delta_ts is not None
    assert scan.minimum_frame_delta_ts > 0
    assert scan.maximum_frame_count_per_pts == 1
    assert scan.origin_pts == 0
    assert scan.last_frame_duration_ts is not None
    assert stream.time_base is not None
    end_pts = scan.last_frame_pts + scan.last_frame_duration_ts
    assert Fraction(end_pts) * stream.time_base == 3
    assert [Fraction(item.source_pts) * item.time_base for item in scan.heartbeats] == [
        Fraction(0),
        Fraction(1),
        Fraction(2),
    ]
    assert scan.scene_frames
    assert all(max(item.width, item.height) <= 320 for item in scan.scene_frames)
    assert all(item.image_path.exists() for item in scan.scene_frames)
    assert all(max(item.width, item.height) <= 960 for item in scan.heartbeats)
    assert all(item.image_path.suffix == ".jpg" for item in scan.heartbeats)
    for heartbeat in scan.heartbeats:
        with Image.open(heartbeat.image_path) as proxy:
            assert proxy.format == "JPEG"
            assert len(proxy.getexif()) == 0


def test_scan_video_partitions_match_uninterrupted_scan(
    tmp_path: Path,
) -> None:
    """partitionの再集約結果が連続Video Scanと同一になること。

    Arrange:
        - scene境界を持つ3秒動画と中央PTSのpartition境界が用意される
    Act:
        - 動画全体と、有限range + EOF rangeの二通りでscanされる
    Assert:
        - timeline端点、heartbeat、scene signal、proxy bytesが一致すること
    """
    # Arrange
    video_path = generate_scene_change_video(tmp_path / "partition-scenes.mkv")
    runtime = FfmpegMediaRuntime()
    stream = runtime.probe(video_path).streams[0]
    assert stream.start_pts is not None
    assert stream.time_base is not None
    boundary_offset = Fraction(3, 2) / stream.time_base
    assert boundary_offset.denominator == 1
    boundary_pts = stream.start_pts + boundary_offset.numerator

    # Act
    uninterrupted = runtime.scan_video(
        video_path,
        stream,
        tmp_path / "uninterrupted",
        heartbeat_interval_seconds=1.0,
        scene_change_threshold=0.25,
        scene_min_interval_seconds=0.5,
        decode_backend="cpu",
    )
    first = runtime.scan_video_partition(
        video_path,
        stream,
        tmp_path / "partition-first",
        media_origin=stream.start_pts * stream.time_base,
        start_pts=stream.start_pts,
        end_pts=boundary_pts,
        heartbeat_interval_seconds=1.0,
        scene_change_threshold=0.25,
        scene_min_interval_seconds=0.5,
        decode_backend="cpu",
    )
    second = runtime.scan_video_partition(
        video_path,
        stream,
        tmp_path / "partition-second",
        media_origin=stream.start_pts * stream.time_base,
        start_pts=boundary_pts,
        end_pts=None,
        heartbeat_interval_seconds=1.0,
        scene_change_threshold=0.25,
        scene_min_interval_seconds=0.5,
        decode_backend="cpu",
    )

    # Assert
    assert isinstance(first, NativeVideoScan)
    assert isinstance(second, NativeVideoScan)
    assert first.origin_pts == uninterrupted.origin_pts
    assert second.last_frame_pts == uninterrupted.last_frame_pts
    assert second.last_frame_duration_ts == uninterrupted.last_frame_duration_ts
    partition_heartbeats = (*first.heartbeats, *second.heartbeats)
    partition_scenes = select_scene_signal_frames(
        (*first.scene_frames, *second.scene_frames),
        0.5,
    )
    assert [
        (frame.source_pts, frame.image_path.read_bytes())
        for frame in partition_heartbeats
    ] == [
        (frame.source_pts, frame.image_path.read_bytes())
        for frame in uninterrupted.heartbeats
    ]
    assert [
        (frame.source_pts, frame.image_path.read_bytes()) for frame in partition_scenes
    ] == [
        (frame.source_pts, frame.image_path.read_bytes())
        for frame in uninterrupted.scene_frames
    ]


def test_scan_partition_allows_no_owned_heartbeat_or_scene(
    tmp_path: Path,
) -> None:
    """signal ownershipが0件の正当なpartitionが成功されること。

    Arrange:
        - heartbeat bucket途中の1秒区間とsceneを選ばないthresholdが用意される
    Act:
        - 有限partitionが実FFmpegでscanされる
    Assert:
        - timeline端点はありheartbeatとsceneは空で返されること
    """
    # Arrange
    video_path = generate_scene_change_video(tmp_path / "empty-signals.mkv")
    runtime = FfmpegMediaRuntime()
    stream = runtime.probe(video_path).streams[0]
    assert stream.start_pts is not None
    assert stream.time_base is not None
    start_offset = Fraction(1) / stream.time_base
    end_offset = Fraction(2) / stream.time_base
    assert start_offset.denominator == 1
    assert end_offset.denominator == 1

    # Act
    scan = runtime.scan_video_partition(
        video_path,
        stream,
        tmp_path / "empty-signal-partition",
        media_origin=stream.start_pts * stream.time_base,
        start_pts=stream.start_pts + start_offset.numerator,
        end_pts=stream.start_pts + end_offset.numerator,
        heartbeat_interval_seconds=10.0,
        scene_change_threshold=1.0,
        scene_min_interval_seconds=0.5,
        decode_backend="cpu",
    )

    # Assert
    assert isinstance(scan, NativeVideoScan)
    assert scan.origin_pts >= stream.start_pts + start_offset.numerator
    assert scan.last_frame_pts < stream.start_pts + end_offset.numerator
    assert scan.heartbeats == ()
    assert scan.scene_frames == ()


def test_scan_partition_seeks_from_media_origin_when_video_starts_later(
    tmp_path: Path,
) -> None:
    """遅延video streamの有限partitionがmedia origin基準でseekされること。

    Arrange:
        - audioより2秒遅く始まるvideoと後半の1秒partitionが用意される
    Act:
        - container全体のmedia originを指定してpartition scanが実行される
    Assert:
        - 要求した半開PTS区間のvideo frameが返されること
    """
    # Arrange
    video_path = generate_delayed_video_with_audio(tmp_path / "delayed-video.mkv")
    runtime = FfmpegMediaRuntime()
    probe = runtime.probe(video_path)
    stream = next(item for item in probe.streams if item.kind == "video")
    assert stream.start_pts is not None
    assert stream.time_base is not None
    origins = tuple(
        item.start_pts * item.time_base
        for item in probe.streams
        if item.start_pts is not None and item.time_base is not None
    )
    media_origin = min(origins)
    start_offset = Fraction(10) / stream.time_base
    end_offset = Fraction(11) / stream.time_base
    assert start_offset.denominator == 1
    assert end_offset.denominator == 1
    start_pts = stream.start_pts + start_offset.numerator
    end_pts = stream.start_pts + end_offset.numerator

    # Act
    scan = runtime.scan_video_partition(
        video_path,
        stream,
        tmp_path / "delayed-video-partition",
        media_origin=media_origin,
        start_pts=start_pts,
        end_pts=end_pts,
        heartbeat_interval_seconds=1.0,
        scene_change_threshold=1.0,
        scene_min_interval_seconds=0.5,
        decode_backend="cpu",
    )

    # Assert
    assert isinstance(scan, NativeVideoScan)
    assert media_origin == 0
    assert stream.start_pts * stream.time_base == 2
    assert scan.origin_pts >= start_pts
    assert scan.last_frame_pts < end_pts


def test_scan_partition_returns_empty_result_after_video_eof(
    tmp_path: Path,
) -> None:
    """映像EOF後の成功した有限scanが空partitionとして返されること。

    Arrange:
        - 3秒で終了するvideoと4秒から5秒の半開PTS区間が用意される
    Act:
        - EOF後のpartition scanが実FFmpegで実行される
    Assert:
        - decoder failureではなく要求rangeに対応する空partitionが返されること
    """
    # Arrange
    video_path = generate_scene_change_video(tmp_path / "empty-after-eof.mkv")
    runtime = FfmpegMediaRuntime()
    stream = runtime.probe(video_path).streams[0]
    assert stream.start_pts is not None
    assert stream.time_base is not None
    start_offset = Fraction(4) / stream.time_base
    end_offset = Fraction(5) / stream.time_base
    assert start_offset.denominator == 1
    assert end_offset.denominator == 1
    start_pts = stream.start_pts + start_offset.numerator
    end_pts = stream.start_pts + end_offset.numerator

    # Act
    scan = runtime.scan_video_partition(
        video_path,
        stream,
        tmp_path / "empty-after-eof",
        media_origin=stream.start_pts * stream.time_base,
        start_pts=start_pts,
        end_pts=end_pts,
        heartbeat_interval_seconds=1.0,
        scene_change_threshold=1.0,
        scene_min_interval_seconds=0.5,
        decode_backend="cpu",
    )

    # Assert
    assert isinstance(scan, EmptyVideoScanPartition)
    assert scan.start_pts == start_pts
    assert scan.end_pts == end_pts


def test_scan_partition_does_not_treat_unparsed_proxy_output_as_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """proxyがあるtiming解析異常が空partitionとして扱われないこと。

    Arrange:
        - proxyを生成するが解釈不能なstderrだけを返す成功processが用意される
    Act:
        - 有限partition scanが実行される
    Assert:
        - EOF確定ではなくdecoder failureが返されること
    """
    # Arrange
    artifact_folder = tmp_path / "unparsed-partition"
    process = MagicMock()
    process.pid = 123
    process.returncode = None
    process.stderr.__iter__.return_value = iter(["unrecognized showinfo output\n"])

    def start_process(*_args: object, **_kwargs: object) -> MagicMock:
        heartbeat_folder = artifact_folder / "heartbeats"
        (heartbeat_folder / "000000000001.jpg").write_bytes(b"proxy")
        return process

    def reap(_process: object) -> tuple[int, float]:
        process.returncode = 0
        return 0, 0.01

    monkeypatch.setattr(
        "src.video_selection.media.ffmpeg_media_runtime.subprocess.Popen",
        start_process,
    )
    monkeypatch.setattr(
        "src.video_selection.media.ffmpeg_media_runtime.wait_for_process",
        reap,
    )
    runtime = FfmpegMediaRuntime()
    stream = MediaStream(
        index=0,
        kind="video",
        codec_name="h264",
        time_base=Fraction(1, 1000),
        start_pts=0,
        duration_ts=3000,
        width=1280,
        height=720,
        sample_rate=None,
        channels=None,
        language=None,
        is_default=True,
        is_forced=False,
    )

    # Act
    with pytest.raises(MediaRuntimeError) as caught:
        runtime.scan_video_partition(
            tmp_path / "source.mkv",
            stream,
            artifact_folder,
            media_origin=Fraction(0),
            start_pts=1000,
            end_pts=2000,
            heartbeat_interval_seconds=1.0,
            scene_change_threshold=0.3,
            scene_min_interval_seconds=0.5,
            decode_backend="cpu",
        )

    # Assert
    assert "timing" in str(caught.value)


def test_scan_video_reaps_decoder_when_stderr_processing_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stderr処理失敗時に実行中decoderが終了・回収されること。

    Arrange:
        - 一行返した後に失敗するstderrを持つ実行中FFmpeg processが用意される
    Act:
        - Video Scanが実行され、その後runtime全体のcancelが要求される
    Assert:
        - 失敗したprocessがSIGTERMされてwaitで回収されること
        - active processから解除され、後続cancelで再度終了されないこと
    """
    # Arrange
    process = MagicMock()
    process.pid = 123
    process.returncode = None

    def failing_stderr() -> Iterator[str]:
        yield "unrecognized stderr line\n"
        raise ValueError("stderr processing failed")

    process.stderr = failing_stderr()
    killed: list[tuple[int, int]] = []
    reaped: list[int] = []

    def reap(active_process: object) -> tuple[int, float]:
        assert active_process is process
        reaped.append(process.pid)
        process.returncode = -15
        return -15, 0.0

    monkeypatch.setattr(
        "src.video_selection.media.ffmpeg_media_runtime.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(
        "src.video_selection.media.ffmpeg_media_runtime.os.kill",
        lambda pid, sent_signal: killed.append((pid, sent_signal)),
    )
    monkeypatch.setattr(
        "src.video_selection.media.ffmpeg_media_runtime.wait_for_process",
        reap,
    )
    runtime = FfmpegMediaRuntime()
    stream = MediaStream(
        index=0,
        kind="video",
        codec_name="h264",
        time_base=Fraction(1, 1000),
        start_pts=0,
        duration_ts=1000,
        width=1280,
        height=720,
        sample_rate=None,
        channels=None,
        language=None,
        is_default=True,
        is_forced=False,
    )

    # Act
    with pytest.raises(ValueError, match="stderr processing failed"):
        runtime.scan_video(
            tmp_path / "source.mkv",
            stream,
            tmp_path / "scan-artifacts",
            heartbeat_interval_seconds=1.0,
            scene_change_threshold=0.25,
            scene_min_interval_seconds=0.5,
            decode_backend="cpu",
        )
    runtime.cancel_video_scans()

    # Assert
    assert killed == [(123, signal.SIGTERM)]
    assert reaped == [123]


def test_cancel_requested_before_scan_prevents_decoder_start(tmp_path: Path) -> None:
    """cancel後に新しいVideo Scan decoderが開始されないこと。

    Arrange:
        - probe済みVideo Sourceとcancel済みMedia Runtimeが用意される
    Act:
        - 同じruntimeでVideo Scan開始が要求される
    Assert:
        - decoder開始前のcancellationとして拒否されること
    """
    # Arrange
    video_path = generate_cfr_video(tmp_path / "cancel-before-scan.mkv")
    runtime = FfmpegMediaRuntime()
    stream = runtime.probe(video_path).streams[0]
    runtime.cancel_video_scans()

    # Act
    with pytest.raises(MediaRuntimeError) as exc_info:
        runtime.scan_video(
            video_path,
            stream,
            tmp_path / "cancelled-scan",
            heartbeat_interval_seconds=1.0,
            scene_change_threshold=0.25,
            scene_min_interval_seconds=0.5,
            decode_backend="cpu",
        )

    # Assert
    assert "cancel" in str(exc_info.value)


def test_extract_video_frame_returns_exact_requested_pts(tmp_path: Path) -> None:
    """指定されたsource PTSの一つのframe artifactが返されること。

    Arrange:
        - 0.5秒地点にsource frameを持つCFR fixtureが用意される
    Act:
        - 既知のsource PTSで一つのframeが抽出される
    Assert:
        - 0.5秒地点のRGB24 frameが返されること
    """
    # Arrange
    video_path = generate_cfr_video(tmp_path / "cfr.mkv")
    runtime = FfmpegMediaRuntime()
    stream = runtime.probe(video_path).streams[0]
    assert stream.time_base is not None
    target_pts = Fraction(1, 2) / stream.time_base
    assert target_pts.denominator == 1

    # Act
    frame = runtime.extract_video_frame(
        video_path,
        stream_index=stream.index,
        pts=target_pts.numerator,
        max_dimension=64,
    )

    # Assert
    assert Fraction(frame.pts) * frame.time_base == Fraction(1, 2)
    assert (frame.width, frame.height, frame.pixel_format) == (64, 48, "rgb24")
    assert len(frame.pixels) == 64 * 48 * 3


def test_extract_original_video_frame_preserves_odd_source_dimensions(
    tmp_path: Path,
) -> None:
    """奇数寸法のsource frameがscaleされず抽出されること。

    Arrange:
        - 65x49のsource frameを持つfixtureが用意される
    Act:
        - original frame extractionがexact PTSで実行される
    Assert:
        - 幅と高さが偶数へ丸められずRGB24で返されること
    """
    # Arrange
    video_path = generate_odd_dimension_video(tmp_path / "odd-dimension.mkv")
    runtime = FfmpegMediaRuntime()
    stream = runtime.probe(video_path).streams[0]

    # Act
    frame = runtime.extract_original_video_frame(
        video_path,
        stream_index=stream.index,
        pts=0,
    )

    # Assert
    assert (frame.width, frame.height, frame.pixel_format) == (65, 49, "rgb24")
    assert len(frame.pixels) == 65 * 49 * 3


def test_extract_original_video_frame_seek_preserves_late_nonzero_frame(
    tmp_path: Path,
) -> None:
    """入力seek後も遅い非ゼロPTSの元寸法frameが完全一致すること。

    Arrange:
        - 5秒開始のVideo Sourceと先頭decode済みの8秒frameが用意される
    Act:
        - 8秒の元寸法frameがexact PTS指定で再抽出される
    Assert:
        - PTS、time base、寸法、RGB pixelが先頭decode結果と一致すること
    """
    # Arrange
    video_path = generate_nonzero_start_video(tmp_path / "nonzero-original.mkv")
    runtime = FfmpegMediaRuntime()
    expected = next(
        frame
        for frame in runtime.scan_video_frames(video_path, 0, 64)
        if Fraction(frame.pts) * frame.time_base == Fraction(8)
    )

    # Act
    actual = runtime.extract_original_video_frame(
        video_path,
        stream_index=0,
        pts=expected.pts,
    )

    # Assert
    assert (
        actual.pts,
        actual.time_base,
        actual.width,
        actual.height,
        actual.pixels,
    ) == (
        expected.pts,
        expected.time_base,
        expected.width,
        expected.height,
        expected.pixels,
    )


def test_scan_video_frame_ranges_preserves_vfr_frames_inside_half_open_ranges(
    tmp_path: Path,
) -> None:
    """複数の半開PTS range内だけからnative VFR frameが返されること。

    Arrange:
        - 0、0.25、0.75、1.0秒にframeを持つVFR fixtureが用意される
    Act:
        - firstとmiddle-lastを覆う複数PTS rangeが一回でscanされる
    Assert:
        - range外frameや固定fps slotが追加されずexact PTSが返されること
    """
    # Arrange
    video_path = generate_vfr_video(tmp_path / "vfr-ranges.mkv")
    runtime = FfmpegMediaRuntime()
    decoder_cpu_seconds: list[float] = []

    # Act
    frames = tuple(
        runtime.scan_video_frame_ranges(
            video_path,
            stream_index=0,
            pts_ranges=((0, 1), (750, 1001)),
            max_dimension=64,
            cpu_seconds_recorder=decoder_cpu_seconds.append,
        )
    )

    # Assert
    assert [frame.pts for frame in frames] == [0, 750, 1000]
    assert sum(decoder_cpu_seconds) > 0


def test_cancel_frame_refinements_terminates_active_range_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frame Refinement中止時にactive range decoderが終了されること。

    Arrange:
        - 中止要求まで終了しないrange decoderとprobe結果が用意される
    Act:
        - 別threadでrange scanが開始され、runtimeへ中止が要求される
    Assert:
        - active decoderへSIGTERMが一度送られること
        - 終了済みdecoderがactive集合から解除されること
    """
    # Arrange
    runtime = FfmpegMediaRuntime()
    stream = MediaStream(
        index=0,
        kind="video",
        codec_name="h264",
        time_base=Fraction(1, 1000),
        start_pts=0,
        duration_ts=1000,
        width=1280,
        height=720,
        sample_rate=None,
        channels=None,
        language=None,
        is_default=True,
        is_forced=False,
    )
    probe = MediaProbe(format_names=("matroska",), streams=(stream,))
    decoder_started = threading.Event()
    release_decoder = threading.Event()
    killed: list[tuple[int, int]] = []

    def iter_blocked_frames(
        _command: list[str],
        _stream_index: int,
        *,
        cpu_seconds_recorder: object = None,
        on_process_started: object = None,
        on_process_finished: object = None,
    ) -> Iterator[object]:
        del cpu_seconds_recorder
        process = MagicMock()
        process.pid = 123
        assert callable(on_process_started)
        assert callable(on_process_finished)
        on_process_started(process)
        decoder_started.set()
        try:
            assert release_decoder.wait(timeout=5)
            yield from ()
        finally:
            on_process_finished(process)

    def terminate_decoder(pid: int, sent_signal: int) -> None:
        killed.append((pid, sent_signal))
        release_decoder.set()

    monkeypatch.setattr(runtime, "probe", lambda _path: probe)
    monkeypatch.setattr(
        "src.video_selection.media.ffmpeg_media_runtime.iter_decoded_video_frames",
        iter_blocked_frames,
    )
    monkeypatch.setattr(
        "src.video_selection.media.ffmpeg_media_runtime.os.kill",
        terminate_decoder,
    )
    failures: list[BaseException] = []

    def scan_ranges() -> None:
        try:
            tuple(
                runtime.scan_video_frame_ranges(
                    tmp_path / "source.mkv",
                    stream_index=0,
                    pts_ranges=((0, 1000),),
                    max_dimension=960,
                )
            )
        except BaseException as error:
            failures.append(error)

    scan_thread = threading.Thread(target=scan_ranges)
    scan_thread.start()
    assert decoder_started.wait(timeout=5)

    # Act
    runtime.cancel_frame_refinements()
    scan_thread.join(timeout=5)
    runtime.cancel_frame_refinements()

    # Assert
    assert not scan_thread.is_alive()
    assert failures == []
    assert killed == [(123, signal.SIGTERM)]


def test_scan_video_frame_ranges_seek_preserves_nonzero_source_frames(
    tmp_path: Path,
) -> None:
    """分離rangeへのinput seek後も非ゼロPTSとRGB frameが保持されること。

    Arrange:
        - 5秒開始で4fpsのVideo Sourceと分離した2 rangeが用意される
        - 全frameを先頭からdecodeした比較結果が用意される
    Act:
        - 複数rangeだけがinput seek付きでscanされる
    Assert:
        - range内のPTSとRGB pixelが先頭decode結果と完全一致すること
    """
    # Arrange
    video_path = generate_nonzero_start_video(tmp_path / "nonzero-start.mkv")
    runtime = FfmpegMediaRuntime()
    all_frames = tuple(runtime.scan_video_frames(video_path, 0, 64))
    ranges = ((6000, 6500), (8000, 8750))
    expected = tuple(
        frame
        for frame in all_frames
        if any(start <= frame.pts < end for start, end in ranges)
    )

    # Act
    actual = tuple(
        runtime.scan_video_frame_ranges(
            video_path,
            stream_index=0,
            pts_ranges=ranges,
            max_dimension=64,
        )
    )

    # Assert
    assert [(frame.pts, frame.pixels) for frame in actual] == [
        (frame.pts, frame.pixels) for frame in expected
    ]


def test_write_mjpeg_proxy_encodes_selected_rgb_frame_without_source_metadata(
    tmp_path: Path,
) -> None:
    """選抜済みRGB frameがmetadataなしMJPEG proxyへ保存されること。

    Arrange:
        - CFR fixtureからexact PTSで抽出されたRGB frameが用意される
    Act:
        - FFmpeg MJPEG q:v=3でcandidate proxyが保存される
    Assert:
        - JPEGが元frame寸法で読めEXIF metadataを持たないこと
    """
    # Arrange
    video_path = generate_cfr_video(tmp_path / "proxy-source.mkv")
    runtime = FfmpegMediaRuntime()
    frame = runtime.extract_video_frame(video_path, 0, 0, 64)
    proxy_path = tmp_path / "candidate.jpg"

    # Act
    encoder_cpu_seconds = runtime.write_mjpeg_proxy(frame, proxy_path, quality=3)

    # Assert
    with Image.open(proxy_path) as proxy:
        assert proxy.format == "JPEG"
        assert proxy.size == (64, 48)
        assert len(proxy.getexif()) == 0
    assert encoder_cpu_seconds > 0


def test_real_cfr_vfr_video_stage_preserves_exact_timeline_and_scene_boundaries(
    tmp_path: Path,
) -> None:
    """実FFmpegのCFR/VFRがVideo Stageのexact成果物へ確定されること。

    Arrange:
        - VFR fixtureと明確なscene changeを持つCFR fixtureが用意される
    Act:
        - 両動画がVideo Order順にVideo Stage processorへ通される
    Assert:
        - VFR durationが5/4秒として保持されること
        - scene境界を持つsegmentがgapとoverlapなく3秒を覆うこと
        - Candidate ProxyだけがCompleted Stageに残りscene画像が残らないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    generate_vfr_video(input_folder / "01-vfr.mkv")
    generate_scene_change_video(input_folder / "02-scenes.mkv")
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
    )
    video_set = discover_video_set(input_folder)

    # Act
    results = VideoStageProcessor(
        FfmpegMediaRuntime(),
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(video_set, configuration)

    # Assert
    assert results[0].scan.timeline.duration.seconds == Fraction(5, 4)
    scene_timeline = results[1].scan.timeline
    assert scene_timeline.duration.seconds == Fraction(3)
    assert len(scene_timeline.segments) >= 2
    assert scene_timeline.segments[0].start == 0
    assert scene_timeline.segments[-1].end == 3
    assert all(
        left.end == right.start
        for left, right in zip(
            scene_timeline.segments,
            scene_timeline.segments[1:],
            strict=False,
        )
    )
    assert not tuple(configuration.processing_cache_folder.rglob(".scene-proxies"))
    assert all(
        candidate.proxy_path is not None and candidate.proxy_path.suffix == ".jpg"
        for result in results
        for candidate in result.extraction.candidates
    )


def test_probe_reports_audio_and_embedded_subtitle_streams(
    tmp_path: Path,
) -> None:
    """multiple audio/subtitle streamの意味とdispositionが返されること。

    Arrange:
        - video、2 audio、2 embedded text subtitleを持つfixtureが用意される
    Act:
        - MediaRuntimeでfixtureがprobeされる
    Assert:
        - stream順、言語、default、forced metadataが区別されること
    """
    # Arrange
    video_path = generate_stream_matrix_video(tmp_path / "streams.mkv")
    runtime = FfmpegMediaRuntime()

    # Act
    streams = runtime.probe(video_path).streams

    # Assert
    assert [stream.kind for stream in streams] == [
        "video",
        "audio",
        "audio",
        "subtitle",
        "subtitle",
    ]
    assert [(stream.language, stream.is_default) for stream in streams[1:3]] == [
        ("jpn", True),
        ("eng", False),
    ]
    assert [
        (stream.language, stream.is_default, stream.is_forced) for stream in streams[3:]
    ] == [
        ("jpn", True, False),
        ("eng", False, True),
    ]


def test_required_av1_and_aac_streams_are_decoded(tmp_path: Path) -> None:
    """required AV1 videoとAAC audioがsemantic artifactへdecodeされること。

    Arrange:
        - AV1 videoとAAC audioを持つ専用fixtureが用意される
    Act:
        - video frameとPCM audioが既存のscan経路でdecodeされる
    Assert:
        - AV1/AAC codecと非空のframe・PCM artifactが返されること
    """
    # Arrange
    video_path = generate_av1_aac_video(tmp_path / "av1-aac.mkv")
    runtime = FfmpegMediaRuntime()
    streams = runtime.probe(video_path).streams

    # Act
    frames = tuple(
        runtime.scan_video_frames(
            video_path,
            stream_index=0,
            max_dimension=64,
        )
    )
    chunks = tuple(
        runtime.scan_pcm_audio(
            video_path,
            stream_index=1,
            sample_rate=16_000,
            frame_sample_count=4_000,
        )
    )

    # Assert
    assert [(stream.kind, stream.codec_name) for stream in streams[:2]] == [
        ("video", "av1"),
        ("audio", "aac"),
    ]
    assert frames
    assert all(frame.pixels for frame in frames)
    assert chunks
    assert all(chunk.pcm_bytes for chunk in chunks)


def test_scan_pcm_audio_returns_contiguous_sample_grid(tmp_path: Path) -> None:
    """選択audioがmono 16kHz signed 16-bit PCMへ連続decodeされること。

    Arrange:
        - 3秒の440Hz audio streamを持つfixtureが用意される
    Act:
        - 4,000 sample単位でPCMがscanされる
    Assert:
        - 48,000 sampleがgapなしのsample gridと非無音artifactで返されること
    """
    # Arrange
    video_path = generate_stream_matrix_video(tmp_path / "streams.mkv")
    runtime = FfmpegMediaRuntime()

    # Act
    chunks = tuple(
        runtime.scan_pcm_audio(
            video_path,
            stream_index=1,
            sample_rate=16_000,
            frame_sample_count=4_000,
        )
    )

    # Assert
    assert sum(chunk.sample_count for chunk in chunks) == 48_000
    assert [chunk.sample_start for chunk in chunks] == list(range(0, 48_000, 4_000))
    assert all(chunk.sample_rate == 16_000 for chunk in chunks)
    assert all(chunk.channel_count == 1 for chunk in chunks)
    assert all(chunk.sample_format == "s16le" for chunk in chunks)
    assert all(
        Fraction(chunk.pts) * chunk.time_base == Fraction(chunk.sample_start, 16_000)
        for chunk in chunks
    )
    samples = [
        value
        for chunk in chunks
        for (value,) in struct.iter_unpack("<h", chunk.pcm_bytes)
    ]
    assert sum(abs(value) for value in samples) / len(samples) > 100


def test_extract_pcm_audio_chunks_returns_canonical_seek_ranges(
    tmp_path: Path,
) -> None:
    """seek付きrange抽出がgapのないcanonical PCM gridを返すこと。

    Arrange:
        - 3秒のAAC audio streamと4,000 sample ownershipが用意される
    Act:
        - EOFまでrangeごとに独立したFFmpeg processで抽出される
    Assert:
        - 48,000 sampleが連続gridとして返され各rangeが非無音であること
    """
    # Arrange
    video_path = generate_stream_matrix_video(tmp_path / "range-streams.mkv")
    runtime = FfmpegMediaRuntime()
    probe = runtime.probe(video_path)
    stream = probe.streams[1]
    origins = tuple(
        item.start_pts * item.time_base
        for item in probe.streams
        if item.start_pts is not None and item.time_base is not None
    )
    media_origin = min(origins)

    # Act
    chunks = []
    sample_start = 0
    while True:
        chunk = runtime.extract_pcm_audio_chunk(
            video_path,
            stream,
            media_origin,
            16_000,
            sample_start,
            4_000,
        )
        if chunk is None:
            break
        chunks.append(chunk)
        if chunk.sample_count < 4_000:
            break
        sample_start += 4_000

    # Assert
    assert sum(chunk.sample_count for chunk in chunks) == 48_000
    assert [chunk.sample_start for chunk in chunks] == list(range(0, 48_000, 4_000))
    assert all(
        chunk.pts * chunk.time_base
        == stream.start_pts * stream.time_base + Fraction(chunk.sample_start, 16_000)
        for chunk in chunks
        if stream.start_pts is not None and stream.time_base is not None
    )
    assert all(any(chunk.pcm_bytes) for chunk in chunks)


def test_extract_pcm_audio_chunk_rejects_observed_timestamp_discontinuity(
    tmp_path: Path,
) -> None:
    """range先頭の観測PTSがsample gridと不連続なら確定されないこと。

    Arrange:
        - 1秒以降のpacket PTSが連続sample gridからずれるaudioが用意される
        - 不連続より後から始まるcheckpoint rangeが指定される
    Act:
        - rangeが独立したFFmpeg processで抽出される
    Assert:
        - 観測PTSが期待値へ置換されずaudio抽出失敗として拒否されること
    """
    # Arrange
    audio_path = generate_discontinuous_audio(tmp_path / "discontinuous-audio.mkv")
    runtime = FfmpegMediaRuntime()
    probe = runtime.probe(audio_path)
    stream = probe.streams[0]
    assert stream.start_pts is not None
    assert stream.time_base is not None
    media_origin = stream.start_pts * stream.time_base

    # Act
    with pytest.raises(MediaRuntimeError, match="timestamp") as captured:
        runtime.extract_pcm_audio_chunk(
            audio_path,
            stream,
            media_origin,
            16_000,
            17_000,
            15_000,
        )

    # Assert
    assert captured.value.reason is MediaRuntimeFailureReason.AUDIO_EXTRACTION_FAILED


def test_scan_pcm_audio_normalizes_quantized_packet_pts_to_sample_grid(
    tmp_path: Path,
) -> None:
    """packet PTSの量子化ずれが連続PCM sample gridへ正規化されること。

    Arrange:
        - 後半packet PTSが3 output sampleずれるaudio fixtureが用意される
    Act:
        - packet境界をまたぐ17,000 sample単位でPCMがscanされる
    Assert:
        - stream開始PTSとsample indexから連続chunk PTSが生成されること
    """
    # Arrange
    audio_path = generate_quantized_audio(tmp_path / "quantized-audio.mkv")
    runtime = FfmpegMediaRuntime()

    # Act
    chunks = tuple(
        runtime.scan_pcm_audio(
            audio_path,
            stream_index=0,
            sample_rate=16_000,
            frame_sample_count=17_000,
        )
    )

    # Assert
    assert [chunk.sample_start for chunk in chunks] == [0, 17_000]
    assert [chunk.sample_count for chunk in chunks] == [17_000, 15_000]
    assert all(
        chunk.pts * chunk.time_base == Fraction(chunk.sample_start, 16_000)
        for chunk in chunks
    )


def test_read_embedded_subtitles_preserves_packet_pts_and_text(
    tmp_path: Path,
) -> None:
    """embedded text subtitleが元packet PTSと本文を保持して返されること。

    Arrange:
        - repository所有の日本語字幕を内蔵したfixtureが用意される
    Act:
        - 選択subtitle streamのeventが読み取られる
    Assert:
        - repo-owned本文と0.5秒・1.5秒の元packet時刻が返されること
    """
    # Arrange
    video_path = generate_stream_matrix_video(tmp_path / "streams.mkv")
    runtime = FfmpegMediaRuntime()
    subtitle_stream = next(
        stream
        for stream in runtime.probe(video_path).streams
        if stream.kind == "subtitle" and stream.language == "jpn"
    )

    # Act
    events = runtime.read_embedded_subtitles(
        video_path,
        stream_index=subtitle_stream.index,
    )

    # Assert
    assert [event.text for event in events] == [
        "これは既定の日本語字幕です",
        "二つ目の字幕イベントです",
    ]
    assert [Fraction(event.pts) * event.time_base for event in events] == [
        Fraction(1, 2),
        Fraction(3, 2),
    ]
    assert [Fraction(event.duration_ts) * event.time_base for event in events] == [
        Fraction(3, 4),
        Fraction(3, 4),
    ]


def test_real_embedded_subtitles_become_exact_context_cues(tmp_path: Path) -> None:
    """実embedded subtitleがVideo Time付きContext Cueへ変換されること。

    Arrange:
        - 日本語non-forced subtitleとaudioを持つ実Matroska fixtureが用意される
    Act:
        - fixtureが3つのsource-local Video Stageへ通される
    Assert:
        - subtitle packet PTSから正確なCue範囲と本文が得られること
        - subtitle優先によりSpeechRuntimeが実行されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    generate_stream_matrix_video(input_folder / "streams.mkv")
    speech_runtime = FakeSpeechRuntime()

    # Act
    result = VideoStageProcessor(
        FfmpegMediaRuntime(),
        speech_runtime,
        RecordingRunObserver(),
    ).process(
        discover_video_set(input_folder),
        EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=tmp_path / "output",
            language="ja",
        ),
    )[0]

    # Assert
    assert [cue.text for cue in result.context.cues] == [
        "これは既定の日本語字幕です",
        "二つ目の字幕イベントです",
    ]
    assert [
        (cue.start, cue.end, cue.timestamp_basis) for cue in result.context.cues
    ] == [
        (Fraction(1, 2), Fraction(5, 4), "source_pts"),
        (Fraction(3, 2), Fraction(9, 4), "source_pts"),
    ]
    assert speech_runtime.transcribe_calls == []


def test_scan_video_frames_reports_corrupt_packet_as_decoder_failure(
    tmp_path: Path,
) -> None:
    """probe可能な破損packetがstable decoder failureへ変換されること。

    Arrange:
        - headerはprobe可能で途中packetが破損したMPEG-TS fixtureが用意される
    Act:
        - MediaRuntimeで全video frameがscanされる
    Assert:
        - decoder_failureとして失敗すること
    """
    # Arrange
    video_path = generate_corrupt_video(tmp_path / "corrupt.ts")
    runtime = FfmpegMediaRuntime()
    stream = runtime.probe(video_path).streams[0]

    # Act
    with pytest.raises(MediaRuntimeError) as captured:
        tuple(
            runtime.scan_video_frames(
                video_path,
                stream_index=stream.index,
                max_dimension=64,
            )
        )

    # Assert
    assert captured.value.reason is MediaRuntimeFailureReason.DECODER_FAILURE
