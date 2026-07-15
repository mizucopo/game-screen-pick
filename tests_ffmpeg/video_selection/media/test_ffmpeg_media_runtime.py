"""system FFmpeg MediaRuntimeのintegration test。"""

import stat
import struct
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from src.video_selection.media.ffmpeg_media_runtime import FfmpegMediaRuntime
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.media_runtime_error import MediaRuntimeError
from src.video_selection.models.media_runtime_failure_reason import (
    MediaRuntimeFailureReason,
)
from src.video_selection.services.discover_video_set import discover_video_set
from src.video_selection.services.video_stage_processor import VideoStageProcessor
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver
from tests_ffmpeg.support.ffmpeg_fixture_factory import (
    generate_av1_aac_video,
    generate_cfr_video,
    generate_corrupt_video,
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
    print("T.C asetnsamples")
    print(" ... ashowinfo")
    print(" ... format")
    print(" ... nullsink")
    print("..C scale")
    print(" ... select")
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

    # Act
    frames = tuple(
        runtime.scan_video_frame_ranges(
            video_path,
            stream_index=0,
            pts_ranges=((0, 1), (750, 1001)),
            max_dimension=64,
        )
    )

    # Assert
    assert [frame.pts for frame in frames] == [0, 750, 1000]


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
    runtime.write_mjpeg_proxy(frame, proxy_path, quality=3)

    # Assert
    with Image.open(proxy_path) as proxy:
        assert proxy.format == "JPEG"
        assert proxy.size == (64, 48)
        assert len(proxy.getexif()) == 0


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
