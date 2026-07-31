"""Context Collection Stageの統合style test。"""

import hashlib
import json
import traceback
from fractions import Fraction
from pathlib import Path

import pytest

from src.video_selection.models.checkpoint_operation import CheckpointOperation
from src.video_selection.models.context_stage_error import ContextStageError
from src.video_selection.models.context_stage_failure_reason import (
    ContextStageFailureReason,
)
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.embedded_subtitle import EmbeddedSubtitle
from src.video_selection.models.media_probe import MediaProbe
from src.video_selection.models.media_runtime_error import MediaRuntimeError
from src.video_selection.models.media_runtime_failure_reason import (
    MediaRuntimeFailureReason,
)
from src.video_selection.models.media_runtime_identity import MediaRuntimeIdentity
from src.video_selection.models.media_stream import MediaStream, MediaStreamKind
from src.video_selection.models.pcm_audio_chunk import PcmAudioChunk
from src.video_selection.models.speech_recognition_result import (
    SpeechRecognitionResult,
)
from src.video_selection.models.speech_segment import SpeechSegment
from src.video_selection.models.speech_word import SpeechWord
from src.video_selection.models.timeline_segment import TimelineSegment
from src.video_selection.models.video_duration import VideoDuration
from src.video_selection.models.video_scan_metrics import VideoScanMetrics
from src.video_selection.models.video_scan_result import VideoScanResult
from src.video_selection.models.video_timeline import VideoTimeline
from src.video_selection.services.build_context_cue_id import build_context_cue_id
from src.video_selection.services.checkpoint_version import checkpoint_version
from src.video_selection.services.context_stage_processor import ContextStageProcessor
from src.video_selection.services.discover_video_set import discover_video_set
from src.video_selection.services.run_progress_tracker import RunProgressTracker
from tests.video_selection.fakes.fake_speech_runtime import FakeSpeechRuntime
from tests.video_selection.fakes.fake_video_stage_media_runtime import (
    FakeVideoStageMediaRuntime,
)
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver


def _rewrite_hash_consistent_artifact(
    checkpoint_folder: Path,
    artifact: dict[str, object],
) -> None:
    """artifactとmanifest recordを同じ破損内容へ揃えて書き換える。"""
    artifact_path = checkpoint_folder / "artifact.json"
    manifest_path = checkpoint_folder / "manifest.json"
    artifact_bytes = (
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    artifact_path.write_bytes(artifact_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_record = next(
        item for item in manifest["artifacts"] if item["path"] == "artifact.json"
    )
    artifact_record["size_bytes"] = len(artifact_bytes)
    artifact_record["sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_non_forced_text_subtitle_is_preferred_without_running_stt(
    tmp_path: Path,
) -> None:
    """non-forced text subtitleがContext Cueへ変換されSTTが抑止されること。

    Arrange:
        - 日本語のnon-forced subtitleとaudioを持つVideo Sourceが用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - subtitleのsource PTSから正確なContext Cueが生成されること
        - SpeechRuntimeが呼ばれないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(
            _stream(0, "video", "ffv1", is_default=True),
            _stream(1, "audio", "pcm_s16le", language="jpn", is_default=True),
            _stream(2, "subtitle", "subrip", language="jpn", is_default=True),
        ),
    )
    media_runtime = FakeVideoStageMediaRuntime(
        media_probe=probe,
        embedded_subtitles=(
            EmbeddedSubtitle(
                stream_index=2,
                pts=125,
                duration_ts=20,
                time_base=Fraction(1, 10),
                text="  千年の物語が始まる  ",
            ),
        ),
    )
    speech_runtime = FakeSpeechRuntime()
    processor = ContextStageProcessor(
        media_runtime,
        speech_runtime,
        RecordingRunObserver(),
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        language="ja",
    )

    # Act
    result = processor.process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=configuration,
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )

    # Assert
    assert len(result.cues) == 1
    cue = result.cues[0]
    assert cue.identifier == (
        "cue_4d787c508cf11a5aa78032601ea1c6f30549bad450d897a29715922fdf58b518"
    )
    assert cue.video_fingerprint == source.fingerprint
    assert cue.source_kind == "embedded_subtitle"
    assert cue.stream_index == 2
    assert cue.start == Fraction(5, 2)
    assert cue.end == Fraction(9, 2)
    assert cue.timestamp_basis == "source_pts"
    assert cue.text == "千年の物語が始まる"
    assert cue.language == "jpn"
    assert cue.reliability == "usable"
    assert cue.diagnostics is None
    assert cue.provenance is not None
    assert cue.provenance.codec_name == "subrip"
    assert cue.provenance.source_pts == 125
    assert cue.provenance.source_time_base == Fraction(1, 10)
    assert cue.provenance.stream_language == "jpn"
    assert cue.provenance.is_default is True
    assert cue.provenance.is_forced is False
    assert cue.provenance.language_source == "stream_metadata"
    assert cue.provenance.chunk_sample_start is None
    assert cue.provenance.chunk_sample_end is None
    assert cue.provenance.speech_runtime_identity is None
    assert cue.provenance.resolved_model_identity is None
    assert [(item.status, item.reason_code) for item in result.outcomes] == [
        ("available", "context_extracted")
    ]
    assert speech_runtime.transcribe_calls == []


def test_hash_consistent_wrong_context_parent_reuses_subtitle_checkpoint(
    tmp_path: Path,
) -> None:
    """別動画を指す親Contextだけが破棄されsubtitle checkpointから修復されること。

    Arrange:
        - subtitle Cueを持つ親Contextと子checkpointが正常に確定される
        - 親Cueの動画fingerprintとIDがhash整合を保って別動画へ改変される
    Act:
        - 同じVideo SourceのContext Stageが再実行される
    Assert:
        - 親Contextだけが再構築されsubtitle抽出は再実行されないこと
        - 修復後の意味的なContext結果が初回結果と一致すること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(
            _stream(0, "video", "ffv1", is_default=True),
            _stream(2, "subtitle", "subrip", language="jpn", is_default=True),
        ),
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        language="ja",
    )
    initial = ContextStageProcessor(
        FakeVideoStageMediaRuntime(
            media_probe=probe,
            embedded_subtitles=(
                EmbeddedSubtitle(
                    stream_index=2,
                    pts=125,
                    duration_ts=20,
                    time_base=Fraction(1, 10),
                    text="千年の物語が始まる",
                ),
            ),
        ),
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=configuration,
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )
    context_root = (
        configuration.processing_cache_folder
        / "videos"
        / source.fingerprint
        / "collect-context"
    )
    context_folder = next(
        path
        for path in context_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    artifact = json.loads(
        (context_folder / "artifact.json").read_text(encoding="utf-8")
    )
    cue = artifact["cues"][0]
    wrong_fingerprint = "b" * 64
    cue["video_fingerprint"] = wrong_fingerprint
    cue["id"] = build_context_cue_id(
        video_fingerprint=wrong_fingerprint,
        source_kind=cue["source_kind"],
        stream_index=cue["stream_index"],
        start=Fraction(
            cue["start"]["numerator"],
            cue["start"]["denominator"],
        ),
        end=Fraction(
            cue["end"]["numerator"],
            cue["end"]["denominator"],
        ),
        text=cue["text"],
    )
    _rewrite_hash_consistent_artifact(context_folder, artifact)
    retry_runtime = FakeVideoStageMediaRuntime(media_probe=probe)

    # Act
    repaired = ContextStageProcessor(
        retry_runtime,
        FakeSpeechRuntime(),
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=configuration,
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )

    # Assert
    assert retry_runtime.subtitle_calls == []
    assert repaired.cues == initial.cues
    assert repaired.outcomes == initial.outcomes
    assert repaired.rejected_speech_diagnostics == (initial.rejected_speech_diagnostics)
    assert repaired.equivalence_groups == initial.equivalence_groups


def test_stt_chunk_emits_external_work_start_event(tmp_path: Path) -> None:
    """STT chunkのblocking処理開始がProgress Eventとして通知されること。

    Arrange:
        - audio 1 chunkとrun開始済みProgress Trackerが用意される
    Act:
        - Context Collection StageのSTT fallbackが実行される
    Assert:
        - speech recognition開始eventがraw transcriptなしで一度通知されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    pcm = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=16000,
        sample_rate=16000,
        channel_count=1,
        sample_format="s16le",
        pts=160000,
        time_base=Fraction(1, 16000),
        pcm_bytes=b"\x00\x00" * 16000,
    )
    observer = RecordingRunObserver()
    progress = RunProgressTracker(observer, clock=lambda: 10.0)
    progress.start_run()
    processor = ContextStageProcessor(
        FakeVideoStageMediaRuntime(media_probe=probe, pcm_audio_chunks=(pcm,)),
        FakeSpeechRuntime(),
        observer,
        progress=progress,
    )

    # Act
    processor.process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=tmp_path / "output",
            language="ja",
        ),
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )

    # Assert
    assert tuple(
        (event.kind, event.reason_code, event.processed_count, event.eta_seconds)
        for event in observer.progress_events
        if event.kind == "external_work_started"
    ) == (("external_work_started", "speech_recognition_started", None, None),)


def test_empty_non_forced_subtitle_is_no_context_without_stt_fallback(
    tmp_path: Path,
) -> None:
    """eventがないnon-forced subtitleがno_contextにされること。

    Arrange:
        - eventを返さないnon-forced subtitle streamとaudioが用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - no_context outcomeが記録されContext Cueが生成されないこと
        - audio STTへfallbackされないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(
            _stream(0, "video", "ffv1", is_default=True),
            _stream(1, "audio", "pcm_s16le", language="jpn", is_default=True),
            _stream(2, "subtitle", "subrip", language="jpn", is_default=True),
        ),
    )
    media_runtime = FakeVideoStageMediaRuntime(
        media_probe=probe,
        embedded_subtitles=(),
    )
    speech_runtime = FakeSpeechRuntime()

    # Act
    result = ContextStageProcessor(
        media_runtime,
        speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=tmp_path / "output",
            language="ja",
        ),
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )

    # Assert
    assert result.cues == ()
    assert [(item.status, item.reason_code) for item in result.outcomes] == [
        ("no_context", "no_subtitle_events")
    ]
    assert speech_runtime.transcribe_calls == []
    assert result.completed_stage is not None
    assert result.completed_stage.semantic_input["checkpoint_contracts"] == {
        "embedded_subtitle_stream": checkpoint_version(
            CheckpointOperation.EMBEDDED_SUBTITLE_STREAM
        )
    }


def test_audio_stt_is_used_when_text_subtitle_is_absent(tmp_path: Path) -> None:
    """subtitle不在時にaudio STTがexact Video TimeのContext Cueへ変換されること。

    Arrange:
        - subtitleがなく日本語audioだけを持つVideo Sourceが用意される
        - SpeechRuntimeがsample位置付きwordを返すよう用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - wordがgap policyでまとめられ正確なVideo Timeへ変換されること
        - subtitle不在とSTT成功が別々のoutcomeとして返されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    pcm = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=160000,
        sample_rate=16000,
        channel_count=1,
        sample_format="s16le",
        pts=160000,
        time_base=Fraction(1, 16000),
        pcm_bytes=b"\x00\x00" * 160000,
    )
    media_runtime = FakeVideoStageMediaRuntime(
        media_probe=probe,
        pcm_audio_chunks=(pcm,),
    )
    speech_runtime = FakeSpeechRuntime(
        (
            SpeechRecognitionResult(
                vad_speech_detected=True,
                detected_language="ja",
                segments=(
                    SpeechSegment(
                        words=(
                            SpeechWord("冒険", 16000, 24000, 0.95),
                            SpeechWord("者", 24000, 24000, 0.94),
                            SpeechWord("が", 24800, 27200, 0.93),
                            SpeechWord("始まる", 28800, 40000, 0.96),
                        ),
                        average_log_probability=-0.5,
                        no_speech_probability=0.01,
                    ),
                ),
            ),
        )
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        language="ja-JP",
    )

    # Act
    result = ContextStageProcessor(
        media_runtime,
        speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=configuration,
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )

    # Assert
    assert len(result.cues) == 1
    cue = result.cues[0]
    assert cue.source_kind == "speech_to_text"
    assert cue.stream_index == 1
    assert cue.start == Fraction(1)
    assert cue.end == Fraction(5, 2)
    assert cue.timestamp_basis == "asr_sample_grid_estimate"
    assert cue.text == "冒険者が始まる"
    assert cue.language == "ja"
    assert cue.reliability == "usable"
    assert cue.diagnostics is not None
    assert cue.diagnostics.average_log_probability == -0.5
    assert cue.diagnostics.no_speech_probability == 0.01
    assert cue.diagnostics.word_probabilities == (0.95, 0.94, 0.93, 0.96)
    assert cue.provenance is not None
    assert cue.provenance.codec_name == "pcm_s16le"
    assert cue.provenance.source_pts == 160000
    assert cue.provenance.source_time_base == Fraction(1, 16000)
    assert cue.provenance.stream_language == "jpn"
    assert cue.provenance.is_default is True
    assert cue.provenance.is_forced is False
    assert cue.provenance.language_source == "speech_recognition"
    assert cue.provenance.chunk_sample_start == 0
    assert cue.provenance.chunk_sample_end == 160000
    assert cue.provenance.speech_runtime_identity == "fake-speech-runtime-v1"
    assert cue.provenance.resolved_model_identity == "hf:" + "0" * 40
    assert cue.provenance.device == "cuda"
    assert cue.provenance.compute_type == "float16"
    assert [(item.status, item.reason_code) for item in result.outcomes] == [
        ("absent", "no_subtitle_stream"),
        ("available", "context_extracted"),
    ]
    assert speech_runtime.transcribe_calls == [pcm]
    assert speech_runtime.transcribe_options == [("ja", True, 5)]


@pytest.mark.parametrize(
    ("vad_speech_detected", "expected_reason"),
    [(False, "vad_no_speech"), (True, "asr_no_speech")],
)
def test_vad_and_asr_no_speech_outcomes_are_distinguished(
    tmp_path: Path,
    vad_speech_detected: bool,
    expected_reason: str,
) -> None:
    """VAD無発話とASR無発話が異なる正常outcomeにされること。

    Arrange:
        - audioと空のSpeech Recognition Resultが用意される
        - VADの発話検出有無が指定される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - no_speech statusにVADまたはASR由来のreasonが記録されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    pcm = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=16000,
        sample_rate=16000,
        channel_count=1,
        sample_format="s16le",
        pts=160000,
        time_base=Fraction(1, 16000),
        pcm_bytes=b"\x00\x00" * 16000,
    )
    speech_runtime = FakeSpeechRuntime(
        (
            SpeechRecognitionResult(
                vad_speech_detected=vad_speech_detected,
                segments=(),
            ),
        )
    )

    # Act
    result = ContextStageProcessor(
        FakeVideoStageMediaRuntime(
            media_probe=probe,
            pcm_audio_chunks=(pcm,),
        ),
        speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=tmp_path / "output",
            language="ja",
        ),
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )

    # Assert
    assert result.cues == ()
    assert [(item.status, item.reason_code) for item in result.outcomes] == [
        ("absent", "no_subtitle_stream"),
        ("no_speech", expected_reason),
    ]
    assert result.outcomes[1].processed_chunk_count == 1


def test_low_reliability_speech_is_isolated_in_private_cache(
    tmp_path: Path,
) -> None:
    """低信頼STT文字列がContext Cueから除外されcache診断だけに保持されること。

    Arrange:
        - 短く低log probabilityのSTT結果を返すVideo Sourceが用意される
    Act:
        - Context Collection Stageが実行され同じsemantic入力で再実行される
    Assert:
        - 文字列が低信頼outcomeと非公開diagnosticへ隔離されること
        - 2回目はSpeechRuntimeを呼ばずcacheから同じdiagnosticが復元されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    pcm = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=16000,
        sample_rate=16000,
        channel_count=1,
        sample_format="s16le",
        pts=160000,
        time_base=Fraction(1, 16000),
        pcm_bytes=b"\x00\x00" * 16000,
    )
    first_media_runtime = FakeVideoStageMediaRuntime(
        media_probe=probe,
        pcm_audio_chunks=(pcm,),
    )
    first_speech_runtime = FakeSpeechRuntime(
        (
            SpeechRecognitionResult(
                vad_speech_detected=True,
                segments=(
                    SpeechSegment(
                        words=(SpeechWord("ん", 8000, 12000, 0.2),),
                        average_log_probability=-0.98,
                        no_speech_probability=0.4,
                    ),
                ),
            ),
        )
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        language="ja",
    )

    # Act
    first_result = ContextStageProcessor(
        first_media_runtime,
        first_speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=configuration,
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )
    cached_speech_runtime = FakeSpeechRuntime()
    cached_result = ContextStageProcessor(
        FakeVideoStageMediaRuntime(media_probe=probe),
        cached_speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=configuration,
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )

    # Assert
    assert first_result.cues == ()
    assert [(item.status, item.reason_code) for item in first_result.outcomes] == [
        ("absent", "no_subtitle_stream"),
        ("low_reliability", "asr_below_policy_threshold"),
    ]
    assert len(first_result.rejected_speech_diagnostics) == 1
    diagnostic = first_result.rejected_speech_diagnostics[0]
    assert diagnostic.stream_index == 1
    assert diagnostic.start == Fraction(1, 2)
    assert diagnostic.end == Fraction(3, 4)
    assert diagnostic.text == "ん"
    assert diagnostic.average_log_probability == -0.98
    assert diagnostic.no_speech_probability == 0.4
    assert diagnostic.word_probabilities == (0.2,)
    assert diagnostic.provenance is not None
    assert diagnostic.provenance.codec_name == "pcm_s16le"
    assert diagnostic.provenance.chunk_sample_start == 0
    assert diagnostic.provenance.chunk_sample_end == 16000
    assert diagnostic.provenance.speech_runtime_identity == "fake-speech-runtime-v1"
    assert diagnostic.provenance.resolved_model_identity == "hf:" + "0" * 40
    assert cached_result.rejected_speech_diagnostics == (
        first_result.rejected_speech_diagnostics
    )
    assert cached_speech_runtime.transcribe_calls == []


def test_forced_subtitle_and_stt_duplicate_are_one_annotation_input(
    tmp_path: Path,
) -> None:
    """forced subtitleとSTTの重複がprovenanceを保った一つの入力にされること。

    Arrange:
        - 同じ時刻と本文を持つforced subtitleとaudio STTが用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - 両方のCueがcache結果に保持されequivalence groupへ関連付けられること
        - source PTSを持つsubtitleだけがannotation入力になること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    forced_subtitle = _stream(
        2,
        "subtitle",
        "subrip",
        language="jpn",
        is_default=True,
        is_forced=True,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(
            _stream(0, "video", "ffv1", is_default=True),
            audio_stream,
            forced_subtitle,
        ),
    )
    pcm = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=48000,
        sample_rate=16000,
        channel_count=1,
        sample_format="s16le",
        pts=160000,
        time_base=Fraction(1, 16000),
        pcm_bytes=b"\x00\x00" * 48000,
    )
    media_runtime = FakeVideoStageMediaRuntime(
        media_probe=probe,
        embedded_subtitles=(
            EmbeddedSubtitle(
                stream_index=2,
                pts=110,
                duration_ts=10,
                time_base=Fraction(1, 10),
                text="助けて！",
            ),
        ),
        pcm_audio_chunks=(pcm,),
    )
    speech_runtime = FakeSpeechRuntime(
        (
            SpeechRecognitionResult(
                vad_speech_detected=True,
                segments=(
                    SpeechSegment(
                        words=(SpeechWord(" 助けて ", 16000, 32000, 0.98),),
                        average_log_probability=-0.2,
                    ),
                ),
            ),
        )
    )

    # Act
    result = ContextStageProcessor(
        media_runtime,
        speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=tmp_path / "output",
            language="ja",
        ),
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )

    # Assert
    assert [cue.source_kind for cue in result.cues] == [
        "embedded_subtitle",
        "speech_to_text",
    ]
    assert len(result.equivalence_groups) == 1
    group = result.equivalence_groups[0]
    assert group.representative_cue_id == result.cues[0].identifier
    assert group.cue_ids == tuple(cue.identifier for cue in result.cues)
    assert result.annotation_cues == (result.cues[0],)


def test_partial_stt_failure_is_fatal_without_publishing_context(
    tmp_path: Path,
) -> None:
    """一部chunkのSTT失敗で成功済みCueもCompleted Stageへ公開されないこと。

    Arrange:
        - 2つのPCM chunkと2件目で秘密の文字列を含むerrorが用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - reason-codedなSTT failureが返されraw文字列がerrorへ含まれないこと
        - 部分的なContext artifactとcompletion manifestが公開されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    chunks = tuple(
        PcmAudioChunk(
            stream_index=1,
            sample_start=sample_start,
            sample_count=16000,
            sample_rate=16000,
            channel_count=1,
            sample_format="s16le",
            pts=160000 + sample_start,
            time_base=Fraction(1, 16000),
            pcm_bytes=b"\x00\x00" * 16000,
        )
        for sample_start in (0, 16000)
    )
    media_runtime = FakeVideoStageMediaRuntime(
        media_probe=probe,
        pcm_audio_chunks=chunks,
    )
    speech_runtime = FakeSpeechRuntime(
        (
            SpeechRecognitionResult(
                vad_speech_detected=True,
                segments=(
                    SpeechSegment(
                        words=(SpeechWord("最初の台詞", 0, 8000, 0.9),),
                        average_log_probability=-0.2,
                    ),
                ),
            ),
        ),
        error_on_call=1,
        error_message="秘密の台詞を解析できませんでした",
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        language="ja",
        speech_chunk_seconds=1.0,
        speech_overlap_seconds=0.0,
    )

    # Act
    with pytest.raises(ContextStageError) as raised:
        ContextStageProcessor(
            media_runtime,
            speech_runtime,
            RecordingRunObserver(),
        ).process(
            video_set=video_set,
            source=source,
            probe=probe,
            scan=_scan(),
            configuration=configuration,
            media_runtime_identity=MediaRuntimeIdentity(
                "6.1.1-test",
                "6.1.1-test",
                "0" * 64,
            ),
        )

    # Assert
    assert raised.value.reason is ContextStageFailureReason.CHUNK_FAILED
    assert "秘密の台詞" not in str(raised.value)
    assert "秘密の台詞" not in "".join(traceback.format_exception(raised.value))
    context_root = (
        configuration.processing_cache_folder
        / "videos"
        / source.fingerprint
        / "collect-context"
    )
    assert not tuple(context_root.rglob("manifest.json"))


def test_completed_stt_chunk_survives_later_chunk_failure(
    tmp_path: Path,
) -> None:
    """後続STT chunk失敗後も完了済みchunkが再利用されること。

    Arrange:
        - 2 chunkの2件目だけ初回に失敗するSpeech Runtimeが用意される
    Act:
        - 初回失敗後に同じContext Collectionが再実行される
    Assert:
        - retryでは2件目だけがSpeech Runtimeへ渡されること
        - 中断なしと同じCue ID、本文、時刻が生成されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    chunks = tuple(
        PcmAudioChunk(
            stream_index=1,
            sample_start=sample_start,
            sample_count=16000,
            sample_rate=16000,
            channel_count=1,
            sample_format="s16le",
            pts=160000 + sample_start,
            time_base=Fraction(1, 16000),
            pcm_bytes=b"\x00\x00" * 16000,
        )
        for sample_start in (0, 16000)
    )
    first_recognition = SpeechRecognitionResult(
        vad_speech_detected=True,
        segments=(
            SpeechSegment(
                words=(SpeechWord("最初の台詞", 0, 8000, 0.9),),
                average_log_probability=-0.2,
            ),
        ),
    )
    second_recognition = SpeechRecognitionResult(
        vad_speech_detected=True,
        segments=(
            SpeechSegment(
                words=(SpeechWord("次の台詞", 0, 8000, 0.9),),
                average_log_probability=-0.2,
            ),
        ),
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        language="ja",
        speech_chunk_seconds=1.0,
        speech_overlap_seconds=0.0,
    )
    runtime_identity = MediaRuntimeIdentity(
        "6.1.1-test",
        "6.1.1-test",
        "0" * 64,
    )
    with pytest.raises(ContextStageError):
        ContextStageProcessor(
            FakeVideoStageMediaRuntime(
                media_probe=probe,
                pcm_audio_chunks=chunks,
            ),
            FakeSpeechRuntime(
                (first_recognition,),
                error_on_call=1,
            ),
            RecordingRunObserver(),
        ).process(
            video_set=video_set,
            source=source,
            probe=probe,
            scan=_scan(),
            configuration=configuration,
            media_runtime_identity=runtime_identity,
        )
    retry_speech_runtime = FakeSpeechRuntime((second_recognition,))
    retry_media_runtime = FakeVideoStageMediaRuntime(
        media_probe=probe,
        pcm_audio_chunks=chunks,
    )

    # Act
    resumed = ContextStageProcessor(
        retry_media_runtime,
        retry_speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=configuration,
        media_runtime_identity=runtime_identity,
    )

    # Assert
    assert [
        (chunk.sample_start, chunk.sample_count)
        for chunk in retry_speech_runtime.transcribe_calls
    ] == [(16000, 16000)]
    assert retry_media_runtime.audio_chunk_calls == []
    assert [(cue.identifier, cue.text, cue.start, cue.end) for cue in resumed.cues] == [
        (
            resumed.cues[0].identifier,
            "最初の台詞",
            Fraction(0),
            Fraction(1, 2),
        ),
        (
            resumed.cues[1].identifier,
            "次の台詞",
            Fraction(1),
            Fraction(3, 2),
        ),
    ]
    assert resumed.outcomes[-1].processed_chunk_count == 2


def test_first_stt_chunk_failure_is_stt_analysis_failed(tmp_path: Path) -> None:
    """最初のSTT chunk失敗がpartial failureと区別されること。

    Arrange:
        - 一つのPCM chunkと初回に失敗するSpeech Runtimeが用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - stt_analysis_failedのstable reasonで失敗されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    pcm = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=16000,
        sample_rate=16000,
        channel_count=1,
        sample_format="s16le",
        pts=160000,
        time_base=Fraction(1, 16000),
        pcm_bytes=b"\x00\x00" * 16000,
    )

    # Act
    with pytest.raises(ContextStageError) as raised:
        ContextStageProcessor(
            FakeVideoStageMediaRuntime(
                media_probe=probe,
                pcm_audio_chunks=(pcm,),
            ),
            FakeSpeechRuntime(error_on_call=0),
            RecordingRunObserver(),
        ).process(
            video_set=video_set,
            source=source,
            probe=probe,
            scan=_scan(),
            configuration=EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=tmp_path / "output",
                language="ja",
            ),
            media_runtime_identity=MediaRuntimeIdentity(
                "6.1.1-test",
                "6.1.1-test",
                "0" * 64,
            ),
        )

    # Assert
    assert raised.value.reason is ContextStageFailureReason.STT_ANALYSIS_FAILED


def test_audio_decoder_failure_is_fatal_without_fallback(tmp_path: Path) -> None:
    """選択audioのdecode失敗が空contextへ変換されずfatalになること。

    Arrange:
        - subtitleがなく選択audioのdecodeで失敗するMediaRuntimeが用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - audio extractionのreason-coded failureがそのまま返されること
        - Context Stage completion manifestが公開されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        language="ja",
    )
    media_runtime = FakeVideoStageMediaRuntime(
        media_probe=probe,
        audio_error=MediaRuntimeError(
            MediaRuntimeFailureReason.AUDIO_EXTRACTION_FAILED,
            "audio streamをPCMへdecodeできませんでした",
        ),
    )

    # Act
    with pytest.raises(MediaRuntimeError) as raised:
        ContextStageProcessor(
            media_runtime,
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(
            video_set=video_set,
            source=source,
            probe=probe,
            scan=_scan(),
            configuration=configuration,
            media_runtime_identity=MediaRuntimeIdentity(
                "6.1.1-test",
                "6.1.1-test",
                "0" * 64,
            ),
        )

    # Assert
    assert raised.value.reason is MediaRuntimeFailureReason.AUDIO_EXTRACTION_FAILED
    context_root = (
        configuration.processing_cache_folder
        / "videos"
        / source.fingerprint
        / "collect-context"
    )
    assert not tuple(context_root.rglob("manifest.json"))


def test_ambiguous_subtitle_stream_is_fatal_without_audio_fallback(
    tmp_path: Path,
) -> None:
    """同順位subtitleが複数ある場合にindex順で推測されずfatalになること。

    Arrange:
        - 同じ言語・dispositionのtext subtitle 2本とaudioが用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - ambiguous subtitleのstable reasonで失敗されること
        - audio STTがfallbackとして実行されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(
            _stream(0, "video", "ffv1", is_default=True),
            _stream(
                1,
                "audio",
                "pcm_s16le",
                language="jpn",
                is_default=True,
                start_pts=100,
            ),
            _stream(2, "subtitle", "subrip", language="jpn"),
            _stream(3, "subtitle", "ass", language="jpn"),
        ),
    )
    speech_runtime = FakeSpeechRuntime()

    # Act
    with pytest.raises(ContextStageError) as raised:
        ContextStageProcessor(
            FakeVideoStageMediaRuntime(media_probe=probe),
            speech_runtime,
            RecordingRunObserver(),
        ).process(
            video_set=video_set,
            source=source,
            probe=probe,
            scan=_scan(),
            configuration=EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=tmp_path / "output",
                language="ja",
            ),
            media_runtime_identity=MediaRuntimeIdentity(
                "6.1.1-test",
                "6.1.1-test",
                "0" * 64,
            ),
        )

    # Assert
    assert raised.value.reason is ContextStageFailureReason.AMBIGUOUS_SUBTITLE_STREAM
    assert speech_runtime.transcribe_calls == []


def test_selected_bitmap_subtitle_is_fatal_without_audio_fallback(
    tmp_path: Path,
) -> None:
    """選択対象のbitmap subtitleが字幕不在へ畳み込まれずfatalになること。

    Arrange:
        - defaultのbitmap subtitleと利用可能なaudioが用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - unsupported bitmap subtitleのstable reasonで失敗されること
        - audio STTが暗黙fallbackとして実行されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(
            _stream(0, "video", "ffv1", is_default=True),
            _stream(
                1,
                "audio",
                "pcm_s16le",
                language="jpn",
                is_default=True,
                start_pts=100,
            ),
            _stream(
                2,
                "subtitle",
                "hdmv_pgs_subtitle",
                language="jpn",
                is_default=True,
            ),
        ),
    )
    speech_runtime = FakeSpeechRuntime()

    # Act
    with pytest.raises(ContextStageError) as raised:
        ContextStageProcessor(
            FakeVideoStageMediaRuntime(media_probe=probe),
            speech_runtime,
            RecordingRunObserver(),
        ).process(
            video_set=video_set,
            source=source,
            probe=probe,
            scan=_scan(),
            configuration=EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=tmp_path / "output",
                language="ja",
            ),
            media_runtime_identity=MediaRuntimeIdentity(
                "6.1.1-test",
                "6.1.1-test",
                "0" * 64,
            ),
        )

    # Assert
    assert raised.value.reason is ContextStageFailureReason.UNSUPPORTED_BITMAP_SUBTITLE
    assert speech_runtime.transcribe_calls == []


def test_no_context_streams_are_normal_and_ignore_unused_speech_identity(
    tmp_path: Path,
) -> None:
    """subtitle・audio不在が正常結果となり未使用STT identityに依存しないこと。

    Arrange:
        - video streamだけを持つVideo Sourceが用意される
    Act:
        - 異なるSpeech Runtime・model identityでContext Stageが2回実行される
    Assert:
        - subtitleとaudioの不在が別outcomeとして返されること
        - STTが実行されず同じStage Fingerprintが再利用されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True),),
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        language="ja",
    )
    first_speech_runtime = FakeSpeechRuntime(
        runtime_identity="speech-runtime-a",
        resolved_model_identity="hf:" + "a" * 40,
    )
    second_speech_runtime = FakeSpeechRuntime(
        runtime_identity="speech-runtime-b",
        resolved_model_identity="hf:" + "b" * 40,
    )

    # Act
    first_result = ContextStageProcessor(
        FakeVideoStageMediaRuntime(media_probe=probe),
        first_speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=configuration,
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )
    second_result = ContextStageProcessor(
        FakeVideoStageMediaRuntime(media_probe=probe),
        second_speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=configuration,
        media_runtime_identity=MediaRuntimeIdentity(
            "8.0-test",
            "8.0-test",
            "9" * 64,
        ),
    )

    # Assert
    assert first_result.cues == ()
    assert [(item.status, item.reason_code) for item in first_result.outcomes] == [
        ("absent", "no_subtitle_stream"),
        ("absent", "no_audio_stream"),
    ]
    assert first_result.completed_stage == second_result.completed_stage
    assert first_speech_runtime.transcribe_calls == []
    assert second_speech_runtime.transcribe_calls == []


def test_resolved_model_identity_change_invalidates_stt_context_cache(
    tmp_path: Path,
) -> None:
    """実行時に解決されたmodel identity変更でSTT cacheが再計算されること。

    Arrange:
        - 同じaudioと異なるResolved Model IdentityのSpeech Runtimeが用意される
    Act:
        - 同じ設定でContext Collection Stageが2回実行される
    Assert:
        - 2回目もSTTが実行され異なるStage Fingerprintが確定されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    pcm = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=16000,
        sample_rate=16000,
        channel_count=1,
        sample_format="s16le",
        pts=160000,
        time_base=Fraction(1, 16000),
        pcm_bytes=b"\x00\x00" * 16000,
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        language="ja",
    )
    first_speech_runtime = FakeSpeechRuntime(
        resolved_model_identity="hf:" + "a" * 40,
    )
    second_speech_runtime = FakeSpeechRuntime(
        resolved_model_identity="hf:" + "b" * 40,
    )

    # Act
    first_result = ContextStageProcessor(
        FakeVideoStageMediaRuntime(media_probe=probe, pcm_audio_chunks=(pcm,)),
        first_speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=configuration,
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )
    second_result = ContextStageProcessor(
        FakeVideoStageMediaRuntime(media_probe=probe, pcm_audio_chunks=(pcm,)),
        second_speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=configuration,
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )

    # Assert
    assert first_speech_runtime.transcribe_calls == [pcm]
    assert second_speech_runtime.transcribe_calls == [pcm]
    assert first_result.completed_stage is not None
    assert second_result.completed_stage is not None
    assert (
        first_result.completed_stage.fingerprint
        != second_result.completed_stage.fingerprint
    )
    assert second_result.completed_stage.semantic_input["checkpoint_contracts"] == {
        "pcm_audio_chunk": checkpoint_version(CheckpointOperation.PCM_AUDIO_CHUNK),
        "speech_recognition_chunk": checkpoint_version(
            CheckpointOperation.SPEECH_RECOGNITION_CHUNK
        ),
    }


def test_stt_cache_uses_resolved_identity_instead_of_configured_model_name(
    tmp_path: Path,
) -> None:
    """同じResolved Model Identityならmodel alias変更後も再利用されること。

    Arrange:
        - 同じaudioとResolved Model Identityを持つSpeech Runtimeが用意される
        - 異なる設定上のSTT model名が用意される
    Act:
        - Context Collection Stageが各model名で実行される
    Assert:
        - 2回目はSTTを実行せず同じCompleted Stageが再利用されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    pcm = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=16000,
        sample_rate=16000,
        channel_count=1,
        sample_format="s16le",
        pts=160000,
        time_base=Fraction(1, 16000),
        pcm_bytes=b"\x00\x00" * 16000,
    )
    resolved_identity = "hf:" + "a" * 40
    first_speech_runtime = FakeSpeechRuntime(
        resolved_model_identity=resolved_identity,
    )
    second_speech_runtime = FakeSpeechRuntime(
        resolved_model_identity=resolved_identity,
    )

    # Act
    first_result = ContextStageProcessor(
        FakeVideoStageMediaRuntime(media_probe=probe, pcm_audio_chunks=(pcm,)),
        first_speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=tmp_path / "output",
            language="ja",
            speech_to_text_model="organization/model-alias-a",
        ),
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )
    second_result = ContextStageProcessor(
        FakeVideoStageMediaRuntime(media_probe=probe, pcm_audio_chunks=(pcm,)),
        second_speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=tmp_path / "output",
            language="ja",
            speech_to_text_model="organization/model-alias-b",
        ),
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )

    # Assert
    assert first_speech_runtime.transcribe_calls == [pcm]
    assert second_speech_runtime.transcribe_calls == []
    assert first_result.completed_stage == second_result.completed_stage


def test_pcm_chunks_overlap_and_midpoint_ownership_removes_duplicate_cues(
    tmp_path: Path,
) -> None:
    """PCM chunkがoverlapされ中央の半開所有境界でCue重複が除かれること。

    Arrange:
        - 2秒ずつ連続する3つのPCM source chunkが用意される
        - 4秒chunk・2秒overlapの双方から同じ時刻のwordが返される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - SpeechRuntimeへ0〜4秒と2〜6秒のoverlap chunkが渡されること
        - overlap中央にmidpointを持つCueが後側chunkにだけ所有されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    source_chunks = tuple(
        PcmAudioChunk(
            stream_index=1,
            sample_start=sample_start,
            sample_count=32000,
            sample_rate=16000,
            channel_count=1,
            sample_format="s16le",
            pts=160000 + sample_start,
            time_base=Fraction(1, 16000),
            pcm_bytes=bytes([index, 0]) * 32000,
        )
        for index, sample_start in enumerate((0, 32000, 64000), start=1)
    )
    speech_runtime = FakeSpeechRuntime(
        (
            SpeechRecognitionResult(
                vad_speech_detected=True,
                segments=(
                    SpeechSegment(
                        words=(SpeechWord("重複台詞", 40000, 56000, 0.9),),
                        average_log_probability=-0.2,
                    ),
                ),
            ),
            SpeechRecognitionResult(
                vad_speech_detected=True,
                segments=(
                    SpeechSegment(
                        words=(SpeechWord("重複台詞", 8000, 24000, 0.9),),
                        average_log_probability=-0.2,
                    ),
                ),
            ),
        )
    )

    # Act
    result = ContextStageProcessor(
        FakeVideoStageMediaRuntime(
            media_probe=probe,
            pcm_audio_chunks=source_chunks,
        ),
        speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=tmp_path / "output",
            language="ja",
            speech_chunk_seconds=4.0,
            speech_overlap_seconds=2.0,
        ),
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )

    # Assert
    assert [
        (chunk.sample_start, chunk.sample_count)
        for chunk in speech_runtime.transcribe_calls
    ] == [(0, 64000), (32000, 64000)]
    assert len(result.cues) == 1
    assert result.cues[0].start == Fraction(5, 2)
    assert result.cues[0].end == Fraction(7, 2)


def test_pcm_overlap_can_span_multiple_source_chunks(tmp_path: Path) -> None:
    """大きいoverlapが複数source chunkを跨いで指定windowにされること。

    Arrange:
        - 1秒ずつ連続する6つのPCM source chunkが用意される
        - 4秒chunk・3秒overlapが設定される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - SpeechRuntimeへ0〜4秒、1〜5秒、2〜6秒のwindowが渡されること
        - 前window内に完全包含される末尾windowが追加されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    source_chunks = tuple(
        PcmAudioChunk(
            stream_index=1,
            sample_start=sample_start,
            sample_count=16000,
            sample_rate=16000,
            channel_count=1,
            sample_format="s16le",
            pts=160000 + sample_start,
            time_base=Fraction(1, 16000),
            pcm_bytes=bytes([index, 0]) * 16000,
        )
        for index, sample_start in enumerate(
            range(0, 96000, 16000),
            start=1,
        )
    )
    speech_runtime = FakeSpeechRuntime()

    # Act
    ContextStageProcessor(
        FakeVideoStageMediaRuntime(
            media_probe=probe,
            pcm_audio_chunks=source_chunks,
        ),
        speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=tmp_path / "output",
            language="ja",
            speech_chunk_seconds=4.0,
            speech_overlap_seconds=3.0,
        ),
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )

    # Assert
    assert [
        (chunk.sample_start, chunk.sample_count)
        for chunk in speech_runtime.transcribe_calls
    ] == [(0, 64000), (16000, 64000), (32000, 64000)]


def test_missing_explicit_subtitle_stream_is_fatal(tmp_path: Path) -> None:
    """存在しない明示subtitle indexがtrack不在へ変換されずfatalになること。

    Arrange:
        - subtitleを持たないVideo Sourceと明示subtitle indexが用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - invalid subtitle streamのstable reasonで失敗されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True),),
    )

    # Act
    with pytest.raises(ContextStageError) as raised:
        ContextStageProcessor(
            FakeVideoStageMediaRuntime(media_probe=probe),
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(
            video_set=video_set,
            source=source,
            probe=probe,
            scan=_scan(),
            configuration=EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=tmp_path / "output",
                language="ja",
                subtitle_stream_index=99,
            ),
            media_runtime_identity=MediaRuntimeIdentity(
                "6.1.1-test",
                "6.1.1-test",
                "0" * 64,
            ),
        )

    # Assert
    assert raised.value.reason is ContextStageFailureReason.INVALID_SUBTITLE_STREAM


def test_ambiguous_audio_stream_is_fatal(tmp_path: Path) -> None:
    """同順位audioが複数ある場合にindex順で推測されずfatalになること。

    Arrange:
        - subtitleがなく同じ言語・dispositionのaudio 2本が用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - ambiguous audio streamのstable reasonで失敗されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(
            _stream(0, "video", "ffv1", is_default=True),
            _stream(1, "audio", "aac", language="jpn", start_pts=100),
            _stream(2, "audio", "aac", language="jpn", start_pts=100),
        ),
    )

    # Act
    with pytest.raises(ContextStageError) as raised:
        ContextStageProcessor(
            FakeVideoStageMediaRuntime(media_probe=probe),
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(
            video_set=video_set,
            source=source,
            probe=probe,
            scan=_scan(),
            configuration=EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=tmp_path / "output",
                language="ja",
            ),
            media_runtime_identity=MediaRuntimeIdentity(
                "6.1.1-test",
                "6.1.1-test",
                "0" * 64,
            ),
        )

    # Assert
    assert raised.value.reason is ContextStageFailureReason.AMBIGUOUS_AUDIO_STREAM


def test_missing_explicit_audio_stream_is_fatal(tmp_path: Path) -> None:
    """存在しない明示audio indexがtrack不在へ変換されずfatalになること。

    Arrange:
        - subtitle・audioを持たないVideo Sourceと明示audio indexが用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - invalid audio streamのstable reasonで失敗されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True),),
    )

    # Act
    with pytest.raises(ContextStageError) as raised:
        ContextStageProcessor(
            FakeVideoStageMediaRuntime(media_probe=probe),
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(
            video_set=video_set,
            source=source,
            probe=probe,
            scan=_scan(),
            configuration=EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=tmp_path / "output",
                language="ja",
                audio_stream_index=99,
            ),
            media_runtime_identity=MediaRuntimeIdentity(
                "6.1.1-test",
                "6.1.1-test",
                "0" * 64,
            ),
        )

    # Assert
    assert raised.value.reason is ContextStageFailureReason.INVALID_AUDIO_STREAM


def test_out_of_range_speech_timestamp_is_fatal(tmp_path: Path) -> None:
    """PCM範囲外のASR timestampがtimestamp driftになること。

    Arrange:
        - 1秒PCMに対して範囲外のwordが用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - timestamp driftのstable reasonで失敗されること
        - Context Cueが部分公開されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    pcm = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=16000,
        sample_rate=16000,
        channel_count=1,
        sample_format="s16le",
        pts=160000,
        time_base=Fraction(1, 16000),
        pcm_bytes=b"\x00\x00" * 16000,
    )
    speech_runtime = FakeSpeechRuntime(
        (
            SpeechRecognitionResult(
                vad_speech_detected=True,
                segments=(
                    SpeechSegment(
                        words=(SpeechWord("範囲外台詞", 8000, 20000, 0.9),),
                        average_log_probability=-0.2,
                    ),
                ),
            ),
        )
    )

    # Act
    with pytest.raises(ContextStageError) as raised:
        ContextStageProcessor(
            FakeVideoStageMediaRuntime(
                media_probe=probe,
                pcm_audio_chunks=(pcm,),
            ),
            speech_runtime,
            RecordingRunObserver(),
        ).process(
            video_set=video_set,
            source=source,
            probe=probe,
            scan=_scan(),
            configuration=EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=tmp_path / "output",
                language="ja",
            ),
            media_runtime_identity=MediaRuntimeIdentity(
                "6.1.1-test",
                "6.1.1-test",
                "0" * 64,
            ),
        )

    # Assert
    assert raised.value.reason is ContextStageFailureReason.TIMESTAMP_DRIFT


def test_zero_duration_speech_is_rejected_without_guessing_interval(
    tmp_path: Path,
) -> None:
    """時間幅0のASR groupが時刻を補間せず低信頼へ隔離されること。

    Arrange:
        - 1秒PCMと同一点のsample境界だけを持つword groupが用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - Context Cueへ採用されず幅0のprivate診断として記録されること
        - source outcomeがasr_zero_durationの非fatal結果になること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    pcm = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=16000,
        sample_rate=16000,
        channel_count=1,
        sample_format="s16le",
        pts=160000,
        time_base=Fraction(1, 16000),
        pcm_bytes=b"\x00\x00" * 16000,
    )
    speech_runtime = FakeSpeechRuntime(
        (
            SpeechRecognitionResult(
                vad_speech_detected=True,
                segments=(
                    SpeechSegment(
                        words=(SpeechWord("時刻なし台詞", 8000, 8000, 0.9),),
                        average_log_probability=-0.2,
                    ),
                ),
            ),
        )
    )

    # Act
    result = ContextStageProcessor(
        FakeVideoStageMediaRuntime(
            media_probe=probe,
            pcm_audio_chunks=(pcm,),
        ),
        speech_runtime,
        RecordingRunObserver(),
    ).process(
        video_set=video_set,
        source=source,
        probe=probe,
        scan=_scan(),
        configuration=EffectiveConfiguration(
            video_input_folder=input_folder,
            output_folder=tmp_path / "output",
            language="ja",
        ),
        media_runtime_identity=MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        ),
    )

    # Assert
    assert result.cues == ()
    assert [(item.status, item.reason_code) for item in result.outcomes] == [
        ("absent", "no_subtitle_stream"),
        ("low_reliability", "asr_zero_duration"),
    ]
    assert len(result.rejected_speech_diagnostics) == 1
    diagnostic = result.rejected_speech_diagnostics[0]
    assert diagnostic.start == Fraction(1, 2)
    assert diagnostic.end == Fraction(1, 2)
    assert diagnostic.reason_code == "asr_zero_duration"
    assert diagnostic.text == "時刻なし台詞"


def test_pcm_origin_drift_is_fatal_without_publishing_context(
    tmp_path: Path,
) -> None:
    """観測PCM originがaudio stream originからずれた場合にfatalになること。

    Arrange:
        - 宣言されたaudio originから2 sampleずれた先頭PCMが用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - timestamp driftのstable reasonで失敗されること
        - Context Stage completion manifestが公開されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    pcm = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=16000,
        sample_rate=16000,
        channel_count=1,
        sample_format="s16le",
        pts=160002,
        time_base=Fraction(1, 16000),
        pcm_bytes=b"\x00\x00" * 16000,
    )
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        language="ja",
    )

    # Act
    with pytest.raises(ContextStageError) as raised:
        ContextStageProcessor(
            FakeVideoStageMediaRuntime(
                media_probe=probe,
                pcm_audio_chunks=(pcm,),
            ),
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(
            video_set=video_set,
            source=source,
            probe=probe,
            scan=_scan(),
            configuration=configuration,
            media_runtime_identity=MediaRuntimeIdentity(
                "6.1.1-test",
                "6.1.1-test",
                "0" * 64,
            ),
        )

    # Assert
    assert raised.value.reason is ContextStageFailureReason.TIMESTAMP_DRIFT
    context_root = (
        configuration.processing_cache_folder
        / "videos"
        / source.fingerprint
        / "collect-context"
    )
    assert not tuple(context_root.rglob("manifest.json"))


def test_discontinuous_pcm_grid_is_reported_as_timestamp_drift(
    tmp_path: Path,
) -> None:
    """PCM sample gridのgapが内部ValueErrorではなくfatal reasonにされること。

    Arrange:
        - 先頭chunkの終端と次chunkの開始に1 sampleのgapが用意される
    Act:
        - Context Collection Stageが実行される
    Assert:
        - timestamp driftのstable reasonで失敗されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "video.mkv").write_bytes(b"video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    audio_stream = _stream(
        1,
        "audio",
        "pcm_s16le",
        language="jpn",
        is_default=True,
        start_pts=100,
    )
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_stream(0, "video", "ffv1", is_default=True), audio_stream),
    )
    chunks = tuple(
        PcmAudioChunk(
            stream_index=1,
            sample_start=sample_start,
            sample_count=16000,
            sample_rate=16000,
            channel_count=1,
            sample_format="s16le",
            pts=160000 + sample_start,
            time_base=Fraction(1, 16000),
            pcm_bytes=b"\x00\x00" * 16000,
        )
        for sample_start in (0, 16001)
    )

    # Act
    with pytest.raises(ContextStageError) as raised:
        ContextStageProcessor(
            FakeVideoStageMediaRuntime(
                media_probe=probe,
                pcm_audio_chunks=chunks,
            ),
            FakeSpeechRuntime(),
            RecordingRunObserver(),
        ).process(
            video_set=video_set,
            source=source,
            probe=probe,
            scan=_scan(),
            configuration=EffectiveConfiguration(
                video_input_folder=input_folder,
                output_folder=tmp_path / "output",
                language="ja",
                speech_chunk_seconds=1.0,
                speech_overlap_seconds=0.0,
            ),
            media_runtime_identity=MediaRuntimeIdentity(
                "6.1.1-test",
                "6.1.1-test",
                "0" * 64,
            ),
        )

    # Assert
    assert raised.value.reason is ContextStageFailureReason.TIMESTAMP_DRIFT


def _stream(
    index: int,
    kind: MediaStreamKind,
    codec_name: str,
    *,
    language: str | None = None,
    is_default: bool = False,
    is_forced: bool = False,
    start_pts: int = 0,
) -> MediaStream:
    return MediaStream(
        index=index,
        kind=kind,
        codec_name=codec_name,
        time_base=Fraction(1, 10),
        start_pts=start_pts,
        duration_ts=200,
        width=64 if kind == "video" else None,
        height=48 if kind == "video" else None,
        sample_rate=16000 if kind == "audio" else None,
        channels=1 if kind == "audio" else None,
        language=language,
        is_default=is_default,
        is_forced=is_forced,
    )


def _scan() -> VideoScanResult:
    duration = Fraction(10)
    return VideoScanResult(
        primary_stream=_stream(0, "video", "ffv1", is_default=True),
        timeline=VideoTimeline(
            origin_pts=100,
            time_base=Fraction(1, 10),
            duration=VideoDuration(duration),
            segments=(TimelineSegment("seg_" + "0" * 64, Fraction(0), duration),),
        ),
        heartbeats=(),
        scene_signals=(),
        metrics=VideoScanMetrics(
            input_duration=duration,
            wall_seconds=0.1,
            cpu_seconds=0.05,
            input_seconds_per_wall_second=100.0,
            decode_backend="cpu",
            decode_pass_count=1,
            heartbeat_count=0,
            heartbeat_bytes=0,
            heartbeat_max_gap_seconds=0.0,
            heartbeat_p95_gap_seconds=0.0,
            scene_signal_count=0,
            timeline_segment_count=1,
        ),
    )
