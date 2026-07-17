"""faster-whisper SpeechRuntime adapterのcontract test。"""

from fractions import Fraction
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import ctranslate2
import numpy as np
import pytest

from src.video_selection.models.pcm_audio_chunk import PcmAudioChunk
from src.video_selection.services.gpu_work_coordinator import GpuWorkCoordinator
from src.video_selection.speech.faster_whisper_speech_runtime import (
    FasterWhisperSpeechRuntime,
)
from tests.video_selection.fakes.fake_faster_whisper_model import (
    FakeFasterWhisperModel,
)


def test_pcm_is_transcribed_to_integer_sample_word_timestamps() -> None:
    """s16le PCMがbackend非依存の整数sample位置へ変換されること。

    Arrange:
        - 既知sampleを持つPCM chunkとfaster-whisper model fakeが用意される
    Act:
        - SpeechRuntimeでword timestamp付きtranscriptionが実行される
    Assert:
        - PCMが正規化waveformとして渡されること
        - 必須optionと整数sample位置付き結果が返されること
    """
    # Arrange
    segment = SimpleNamespace(
        words=(SimpleNamespace(word=" 冒険", start=0.5, end=1.0, probability=0.9),),
        avg_logprob=-0.4,
        no_speech_prob=0.1,
    )
    info = SimpleNamespace(language="ja", duration_after_vad=1.0)
    model = FakeFasterWhisperModel((segment,), info)
    runtime = FasterWhisperSpeechRuntime(
        model,
        runtime_identity="speech-runtime:test",
        resolved_model_identity="hf:" + "a" * 40,
    )
    pcm = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=16000,
        sample_rate=16000,
        channel_count=1,
        sample_format="s16le",
        pts=0,
        time_base=Fraction(1, 16000),
        pcm_bytes=(-32768).to_bytes(2, "little", signed=True)
        + (0).to_bytes(2, "little", signed=True)
        + (32767).to_bytes(2, "little", signed=True)
        + b"\x00\x00" * 15997,
    )

    # Act
    result = runtime.transcribe(
        pcm,
        language="ja",
        vad_filter=True,
        beam_size=5,
    )

    # Assert
    assert model.audio is not None
    assert model.audio.shape == (16000,)
    np.testing.assert_allclose(
        model.audio[:3],
        np.array([-1.0, 0.0, 32767 / 32768], dtype=np.float32),
    )
    assert model.options == {
        "language": "ja",
        "vad_filter": True,
        "beam_size": 5,
        "word_timestamps": True,
        "condition_on_previous_text": False,
    }
    assert result.vad_speech_detected is True
    assert result.detected_language == "ja"
    assert len(result.segments) == 1
    output_segment = result.segments[0]
    assert output_segment.average_log_probability == -0.4
    assert output_segment.no_speech_probability == 0.1
    assert output_segment.words[0].text == " 冒険"
    assert output_segment.words[0].start_sample == 8000
    assert output_segment.words[0].end_sample == 16000
    assert output_segment.words[0].probability == 0.9


def test_zero_duration_backend_token_is_preserved_as_sample_boundary() -> None:
    """量子化で同一点になったword tokenのsample境界が保持されること。

    Arrange:
        - startとendが同じfaster-whisper word tokenが用意される
    Act:
        - SpeechRuntimeでword timestamp付きtranscriptionが実行される
    Assert:
        - tokenが推測で延長されず同じ整数sample境界として返されること
    """
    # Arrange
    segment = SimpleNamespace(
        words=(SimpleNamespace(word="者", start=0.5, end=0.5, probability=0.9),),
        avg_logprob=-0.4,
        no_speech_prob=0.1,
    )
    runtime = FasterWhisperSpeechRuntime(
        FakeFasterWhisperModel(
            (segment,),
            SimpleNamespace(language="ja", duration_after_vad=1.0),
        ),
        runtime_identity="speech-runtime:test",
        resolved_model_identity="hf:" + "a" * 40,
    )
    pcm = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=16000,
        sample_rate=16000,
        channel_count=1,
        sample_format="s16le",
        pts=0,
        time_base=Fraction(1, 16000),
        pcm_bytes=b"\x00\x00" * 16000,
    )

    # Act
    result = runtime.transcribe(
        pcm,
        language="ja",
        vad_filter=True,
        beam_size=5,
    )

    # Assert
    word = result.segments[0].words[0]
    assert word.start_sample == 8000
    assert word.end_sample == 8000


def test_resolved_local_model_is_loaded_without_backend_download(
    tmp_path: Path,
) -> None:
    """解決済みlocal artifactだけがdownload禁止でloadされること。

    Arrange:
        - ModelRuntimeが用意したlocal model folderとmodel loader fakeが用意される
    Act:
        - faster-whisper SpeechRuntimeがlocal artifactから構築される
    Assert:
        - local_files_onlyでdevice・compute typeがloaderへ渡されること
        - model identityと導出済みruntime identityが公開されること
    """
    # Arrange
    model_artifact = tmp_path / "resolved-model"
    model_artifact.mkdir()
    model = FakeFasterWhisperModel(
        (),
        SimpleNamespace(language="ja", duration_after_vad=0.0),
    )
    calls: list[tuple[Path, str, str, bool]] = []

    def load_model(
        path: Path,
        device: str,
        compute_type: str,
        local_files_only: bool,
    ) -> FakeFasterWhisperModel:
        calls.append((path, device, compute_type, local_files_only))
        return model

    # Act
    runtime = FasterWhisperSpeechRuntime.load_local(
        model_artifact,
        resolved_model_identity="hf:" + "b" * 40,
        device="cuda",
        compute_type="float16",
        model_loader=load_model,
    )

    # Assert
    assert calls == [(model_artifact, "cuda", "float16", True)]
    assert runtime.resolved_model_identity == "hf:" + "b" * 40
    assert runtime.runtime_identity.startswith("speech_")
    assert len(runtime.runtime_identity) == 71


def test_runtime_identity_changes_with_gpu_runtime_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU runtime capability変更でSpeech Runtime Identityが変わること。

    Arrange:
        - 同じlocal modelと異なるCTranslate2 CUDA compute capabilityが用意される
    Act:
        - 各capabilityでSpeech Runtimeが構築される
    Assert:
        - runtime versionが同じでも異なるidentityが導出されること
    """
    # Arrange
    model_artifact = tmp_path / "resolved-model"
    model_artifact.mkdir()
    model = FakeFasterWhisperModel(
        (),
        SimpleNamespace(language="ja", duration_after_vad=0.0),
    )
    cuda_compute_types = {"float16", "int8_float16"}
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)

    def supported_compute_types(
        device: str,
        device_index: int = 0,
    ) -> set[str]:
        del device_index
        return cuda_compute_types if device == "cuda" else {"float32"}

    monkeypatch.setattr(
        ctranslate2,
        "get_supported_compute_types",
        supported_compute_types,
    )

    def load_model(
        path: Path,
        device: str,
        compute_type: str,
        local_files_only: bool,
    ) -> FakeFasterWhisperModel:
        del path, device, compute_type, local_files_only
        return model

    # Act
    first = FasterWhisperSpeechRuntime.load_local(
        model_artifact,
        resolved_model_identity="hf:" + "a" * 40,
        device="cuda",
        compute_type="float16",
        model_loader=load_model,
    )
    cuda_compute_types.remove("int8_float16")
    second = FasterWhisperSpeechRuntime.load_local(
        model_artifact,
        resolved_model_identity="hf:" + "a" * 40,
        device="cuda",
        compute_type="float16",
        model_loader=load_model,
    )

    # Assert
    assert first.runtime_identity != second.runtime_identity


def test_transcription_waits_for_shared_gpu_lease() -> None:
    """STTが共有GPU leaseを取得してからmodelを実行すること。

    Arrange:
        - Vision相当workが保持中の実coordinatorとSpeech Runtimeが用意される
    Act:
        - 別threadからSTTが要求され、先行leaseが解放される
    Assert:
        - 解放前はmodelが呼ばれず、解放後にtranscriptionが完了すること
    """
    # Arrange
    coordinator = GpuWorkCoordinator()
    lease_started = Event()
    release_lease = Event()
    model = FakeFasterWhisperModel(
        (),
        SimpleNamespace(language="ja", duration_after_vad=0.0),
    )
    runtime = FasterWhisperSpeechRuntime(
        model,
        runtime_identity="speech-runtime:test",
        resolved_model_identity="hf:" + "a" * 40,
        gpu_coordinator=coordinator,
    )
    pcm = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=1,
        sample_rate=16000,
        channel_count=1,
        sample_format="s16le",
        pts=0,
        time_base=Fraction(1, 16000),
        pcm_bytes=b"\x00\x00",
    )
    failures: list[BaseException] = []

    def hold_gpu_lease() -> None:
        def wait_for_release() -> None:
            lease_started.set()
            if not release_lease.wait(timeout=1.0):
                msg = "GPU leaseを解放できませんでした"
                raise RuntimeError(msg)

        try:
            coordinator.run("vision_inference", wait_for_release)
        except BaseException as error:
            failures.append(error)

    def transcribe() -> None:
        try:
            runtime.transcribe(
                pcm,
                language="ja",
                vad_filter=True,
                beam_size=5,
            )
        except BaseException as error:
            failures.append(error)

    holder = Thread(target=hold_gpu_lease)
    worker = Thread(target=transcribe)

    # Act
    holder.start()
    assert lease_started.wait(timeout=1.0)
    worker.start()
    worker.join(timeout=0.05)
    blocked_before_release = worker.is_alive() and model.audio is None
    release_lease.set()
    holder.join(timeout=1.0)
    worker.join(timeout=1.0)

    # Assert
    assert blocked_before_release is True
    assert failures == []
    assert model.audio is not None
    assert holder.is_alive() is False
    assert worker.is_alive() is False
