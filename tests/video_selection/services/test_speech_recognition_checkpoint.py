"""SpeechRecognitionCheckpointの再利用test。"""

import hashlib
import json
from fractions import Fraction
from pathlib import Path

from src.video_selection.models.pcm_audio_chunk import PcmAudioChunk
from src.video_selection.models.speech_recognition_result import (
    SpeechRecognitionResult,
)
from src.video_selection.models.speech_segment import SpeechSegment
from src.video_selection.models.speech_word import SpeechWord
from src.video_selection.services.speech_recognition_checkpoint import (
    SpeechRecognitionCheckpoint,
)


def test_completed_chunk_is_not_inferred_again(tmp_path: Path) -> None:
    """完了済みPCM chunkのSpeech Runtime推論が再実行されないこと。

    Arrange:
        - 一つのPCM chunkと決定的なRecognition Resultが用意される
    Act:
        - 同じsemantic依存でcheckpointが2回解決される
    Assert:
        - recognize callbackは初回だけ呼ばれ同じ結果が返されること
    """
    # Arrange
    chunk = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=4,
        sample_rate=4,
        channel_count=1,
        sample_format="s16le",
        pts=0,
        time_base=Fraction(1, 4),
        pcm_bytes=b"\x00\x00" * 4,
    )
    expected = SpeechRecognitionResult(
        vad_speech_detected=True,
        detected_language="ja",
        segments=(
            SpeechSegment(
                words=(
                    SpeechWord(
                        text="再開",
                        start_sample=0,
                        end_sample=2,
                        probability=0.9,
                    ),
                ),
                average_log_probability=-0.1,
                no_speech_probability=0.01,
            ),
        ),
    )
    calls = 0

    def recognize() -> SpeechRecognitionResult:
        nonlocal calls
        calls += 1
        return expected

    def checkpoint() -> SpeechRecognitionCheckpoint:
        return SpeechRecognitionCheckpoint(
            tmp_path,
            source_fingerprint="c" * 64,
            recognition_semantic_input={
                "speech_runtime_identity": "runtime-a",
                "resolved_model_identity": "model-a",
            },
        )

    # Act
    first = checkpoint().resolve(chunk, recognize)
    second = checkpoint().resolve(chunk, recognize)

    # Assert
    assert first == expected
    assert second == expected
    assert calls == 1


def test_changed_pcm_bytes_invalidate_completed_recognition(tmp_path: Path) -> None:
    """同じsample rangeでもPCM bytesが変わればSTTが再実行されること。

    Arrange:
        - timingが同じでPCM bytesだけが異なる二つのchunkが用意される
    Act:
        - 同じSpeech Recognition依存で両chunkが順に解決される
    Assert:
        - 古い認識結果は再利用されず両方が推論されること
    """
    # Arrange
    first_chunk = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=4,
        sample_rate=4,
        channel_count=1,
        sample_format="s16le",
        pts=0,
        time_base=Fraction(1, 4),
        pcm_bytes=b"\x00\x00" * 4,
    )
    second_chunk = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=4,
        sample_rate=4,
        channel_count=1,
        sample_format="s16le",
        pts=0,
        time_base=Fraction(1, 4),
        pcm_bytes=b"\x01\x00" * 4,
    )
    checkpoint = SpeechRecognitionCheckpoint(
        tmp_path,
        source_fingerprint="e" * 64,
        recognition_semantic_input={
            "speech_runtime_identity": "runtime-a",
            "resolved_model_identity": "model-a",
        },
    )
    calls = 0

    def recognize() -> SpeechRecognitionResult:
        nonlocal calls
        calls += 1
        return SpeechRecognitionResult(
            vad_speech_detected=False,
            detected_language=None,
            segments=(),
        )

    # Act
    checkpoint.resolve(first_chunk, recognize)
    checkpoint.resolve(second_chunk, recognize)

    # Assert
    assert calls == 2


def test_domain_invalid_chunk_is_recomputed_instead_of_reused(
    tmp_path: Path,
) -> None:
    """hash整合済みでもchunk外時刻を持つ認識結果が再計算されること。

    Arrange:
        - 正常なSpeech Recognition checkpointが用意される
        - word時刻とmanifest hashがchunk外の値へ改変される
    Act:
        - 同じPCM chunkのrecognitionが再解決される
    Assert:
        - 不正checkpointが再利用されずSpeech Runtimeが再実行されること
    """
    # Arrange
    chunk = PcmAudioChunk(
        stream_index=1,
        sample_start=0,
        sample_count=4,
        sample_rate=4,
        channel_count=1,
        sample_format="s16le",
        pts=0,
        time_base=Fraction(1, 4),
        pcm_bytes=b"\x00\x00" * 4,
    )
    expected = SpeechRecognitionResult(
        vad_speech_detected=True,
        detected_language="ja",
        segments=(
            SpeechSegment(
                words=(SpeechWord("再開", 0, 2, 0.9),),
                average_log_probability=-0.1,
                no_speech_probability=0.01,
            ),
        ),
    )
    checkpoint = SpeechRecognitionCheckpoint(
        tmp_path,
        source_fingerprint="d" * 64,
        recognition_semantic_input={"resolved_model_identity": "model-a"},
    )
    checkpoint.resolve(chunk, lambda: expected)
    artifact_path = next(tmp_path.rglob("artifact.json"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["segments"][0]["words"][0]["start_sample"] = -1
    artifact_bytes = (
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    artifact_path.write_bytes(artifact_bytes)
    manifest_path = artifact_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item for item in manifest["artifacts"] if item["path"] == "artifact.json"
    )
    record["size_bytes"] = len(artifact_bytes)
    record["sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    calls = 0

    def recognize() -> SpeechRecognitionResult:
        nonlocal calls
        calls += 1
        return expected

    # Act
    actual = checkpoint.resolve(chunk, recognize)

    # Assert
    assert actual == expected
    assert calls == 1
