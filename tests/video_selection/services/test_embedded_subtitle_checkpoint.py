"""EmbeddedSubtitleCheckpointの再利用test。"""

from fractions import Fraction
from pathlib import Path

from src.video_selection.models.embedded_subtitle import EmbeddedSubtitle
from src.video_selection.services.embedded_subtitle_checkpoint import (
    EmbeddedSubtitleCheckpoint,
)


def test_completed_stream_is_not_extracted_again(tmp_path: Path) -> None:
    """選択subtitle stream完了後の再開でruntimeが再呼出されないこと。

    Arrange:
        - 一つのsubtitle eventを返すextractorが用意される
    Act:
        - 同じsemantic依存でcheckpointが2回解決される
    Assert:
        - extractorは初回だけ呼ばれ同じevent列が返されること
    """
    # Arrange
    calls = 0
    expected = (
        EmbeddedSubtitle(
            stream_index=2,
            pts=100,
            duration_ts=20,
            time_base=Fraction(1, 1000),
            text="再開できる字幕",
        ),
    )

    def extract() -> tuple[EmbeddedSubtitle, ...]:
        nonlocal calls
        calls += 1
        return expected

    def checkpoint() -> EmbeddedSubtitleCheckpoint:
        return EmbeddedSubtitleCheckpoint(
            tmp_path,
            source_fingerprint="b" * 64,
            stream_index=2,
            extraction_semantic_input={"media_runtime_identity": "runtime-a"},
        )

    # Act
    first = checkpoint().resolve(extract)
    second = checkpoint().resolve(extract)

    # Assert
    assert first == expected
    assert second == expected
    assert calls == 1
