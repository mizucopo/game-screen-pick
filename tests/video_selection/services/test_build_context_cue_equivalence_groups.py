"""Context Cue Equivalence Group構築のtest。"""

from fractions import Fraction

from src.video_selection.models.context_cue import ContextCue, ContextSourceKind
from src.video_selection.models.context_stage_result import ContextStageResult
from src.video_selection.services.build_context_cue_equivalence_groups import (
    build_context_cue_equivalence_groups,
)


def test_long_stt_cue_does_not_collapse_distinct_subtitle_occurrences() -> None:
    """長いSTT Cueで別時刻のsubtitle occurrenceが推移的に畳まれないこと。

    Arrange:
        - 同文のsubtitle Cue 2件と両方へ重なるSTT Cueが用意される
    Act:
        - Context Cue Equivalence Groupが構築される
    Assert:
        - STT Cueが一方のsubtitle Cueとのpairだけにされること
        - 別時刻のsubtitle Cueがannotation入力に保持されること
    """
    # Arrange
    first_subtitle = _cue(
        "subtitle-1",
        "embedded_subtitle",
        Fraction(1),
        Fraction(2),
    )
    second_subtitle = _cue(
        "subtitle-2",
        "embedded_subtitle",
        Fraction(3),
        Fraction(4),
    )
    speech = _cue(
        "speech",
        "speech_to_text",
        Fraction(1),
        Fraction(4),
    )
    cues = (first_subtitle, second_subtitle, speech)

    # Act
    groups = build_context_cue_equivalence_groups(cues)
    result = ContextStageResult(
        cues=cues,
        outcomes=(),
        equivalence_groups=groups,
    )

    # Assert
    assert len(groups) == 1
    assert groups[0].representative_cue_id == first_subtitle.identifier
    assert groups[0].cue_ids == (first_subtitle.identifier, speech.identifier)
    assert result.annotation_cues == (first_subtitle, second_subtitle)


def _cue(
    identifier: str,
    source_kind: ContextSourceKind,
    start: Fraction,
    end: Fraction,
) -> ContextCue:
    return ContextCue(
        identifier=identifier,
        source_kind=source_kind,
        start=start,
        end=end,
        text="助けて",
    )
