"""Representative Frame比較に必要な構造化観測。"""

from dataclasses import dataclass
from typing import Literal, cast, get_args

CandidateFrameContentKind = Literal[
    "gameplay_action",
    "gameplay_idle",
    "event_dialogue",
    "event_action",
    "event_setup",
    "document",
    "shop",
    "map",
    "save",
    "tutorial_help",
    "other_interface",
    "title",
    "other",
]
PrimarySubjectVisibility = Literal["clear", "partial", "absent"]
TransientObstruction = Literal["none", "partial", "severe"]
CharacterBodyVisibility = Literal["clear", "partial", "absent"]

CANDIDATE_FRAME_CONTENT_KINDS = cast(
    tuple[CandidateFrameContentKind, ...],
    get_args(CandidateFrameContentKind),
)
PRIMARY_SUBJECT_VISIBILITIES = cast(
    tuple[PrimarySubjectVisibility, ...],
    get_args(PrimarySubjectVisibility),
)
TRANSIENT_OBSTRUCTIONS = cast(
    tuple[TransientObstruction, ...],
    get_args(TransientObstruction),
)
CHARACTER_BODY_VISIBILITIES = cast(
    tuple[CharacterBodyVisibility, ...],
    get_args(CharacterBodyVisibility),
)


@dataclass(frozen=True)
class RepresentativeFrameEvidence:
    """独立評価済みframeの内容、対象可視性、遮蔽を保持する。"""

    content_kind: CandidateFrameContentKind
    primary_subject_visibility: PrimarySubjectVisibility
    opponent_body_visibility: CharacterBodyVisibility
    transient_obstruction: TransientObstruction

    def __post_init__(self) -> None:
        """Representative Frame比較用enumだけを受理する。"""
        if (
            self.content_kind not in CANDIDATE_FRAME_CONTENT_KINDS
            or self.primary_subject_visibility not in PRIMARY_SUBJECT_VISIBILITIES
            or self.opponent_body_visibility not in CHARACTER_BODY_VISIBILITIES
            or self.transient_obstruction not in TRANSIENT_OBSTRUCTIONS
        ):
            msg = "Representative Frame Evidenceが不正です"
            raise ValueError(msg)
