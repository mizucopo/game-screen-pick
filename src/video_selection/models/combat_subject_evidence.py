"""戦闘対象を名前に依存せず比較するための画像内根拠。"""

from dataclasses import dataclass
from typing import Literal, cast, get_args

CombatSubjectBodyPlan = Literal[
    "unknown",
    "humanoid",
    "quadruped",
    "serpentine",
    "fish_like",
    "insectoid",
    "avian",
    "amorphous",
    "plant_like",
    "mechanical",
    "structure",
    "multi_part",
    "other",
]
CombatSubjectScale = Literal["unknown", "small", "peer", "large", "enormous"]
CombatSubjectSurface = Literal[
    "unknown",
    "organic",
    "furred",
    "scaled",
    "armored",
    "skeletal",
    "mechanical",
    "plant",
    "elemental",
    "other",
]
CombatSubjectColor = Literal[
    "black",
    "white",
    "gray",
    "brown",
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "pink",
    "multicolor",
]
CombatSubjectTrait = Literal[
    "armor",
    "bulbous_body",
    "elongated_body",
    "floating",
    "glowing_core",
    "horns",
    "large_eye",
    "large_mouth",
    "machine_parts",
    "multiple_heads",
    "multiple_limbs",
    "plant_parts",
    "shell",
    "spikes",
    "tail",
    "tentacles",
    "weapon",
    "wings",
]
CombatSubjectDistinctiveness = Literal[
    "unclear",
    "generic",
    "distinctive",
]

COMBAT_SUBJECT_BODY_PLANS = cast(
    tuple[CombatSubjectBodyPlan, ...],
    get_args(CombatSubjectBodyPlan),
)
COMBAT_SUBJECT_SCALES = cast(
    tuple[CombatSubjectScale, ...],
    get_args(CombatSubjectScale),
)
COMBAT_SUBJECT_SURFACES = cast(
    tuple[CombatSubjectSurface, ...],
    get_args(CombatSubjectSurface),
)
COMBAT_SUBJECT_COLORS = cast(
    tuple[CombatSubjectColor, ...],
    get_args(CombatSubjectColor),
)
COMBAT_SUBJECT_TRAITS = cast(
    tuple[CombatSubjectTrait, ...],
    get_args(CombatSubjectTrait),
)
COMBAT_SUBJECT_DISTINCTIVENESSES = cast(
    tuple[CombatSubjectDistinctiveness, ...],
    get_args(CombatSubjectDistinctiveness),
)


@dataclass(frozen=True)
class CombatSubjectEvidence:
    """一枚の画像から観測した戦闘対象の外見特徴を保持する。"""

    body_plan: CombatSubjectBodyPlan
    scale: CombatSubjectScale
    surface: CombatSubjectSurface
    colors: tuple[CombatSubjectColor, ...]
    traits: tuple[CombatSubjectTrait, ...]
    distinctiveness: CombatSubjectDistinctiveness

    def __post_init__(self) -> None:
        """有限enumと安定順の外見特徴だけを受理する。"""
        normalized_colors = tuple(sorted(set(self.colors)))
        normalized_traits = tuple(sorted(set(self.traits)))
        if (
            self.body_plan not in COMBAT_SUBJECT_BODY_PLANS
            or self.scale not in COMBAT_SUBJECT_SCALES
            or self.surface not in COMBAT_SUBJECT_SURFACES
            or any(color not in COMBAT_SUBJECT_COLORS for color in normalized_colors)
            or any(trait not in COMBAT_SUBJECT_TRAITS for trait in normalized_traits)
            or self.distinctiveness not in COMBAT_SUBJECT_DISTINCTIVENESSES
            or len(normalized_colors) > 2
            or len(normalized_traits) > 4
            or self.distinctiveness == "distinctive"
            and (
                self.body_plan == "unknown"
                or self.scale == "unknown"
                or self.surface == "unknown"
                or not normalized_colors
                or not normalized_traits
            )
        ):
            raise ValueError("Combat Subject Evidenceが不正です")
        object.__setattr__(self, "colors", normalized_colors)
        object.__setattr__(self, "traits", normalized_traits)

    @property
    def can_identify_subject(self) -> bool:
        """別の画像と同じ対象だと比較できる十分な外見根拠があるかを返す。"""
        return (
            self.distinctiveness == "distinctive"
            and self.body_plan != "unknown"
            and self.scale != "unknown"
            and self.surface != "unknown"
            and bool(self.colors)
            and bool(self.traits)
        )


UNCLEAR_COMBAT_SUBJECT_EVIDENCE = CombatSubjectEvidence(
    body_plan="unknown",
    scale="unknown",
    surface="unknown",
    colors=(),
    traits=(),
    distinctiveness="unclear",
)
