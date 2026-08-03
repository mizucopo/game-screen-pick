"""Semantic Duplicate Groupを支持するprivacy-safeな根拠。"""

from typing import Literal, cast, get_args

SemanticDuplicateBasis = Literal[
    "combat_encounter_sequence",
    "title_semantics",
    "visual_role_similarity",
]

SEMANTIC_DUPLICATE_BASES = cast(
    tuple[SemanticDuplicateBasis, ...],
    get_args(SemanticDuplicateBasis),
)
