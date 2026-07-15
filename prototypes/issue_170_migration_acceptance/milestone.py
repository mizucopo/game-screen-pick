"""Migration milestone。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Milestone:
    """一つのimplementation Issueを表す。"""

    number: int
    title: str
