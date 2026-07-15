"""walking skeletonのResolved Model Identity。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedModelIdentity:
    """fake ModelRuntimeが解決する最小model identity。"""

    identifier: str
