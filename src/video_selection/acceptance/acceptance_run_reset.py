"""target acceptanceの利用者向けrun reset名。"""

from typing import Literal, get_args

AcceptanceRunReset = Literal[
    "parallelism-baseline",
    "fresh-processing",
    "cache-reuse",
]

ACCEPTANCE_RUN_RESETS = get_args(AcceptanceRunReset)
