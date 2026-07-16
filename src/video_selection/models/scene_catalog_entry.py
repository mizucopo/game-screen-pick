"""共有Scene Catalogの一つのscene。"""

import re
from dataclasses import dataclass
from typing import Literal

SceneSelectionRole = Literal["ordinary", "cinematic", "recurring_gameplay"]

_SCENE_SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True)
class SceneCatalogEntry:
    """scene slug、表示名、説明、選定roleを保持する。"""

    slug: str
    display_name: str
    description: str
    selection_role: SceneSelectionRole

    def __post_init__(self) -> None:
        """公開可能なscene fieldだけを受理する。"""
        if (
            _SCENE_SLUG_PATTERN.fullmatch(self.slug) is None
            or not self.display_name.strip()
            or not self.description.strip()
            or self.selection_role
            not in {"ordinary", "cinematic", "recurring_gameplay"}
        ):
            msg = "Scene Catalog entryが不正です"
            raise ValueError(msg)
