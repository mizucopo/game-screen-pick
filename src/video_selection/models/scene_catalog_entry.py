"""共有Scene Catalogの一つのscene。"""

import re
from dataclasses import dataclass
from typing import Literal, cast, get_args

SceneSelectionRole = Literal["ordinary", "cinematic", "recurring_gameplay"]
SCENE_SELECTION_ROLES = cast(
    tuple[SceneSelectionRole, ...],
    get_args(SceneSelectionRole),
)

_SCENE_SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def is_valid_scene_slug(value: str) -> bool:
    """Scene Catalogで公開可能なslugであることを返す。"""
    return _SCENE_SLUG_PATTERN.fullmatch(value) is not None


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
            not is_valid_scene_slug(self.slug)
            or not self.display_name.strip()
            or not self.description.strip()
            or self.selection_role not in SCENE_SELECTION_ROLES
        ):
            msg = "Scene Catalog entryが不正です"
            raise ValueError(msg)
