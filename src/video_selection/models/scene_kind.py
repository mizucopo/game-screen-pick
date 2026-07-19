"""Scene Catalogで共有する内容種別。"""

from typing import Literal, cast, get_args

SceneKind = Literal["combat", "exploration", "interface", "event", "other"]
SCENE_KINDS = cast(tuple[SceneKind, ...], get_args(SceneKind))
