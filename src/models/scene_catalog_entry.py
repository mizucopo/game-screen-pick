"""scene catalog の1項目."""

import re
from dataclasses import dataclass

_SCENE_SLUG_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


@dataclass(frozen=True)
class SceneCatalogEntry:
    """実行内で使うscene定義."""

    slug: str
    display_name: str
    description: str

    def __post_init__(self) -> None:
        """scene定義の妥当性を検証する."""
        if not _SCENE_SLUG_PATTERN.fullmatch(self.slug):
            msg = (
                "scene slugは小文字英数字、underscore、hyphenのみである必要があります: "
                f"{self.slug}"
            )
            raise ValueError(msg)
        if not self.display_name.strip():
            msg = "scene display_nameは必須です"
            raise ValueError(msg)
        if not self.description.strip():
            msg = "scene descriptionは必須です"
            raise ValueError(msg)
