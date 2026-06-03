"""scene catalog の1項目."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneCatalogEntry:
    """実行内で使うscene定義."""

    slug: str
    display_name: str
    description: str

    def __post_init__(self) -> None:
        """scene定義の妥当性を検証する."""
        if not self.slug.strip():
            msg = "scene slugは必須です"
            raise ValueError(msg)
        if not self.display_name.strip():
            msg = "scene display_nameは必須です"
            raise ValueError(msg)
        if not self.description.strip():
            msg = "scene descriptionは必須です"
            raise ValueError(msg)
