"""一回のrunでfreezeされた全model role。"""

from dataclasses import dataclass

from .model_role import ModelRole
from .resolved_model import ResolvedModel


@dataclass(frozen=True)
class ResolvedModels:
    """3 roleの重複のないResolved Modelを保持する。"""

    items: tuple[ResolvedModel, ...]

    def __post_init__(self) -> None:
        """全roleが一度ずつ含まれることを検証する。"""
        roles = tuple(item.role for item in self.items)
        if len(roles) != len(ModelRole) or set(roles) != set(ModelRole):
            msg = "Resolved Modelsには全model roleが一度ずつ必要です"
            raise ValueError(msg)

    def for_role(self, role: ModelRole) -> ResolvedModel:
        """指定roleのfreeze済みmodelを返す。"""
        return next(item for item in self.items if item.role is role)

    def semantic_input(self) -> dict[str, object]:
        """全roleのmodel lifecycle Stage入力を返す。"""
        return {
            item.role.value: item.semantic_input()
            for item in sorted(self.items, key=lambda value: value.role.value)
        }

    def provenance(self) -> dict[str, dict[str, object]]:
        """全roleのpathなしprovenanceを返す。"""
        return {
            item.role.value: item.provenance()
            for item in sorted(self.items, key=lambda value: value.role.value)
        }
