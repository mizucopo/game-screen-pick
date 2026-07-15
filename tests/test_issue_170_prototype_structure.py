"""Issue #170 prototypeのcode organization test。"""

import ast
from pathlib import Path


def test_migration_gate_modules_define_at_most_one_class() -> None:
    """migration gateの各moduleにclassが最大一つだけ定義されること。

    Arrange:
        - Issue #170 prototypeのPython moduleが列挙される
    Act:
        - 各moduleのtop-level class定義数が数えられる
    Assert:
        - one class per file規約が満たされること
    """
    # Arrange
    prototype_folder = (
        Path(__file__).parents[1] / "prototypes" / "issue_170_migration_acceptance"
    )
    modules = tuple(prototype_folder.glob("*.py"))

    # Act
    class_counts = {
        module.name: sum(
            isinstance(node, ast.ClassDef)
            for node in ast.parse(module.read_text(encoding="utf-8")).body
        )
        for module in modules
    }

    # Assert
    assert all(count <= 1 for count in class_counts.values()), class_counts
