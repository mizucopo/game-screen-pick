"""Migration全体のstate。"""

from dataclasses import dataclass

from evidence import Evidence


@dataclass(frozen=True)
class MigrationState:
    """migrationとpublic interfaceの全関連state。"""

    completed_issues: frozenset[int] = frozenset()
    passed_pr_gates: frozenset[int] = frozenset()
    evidence: frozenset[Evidence] = frozenset()
    public_cli: str = "screenshot"
    package_version: str = "1.5.2"
    legacy_code_present: bool = True
    legacy_adrs_active: bool = True
