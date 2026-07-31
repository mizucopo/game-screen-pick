"""Canonical Selection Report reader schema compatibilityのtest。"""

import json
import tomllib
from pathlib import Path

import pytest

from src.video_selection.services.build_canonical_selection_report import (
    REPORT_SCHEMA_VERSION,
)
from src.video_selection.services.validate_report_schema_compatibility import (
    validate_report_schema_compatibility,
)


def test_same_major_accepts_unknown_fields_and_enum_values() -> None:
    """対応major内の未知fieldと未知enum値がreader gateで拒否されないこと。

    Arrange:
        - report@2の将来minorと未知field、未知enum値を持つobjectが用意される
    Act:
        - reader schema compatibility gateが実行される
    Assert:
        - schema identityだけが検証されobjectが変更されないこと
    """
    # Arrange
    report: dict[str, object] = {
        "schema": {"name": "game-screen-pick/report", "version": "2.8.0"},
        "future_field": {"future_enum": "new_value"},
    }
    expected = dict(report)

    # Act
    validate_report_schema_compatibility(report)

    # Assert
    assert report == expected


def test_unknown_major_is_rejected() -> None:
    """未対応major versionがreader gateで拒否されること。

    Arrange:
        - report@3のschema identityを持つobjectが用意される
    Act:
        - reader schema compatibility gateが実行される
    Assert:
        - 未対応major versionとして拒否されること
    """
    # Arrange
    report: dict[str, object] = {
        "schema": {"name": "game-screen-pick/report", "version": "3.0.0"}
    }

    # Act
    # Assert
    with pytest.raises(ValueError, match="未対応major"):
        validate_report_schema_compatibility(report)


def test_report_schema_version_is_independent_from_package_version() -> None:
    """report schema versionがproject package versionと独立していること。

    Arrange:
        - source treeのproject metadataとreport producer versionが用意される
    Act:
        - 両方のversionが読み取られる
    Assert:
        - report schemaが2.0.0でpackage versionとは異なること
    """
    # Arrange
    project_root = Path(__file__).parents[3]

    # Act
    project = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    package_version = project["project"]["version"]

    # Assert
    assert REPORT_SCHEMA_VERSION == "2.0.0"
    assert package_version != REPORT_SCHEMA_VERSION


def test_historical_report_schema_remains_available_after_major_upgrade() -> None:
    """major更新後も旧report schemaがversion別pathで保持されること。

    Arrange:
        - schema配置先と履歴1.0.0・現行2.0.0のversion別pathが用意される
    Act:
        - 両schema documentが読み込まれる
    Assert:
        - 各pathが対応するschema versionを保持すること
    """
    # Arrange
    schema_folder = Path(__file__).parents[3] / "src/video_selection/schemas"
    historical_path = schema_folder / "report-1.0.0.schema.json"
    current_path = schema_folder / "report-2.0.0.schema.json"

    # Act
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    current = json.loads(current_path.read_text(encoding="utf-8"))

    # Assert
    assert historical["properties"]["schema"]["properties"]["version"]["const"] == (
        "1.0.0"
    )
    assert current["properties"]["schema"]["properties"]["version"]["const"] == "2.0.0"
