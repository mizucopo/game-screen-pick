import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _quality_project(tmp_path: Path, has_tests: bool) -> None:
    (tmp_path / "pyproject.toml").write_text((_ROOT / "pyproject.toml").read_text())
    for directory in ("src", "tests", "stubs"):
        (tmp_path / directory).mkdir()
    (tmp_path / "src/__init__.py").touch()
    (tmp_path / "stubs/__init__.py").touch()
    (tmp_path / "src/demo.py").write_text("def answer() -> int:\n    return 42\n")
    (tmp_path / "tests/support.py").write_text("VALUE = 42\n")
    # Include the template runner when present so this checks the real task.
    runner = _ROOT / "tests/run_pytest.py"
    if runner.exists():
        (tmp_path / "tests/run_pytest.py").write_bytes(runner.read_bytes())
    if has_tests:
        (tmp_path / "tests/test_demo.py").write_text(
            "from src.demo import answer\n\n\n"
            "def test_answer() -> None:\n    assert answer() == 42\n"
        )


def _run_quality_gate(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(Path(sys.executable).with_name("task")), "check"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        },
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("has_tests", [False, True])
def test_quality_gate_requires_a_test_suite(tmp_path: Path, has_tests: bool) -> None:
    _quality_project(tmp_path, has_tests)
    result = _run_quality_gate(tmp_path)
    assert result.returncode == (0 if has_tests else 5), result.stdout + result.stderr
    assert ("1 passed" if has_tests else "no tests ran") in result.stdout


@pytest.mark.parametrize(
    ("script", "diagnostic"),
    [
        ("VALUE = missing_name\n", "F821"),
        ("VALUE=42\n", "reformat"),
    ],
)
def test_quality_gate_checks_release_controller(
    tmp_path: Path, script: str, diagnostic: str
) -> None:
    _quality_project(tmp_path, has_tests=True)
    controller = tmp_path / ".github/scripts/release.py"
    controller.parent.mkdir(parents=True)
    controller.write_text(script)
    result = _run_quality_gate(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert ".github/scripts/release.py" in result.stdout
    assert diagnostic in result.stdout
