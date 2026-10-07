import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/release.yml"


def _step_script(name: str) -> str:
    step = _WORKFLOW.read_text().split(f"      - name: {name}\n", 1)[1]
    step = re.split(r"\n(?:      - name:|  [a-z])", step, maxsplit=1)[0]
    match = re.search(r"^        run: (.*)$", step, re.MULTILINE)
    assert match is not None
    return (
        textwrap.dedent(step.split("        run: |\n", 1)[1])
        if match.group(1) == "|"
        else match.group(1)
    )


@pytest.mark.parametrize(
    ("version", "is_prerelease"),
    [
        ("1.20.0", False),
        ("1.20.0-rc.1", True),
        ("1.20.0-rc.1+build.2", True),
        ("1.20.0+build-with-hyphens", False),
    ],
)
def test_release_creation_marks_prereleases(
    tmp_path: Path, version: str, is_prerelease: bool
) -> None:
    script = _step_script("Create GitHub Release")
    capture = tmp_path / "gh-arguments.json"
    gh = tmp_path / "gh"
    gh.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGUMENTS_FILE"\n')
    gh.chmod(0o755)
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "TAG": version,
            "VERSION": version,
            "ARGUMENTS_FILE": str(capture),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    arguments = capture.read_text().splitlines()
    assert arguments[:3] == ["release", "create", version]
    assert "--latest=false" in arguments
    assert ("--prerelease" in arguments) is is_prerelease


@pytest.mark.parametrize(
    ("version", "is_prerelease"),
    [
        ("1.20.0", False),
        ("1.20.0-rc.1", True),
        ("1.20.0-rc.1+build.2", True),
        ("1.20.0+build-with-hyphens", False),
    ],
)
def test_prerelease_skips_latest_lookup_without_a_prior_stable_release(
    tmp_path: Path, version: str, is_prerelease: bool
) -> None:
    python = tmp_path / "python3"
    python.write_text(
        '#!/bin/sh\nprintf "called\\n" > "$LATEST_CALLED"\n'
        'if [ "$HAS_STABLE_RELEASE" = false ]; then exit 9; fi\n'
        'printf "promote_latest=true\\n" >> "$GITHUB_OUTPUT"\n'
    )
    python.chmod(0o755)
    output = tmp_path / "output"
    called = tmp_path / "called"
    result = subprocess.run(
        ["bash", "-e", "-c", _step_script("Check newest completed release")],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "VERSION": version,
            "GITHUB_OUTPUT": str(output),
            "LATEST_CALLED": str(called),
            "HAS_STABLE_RELEASE": str(not is_prerelease).lower(),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert called.exists() is not is_prerelease
    assert output.read_text().strip() == (
        "promote_latest=false" if is_prerelease else "promote_latest=true"
    )
