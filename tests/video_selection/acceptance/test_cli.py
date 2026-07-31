"""target acceptance CLIのtest。"""

import io
from pathlib import Path

import pytest

from src.video_selection.acceptance.cli import main


def test_cli_requires_explicit_suite(tmp_path: Path) -> None:
    """suite未指定がargparse usage errorのexit 2になること。

    Arrange:
        - profileだけを指定したCLI引数が用意される
    Act:
        - target acceptance CLIの解析が実行される
    Assert:
        - runner実行前にSystemExit 2が返されること
    """
    # Arrange
    arguments = ["--profile", str(tmp_path / "target.toml")]

    # Act
    # Assert
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_cli_passes_profile_suite_reset_and_review_to_runner(
    tmp_path: Path,
) -> None:
    """検証済みCLI値がrunnerへ渡されstable exit codeが保持されること。

    Arrange:
        - full suite、reset、human reviewを持つCLI引数が用意される
    Act:
        - 注入されたtarget runnerが実行される
    Assert:
        - Pathとflagが一度渡されpending exit 3が返されること
    """
    # Arrange
    profile = tmp_path / "target.toml"
    review = tmp_path / "review.json"
    captured: list[tuple[Path, str, bool, str | None, Path | None]] = []

    def run_target_suite(
        profile_path: Path,
        suite: str,
        reset_suite: bool,
        reset_run: str | None,
        human_review_path: Path | None,
    ) -> int:
        captured.append(
            (profile_path, suite, reset_suite, reset_run, human_review_path),
        )
        return 3

    # Act
    result = main(
        [
            "--profile",
            str(profile),
            "--suite",
            "full",
            "--reset-suite",
            "--human-review",
            str(review),
        ],
        run_target_suite=run_target_suite,
    )

    # Assert
    assert result == 3
    assert captured == [(profile, "full", True, None, review)]


def test_cli_passes_user_facing_run_reset_to_runner(tmp_path: Path) -> None:
    """利用者向けrun reset名がrunnerへ渡されること。

    Arrange:
        - fresh processingだけをresetするrelease CLI引数が用意される
    Act:
        - 注入されたtarget runnerが実行される
    Assert:
        - `fresh-processing`が変更されずrunnerへ渡されること
    """
    # Arrange
    profile = tmp_path / "target.toml"
    captured: list[tuple[Path, str, bool, str | None, Path | None]] = []

    def run_target_suite(
        profile_path: Path,
        suite: str,
        reset_suite: bool,
        reset_run: str | None,
        human_review_path: Path | None,
    ) -> int:
        captured.append(
            (profile_path, suite, reset_suite, reset_run, human_review_path),
        )
        return 3

    # Act
    result = main(
        [
            "--profile",
            str(profile),
            "--suite",
            "release",
            "--reset-run",
            "fresh-processing",
        ],
        run_target_suite=run_target_suite,
    )

    # Assert
    assert result == 3
    assert captured == [
        (profile, "release", False, "fresh-processing", None),
    ]


def test_cli_rejects_suite_and_run_reset_together(tmp_path: Path) -> None:
    """suite全体resetとrun resetの同時指定が拒否されること。

    Arrange:
        - `--reset-suite`と`--reset-run`を併記した引数が用意される
    Act:
        - target acceptance CLIの解析が実行される
    Assert:
        - runner実行前にSystemExit 2になること
    """
    # Arrange
    arguments = [
        "--profile",
        str(tmp_path / "target.toml"),
        "--suite",
        "full",
        "--reset-suite",
        "--reset-run",
        "parallelism-baseline",
    ]

    # Act
    with pytest.raises(SystemExit) as error:
        main(arguments)

    # Assert
    assert error.value.code == 2


def test_cli_maps_validation_interrupt_and_operation_failures() -> None:
    """runner境界のfailure種別がstable exit codeへ変換されること。

    Arrange:
        - validation、interrupt、operation errorを投げるrunnerが用意される
    Act:
        - 各runnerで同じvalid CLIが実行される
    Assert:
        - 順にexit 2、130、1が返されoperation detailが公開されないこと
    """
    # Arrange
    arguments = ["--profile", "/private/target.toml", "--suite", "release"]
    stderr = io.StringIO()

    def validation_failure(
        _profile: Path,
        _suite: str,
        _reset: bool,
        _reset_run: str | None,
        _review: Path | None,
    ) -> int:
        raise ValueError("profile schema mismatch")

    def interrupted(
        _profile: Path,
        _suite: str,
        _reset: bool,
        _reset_run: str | None,
        _review: Path | None,
    ) -> int:
        raise KeyboardInterrupt

    def operation_failure(
        _profile: Path,
        _suite: str,
        _reset: bool,
        _reset_run: str | None,
        _review: Path | None,
    ) -> int:
        raise RuntimeError("private operation detail")

    # Act
    validation_exit = main(
        arguments,
        run_target_suite=validation_failure,
        stderr=stderr,
    )
    interrupt_exit = main(arguments, run_target_suite=interrupted, stderr=stderr)
    operation_exit = main(
        arguments,
        run_target_suite=operation_failure,
        stderr=stderr,
    )

    # Assert
    assert (validation_exit, interrupt_exit, operation_exit) == (2, 130, 1)
    assert "profile schema mismatch" in stderr.getvalue()
    assert "private operation detail" not in stderr.getvalue()
