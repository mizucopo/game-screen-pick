"""supported target向け内部acceptance CLI。"""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from .target_suite_runner import TargetSuiteRunner

RunTargetSuite = Callable[[Path, str, bool, Path | None], int]

_EXIT_CODES = {0, 1, 2, 3, 130}


def main(
    argv: Sequence[str] | None = None,
    *,
    run_target_suite: RunTargetSuite | None = None,
    stderr: TextIO | None = None,
) -> int:
    """CLI引数を検証しtarget suiteのstable exit codeを返す。"""
    arguments = _parser().parse_args(argv)
    execute = run_target_suite or _run_target_suite
    error_stream = stderr or sys.stderr
    try:
        exit_code = execute(
            arguments.profile,
            arguments.suite,
            arguments.reset_suite,
            arguments.human_review,
        )
    except KeyboardInterrupt:
        return 130
    except ValueError as error:
        print(f"acceptance-target: {error}", file=error_stream)
        return 2
    except Exception:
        print("acceptance-target: target operation failed", file=error_stream)
        return 1
    if exit_code not in _EXIT_CODES:
        print("acceptance-target: invalid runner exit code", file=error_stream)
        return 1
    return exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acceptance-target",
        description="WSL2 supported targetでcold/warm acceptanceを実行します。",
    )
    parser.add_argument(
        "--profile",
        required=True,
        type=Path,
        metavar="PATH",
        help="repository外のtarget-only acceptance profile",
    )
    parser.add_argument(
        "--suite",
        required=True,
        choices=("release", "full"),
        help="実行する固定suite",
    )
    parser.add_argument(
        "--reset-suite",
        action="store_true",
        help="同じsuiteのdurable stateを明示的に破棄してcoldから再実行",
    )
    parser.add_argument(
        "--human-review",
        type=Path,
        metavar="PATH",
        help="記入済みhuman review worksheet（省略時はartifact内）",
    )
    return parser


def _run_target_suite(
    profile_path: Path,
    suite: str,
    reset_suite: bool,
    human_review_path: Path | None,
) -> int:
    return TargetSuiteRunner().run(
        profile_path=profile_path,
        suite=suite,
        reset_suite=reset_suite,
        human_review_path=human_review_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
