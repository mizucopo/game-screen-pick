"""supported target向け内部acceptance CLI。"""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from .acceptance_run_reset import ACCEPTANCE_RUN_RESETS, AcceptanceRunReset
from .target_suite_runner import TargetSuiteRunner

RunTargetSuite = Callable[
    [Path, str, bool, AcceptanceRunReset | None, Path | None],
    int,
]

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
            arguments.reset_run,
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
        description=(
            "WSL2 supported targetで比較runとcold/warm acceptanceを実行します。"
        ),
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
    reset_group = parser.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--reset-suite",
        action="store_true",
        help="同じsuiteのdurable stateを明示的に破棄して先頭runから再実行",
    )
    reset_group.add_argument(
        "--reset-run",
        choices=ACCEPTANCE_RUN_RESETS,
        help=(
            "指定runと依存する後続runだけを再測定: "
            "parallelism-baseline（fullの並列基準）、"
            "fresh-processing（cacheなし本処理）、"
            "cache-reuse（同一cache再利用）"
        ),
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
    reset_run: AcceptanceRunReset | None,
    human_review_path: Path | None,
) -> int:
    return TargetSuiteRunner().run(
        profile_path=profile_path,
        suite=suite,
        reset_suite=reset_suite,
        reset_run=reset_run,
        human_review_path=human_review_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
