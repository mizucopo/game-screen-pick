"""PROTOTYPE: timeline domain modelのcaseを切り替えるTUI。"""

from __future__ import annotations

import argparse
import json

from domain_model import prototype_cases

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    """CLI引数を返す。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def render(cases: list[dict[str, object]], index: int) -> None:
    """選択中caseの全状態を一画面へ表示する。"""
    print("\033[2J\033[H", end="")
    print(f"{BOLD}Timeline Segment and Candidate Moment PROTOTYPE{RESET}")
    print(f"{DIM}case {index + 1}/{len(cases)}{RESET}")
    print(json.dumps(cases[index], ensure_ascii=False, indent=2))
    print(f"\n{BOLD}[n]{RESET} next  {BOLD}[p]{RESET} previous  {BOLD}[q]{RESET} quit")


def run_tui(cases: list[dict[str, object]]) -> None:
    """入力に応じてprototype caseを切り替える。"""
    index = 0
    while True:
        render(cases, index)
        action = input("> ").strip().casefold()
        if action == "q":
            return
        if action == "n":
            index = (index + 1) % len(cases)
        if action == "p":
            index = (index - 1) % len(cases)


def main() -> None:
    """prototypeを起動する。"""
    args = parse_args()
    cases = prototype_cases()
    if args.all:
        print(json.dumps(cases, ensure_ascii=False, indent=2))
        return
    run_tui(cases)


if __name__ == "__main__":
    main()
