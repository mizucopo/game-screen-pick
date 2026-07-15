"""PROTOTYPE: migration gateを手動操作するTUI。"""

from __future__ import annotations

import argparse
import sys

from migration_gate import (
    CUTOVER_EVIDENCE,
    MILESTONES,
    attempt_cutover,
    complete_next_issue,
    next_milestone,
    pass_next_pr_gate,
    record_next_evidence,
    validate_state,
)
from migration_state import MigrationState
from transition import Transition

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[H"


def parse_args() -> argparse.Namespace:
    """prototype引数を返す。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def render(state: MigrationState, transition: Transition | None) -> str:
    """全関連stateを一画面へ描画する。"""
    milestone = next_milestone(state)
    lines = [
        f"{BOLD}Issue #170 migration gate prototype{RESET}",
        "",
        f"{BOLD}Public CLI:{RESET} {state.public_cli}",
        f"{BOLD}Package version:{RESET} {state.package_version}",
        f"{BOLD}Legacy code present:{RESET} {state.legacy_code_present}",
        f"{BOLD}Legacy ADRs active:{RESET} {state.legacy_adrs_active}",
        f"{BOLD}Next Issue:{RESET} "
        + (f"{milestone.number} — {milestone.title}" if milestone else "none"),
        "",
        f"{BOLD}Implementation Issues{RESET}",
    ]
    for item in MILESTONES:
        if item.number in state.completed_issues:
            marker = "✓"
        elif item == milestone:
            marker = "→"
        else:
            marker = "·"
        pr_gate = "PR✓" if item.number in state.passed_pr_gates else "PR·"
        lines.append(f"  {marker} {item.number:02d} [{pr_gate}] {item.title}")

    lines.extend(("", f"{BOLD}Cutover evidence{RESET}"))
    for evidence in CUTOVER_EVIDENCE:
        marker = "✓" if evidence in state.evidence else "·"
        lines.append(f"  {marker} {evidence.value}")

    if transition is not None:
        result = "accepted" if transition.accepted else "blocked"
        lines.extend(("", f"{BOLD}Last action:{RESET} {result} — {transition.message}"))
        lines.extend(f"  ! {blocker}" for blocker in transition.blockers)

    lines.extend(
        (
            "",
            f"{BOLD}[p]{RESET}{DIM} pass PR gate  {RESET}"
            f"{BOLD}[e]{RESET}{DIM} record evidence  {RESET}"
            f"{BOLD}[n]{RESET}{DIM} complete next Issue{RESET}",
            f"{BOLD}[c]{RESET}{DIM} attempt cutover  {RESET}"
            f"{BOLD}[r]{RESET}{DIM} reset  {RESET}"
            f"{BOLD}[q]{RESET}{DIM} quit{RESET}",
        )
    )
    return "\n".join(lines)


def run_tui() -> None:
    """一行入力でstate machineを操作する。"""
    state = MigrationState()
    transition: Transition | None = None
    while True:
        print(CLEAR + render(state, transition), flush=True)
        key = sys.stdin.readline().strip().lower()
        if key == "q" or key == "":
            return
        if key == "p":
            transition = pass_next_pr_gate(state)
        elif key == "e":
            transition = record_next_evidence(state)
        elif key == "n":
            transition = complete_next_issue(state)
        elif key == "c":
            transition = attempt_cutover(state)
        elif key == "r":
            state = MigrationState()
            transition = Transition(state, True, "state reset")
            continue
        else:
            transition = Transition(state, False, f"unknown action: {key}")
        state = transition.state


def run_demo() -> None:
    """early rejectionとatomic cutoverを非対話で実演する。"""
    state = MigrationState()
    early = attempt_cutover(state)
    assert not early.accepted
    assert early.state == state

    while len(state.completed_issues) < 12:
        milestone = next_milestone(state)
        assert milestone is not None
        state = pass_next_pr_gate(state).state
        while True:
            result = complete_next_issue(state)
            if result.accepted:
                state = result.state
                break
            evidence_result = record_next_evidence(state)
            if not evidence_result.accepted:
                raise AssertionError(result.blockers)
            state = evidence_result.state

    state = pass_next_pr_gate(state).state
    final = attempt_cutover(state)
    assert final.accepted
    validate_state(final.state)
    print("early cutover: blocked")
    print(f"early blockers: {len(early.blockers)}")
    print("all Issues 1-12 and evidence: complete")
    print("final cutover: accepted")
    print("public CLI: video-set")
    print("package version: 2.0.0")
    print("legacy code present: false")
    print("legacy ADRs active: false")


def main() -> None:
    """選択されたprototype modeを実行する。"""
    if parse_args().demo:
        run_demo()
    else:
        run_tui()


if __name__ == "__main__":
    main()
