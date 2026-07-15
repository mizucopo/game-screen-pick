"""PROTOTYPE: Video selector migrationのcutover gateを表現する。"""

from __future__ import annotations

from dataclasses import replace

from evidence import Evidence
from migration_state import MigrationState
from milestone import Milestone
from transition import Transition

MILESTONES = (
    Milestone(1, "fake walking skeleton"),
    Milestone(2, "versioned config and internal CLI adapter"),
    Milestone(3, "Video Set identity, lock, cache, legacy cleanup"),
    Milestone(4, "FFmpeg MediaRuntime and generated fixtures"),
    Milestone(5, "frame extraction, timeline, neutral analysis"),
    Milestone(6, "Context Cue and SpeechRuntime"),
    Milestone(7, "ModelRuntime"),
    Milestone(8, "VisionRuntime"),
    Milestone(9, "deterministic Video Set selector"),
    Milestone(10, "WebP, canonical report, atomic publication"),
    Milestone(11, "progress, reason codes, interruption resume"),
    Milestone(12, "target acceptance and performance"),
    Milestone(13, "public v2 cutover and legacy deletion"),
)

CUTOVER_EVIDENCE = tuple(Evidence)
ISSUE_EVIDENCE = {
    1: frozenset({Evidence.FAKE_E2E}),
    4: frozenset({Evidence.FFMPEG_INTEGRATION}),
    11: frozenset({Evidence.INTERRUPTION_MATRIX}),
    12: frozenset(CUTOVER_EVIDENCE),
}


def next_milestone(state: MigrationState) -> Milestone | None:
    """次の未完了milestoneを返す。"""
    return next(
        (
            milestone
            for milestone in MILESTONES
            if milestone.number not in state.completed_issues
        ),
        None,
    )


def pass_next_pr_gate(state: MigrationState) -> Transition:
    """次IssueのPR gateを成功にする。"""
    milestone = next_milestone(state)
    if milestone is None:
        return Transition(state, False, "all Issues are already complete")
    updated = replace(
        state,
        passed_pr_gates=state.passed_pr_gates | {milestone.number},
    )
    return Transition(updated, True, f"Issue {milestone.number} PR gate passed")


def record_next_evidence(state: MigrationState) -> Transition:
    """次の不足evidenceを記録する。"""
    missing = next(
        (evidence for evidence in CUTOVER_EVIDENCE if evidence not in state.evidence),
        None,
    )
    if missing is None:
        return Transition(state, False, "all acceptance evidence is recorded")
    updated = replace(state, evidence=state.evidence | {missing})
    return Transition(updated, True, f"recorded: {missing.value}")


def complete_next_issue(state: MigrationState) -> Transition:
    """必要gateを満たした次Issueを完了する。"""
    milestone = next_milestone(state)
    if milestone is None:
        return Transition(state, False, "all Issues are already complete")
    if milestone.number == 13:
        return attempt_cutover(state)

    blockers = _issue_blockers(state, milestone.number)
    if blockers:
        return Transition(
            state,
            False,
            f"Issue {milestone.number} cannot complete",
            blockers,
        )

    updated = replace(
        state,
        completed_issues=state.completed_issues | {milestone.number},
    )
    validate_state(updated)
    return Transition(updated, True, f"Issue {milestone.number} completed")


def attempt_cutover(state: MigrationState) -> Transition:
    """全gate通過時だけpublic cutoverをatomicに行う。"""
    blockers = _cutover_blockers(state)
    if blockers:
        return Transition(state, False, "public cutover blocked", blockers)

    updated = replace(
        state,
        completed_issues=state.completed_issues | {13},
        public_cli="video-set",
        package_version="2.0.0",
        legacy_code_present=False,
        legacy_adrs_active=False,
    )
    validate_state(updated)
    return Transition(updated, True, "public v2 cutover completed atomically")


def _issue_blockers(state: MigrationState, issue_number: int) -> tuple[str, ...]:
    blockers: list[str] = []
    expected_previous = set(range(1, issue_number))
    missing_previous = sorted(expected_previous - state.completed_issues)
    if missing_previous:
        blockers.append(f"previous Issues incomplete: {missing_previous}")
    if issue_number not in state.passed_pr_gates:
        blockers.append(f"Issue {issue_number} PR gate not passed")
    required = ISSUE_EVIDENCE.get(issue_number, frozenset())
    missing_evidence = [
        evidence.value
        for evidence in CUTOVER_EVIDENCE
        if evidence in required - state.evidence
    ]
    if missing_evidence:
        blockers.append(f"missing evidence: {missing_evidence}")
    return tuple(blockers)


def _cutover_blockers(state: MigrationState) -> tuple[str, ...]:
    blockers = list(_issue_blockers(state, 13))
    missing_evidence = [
        evidence.value
        for evidence in CUTOVER_EVIDENCE
        if evidence not in state.evidence
    ]
    if missing_evidence:
        blockers.append(f"cutover evidence missing: {missing_evidence}")
    return tuple(blockers)


def validate_state(state: MigrationState) -> None:
    """partial public cutoverが存在しないことを検証する。"""
    public_state = (
        state.public_cli,
        state.package_version,
        state.legacy_code_present,
        state.legacy_adrs_active,
    )
    screenshot_state = ("screenshot", "1.5.2", True, True)
    video_state = ("video-set", "2.0.0", False, False)
    if public_state not in {screenshot_state, video_state}:
        raise AssertionError(f"partial public cutover: {public_state}")
    if public_state == video_state and state.completed_issues != set(range(1, 14)):
        raise AssertionError("video CLI was exposed before every Issue completed")
