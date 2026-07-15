# PROTOTYPE — Video selector migration gate

This throwaway logic prototype answers one question for Issue #170:

> Can the 13-PR migration remain internally incremental while making it impossible to expose the Video Set CLI before every implementation, test, target-performance, and human-quality gate has passed?

The prototype keeps all state in memory. It models the ordered implementation Issues, their PR gates, cutover evidence, package version, public CLI, legacy code, and ADR status. Filesystem, production code, GitHub Issues, and external runtimes are not changed.

Run from the repository root:

```bash
uv run task prototype-issue-170
```

Controls:

- `p`: pass the PR gate for the next Issue
- `e`: record the next missing acceptance evidence
- `n`: complete the next Issue when its gates are satisfied
- `c`: attempt the public cutover immediately
- `r`: reset all in-memory state
- `q`: quit

An early cutover must list its blockers and leave the public screenshot CLI, package version, legacy code, and ADR state unchanged. A successful Issue 13 transition must change all four atomically.

For a non-interactive invariant demonstration:

```bash
uv run python prototypes/issue_170_migration_acceptance/prototype.py --demo
```

## Verdict

Confirmed by the non-interactive demonstration:

- an early cutover is rejected without mutating public state;
- implementation Issues cannot complete out of order or before their PR and
  evidence gates;
- the public CLI, package version, legacy code, and legacy ADR state change in
  one Issue 13 transition only after Issues 1–12 and every evidence gate pass.

The state model therefore supports an incremental internal migration with one
atomic public cutover. This prototype validates gate ordering only; the durable
acceptance requirements and traceability live in the Issue #170 design docs.
