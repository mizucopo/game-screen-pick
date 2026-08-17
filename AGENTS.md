# Repository guidance

## Work boundaries

- Do not make implementation changes directly on `main`.
- Track implementation changes in GitHub Issues and use a corresponding non-`main` branch.
- Do not weaken quality checks or test configuration to make a failing change pass.

## Project context

- For issue creation and management, follow `docs/agents/issue-tracker.md`.
- For issue triage, use the label mapping in `docs/agents/triage-labels.md`.
- Before changing domain terminology or architecture, follow `docs/agents/domain.md` and its referenced context and ADRs.

## Template updates

- Generated configuration and source files remain Copier-managed.
- Update from `repo-template/main` with `copier update --trust --defaults --vcs-ref HEAD` on a clean non-`main` branch.
- Review the complete diff and resolve every Copier conflict before running the repository quality gate.
- Do not use `copier recopy` for routine updates.

## Python quality check

```bash
uv run task check
```

Apply automatic Ruff fixes separately with `uv run task fix`, then rerun the non-mutating check.
