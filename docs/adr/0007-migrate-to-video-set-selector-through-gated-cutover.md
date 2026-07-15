# Migrate to the Video Set Selector Through a Gated Cutover

## Status

Accepted. Supersedes ADR 0001, ADR 0002, and ADR 0003 for the v2.0 product.

## Context

ADRs 0004〜0006 define a video-only selector whose input identity, staged cache,
semantic evaluation, deterministic selection, artifacts, configuration, and runtime
contract differ from the current screenshot selector. Exposing those parts one at a
time would create a temporary public interface that neither design intends to support.
A single large rewrite, however, would delay integration evidence until every external
runtime and domain stage had changed.

The migration also needs evidence from real FFmpeg in CI and from the supported Windows
11/WSL2/RTX target. The full 50-hour input is too costly for every PR, while fake-only
tests cannot validate timestamps, codecs, runtime identity, GPU use, or selection quality.

## Decision

Implement the video pipeline through 12 ordered, always-green internal slices, then use
a thirteenth PR for one public cutover. Before that final PR, the installed CLI and
documentation continue to expose the screenshot selector. Internal adapters and the
target acceptance harness do not become compatibility surfaces.

The final PR atomically changes the public CLI to Video Set input, sets package version
2.0.0, removes all screenshot-specific code and compatibility aliases, updates README,
and leaves ADRs 0001〜0003 as superseded history. Recognized legacy processing caches
are deleted after the input lock is acquired; no cache migration layer is retained.

Verification is split into three levels: external-free unit/contract/fake E2E on every
PR, generated-fixture real-FFmpeg integration as a required PR check, and real
Ollama/STT target acceptance before cutover/release and after relevant runtime or
performance changes. A versioned acceptance record, explicit performance budgets, a
human quality gate, and a traceability matrix are cutover evidence.

The complete slices, budgets, fault matrix, privacy contract, and requirement mapping
are defined in [`docs/migration-acceptance.md`](../migration-acceptance.md). The gate
ordering was checked with the throwaway
[`issue_170_migration_acceptance` prototype](../../prototypes/issue_170_migration_acceptance/README.md).

## Consequences

- Reviewable PRs can integrate domain and runtime seams without exposing an unstable CLI.
- Public migration is intentionally breaking and has no screenshot compatibility mode.
- Real target evidence is mandatory but kept out of normal PR latency.
- The final cutover cannot proceed when any implementation, performance, privacy, or
  human-quality requirement is orphaned or missing evidence.
- v1 ADRs remain useful history, while ADRs 0004〜0007 are the active v2.0 decisions.

