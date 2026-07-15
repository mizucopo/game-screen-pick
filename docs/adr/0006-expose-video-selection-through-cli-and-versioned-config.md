# Expose Video Selection Through CLI and Versioned Config

## Context

The future Video Set selector needs one public operating contract for the processing, cache, selection, and report decisions in ADRs 0004 and 0005 and Issues 162–168. The current screenshot-input CLI mixes run intent, model selection, resource tuning, and legacy names. Carrying that surface forward would obscure which values belong to a run, which affect reusable Processing Stages, and how an operator can reproduce a result.

The reference runtime is Windows 11 Pro with WSL2 Ubuntu, system FFmpeg, CUDA speech-to-text, and Windows native Ollama reached explicitly from WSL2. Model tags can move over time, but requiring users to copy content hashes into TOML would make normal upgrades cumbersome. Cache reuse still needs the exact artifact identity that actually ran.

This ADR defines the public CLI, TOML, model-resolution, progress, and error contract for the future Video Set selector. It does not implement the selector or change the existing screenshot-input command.

## One command and two positional paths

The public command has no subcommands:

```text
game-screen-pick [OPTIONS] <VIDEO_INPUT_FOLDER> <OUTPUT_FOLDER>
```

The two positional paths define the discovery/cache owner and the atomic publication target. The CLI exposes only paths, per-run Selection Intent, and explicit operational actions:

| Option | Contract |
|---|---|
| `-n, --image-count INTEGER` | requested output count, built-in default 100 |
| `-r, --recursive` | explicitly enable recursive discovery |
| `--no-recursive` | explicitly disable recursive discovery and override TOML |
| `--config PATH` | read exactly this TOML file |
| `--scene-hint TEXT` | per-run Scene Hint |
| `--spoiler-sensitivity low\|medium\|high` | soft Spoiler Penalty profile, default `medium` |
| `--similarity-threshold FLOAT` | initial visual-similarity ceiling, default 0.72 |
| `--ollama-host URL` | explicit Ollama endpoint override |
| `--reset-cache` | reset this Video Input Folder's processing cache after safe preflight |
| `--debug` | include a sanitized stack trace in failures |

Model, extraction, context, and parallelism settings belong to TOML. There is no host/server autodiscovery and no runtime switch between Windows and WSL Ollama.

The screenshot CLI is not a compatibility surface for this replacement. In particular, `--num`, `--similarity`, and `--ollama-scene-hint` do not remain as aliases. Issue 170 owns the implementation migration and removal boundary.

## Effective Configuration

Configuration is resolved independently for every key in this exact order:

```text
explicit CLI > explicit TOML > environment > built-in default
```

Only `OLLAMA_HOST` is a public environment variable. An explicit `[ollama].host` therefore overrides `OLLAMA_HOST`. Boolean CLI values preserve an unspecified state so that an absent flag does not accidentally override TOML; `--recursive` and `--no-recursive` provide both explicit values.

TOML is read only when `--config` is supplied. There is no current-directory, home-directory, or input-folder discovery. The document must contain `config_version = "1.0.0"`. Unknown sections, keys, types, enum values, ranges, or config versions are fatal usage/config errors before input discovery, network access, cache reset, or output creation.

The complete v1 schema and built-in defaults are published in [`docs/configuration.md`](../configuration.md) and [`docs/examples/video-selection.toml`](../examples/video-selection.toml). The main groups are:

- `[input]` and `[selection]` for recursive discovery and run intent;
- `[frame_extraction]` and `[candidate_moments]` for local Video Stage capacity;
- `[context]` and `[speech_to_text]` for stream choice and STT operation;
- `[ollama]` for endpoint, timeout, and parallel requests;
- `[models]` and its three role tables for model lifecycle and operation-specific settings.

Word timestamps, exact Video Time mapping, overlap ownership, Context Cue reliability, retry semantics, schema validation, selection weights, and output encoding remain versioned domain/policy contracts rather than switches that can disable correctness.

## Independent model roles

There are three model roles:

| Role | Default | Runtime boundary |
|---|---|---|
| Scene Catalog | `qwen3-vl:8b-instruct`, `num_ctx = 32768` | Ollama vision + structured output |
| Candidate Annotation | `qwen3-vl:8b-instruct`, `num_ctx = 32768` | Ollama vision + structured output |
| Speech to text | `dropbox-dash/faster-whisper-large-v3-turbo`, CUDA, float16, beam 5 | faster-whisper + CTranslate2 |

Scene Catalog and Candidate Annotation may use the same default model, but their settings and Stage Fingerprints are independent. The STT model and execution profile are configurable; Issue 166's tested faster-whisper/CTranslate2 path is a default profile and minimum runtime boundary, not a fixed model revision.

No role automatically falls back to a different model. A configured model either resolves and passes its capability checks or that operation fails.

## Runtime-resolved model identity

TOML contains model names, not `expected_digest`, commit SHA, or `revision` hashes. Before a model-dependent Stage begins, the runtime:

1. records any complete local identity;
2. applies the configured Model Upgrade Policy;
3. resolves the complete Ollama manifest digest or Hugging Face commit SHA;
4. proves the artifact is complete and loadable with the required capability;
5. freezes that Resolved Model Identity for the rest of the run.

The complete identity is stored in the Stage Fingerprint, cache manifest, and `report.json`; `report.md` shows only an abbreviated identity. Configuration name, local identity before update, update result, and execution identity remain separate provenance fields.

A changed identity invalidates only model-dependent Stages. Unrelated Video Stage artifacts remain reusable, and Completed Stages for old identities coexist with new ones. If a pull or metadata check resolves to the same identity, cache reuse is preserved. Model-update time and the `auto_upgrade` setting itself are diagnostics, not semantic Stage Fingerprint inputs.

## Model Upgrade Policy

`[models].auto_upgrade` applies to all three roles and defaults to `true`:

- With `true`, each distinct Ollama tag is synchronized once through `/api/pull`; each Hugging Face model resolves remote `main` once and downloads the immutable resolved snapshot.
- With `false`, a complete local artifact is used without a network update check. A missing artifact is still downloaded automatically as bootstrap.
- If an update attempt is unavailable because of offline operation, timeout, registry/Hub failure, or authorization, a complete and loadable local artifact is used with a warning and `update_status = "unavailable"`.
- If no usable local artifact exists, bootstrap or update failure is fatal.
- Partial downloads are never execution identities, and `--reset-cache` never removes a model store.

Ollama has no documented read-only remote-digest check, so its `auto_upgrade = true` behavior is a mutating pull/sync, not notification-only polling. When both Ollama roles use the same tag, it is pulled once and the same post-pull identity is frozen for both.

Model generation uses temperature zero, but this is not treated as bit-for-bit determinism. The first response that passes schema and domain validation is atomically cached. A matching Stage Fingerprint reuses it without majority sampling or regeneration. Resetting processing cache may therefore produce a semantically valid result with different wording.

## Runtime and preflight

The v1 project floors are:

- Python 3.13;
- FFmpeg and ffprobe 6.1.1 from the same build;
- Ollama server 0.31.2;
- faster-whisper 1.2.1;
- CTranslate2 4.8.1.

Newer versions are allowed, and the actual version is provenance and a relevant Stage Fingerprint input. Version comparison does not replace capability probes. Preflight separately validates configuration, paths, Video Set snapshot, cache write access and lock, FFmpeg/ffprobe operations and codecs, stream selection, Ollama reachability/version/model capabilities/context/structured output, and configured STT device/model load.

The reference deployment is WSL2 connecting to Windows native Ollama by an explicit URL. A WSL gateway address is not a built-in default because it can change. Mirrored networking with localhost is preferred. Binding Ollama to all interfaces requires a Windows Firewall/network-profile restriction because it may expose the API beyond WSL.

## Resume and reset

Resume is automatic and has no `--resume` flag. Only an atomic Completed Stage with a valid manifest and artifacts is reused. An interrupted Stage is rerun while completed upstream Stages and other Video Stages remain available.

`--reset-cache` removes only `<VIDEO_INPUT_FOLDER>/.game-screen-pick/cache/` after all non-destructive validation and the non-waiting input lock succeed. It does not modify Output Folder or either model store. Automatic retention, size limits, and pruning remain outside v1.

## Progress and errors

Progress is Stage-local rather than a fabricated whole-run percentage. It reports Stage index/name, Video Order/count/path, processed and currently known totals, cache hit/miss/recompute, elapsed time, reliable Stage ETA, model-download bytes/percent, and stable Stage events. TTY output may update in place; redirected or CI output is line-oriented. Long external work emits a Stage event or heartbeat instead of remaining silent.

Progress, warnings, and errors go to stderr. V1 does not stream a machine-readable report to stdout.

Exit codes are:

| Code | Meaning |
|---:|---|
| 0 | success, including Selection Shortfall as `completed_with_warnings` |
| 1 | operational, preflight, Stage, or publication failure |
| 2 | CLI or TOML usage/validation error |
| 130 | Ctrl+C |

Every failure includes a stable reason code, safe observed values, remediation, and cache-reuse guidance. Stack traces are hidden normally and sanitized under `--debug`. Credentials, environment dumps, absolute paths, prompt bodies, raw model responses, and raw Context Cue text are never printed.

A fatal failure publishes no Output Folder. Selection Shortfall remains the only warning-success path and atomically publishes its valid subset and report.

## Documentation boundary

README provides the future quickstart, runtime summary, implementation-status warning, and links. Detailed contracts are split into:

- [`docs/video-input.md`](../video-input.md) for CLI and discovery;
- [`docs/configuration.md`](../configuration.md) for schema, precedence, and models;
- [`docs/operations.md`](../operations.md) for runtime, cache, progress, and errors;
- [`docs/report.md`](../report.md) for public artifacts;
- [`docs/research/issue-169-runtime-model-contract.md`](../research/issue-169-runtime-model-contract.md) for primary-source evidence and measured capability.

These documents describe a future interface until Issue 170 turns the design into implementation tickets. They must not imply that the current screenshot selector accepts the new command.

## Consequences

- Common runs stay short while reproducibility-sensitive settings remain versioned and reviewable in TOML.
- Exact precedence and strict unknown-key rejection prevent silent configuration drift.
- Users do not maintain hash pins, but reports and caches still bind to the exact model artifact that ran.
- Default automatic upgrades follow model improvements while a valid local model keeps offline runs usable with an explicit warning.
- A mutable configured tag may intentionally cause model-dependent recomputation on a later run.
- Automatic bootstrap means `auto_upgrade = false` is not a permanent offline guarantee when a configured model is absent.
- Stage-local progress and reason-coded failures remain truthful across a long, variable-cost Video Set.
- The breaking screenshot-to-video migration is explicit and remains implementation work, not an undocumented compatibility shim.

The 24-image Scene Catalog capability probe is documented in [`prototypes/issue_169_runtime_contract`](../../prototypes/issue_169_runtime_contract/README.md).
