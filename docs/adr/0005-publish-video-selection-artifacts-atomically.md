# Publish Video Selection Artifacts Atomically

## Context

The future Video Set selector must publish artifacts that serve two different jobs:

- a human reviews selected images and their selection reasons in `report.md`;
- tools and selector diagnostics consume the complete contract in `report.json`.

The artifacts must identify the exact source frames after videos or output folders move, explain model-derived semantics separately from deterministic selection, retain enough provenance to diagnose cache and policy behavior, and avoid leaking host paths or raw model material.

This ADR defines the public artifact contract for the future Video Set selector. It does not implement that selector or change the existing screenshot-input flow.

## Artifact layout

Each successful run publishes one Output Folder:

```text
output/
├── images/
│   ├── 0001_opening_38f1a9c2e642.webp
│   └── 0002_exploration_6a4d812e6241.webp
├── report.json
└── report.md
```

`report.json` is the Canonical Selection Report and the only machine-readable source of truth. `report.md` is a deterministic human projection of the same validated report object. Markdown is not a parser contract, and JSON object key order has no meaning.

## Stable image identity and output names

A Frame Candidate ID is derived from the Video Fingerprint and exact Video Time with the versioned `video-fingerprint-video-time-sha256-v1` derivation. Its public form is:

```text
frm_<64 lowercase hexadecimal characters>
```

The complete ID is stored in `report.json`. It does not depend on source path, output path, Scene, or selection order.

A Selected Image Output Name has the form:

```text
<selection-index>_<scene-slug>_<frame-digest-prefix>.webp
```

- selection index is zero-padded to at least four digits and expands when needed;
- Scene Slug is the selected candidate's lowercase stable slug;
- the normal digest prefix is the first 12 characters after `frm_`;
- if two selected IDs share that prefix in one run, only those names expand to the complete 64-character digest.

The name is human navigation, not stable identity. Reordering selection or changing Scene may change the file name without changing the Frame Candidate ID.

## Selected image encoding

Video Set output uses one fixed v1 encoding:

- lossy WebP;
- quality 95;
- source frame width and height;
- embedded metadata stripped;
- no user-configurable format or quality in v1.

Three representative 3840x2160 frames from the supplied Video Set were compared: a title/logo frame, a dialogue frame, and an action frame with small HUD elements. WebP quality 95 produced files of approximately 0.90 to 1.25 MB compared with 9.4 to 15.5 MB PNG inputs. Whole-frame SSIM was 0.9933 to 0.9951 and average PSNR was 49.9 to 51.0 dB. The test frames remain temporary and are not committed.

Each selected output record includes relative path, SHA-256, width, height, and byte count. The run-level image contract records encoding, quality, size policy, metadata policy, ID derivation, and filename pattern.

## Source video contract

Both reports identify a source with:

- Video Order;
- Video Source ID, normally shortened to a 12-character fingerprint prefix;
- Report Source Path relative to the Video Input Folder.

If two source fingerprints share that prefix, only the colliding IDs expand to the complete 64-character fingerprint. Source IDs must remain unique within a report.

The relative path uses `/`, contains no `..`, and never acts as Video Identity. Absolute input, cache, staging, and output paths are omitted.

`report.json` additionally stores the complete 64-character whole-file SHA-256 Video Fingerprint and its algorithm. `report.md` omits that full fingerprint to keep the human report compact. Sharing `report.json` therefore allows exact recordings to be correlated by hash.

## Exact time contract

For a selected image or published near miss, `report.json` stores:

```json
{
  "source_pts": 189900,
  "origin_pts": 9000,
  "time_base": {
    "numerator": 1,
    "denominator": 90000
  },
  "offset_seconds": {
    "numerator": 201,
    "denominator": 100
  },
  "display": "00:00:02.010"
}
```

The reduced `offset_seconds` rational is authoritative. Float seconds and frame index are not part of the public contract. The display value uses unbounded hours, does not wrap at 24 hours, and rounds half-up to milliseconds.

Context Cue records retain their source timestamp basis. When container cue timing is not losslessly aligned to an integer source PTS grid, the record omits reconstructed PTS values and uses the authoritative reduced `offset_seconds` form instead.

`report.json` also keeps Candidate Moment ID and Timeline Segment ID for selected images. `report.md` shows only Video Order, Report Source Path, Video Source ID, and display time.

## Canonical JSON structure

The contract started at `game-screen-pick/report@1.0.0`. Version `2.0.0` added required conditional-coverage diagnostics and incremented the major version. The current producer is `game-screen-pick/report@2.1.0`; it adds optional Semantic Duplicate Group evidence and the `semantic_duplicate` rejection enum. Immutable `report-1.0.0.schema.json` and `report-2.0.0.schema.json` remain available beside the current schema:

| Field | Purpose |
|---|---|
| `schema` | report contract name and version |
| `run` | run ID, status, UTC timestamps, requested and selected counts, warnings |
| `artifacts` | relative artifact paths, image, publication, and projection contracts |
| `video_set` | Video Set identity, time policy, ordered sources, exact durations |
| `selection_summary` | candidate funnel, final similarity pass, shortfall, type targets and actuals, conditional-coverage counts and reallocation |
| `rejection_summary` | counts for every unselected reason |
| `selected` | complete selected-image records in selection order |
| `near_miss_publication` | deterministic bound and ordering policy |
| `near_misses` | bounded concrete rejected candidates |
| `context_cues` | metadata-only Report Context Evidence referenced by published candidates |
| `provenance` | selection, tool, runtime, model, contract, and Stage provenance |
| `privacy` | explicit inclusion and omission policy |

Arrays whose order carries meaning state that meaning in their field contract. Object key order does not carry meaning.

## Selected-image explanation boundary

Each selected record keeps these sources separate:

- `classification`: Scene, Scene Selection Role, Blog Image Type, Explanation Value, screen-text kind, Spoiler Risk, and optional spoiler evidence;
- `annotation.summary`: model-derived description of what the frame shows;
- `annotation.representative_frame_reason`: model-derived reason this frame represents the Candidate Moment;
- `selection.reason_codes`: deterministic selector facts;
- `selection.decision_explanation`: Japanese text rendered locally from reason codes and numeric selection components;
- optional `selection.semantic_group`: the deterministic privacy-safe group ID and basis when this image represents a Semantic Duplicate Group.

The selector explanation records its renderer version. It is not free-form model output. The report does not publish model confidence, chain of thought, reasoning traces, or raw responses.

`report.md` labels the three human explanations explicitly as model summary, model Representative Frame reason, and selector decision. This prevents a model semantic judgment from being mistaken for the final selection rule.

## Context and spoiler disclosure

Report Context Evidence contains only:

- Context Cue ID;
- Video Source ID;
- source kind;
- exact basis-specific time range;
- reliability diagnostics and policy;
- Context Cue Relevance through the selected annotation reference.

Raw embedded subtitle or speech-to-text content remains in processing cache only. Reports do not quote Context Cue text or generated screen-text transcriptions. Annotation summaries and selector explanations must not reproduce those quotations.

Spoiler Risk is always visible as `none`, `low`, `medium`, or `high`. For non-`none` risk, `report.json` stores a short Candidate Annotation evidence summary. `report.md` puts that summary in a closed `<details>` block. Screen text and Context Cue text are not quoted. Spoiler Penalty remains a separate deterministic selection value.

## Rejections and near misses

`rejection_summary` counts every unselected Blog Candidate by stable reason code.

The maximum number of concrete near misses in `report.json` is:

```text
min(total rejected, 100, max(20, requested image count * 2))
```

The set first includes at least one candidate for every rejection reason, then fills remaining slots by descending counterfactual Marginal Selection Utility with the normal deterministic tie-break. `report.md` shows at most the first 10. Full rejected-candidate detail remains in processing cache.

Near misses retain full Frame Candidate ID, exact source/time, classification, counterfactual utility components, reason code, and reason-specific references such as the blocking selected ID or nearest selected similarity. A `semantic_duplicate` near miss also retains the same Semantic Duplicate Group ID and privacy-safe basis as its blocking selected representative. The validator requires that reference to resolve to exactly one selected representative; raw names, model responses, and comparison prose are not published.

## Stage provenance

`report.json` records each Processing Stage with:

- status;
- complete Stage Fingerprint;
- upstream Stage Fingerprints;
- cache hits, misses, and recomputed count;
- duration, attempt count, validation failures, and applicable token counts;
- normalized result-affecting settings;
- references to tool, runtime, model digest, prompt, schema, and policy versions.

`report.md` shows abbreviated fingerprints, the main cache counts, duration, model/tool versions, and report schema version. Absolute paths, environment variables, credentials, prompt bodies, raw responses, and stack traces are excluded.

## Human Markdown projection

`report.md` uses this order:

1. run identity, status, and Selection Shortfall warning when applicable;
2. requested/selected/video/candidate summary and Blog Image Type mix;
3. gallery of selected images and explanations;
4. bounded near misses;
5. reproduction appendix with rejection counts, selector settings, score diagnostics, Stage summary, models, tools, and privacy omissions;
6. link to `report.json` for complete machine-readable detail.

Each selected WebP is embedded and linked to itself:

```markdown
[![01 — オープニング](images/0001_opening_38f1a9c2e642.webp)](images/0001_opening_38f1a9c2e642.webp)
```

The alt text contains selection index and Scene Display Name only. No separate thumbnail artifacts are generated.

## Schema evolution

The canonical schema name is `game-screen-pick/report`, starting at `1.0.0`, and follows Semantic Versioning:

- major: field removal, rename, type or semantic change, or a new required field;
- minor: optional field or enum value addition;
- patch: documentation, constraints, or producer fixes that do not change instance structure.

Readers reject an unsupported major version. For a supported major, the reader selects that major's retained baseline schema, validates its known required structure, and allows additional fields and string enum values introduced by later minor versions. Version-specific relationships and Markdown sections absent from an older major are not imposed on it. The producer continues to validate new output against the exact current schema. Historical schemas remain available, reports are immutable, and the producer does not rewrite or automatically migrate old reports.

## Atomic publication and failure

The Output Folder must be absent or empty at preflight. An existing empty target is rechecked and removed before processing owns the path.

All selected images, `report.json`, and `report.md` are written to a hidden sibling staging directory on the same filesystem. Before publication, the producer:

1. flushes staged files;
2. validates `report.json` against its exact schema;
3. verifies selected counts, image hashes, dimensions, and relative paths;
4. verifies that Markdown IDs, paths, counts, reasons, and links match the canonical report;
5. renames the staging directory to the final Output Folder once.

If atomic directory rename is unavailable, preflight fails. There is no file-by-file or completion-marker fallback.

A fatal Processing Stage, invalid report, renderer failure, or cross-artifact mismatch publishes no Output Folder. A Selection Shortfall is a successful `completed_with_warnings` run and atomically publishes the selected subset, warning, final similarity pass, and reason counts.

## Consequences

- Human review remains gallery-first while diagnostics stay available in an appendix and JSON.
- Reports remain relocatable because every artifact and source display path is relative.
- Stable IDs and exact time survive output reorder and path changes.
- A shared JSON report intentionally exposes whole-video fingerprints and hardware/runtime provenance but not host paths or raw text/model material.
- Fixed WebP encoding keeps v1 simple; adding configurable output formats is a future contract change.
- Atomic directory publication is stricter than partial fallback and may reject filesystems without the required rename semantics.

The interactive prototype and sample Canonical Selection Report are in [`prototypes/issue_168_report_contract`](../../prototypes/issue_168_report_contract/README.md).
