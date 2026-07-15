# Selector監査レポート

`run_20260714T093214Z_7f3a2c` · status `completed_with_warnings`

## Selection funnel

| Stage | Input | Kept | Main exclusion |
|---|---:|---:|---|
| Candidate Moment discovery | 52 | 52 | — |
| Frame refinement | 52 | 42 | `no_valid_frame` 10 |
| Candidate Annotation | 42 | 42 | failure 0 |
| Final selection | 42 | 5 | similarity 35, title limit 2 |

Requested 6 / selected 5。すべてのCandidate Momentを使い切ったためSelection Shortfallとして正常終了した。

## Decision ledger

| Frame ID | Decision | Type | Base | Coverage | Spoiler | Temporal | Marginal |
|---|---|---|---:|---:|---:|---:|---:|
| `frm_38f1a9c2` | selected 01 | title | 0.785 | +0.050 | 0 | 0 | **0.835** |
| `frm_6a4d812e` | selected 02 | normal_gameplay | 0.815 | +0.100 | 0 | 0 | **0.915** |
| `frm_b7206e55` | selected 03 | event | 0.791 | +0.100 | -0.010 | 0 | **0.881** |
| `frm_18d0ab44` | selected 04 | menu | 0.710 | +0.100 | 0 | -0.018 | **0.792** |
| `frm_d9c3f271` | selected 05 | normal_gameplay | 0.803 | +0.100 | -0.040 | 0 | **0.863** |
| `frm_10a83f09` | `title_limit` | title | 0.774 | +0.050 | 0 | 0 | 0.824 |
| `frm_74e2150b` | `visual_near_duplicate` | normal_gameplay | 0.802 | +0.100 | 0 | 0 | 0.902 |
| `frm_885c4d13` | `similarity_ceiling` | normal_gameplay | 0.798 | +0.100 | -0.040 | 0 | 0.858 |

## Stage provenance

| Stage | Fingerprint | Cache | Duration | Contract |
|---|---|---|---:|---|
| `video_scan` | `stg_4118…` | 3/3 hit | 0.42s | `video-scan-v1` |
| `context_cues` | `stg_dccc…` | 2/3 hit | 4.81s | `context-cue-v1` |
| `scene_catalog` | `stg_3112…` | hit | 0.03s | prompt/schema `v1` |
| `candidate_annotation` | `stg_7624…` | 38/42 hit | 9.31s | prompt/schema `v1` |
| `final_selection` | `stg_d973…` | miss | 0.08s | `video-set-selection-v1` |

## Model and tool provenance

- Ollama `0.31.2`
- Scene Catalog / Candidate Annotation: `qwen3-vl:8b-instruct`, digest `sha256:7b8f…`
- STT: `faster-whisper 1.2.1`, model snapshot `0a363e9…`
- FFmpeg `6.1.1`, ffprobe `6.1.1`
- Report schema `game-screen-pick/report@1.0.0`

## Deliberately omitted

- absolute input/cache/output paths
- model reasoning trace
- generated screen-text quotation
- raw model response
- raw Context Cue text（処理cacheだけに保持）

完全なmachine-readable ledgerは[`report.json`](report.sample.json)を参照する。
