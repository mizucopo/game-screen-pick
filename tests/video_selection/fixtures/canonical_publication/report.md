# 画像選定レポート

`run_20260716T120000Z_fixture` · 2026-07-16T12:00:00Z

> [!WARNING]
> 要求2枚に対して1枚を選択しました。

## Summary

| Requested | Selected | Videos | Duration | Candidate Moments |
|---:|---:|---:|---:|---:|
| 2 | 1 | 1 | 00:00:10.000 | 2 |

| Blog Image Type | Target | Actual |
|---|---:|---:|
| normal_gameplay | 1 | 1 |
| event | 1 | 0 |
| menu | 0 | 0 |
| title | 0 | 0 |
| other | 0 | 0 |

Conditional coverage: `applies=false` (requested >= 10)

| Coverage facet | Eligible | Minimum | Actual | Reallocated |
|---|---:|---:|---:|---|
| ordinary_combat | 0 | 0 | 0 | `false` |
| event | 0 | 0 | 0 | `false` |

## Selected images

### 01 — 遺跡の広さとHUDが分かる通常play。

[![01 — 探索](images/0001_test-scene_aaaaaaaaaaaa.webp)](images/0001_test-scene_aaaaaaaaaaaa.webp)

`frm_aaaaaaaaaaaa…` · `images/0001_test-scene_aaaaaaaaaaaa.webp`

- **画像の説明（model）**: 遺跡の広さとHUDが分かる通常play。
- **Representative Frameの理由（model）**: 構図と情報量が最も明瞭なframe。
- **採用理由（selector）**: 高い画質、高い説明価値、normal_gameplay coverageにより選択された。
- **Reason codes**: `high_quality, high_explanation_value, normal_gameplay_coverage`
- **Source**: Video 1 `chapter-01.mkv` (`vid_9b84276662ce`) · `00:00:01.500`
- **Classification**: `test-scene` · `recurring_gameplay` · `normal_gameplay` · spoiler `none`
- **Context**: `strong` · `cue_cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc`
- **Utility**: 0.910000

## Near misses

| Candidate | Counterfactual utility | Not selected because |
|---|---:|---|
| `frm_bbbbbbbb…` 似た構図の遺跡探索frame。 | 0.820000 | `similarity_ceiling (0.985)` |

## Reproduction appendix

### Selection funnel

| Stage | Input | Kept |
|---|---:|---:|
| Candidate Moment discovery | 2 | 2 |
| Frame refinement | 2 | 2 |
| Candidate Annotation | 2 | 2 |
| Final selection | 2 | 1 |

Requested 2 / selected 1。 all_candidate_moments_exhausted=`true`。

### Decision ledger

| Frame ID | Decision | Type | Base | Coverage | Spoiler | Temporal | Marginal | Reasons |
|---|---|---|---:|---:|---:|---:|---:|---|
| `frm_aaaaaaaaaaaa…` | selected 01 | normal_gameplay | 0.810000 | 0.100000 | 0.000000 | 0.000000 | **0.910000** | `high_quality, high_explanation_value, normal_gameplay_coverage` |
| `frm_bbbbbbbbbbbb…` | `similarity_ceiling` | normal_gameplay | 0.720000 | 0.100000 | 0.000000 | 0.000000 | 0.820000 | — |

### Stage provenance

| Stage | Fingerprint | Cache | Duration | Contracts |
|---|---|---|---:|---|
| `final_selection` | `stg_66666666…` | 0 hit / 1 miss | 0.080s | `video_set_selection_policy` |

### Model and tool provenance

- Report schema: `game-screen-pick/report@2.2.0`
- candidate_annotation: `qwen3-vl:8b-instruct` @ `ollama:sha256:81bf…` (not_requested)
- scene_catalog: `qwen3-vl:8b-instruct` @ `ollama:sha256:81bf…` (not_requested)
- speech_to_text: `dropbox-dash/faster-whisper-large-v3-turbo` @ `hf:b866f3b8500eb44…` (not_requested)
- ffmpeg: `6.1.1`
- ffprobe: `6.1.1`
- ollama: `0.31.2`
- faster_whisper: `1.2.1`
- ctranslate2: `4.8.1`

### Deliberately omitted

- absolute input/cache/output paths
- environment variables and credentials
- prompt bodies, model reasoning traces, raw model responses, and stack traces
- generated screen-text quotations
- raw Context Cue text（processing cacheだけに保持）

完全なscore内訳、PTS/time base、Stage provenanceは[`report.json`](report.json)を参照する。
