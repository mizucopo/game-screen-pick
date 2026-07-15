# 画像選定レポート

`run_20260714T093214Z_7f3a2c` · 2026-07-14 09:32:14 UTC

> [!WARNING]
> 6枚の要求に対して5枚を選択しました。残りの候補はvisual similarityまたはtitle上限のため採用できませんでした。

## Summary

| Requested | Selected | Videos | Duration | Candidate Moments |
|---:|---:|---:|---:|---:|
| 6 | 5 | 3 | 03:44:18.420 | 52 |

| Blog Image Type | Target | Actual |
|---|---:|---:|
| normal_gameplay | 4 | 2 |
| event | 2 | 1 |
| menu | 0 | 1 |
| title | 0 | 1 |

## Selected images

### 01 — 旅立ちを示すタイトル画面

[![01 — オープニング](images/0001_opening_38f1a9c2e642.webp)](images/0001_opening_38f1a9c2e642.webp)

`frm_38f1a9c2e642…` · `images/0001_opening_38f1a9c2e642.webp`

- **画像の説明（モデル）**: 作品名と旅立ちの雰囲気を示すタイトル画面。
- **代表frameの理由（モデル）**: logoと背景が同時に読みやすいframe。
- **採用理由（selector）**: 作品を識別できるtitle候補のうち最も品質が高く、title未採用bonusを得た。
- **Source**: Video 1 `01-opening.mp4` (`vid_a1e9c4a7019d`) · `00:00:02.010`
- **Classification**: `opening` · `ordinary` · `title` · spoiler `none`
- **Context**: `unavailable`
- **Utility**: 0.835

### 02 — 広い遺跡を探索する通常play

[![02 — 探索](images/0002_exploration_6a4d812e6241.webp)](images/0002_exploration_6a4d812e6241.webp)

`frm_6a4d812e6241…` · `images/0002_exploration_6a4d812e6241.webp`

- **画像の説明（モデル）**: 遺跡の広さと探索中のHUDが分かる通常play。
- **代表frameの理由（モデル）**: 遺跡の構造とplayer位置が最も明瞭なframe。
- **採用理由（selector）**: 高い画質と説明価値を持ち、normal_gameplayのcoverageを満たす。
- **Source**: Video 1 `01-opening.mp4` (`vid_a1e9c4a7019d`) · `00:12:34.567`
- **Classification**: `exploration` · `recurring_gameplay` · `normal_gameplay` · spoiler `none`
- **Context**: `weak` · `cue_0124d3` · embedded subtitle `00:12:32.400–00:12:36.800` · usable
- **Utility**: 0.915

### 03 — 次の目的地が示される会話event

[![03 — 会話](images/0003_conversation_b7206e55f3aa.webp)](images/0003_conversation_b7206e55f3aa.webp)

`frm_b7206e55f3aa…` · `images/0003_conversation_b7206e55f3aa.webp`

- **画像の説明（モデル）**: 仲間との会話で次の目的地が示されるevent。
- **代表frameの理由（モデル）**: 話者と目的地の背景が同時に分かるframe。
- **採用理由（selector）**: 説明価値highと強いContext Cue relevanceがevent coverageに寄与した。
- **Source**: Video 2 `02-forest.mp4` (`vid_c7bd10eb3220`) · `00:45:10.120`
- **Classification**: `conversation` · `cinematic` · `event` · spoiler `low`
- **Context**: `strong` · `cue_44a901` / `cue_44a902` · audio STT `00:45:07.804–00:45:13.588` · usable
- **Utility**: 0.881

<details>
<summary>Spoiler evidence（モデル）</summary>

次の目的地を示す軽微な進行情報。

</details>

### 04 — 装備構成が分かるmenu

[![04 — 装備](images/0004_equipment_18d0ab449cc7.webp)](images/0004_equipment_18d0ab449cc7.webp)

`frm_18d0ab449cc7…` · `images/0004_equipment_18d0ab449cc7.webp`

- **画像の説明（モデル）**: 装備の種類と構成が読み取れるmenu。
- **代表frameの理由（モデル）**: 項目とcharacter previewが明瞭なframe。
- **採用理由（selector）**: menuのsoft coverageを満たし、既選択画像との視覚重複が小さい。
- **Source**: Video 2 `02-forest.mp4` (`vid_c7bd10eb3220`) · `01:11:02.333`
- **Classification**: `equipment` · `ordinary` · `menu` · spoiler `none`
- **Context**: `none`
- **Utility**: 0.792

### 05 — 終盤の特徴的なboss戦

[![05 — 戦闘](images/0005_battle_d9c3f271dd12.webp)](images/0005_battle_d9c3f271dd12.webp)

`frm_d9c3f271dd12…` · `images/0005_battle_d9c3f271dd12.webp`

- **画像の説明（モデル）**: 終盤areaで固有bossと戦う通常play。
- **代表frameの理由（モデル）**: boss、player、battle HUDが同時に明瞭なframe。
- **採用理由（selector）**: 後半位置自体は減点せず、medium spoiler penalty後も高い品質と説明価値が残った。
- **Source**: Video 3 `03-citadel.mp4` (`vid_f4c22385e80e`) · `00:18:42.900`
- **Classification**: `battle` · `recurring_gameplay` · `normal_gameplay` · spoiler `medium`
- **Context**: `strong` · `cue_9bc1aa` · audio STT `00:18:40.500–00:18:44.200` · usable
- **Utility**: 0.863

<details>
<summary>Spoiler evidence（モデル）</summary>

固有bossと終盤固有areaが表示される。

</details>

## Near misses

| Candidate | Counterfactual utility | Not selected because |
|---|---:|---|
| `frm_10a83f09` 別のタイトル画面 | 0.824 | `title_limit` |
| `frm_74e2150b` 遺跡探索の近似frame | 0.902 | `visual_near_duplicate` (0.997) |
| `frm_885c4d13` 同じboss戦の直後 | 0.858 | `similarity_ceiling` (0.985 > 0.98) |

## Reproduction appendix

### Selection funnel

| Stage | Input | Kept | Main exclusion |
|---|---:|---:|---|
| Candidate Moment discovery | 52 | 52 | — |
| Frame refinement | 52 | 42 | `no_valid_frame` 10 |
| Candidate Annotation | 42 | 42 | failure 0 |
| Final selection | 42 | 5 | visual similarity 35, title limit 2 |

Requested 6 / selected 5。すべてのCandidate Momentを使い切ったためSelection Shortfallとして正常終了した。

### Decision ledger

| Frame ID | Decision | Type | Base | Coverage | Spoiler | Temporal | Marginal |
|---|---|---|---:|---:|---:|---:|---:|
| `frm_38f1a9c2…` | selected 01 | title | 0.785 | +0.050 | 0 | 0 | **0.835** |
| `frm_6a4d812e…` | selected 02 | normal_gameplay | 0.815 | +0.100 | 0 | 0 | **0.915** |
| `frm_b7206e55…` | selected 03 | event | 0.791 | +0.100 | -0.010 | 0 | **0.881** |
| `frm_18d0ab44…` | selected 04 | menu | 0.710 | +0.100 | 0 | -0.018 | **0.792** |
| `frm_d9c3f271…` | selected 05 | normal_gameplay | 0.803 | +0.100 | -0.040 | 0 | **0.863** |
| `frm_10a83f09…` | `title_limit` | title | 0.774 | +0.050 | 0 | 0 | 0.824 |
| `frm_74e2150b…` | `visual_near_duplicate` | normal_gameplay | 0.802 | +0.100 | 0 | 0 | 0.902 |
| `frm_885c4d13…` | `similarity_ceiling` | normal_gameplay | 0.798 | +0.100 | -0.040 | 0 | 0.858 |

### Stage provenance

| Stage | Fingerprint | Cache | Duration | Contract |
|---|---|---|---:|---|
| `video_scan` | `stg_4118…` | 3/3 hit | 0.42s | `video-scan-v1` |
| `context_cues` | `stg_dccc…` | 2/3 hit | 4.81s | `context-cue-v1` |
| `scene_catalog` | `stg_3112…` | hit | 0.03s | prompt/schema `v1` |
| `candidate_annotation` | `stg_7624…` | 38/42 hit | 9.31s | prompt/schema `v1` |
| `final_selection` | `stg_d973…` | miss | 0.08s | `video-set-selection-v1` |

### Model and tool provenance

- Report schema: `game-screen-pick/report@1.0.0`
- Selection policy: `video-set-selection-v1`
- Spoiler Sensitivity: `medium`
- Ollama: `0.31.2`
- Scene Catalog / Candidate Annotation: `qwen3-vl:8b-instruct`, digest `sha256:7b8f…`
- STT: `faster-whisper 1.2.1`, model snapshot `0a363e9…`
- FFmpeg / ffprobe: `6.1.1`

### Deliberately omitted

- absolute input/cache/output paths
- environment variables and credentials
- prompt bodies, model reasoning traces, raw model responses, and stack traces
- generated screen-text quotations
- raw Context Cue text（処理cacheだけに保持）

完全なscore内訳、PTS/time base、stage provenanceは[`report.json`](report.sample.json)を参照する。
