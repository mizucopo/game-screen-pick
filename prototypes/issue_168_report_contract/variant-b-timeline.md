# Video Set選定タイムライン

`run_20260714T093214Z_7f3a2c` · Video Set `vset_59ac118e`

> [!WARNING]
> Selection Shortfall: requested 6 / selected 5

## Video Set

| Order | Source video | Duration | Selected |
|---:|---|---:|---:|
| 1 | `01-opening.mp4` (`vid_a1e9c4`) | 00:58:04.200 | 2 |
| 2 | `02-forest.mp4` (`vid_c7bd10`) | 01:34:11.700 | 2 |
| 3 | `03-citadel.mp4` (`vid_f4c223`) | 01:12:02.520 | 1 |

## Video 1 — 01-opening.mp4

```text
00:00:00 ├●────────────●────────────────────────────────────────┤ 00:58:04
          01           02
```

- **01 · 00:00:02.010** — 旅立ちを示すタイトル画面 (`title`, spoiler `none`)
- **02 · 00:12:34.567** — 広い遺跡を探索する通常play (`normal_gameplay`, spoiler `none`)

## Video 2 — 02-forest.mp4

```text
00:00:00 ├────────────────────────●──────────────●──────────────┤ 01:34:11
                                  03             04
```

- **03 · 00:45:10.120** — 次の目的地が示される会話event (`event`, spoiler `low`)
- **04 · 01:11:02.333** — 装備構成が分かるmenu (`menu`, spoiler `none`)

## Video 3 — 03-citadel.mp4

```text
00:00:00 ├──────────────●───────────────────────────────────────┤ 01:12:02
                        05
```

- **05 · 00:18:42.900** — 終盤の特徴的なboss戦 (`normal_gameplay`, spoiler `medium`)

## Video Set Progress

```text
0% ├●──●──────────────────●──────●────────────────●────┤ 100%
    01 02                 03     04               05
```

後半位置自体への減点はない。各選択の正確な`source_pts`、`origin_pts`、`time_base`とTemporal Diversity Penaltyは[`report.json`](report.sample.json)を参照する。

## Not selected near these positions

- Video 1 `00:12:35.033` — `frm_74e2150b`: selected 02とのsimilarity 0.997
- Video 3 `00:18:43.367` — `frm_885c4d13`: final ceiling 0.98に対してsimilarity 0.985
