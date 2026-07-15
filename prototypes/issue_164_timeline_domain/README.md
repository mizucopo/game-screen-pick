# PROTOTYPE — timeline segmentとcandidate momentのドメインモデルを確定する

このprototypeは本番実装ではありません。次の問いを具体例で確認するための一次資料です。

> Video Source、Video Time、Timeline Segment、Candidate Moment、Frame Candidateのidentityと関係は、VFR、segment境界、重なるrefinement window、複数動画でも曖昧にならないか。

## 実行

```bash
uv run python prototypes/issue_164_timeline_domain/prototype.py
```

`[n]` と `[p]` でcaseを切り替え、`[q]` で終了します。全caseをJSONで一度に表示する場合は次を実行します。

```bash
uv run python prototypes/issue_164_timeline_domain/prototype.py --all
```

## 確認するcase

- 10秒境界のCandidate Momentが後側の半開区間へ一意に属する
- 9.75秒のFrame Candidateは前側segmentに属しながら、10秒のCandidate Momentから参照できる
- 同じanchorのheartbeatとscene signalが一つのCandidate Momentへ統合される
- 重なる二つのCandidate Momentが同じFrame Candidateを共有する
- VFRのframe位置とidentityがfloat秒やframe indexに依存しない
- 同じPTSでもVideo Fingerprintが異なれば別Frame Candidateになる
- Scene CatalogがVideoごとではなくVideo Set Stageに一つだけ属する
- Candidate Moment上限がVideo Durationと密度から算出され、適格候補の採用を強制しない
- 別動画の同一ロード画面を各Video Stageでは独立して保持し、Video Set Stageだけが横断重複として扱う

## 確定した候補上限の契約

Candidate Moment上限は `ceil(Video Duration（分） × Candidate Moment Density（件/分）)` とする。固定件数、Video Setの本数・順序、要求出力枚数には依存させず、上限内の適格候補がなければ0件を許す。

Candidate Moment DensityはVideo Stageの結果に影響する設定としてStage Fingerprintへ含める。prototypeの毎分2件は計算例であり既定値ではなく、既定密度は段階別Ollama評価の責務とschemaを確定する際の実測で決める。

## 確定したStage境界

Video Stageは同一動画内で完結する時間構造、Candidate Moment Density、refinement、無効frame除外、source-localな重複整理、Neutral Image Analysisを所有する。別動画が追加・削除・並べ替えされても、その成果物は変わらない。

Video Set Stageは共有Scene Catalog、全動画横断の順位比較、視覚的重複排除、Video Orderを使う時間的多様性、spoilerとblog image typeを含む最終選定を所有する。動画ごとの最低採用枠は設けず、具体的な順位規則は複数動画の進行・ネタバレ・soft coverage選定規則を確定する作業で決める。
