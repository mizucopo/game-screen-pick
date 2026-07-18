# Video Set最終選定Stage

この文書は、公開前の動画入力selectorが注釈済みBlog Candidateから要求枚数を決定的に選ぶ内部契約を説明します。installed public CLIはIssue #190までscreenshot入力のままです。

## Module seam

`select_video_set_images`は、Candidate AnnotationとNeutral Image Analysisを結合したBlog Candidate集合を受け取り、次を一つの結果として返します。

- 選択順付きのBlog Candidate
- Base、coverage、spoiler、temporal、marginal utility
- 選択時のsimilarity passと最も近い選択画像とのcosine similarity
- Variant Group、stable reason code、tie-break使用有無
- Blog Image Typeの目標と実績
- 未採用候補のstable rejection、blocking ID、Counterfactual Selection Score
- Selection Shortfall、最終similarity ceiling、Major Spoiler件数境界

Candidate Annotationのsummary、spoiler evidence、Context Cue本文は採点式に使いません。Quality ScoreはNeutral Image Analysis、Explanation Value・Context Cue Relevance・Spoiler RiskはCandidate Annotationから取得し、modelに最終scoreや採否を問い合わせません。

## Utilityとcoverage

Selection Base Utilityは次の固定式です。

```text
0.70 * Quality Score
+ 0.25 * Explanation Value
+ 0.05 * Context Cue Relevance
```

Explanation Valueは`none=0`、`low=1/3`、`medium=2/3`、`high=1`、Context Cue Relevanceは`unavailable/none=0`、`weak=0.5`、`strong=1`へ変換します。

Explanation Valueが`none`の候補はCounterfactual Selection Scoreまで計算しますが、要求枚数を満たすための選択対象にはしません。これはCandidate Annotationに最終採否を委ねる処理ではなく、検証済みenumへ`video-set-selection-v2`が適用する決定的な適格性境界です。未採用理由は既存の`lower_marginal_utility`として公開され、shortfallでも穴埋めしません。

要求枚数に対する`normal_gameplay=70%`、`event=25%`、`menu=5%`の目標は最大剰余法で丸めます。同率はこのtype順です。目標未達候補へ`+0.10`、最初のtitleへ`+0.05`を加えますが、超過を減点せず、候補が偏っていてもhard quotaにしません。titleだけは最大1枚です。

## Spoilerと単調性

ADR 0004の表に従うSpoiler Penaltyは候補単体へのsoft penaltyです。greedyなcoverage・diversityの相互作用によって高い感度のMajor Spoiler件数が逆に増えないよう、次の順で同じ候補集合を再選定します。

1. `low`を件数上限なしで選ぶ。
2. `medium`を`low`のMajor Spoiler選択数以下で選ぶ。
3. `high`を`medium`のMajor Spoiler選択数以下で選ぶ。

このguardで未採用になった候補は`spoiler_monotonicity_guard`として説明します。guardで採用不能な候補はrecurring gameplayの未代表Variant Groupへ機会を与える判定から除外します。Video Set ProgressだけでSpoiler Riskを変更しません。

## 視覚・時間的多様性

通常のsimilarity ceilingは設定値から開始し、`+0.03`、`+0.06`、`+0.10`、`+0.15`の決定的passを適用して、必ず終端`0.98`へ進みます。上限で同じ値になるpassは重複させません。cosine similarityが`0.995`を超える組はVisual Near-Duplicateとして、shortfall時にも同時採用しません。

同じsceneでsimilarityが`0.95`以上の連結成分を安定したVariant Groupにします。`recurring_gameplay`で同じGroupの2枚目を選ぶ前に、そのpassで適格な未代表Groupへ一度ずつ機会を与えます。全Groupの代表後はVisual Near-Duplicateを除き`0.98`までvariantを広げます。旧Cinematic Soft Capは適用しません。

Temporal Diversity Penaltyは、要求枚数`N`と最も近い選択済みVideo Set Progress距離`d`から次の式で求めます。

```text
0.08 * max(0, 1 - d / (1 / N))
```

Video Orderや後半位置そのものへの加点・減点、動画ごとの最低枠はありません。

## 決定性、Shortlist拡張、shortfall

各greedy stepはMarginal Selection Utility、低いSpoiler Penalty、高いQuality Score、低い最大visual similarity、Video Order、Video Time、Frame Candidate IDの順で比較します。入力tupleの列挙順は結果へ影響しません。

`select_from_shortlist_batches`は初期注釈batchでshortfallになった場合だけ次の決定的batchを受け取り、拡張済みpoolを空の選択状態から再計算します。以前の緩和passで選んだ候補を固定しません。batchの生成、Candidate Annotation、batch sizeの性能上限は呼び出し側とIssue #189が所有します。

全Candidate Momentを使い切っても不足する場合は、選べた画像だけを正常結果として返します。Explanation Valueが`none`の候補、2枚目のtitle、Visual Near-Duplicate、不適格frame、未完了Annotationでは穴埋めしません。未採用候補のSimilarity Ceilingは要求数を満たした時点、またはshortfallで最後まで到達した実際の最終passを基準にします。未採用候補はCounterfactual Selection Scoreの降順と同じstable tie-breakで返し、主因を次のenumで示します。

- `title_limit`
- `visual_near_duplicate`
- `similarity_ceiling`
- `spoiler_monotonicity_guard`
- `lower_marginal_utility`

内部Video Selection Applicationはこのshortlist拡張と決定的selectorを実行し、`select-images`を`video-set-selection-v2`としてCompleted Stageへ確定する。旧first-N fakeはwalking-skeleton専用applicationへ隔離され、public CLIはIssue #190までscreenshot入力のままである。
