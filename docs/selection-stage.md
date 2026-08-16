# Video Set最終選定Stage

この文書は、公開前の動画入力selectorが注釈済みBlog Candidateから要求枚数を決定的に選ぶ内部契約を説明します。installed public CLIはIssue #190までscreenshot入力のままです。

## Module seam

`select_video_set_images`は、Candidate AnnotationとNeutral Image Analysisを結合したBlog Candidate集合を受け取り、次を一つの結果として返します。

- 選択順付きのBlog Candidate
- Base、coverage、spoiler、temporal、marginal utility
- 選択時のsimilarity passと最も近い選択画像とのcosine similarity
- Variant Group、Semantic Duplicate Groupと判定根拠、stable reason code、tie-break使用有無
- Blog Image Typeの目標と実績
- 通常戦闘・イベントの有効候補数、条件付き最低数、実績、再配分
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

Explanation Valueが`none`の候補はCounterfactual Selection Scoreまで計算しますが、要求枚数を満たすための選択対象にはしません。これはCandidate Annotationに最終採否を委ねる処理ではなく、検証済みenumへ`video-set-selection-v7`が適用する決定的な適格性境界です。未採用理由は既存の`lower_marginal_utility`として公開され、shortfallでも穴埋めしません。

要求枚数に対する`normal_gameplay=70%`、`event=25%`、`menu=5%`の目標は最大剰余法で丸めます。同率はこのtype順です。目標未達候補へ`+0.10`、最初のtitleへ`+0.05`を加えますが、超過を減点せず、候補が偏っていてもhard quotaにしません。titleだけは最大1枚です。

要求枚数が10枚以上なら、説明価値と既存の適格性を満たす候補がある場合に限り、`ordinary_combat`と`event`をそれぞれ最低1枚選びます。`ordinary_combat`は`normal_gameplay`のうち、Combat Encounter Kindが`ordinary`で、一般敵の群れ・編成または通常遭遇を直接示す`ordinary_*`のCombat Encounter Basisがある候補です。主要戦闘の根拠がないことや、敵名・HP barだけでは通常戦闘の根拠になりません。`major`、`uncertain`、探索、移動、障害物破壊はこのfacetに含めません。Spoiler Riskは物語上のネタバレ評価に限定し、通常戦闘か主要戦闘かの判定には使いません。`event`はBlog Image Typeが`event`の候補ですが、Screen Text KindまたはRepresentative Frame Evidenceがtitleを示す候補は誤分類として除外します。

未充足facetの適格候補を通常のutility候補より先に比較します。複数facetが未充足なら、終端similarity ceilingで各facetから1件ずつ両立する組合せを先に確認し、必要な未代表Variant Groupの代表を含む組合せ全体が残り出力枠へ収まる場合だけ候補として残します。現在のsimilarity passでは選べなくても後続passが残る間は枠を保持します。候補が同じrecurring gameplayの既選択Variant Groupに属し、別の未代表Groupが選択の前提になる場合は、最低枠を残せる範囲でその前提Groupを通常候補より先に選びます。終端passでも重複、spoiler guardなどの既存制約に反する場合、または有効候補が存在しない場合は枠を解放します。終端で最低枠を解放した場合だけでなく、緩和ceilingで最後の最低枠を満たした場合も、選択済み画像を保持して設定されたbase similarity ceilingから残りの通常選定を再開します。これにより、最低枠の可否確認にだけ使った緩和ceilingで類似画像が代替候補を押し出しません。残り枚数は固定内訳にせず、従来のMarginal Selection Utilityと候補供給に応じて動的に配分します。

## Spoilerと単調性

ADR 0004の表に従うSpoiler Penaltyは候補単体へのsoft penaltyです。greedyなcoverage・diversityの相互作用によって高い感度のMajor Spoiler件数が逆に増えないよう、次の順で同じ候補集合を再選定します。

1. `low`を件数上限なしで選ぶ。
2. `medium`を`low`のMajor Spoiler選択数以下で選ぶ。
3. `high`を`medium`のMajor Spoiler選択数以下で選ぶ。

このguardで未採用になった候補は`spoiler_monotonicity_guard`として説明します。guardで採用不能な候補はrecurring gameplayの未代表Variant Groupへ機会を与える判定から除外します。Video Set ProgressだけでSpoiler Riskを変更しません。

## 視覚・時間的多様性

通常のsimilarity ceilingは設定値から開始し、`+0.03`、`+0.06`、`+0.10`、`+0.15`の決定的passを適用します。設定値が`0.97`以下なら自動緩和の終端は`0.97`です。`0.97`を超える設定値を利用者が明示した場合は、その設定値を緩和せずに終端として使います。上限で同じ値になるpassは重複させません。cosine similarityが`0.995`を超える組はVisual Near-Duplicateとして、shortfall時にも同時採用しません。

同じsceneでsimilarityが`0.95`以上の連結成分を安定したVariant Groupにします。`recurring_gameplay`で同じGroupの2枚目を選ぶ前に、そのpassで適格な未代表Groupへ一度ずつ機会を与えます。全Groupの代表後も、利用者がより緩い設定値を明示していなければ`0.97`までしかvariantを広げません。旧Cinematic Soft Capは適用しません。

### 分類揺れをまたぐSemantic Duplicate Group

`video-set-selection-v7`はVariant Groupとglobal similarity ceilingの外側にSemantic Duplicate Groupを設けます。同じGroupは要求枚数不足時にも最大1枚で、Marginal Selection Utilityと通常のstable tie-breakで最初に選ばれた候補が代表です。この上限はConditional Coverage Minimumと`recurring_gameplay`のVariant Expansionより強く、未代表の別戦闘対象、別遭遇、通常戦闘、eventを同一Groupの2枚目より先に残します。

Group判定は次の根拠を使います。候補が複数の根拠に属する場合でも、元Groupが統合component全memberを実際に含むときだけ、`combat_subject_appearance`、`combat_encounter_sequence`、`title_semantics`、`visual_role_similarity`の順で公開basisを一つに統合します。全memberを説明する元Groupがない場合は、同じ優先順で元Groupを処理し、先に使われたmemberを除いた残余が2件以上なら同じbasisの残余Groupとして保持します。basisを無関係なmemberへ推移的に拡張せず、未使用member間の低優先重複制約も失いません。`combat_subject_appearance`は単一frameの完全一致ではなく、遭遇内の独立観測から集約された有限enum evidenceを公開します。公開basisとは別に統合前のCombat Encounter Groupを保持し、Encounter edge間の未注釈MomentがなくなるまでShortlistを拡張します。

- `title_semantics`: Blog Image Type、Screen Text Kind、Representative Frame Evidenceのいずれかがtitleを示す候補をVideo Set全体で一つにする。`event`などへの誤分類でも既存のtitle最大1枚を回避できない。
- `combat_subject_appearance`: 各Combat Encounter Groupについて、Candidate Momentごとに最高品質の完全な`distinctive` Evidenceを一つだけ数え、body plan・scale・surfaceの一意な最頻値と、複数Momentで2回以上反復したcolor・traitからCombat Encounter Subject Profileを作る。孤立したeffect、部分表示、色変化はProfileを上書きしない。異なるEncounter Profile間でbody plan・scale・surfaceが一致し、colorとtraitがそれぞれ一つ以上共通し、各Profileを支持する候補の少なくとも一組がNeutral視覚類似度0.80以上となる完全結合だけを同じ戦闘対象にする。完全結合はProfileの各組へ適用し、3件以上の全Profileに共通する単一のcolorまたはtraitまでは要求しない。Scene Slug、敵名、動画、遭遇時刻、公開用summaryの一致は要求しない。`generic`、`unclear`、不完全な根拠、title画像は動画横断同一性を成立させない。
- `combat_encounter_sequence`: 同じVideo Sourceの全候補をVideo Time順に並べ、非`major`候補で遭遇を区切ってから、`major`候補をScene Slugの連続runへ分ける。同じSlugに挟まれた1件だけのSlug揺れは、前後がそれぞれ15秒以内の場合だけ同じrunへ吸収する。各runは原則一つのGroupであり、単発の`distinctive`矛盾、`generic`、`unclear`では分割しない。完全な具体的Evidenceが互換なclusterを作り、互いに異なるclusterが少なくとも二つあり、その各clusterが異なるCandidate Momentで2回以上裏付けられ、集約後の各Profileが互いに非互換な場合だけ対象別に分割する。集約Profileを作れないclusterや互換なProfileの組が一つでもあればrun全体を一つのGroupに保つ。同じMomentの兄弟frameは1回と数える。分割後の単発・不明観測はVideo Timeが最も近い確認済み対象へ安定tie-break付きで属させ、第三の未確認対象として選択しない。`generic`はこの遭遇内の反復根拠には使えるが、動画横断同一性には使わない。
- `visual_role_similarity`: titleでも`major`でもない候補について、同じVideo Source、30秒以内、同じRepresentative Frame content kind、同じCombat Encounter Kind、Neutral視覚類似度0.93以上をすべて満たす組だけを完全結合のGroupにする。`recurring_gameplay`ではさらに、独立評価された画像summaryのUnicode・大小文字・句読点を正規化した値が一致する場合だけ畳み、異なる技・敵・結果の説明を維持する。

Semantic Group IDはbasisと全memberのFrame Candidate IDから決定的に作ります。選択代表には`semantic_group_representative`、除外候補には`semantic_duplicate`、blocking selected ID、同じGroup IDとbasisを記録します。Combat Subject Groupの公開根拠は、非`major`候補を含む元の遭遇境界を保持したProfileから算出します。全Profileで一致するbody plan・scale・surfaceと、全Profileに共通するcolor・traitだけを有限enum tokenとして記録し、colorまたはtraitの全体積集合が空でも完全結合Groupを破棄しません。単発矛盾、自由文、固有名、model responseは診断へ出しません。Group判定は自由文の固有名を正解として扱わず、Neutral特徴だけで異なる敵をまとめることもしません。

Temporal Diversity Penaltyは、要求枚数`N`と最も近い選択済みVideo Set Progress距離`d`から次の式で求めます。

```text
0.08 * max(0, 1 - d / (1 / N))
```

Video Orderや後半位置そのものへの加点・減点、動画ごとの最低枠はありません。

## 決定性、Shortlist拡張、shortfall

各greedy stepはMarginal Selection Utility、低いSpoiler Penalty、高いQuality Score、低い最大visual similarity、Video Order、Video Time、Frame Candidate IDの順で比較します。入力tupleの列挙順は結果へ影響しません。

`select_from_shortlist_batches`は初期注釈batchでshortfallになった場合に加え、要求枚数が10枚以上で既知の条件付きfacetがまだ見つからない、または最低枠が未充足の場合も次の決定的batchを受け取ります。さらに、`combat_encounter_sequence`の同一Groupへ入った候補間に、完全なsource別Candidate Moment時系列上で未注釈Momentが残る場合は選定を確定せず、そのMomentを含むbatchが消費されるまで拡張します。これにより、後続batchの非主要場面を見ないまま同名Scene Slugの別遭遇を畳みません。全候補を一律に注釈するのではなく、要求数、最低枠、観測済み遭遇境界が揃うか、全Candidate Momentを使い切るまで拡張し、拡張済みpoolを空の選択状態から再計算します。以前の緩和passで選んだ候補を固定しません。batchの生成、Candidate Annotation、batch sizeの性能上限は呼び出し側とIssue #189が所有します。

各batchの選定後も確定条件を満たさなかった場合は、その累積Candidate件数をShortlist Selection Frontierとしてatomicに保存します。Frontierは選択候補やscoreを固定する成果物ではなく、「この境界では継続が必要だった」という確認結果だけを保持します。中断後はmanifest、artifact hash・size、engine version、全選定意味入力のfingerprint、累積件数を検証し、連続する確認済み境界を一つへまとめて最初の未確認境界だけを拡張済みpoolから再選定します。全境界が確認済みなら全Candidate末尾を一度だけ再選定します。破損または意味入力が異なるFrontierはその境界だけを再計算し、中断なしと同じCandidate ID・順序・shortfallを返します。

全Candidate Momentを使い切っても不足する場合は、選べた画像だけを正常結果として返します。Explanation Valueが`none`の候補、Semantic Duplicate Groupの2枚目、2枚目のtitle、Visual Near-Duplicate、不適格frame、未完了Annotationでは穴埋めしません。未採用候補のSimilarity Ceilingは要求数を満たした時点、またはshortfallで最後まで到達した実際の最終passを基準にします。未採用候補はCounterfactual Selection Scoreの降順と同じstable tie-breakで返し、主因を次のenumで示します。

- `title_limit`
- `semantic_duplicate`
- `visual_near_duplicate`
- `similarity_ceiling`
- `spoiler_monotonicity_guard`
- `lower_marginal_utility`

内部Video Selection Applicationはこのshortlist拡張と決定的selectorを実行し、`select-images`を`video-set-selection-v7`、cache artifactを`game-screen-pick/video-set-selection@3.0.0`としてCompleted Stageへ確定する。選定前cache keyには、Video Stage、全候補のCatalog fingerprint、Primaryと同一Momentの代替最大2枚を含む全一枚Annotation fingerprint、Combat Representative Fallback policy version、完全時系列contract、設定、batch境界を含める。Shortlist Selection Frontierは同じ選定前cache keyと累積件数を意味入力に持ち、`shortlist-selection-frontier-v1`のengine versionで独立して失効する。fallbackの並列数と完了順は含めない。warm runはこのkeyをselector実行前に検索し、coldで実際に使用したbatch境界までCatalog／Annotationをcacheから復元した後、同じ完全時系列条件でRepresentative Frameを再集約し、score、reason、coverage、Semantic Duplicate Groupを含む選定結果をartifactから結び直す。selectorを実行したrunをcache reuseとして数えない。`video-set-selection-v6`以前のSelection Stageは新policyへ再利用しない。Candidate Annotation contractは変えないため、現行`game-screen-pick/candidate-annotation@5.0.0`とVideo Identity、Video Stage、Context Cueなど不変の上流Stageはfingerprint一致時に再利用する。

旧first-N fakeはwalking-skeleton専用applicationへ隔離され、public CLIはIssue #190までscreenshot入力のままである。
