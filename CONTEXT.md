# Game Screen Pick

ゲーム動画から、ブログで使いやすい画像を選び出すための文脈。

## Language

**Video Set**:
1回のブログ画像選定でまとめて扱う、1本以上の順序を持つ入力動画の集合。実行中は構成・順序・各Videoの内容が変わらないsnapshotとして扱い、保存場所である input folderや選定設定とは区別する。
_Avoid_: input folder, video folder, run, file list

**Video Input Folder**:
Video Setを発見し、そのcacheを保持するためにユーザーが指定するルートフォルダ。Video Setそのもののidentityではない。
_Avoid_: Video Set, Output Folder, cache key

**Video Set Fingerprint**:
Video Setの構成とVideo Orderをcacheやreportで参照するための、順序付きVideo Fingerprint列から導出される安定した識別子。
_Avoid_: input path hash, unordered file set, global setting hash

**Video Order**:
Video Set 内のゲーム進行順。入力ルートからの相対パスの自然順で決まり、更新日時やファイル列挙順には依存しない。
_Avoid_: filesystem order, mtime order, discovery order

**Video Source**:
Video Fingerprintで識別される、一つのVideo Identityとその有効なpresentation timelineの組。Video Input Folder内のpathやVideo Orderとは独立する。
_Avoid_: input file path, ordered Video Set member, path-based identity

**Video Identity**:
動画内容によって決まる、個々の Video の安定した同一性。ファイル名、配置、更新日時が変わっても内容が同じなら維持され、内容が変われば新しい identity になる。
_Avoid_: file path, filename, mtime, Video Order

**Duplicate Video**:
一つのVideo Set内で、複数のpathが同じVideo Identityを指している入力不整合。
_Avoid_: similar video, repeated scene, same filename

**Video Fingerprint**:
Video Identity をcacheやreportで参照するための、動画内容から導出される安定した識別子。
_Avoid_: path hash, file stat, stage setting hash

**Processing Stage**:
再開可能な画像選定を構成する、入力と再利用可能な成果物の境界が明示された処理単位。
_Avoid_: arbitrary function, progress message, whole run

**Stage Fingerprint**:
Processing Stage の成果物に影響する上流成果物と、そのStage固有の設定・versionだけから導出される識別子。
_Avoid_: global config hash, Video Fingerprint, unrelated downstream setting

**Completed Stage**:
成果物と完了manifestがatomicに確定し、再利用できる Processing Stage。完了manifestのない部分成果物は含まない。
_Avoid_: partial cache, in-progress stage, progress checkpoint

**Video Stage**:
一つのVideo Identityだけを対象とし、Video Setの構成やVideo Orderから独立して再利用できる Processing Stage。同一動画内で完結する時間構造、候補密度、frame refinement、Neutral Image Analysisを所有する。
_Avoid_: Video Set Stage, cross-video selection, whole-run stage

**Video Set Stage**:
順序付きのVideo Setと各Video Stageの成果物を入力にして、Scene Catalog、Candidate Annotation、動画横断の比較と多様性、最終選定を所有する Processing Stage。各Video Sourceからの最低採用数は持たない。
_Avoid_: Video Stage, per-video processing, per-video selection quota

**Video Time**:
一つのVideo Sourceのpresentation timeline上の正確な位置。source PTSとtime baseから最初の表示可能frameを0として導出され、float秒やframe indexとは区別する。
_Avoid_: frame index, float timestamp, wall-clock time

**Video Duration**:
一つのVideo Sourceの有効なpresentation timelineが持つ、0から終端までの正確で正の時間長。
_Avoid_: container duration, last frame index, wall-clock duration

**Timeline Segment**:
一つのVideo Sourceのtimeline全体をgapや重複なく覆う、順序付きの半開区間。各Candidate Momentはanchor時刻によって必ず一つのTimeline Segmentに属する。
_Avoid_: overlapping window, scene, refinement window

**Candidate Moment**:
一つのVideo Source内で、ブログに有用なframeがanchor Video Timeの周辺に存在すると判断された時間上の候補。複数の検出根拠をまとめ、refinement後に有効なFrame Candidateがない状態も保持する。
_Avoid_: extracted image, Frame Candidate, scene event

**Context Cue**:
一つのVideo SourceのVideo Time区間に対応付けられた、内蔵text subtitleまたは音声の文字起こしから得る文脈テキスト。視覚的なCandidate Momentへの加点根拠に限り、単独ではCandidate Momentを生成せずframeの採否も決めない。
_Avoid_: external subtitle, raw ASR segment, independent candidate, prompt text

**Candidate Moment Density**:
Video Durationに比例して、一つのVideo Sourceが保持できるCandidate Moment数を定める上限率。既定値は毎分2件で、上限は採用ノルマではなく、適格なCandidate Momentがなければ0件になり得る。
_Avoid_: fixed per-video count, per-video selection quota, requested-output multiplier

**Frame Candidate**:
Candidate Moment周辺のrefinementで有効と判断された、一つのVideo Source上の正確なsource frame。同じframeを複数のCandidate Momentが参照でき、proxy画像や出力画像とは区別する。
_Avoid_: Candidate Moment, cached proxy, output image

**Representative Frame**:
Selection Shortlist内の一つのCandidate Momentが参照する1から3件のFrame Candidateから、Candidate Annotationがブログ上の意味を最も表すものとして選んだframe。画像品質の再評価や最終採用を意味しない。
_Avoid_: highest Quality Score, selected output, Frame Refinement

**Cross-Video Diversity**:
Video Set全体で、視覚的に重複するframeや進行上の一部へ偏ったframeを最終選定から抑える性質。特定のVideo Sourceへ採用枠を保証するものではない。
_Avoid_: per-video quota, source-local deduplication, equal allocation

**Video Set Progress**:
Video Orderに従って各Video Durationを連結したVideo Set全体におけるCandidate Momentの進行位置。先行するVideo Durationの合計と現在のVideo TimeをVideo Set全体の長さで正規化した0以上1未満の値で、単独では候補の有用性やSpoiler Riskを表さない。
_Avoid_: per-video position, story importance, spoiler score, selection quota

**Temporal Diversity Penalty**:
要求枚数を`N`としたとき、選択済み候補との最短Video Set Progress距離が`1/N`未満の候補へ最大0.08を線形に適用するsoft penalty。進行位置そのものへの減点や時間帯ごとの採用枠ではない。
_Avoid_: late-video penalty, timeline bucket quota, per-video quota

**Scene Catalog Representative Set**:
Video Set全体のNeutral Image Analysisから、品質、見た目の多様性、頻出patternを表すFrame Candidateを最大24件選んだScene Catalog専用の入力集合。Selection Shortlistと要求出力枚数から独立し、要求枚数の変更だけではScene Catalogを変えない。
_Avoid_: Selection Shortlist, selected output, per-video representatives

**Candidate Annotation**:
Selection Shortlist内の一つのCandidate Momentについて、1から3件の有効なFrame Candidate、共有Scene Catalog、近傍Context Cue、Selection Intent、Video Set内の進行位置を入力にし、Representative Frameと意味情報を構造化して返すVideo Set StageのOllama評価。画像品質、最終score、soft coverage、最終採否は決めない。
_Avoid_: Candidate Scoring, Frame Refinement, Neutral Image Analysis, final selection

**Scene**:
ブログ用の画像選択で使う、画像内容を表すカテゴリ。ゲームジャンルや入力画像群に応じて決まる。
_Avoid_: play/event density bucket, fixed category

**Scene Slug**:
scene を表す小文字英数字の安定名。出力ファイル名、レポート、カテゴリ集計に使われる。
_Avoid_: localized category name

**Scene-numbered Output Name**:
選択された画像に付ける標準の出力ファイル名。scene slug と scene 内の連番で構成される。
_Avoid_: original filename output, optional rename mode

**Output Folder**:
選択された画像とレポートを書き出す実行ごとの保存先。処理開始前に空であり、input folderと同一または相互の配下であってはならない。
_Avoid_: append destination, overwrite target, resumable output

**Scene Display Name**:
scene を人が読みやすいように表す日本語名。ブログ用の画像選択やレポート表示で使われる。
_Avoid_: filename prefix, report key

**Scene Catalog**:
Scene Catalog Representative Setから作る、一つのVideo Setを横断して共有するsceneとScene Selection Roleの一覧。3から8個のsceneと分類の逃げ先である`other`で構成され、Videoごとには分割しない。
_Avoid_: fixed scene list, free-form per-image labels, per-video catalog

**Scene Description**:
画像がその scene に分類された理由を、ブログ用の画像選択に役立つように短く説明する文章。
_Avoid_: internal reasoning, model trace

**Scene Selection Role**:
scene ごとに、最終選択での扱いを表す役割。値は `ordinary`、`cinematic`、`recurring_gameplay` の3種類で、other scene、その他や不明なroleは通常配分で扱う。
_Avoid_: scene label, manual quota, content reject reason, failure mode

**Scene Hint**:
scene catalog を作るときに、ユーザーがゲームジャンルやブログ画像選択の意図を補足する短い説明。
_Avoid_: fixed scene list, selection rule

**Selection Intent**:
ブログ画像として何を重視して選ぶかを表す実行ごとの意図。scene hint は selection intent を補足する入力であり、変わると scene catalog や画像分類も変わり得る。
_Avoid_: image analysis setting, cache option

**Blog Image Type**:
Representative Frameがブログ内で主に果たす説明上の役割。値は`normal_gameplay`、`event`、`menu`、`title`、`other`で、操作可否ではなく画面とCandidate Momentの主目的からCandidate Annotationが付与する。探索や戦闘に短い台詞・HUD表示が重なったものは`normal_gameplay`、会話や演出そのものが主体なら`event`として扱い、最終的なsoft coverageは決定的なVideo Set selectorが扱う。
_Avoid_: Scene, Scene Selection Role, hard quota, final selection

**Blog Image Type Soft Coverage**:
最終選定が通常時に目指すBlog Image Typeの構成。`normal_gameplay` 70%、`event` 25%、`menu` 5%を目安とし、`other`と`title`には予約枠を設けない。候補の有用性や不足に応じて構成比の超過を許すため固定quotaではなく、`title`だけは有用な候補を最大1枚まで選べる。
_Avoid_: hard quota, per-video quota, Cinematic Soft Cap, guaranteed title image

**Blog Image Type Coverage Bonus**:
最大剰余法で丸めたBlog Image Type Soft Coverageの目標枚数へ未達の`normal_gameplay`、`event`、`menu`候補に0.10、まだtitleを選んでいないときの`title`候補に0.05を加えるsoft bonus。目標到達後はbonusを外すだけで、type超過へのpenaltyや`other`の予約枠は設けない。
_Avoid_: hard quota, overflow penalty, guaranteed title image

**Explanation Value**:
Representative FrameとそのCandidate Momentがブログ本文でplayや出来事を説明できる度合い。値は`none`、`low`、`medium`、`high`で、Candidate Annotationが意味評価として付与するが、最終scoreや採否そのものではない。
_Avoid_: Quality Score, model confidence, final selection score

**Screen Text Kind**:
Representative Frame内で意味を持つ画面内テキストの役割。値は`none`、`dialogue`、`menu`、`title`、`hud`、`other`で、生成された逐語転記は含めない。
_Avoid_: Context Cue, OCR transcript, generated quotation, Blog Image Type

**Context Cue Relevance**:
Candidate Annotationへ渡したContext Cueが、Representative FrameとCandidate Momentの説明をどれだけ補強するかを表す`unavailable`、`none`、`weak`、`strong`の評価。補強に使ったContext Cue IDを伴うが、単独でframeを適格にしない。
_Avoid_: Context Cue reliability, frame acceptance, independent candidate score

**Spoiler Risk**:
Representative FrameとCandidate Momentが物語上の重要情報を明かす可能性を表す`none`、`low`、`medium`、`high`の意味評価。汎用的な探索・戦闘は`none`、軽微な進行情報は`low`、固有ボスや終盤固有エリアなどは`medium`、Major Spoiler Signalの具体的な意味証拠があるものは`high`とし、Candidate Annotationが付与する。利用者設定に応じた減点は決定的なVideo Set selectorが扱う。
_Avoid_: spoiler sensitivity, spoiler penalty, late-video hard reject

**Major Spoiler Signal**:
エンディング、最終ボスの正体・形態、主要人物の生死、裏切り・犯人・真の正体、物語の中心的な種明かしを画像、画面内テキスト、Context Cueが具体的に示すこと。Video Set内の進行位置だけではMajor Spoiler Signalにならない。
_Avoid_: late-video position, generic battle, ordinary progression detail

**Spoiler Sensitivity**:
Spoiler Riskを最終選定でどれだけ避けるかを表す実行ごとの`low`、`medium`、`high`設定。既定値は`medium`で、値を高くしても候補を除外するhard policyにはしない。
_Avoid_: Spoiler Risk, story progress, hard reject

**Spoiler Penalty**:
Spoiler SensitivityとSpoiler Riskの組み合わせから、0から1の選定utilityに適用する決定的な減点。`low`ではriskが`medium`、`high`のとき0.02、0.05、`medium`では`low`、`medium`、`high`のとき0.01、0.04、0.10、`high`では0.02、0.08、0.18とし、riskが`none`なら常に0とする。
_Avoid_: hard reject, late-video penalty, model confidence

**Quality Score**:
blog candidate がブログ画像としてどれだけ使いやすいかを表す評価値。scene の種類やゲームジャンルの指示ではなく、画像そのものの見やすさを表す。
_Avoid_: scene hint, user-facing mode, selection profile

**Selection Base Utility**:
Blog Candidate単体の有用性を0から1で表す決定的な値。Quality Scoreを70%、Explanation Valueを25%、Context Cue Relevanceを5%として合成し、動画内位置、Blog Image Typeの構成、視覚・時間的多様性、Spoiler Penaltyは含めない。
_Avoid_: final selection score, model confidence, diversity bonus, spoiler-adjusted utility

**Marginal Selection Utility**:
greedyなVideo Set selectorが次の1枚を選ぶたびに再計算する値。Selection Base UtilityからSpoiler PenaltyとTemporal Diversity Penaltyを引き、Blog Image Type Coverage Bonusを加える。視覚類似度はutilityではなく適格条件として扱い、同点はSpoiler Penalty、Quality Score、選択済み画像との最大視覚類似度、Video Order、Video Time、Frame Candidate IDの順で解消する。
_Avoid_: static candidate score, Ollama output, global optimization result

**Blog Candidate**:
Candidate Annotationが完了し、Representative Frameと最終選定に必要な意味情報を持つCandidate Moment。明らかな暗転、白飛び、単色画面、遷移フレームはVideo Stageで既に除外されている。
_Avoid_: all Candidate Moments, Selection Shortlist, selected output

**Selection Shortlist**:
有効なFrame Candidateを持つCandidate Momentのうち、Neutral Image Analysisによる品質と見た目の多様性から、Candidate Annotationへ進めるものをVideo Set全体でlocalに絞った集合。
_Avoid_: all Candidate Moments, annotated Blog Candidate, selected output

**Selection Shortfall**:
有効な未注釈Candidate Momentを決定的なshortlist順で追加し、許可された視覚類似度緩和をすべて適用しても、適格なBlog Candidateが要求枚数に満たない状態。選べた画像とshortfall理由をreportへ出してwarning付きで正常終了し、Ollama Stage Failureとは区別する。
_Avoid_: Candidate Annotation failure, silent omission, fabricated output, invalid-frame fallback

**Neutral Image Analysis**:
scene や selection intent に依存せず、Frame Candidateそのものから得られる特徴と品質評価。画像の内容分類ではなく、blog candidate 判定や動画横断の類似度判定の土台になる。
_Avoid_: scene classification, selection intent

**Transition Frame**:
シーン移動や画面切り替えの途中に現れる、ブログ画像として説明価値が低い一時的な画面。
_Avoid_: event scene, cutscene

**Cinematic Scene**:
ゲームの進行操作より演出、会話、イベントの見せ場を主に写した scene。ブログ画像として少量は有用だが、入力全体の代表性を崩さないよう通常 gameplay より控えめに扱う。
_Avoid_: transition frame, hard reject, movie frame

**Cinematic Soft Cap**:
既存の画像入力selectorで、すべてのcinematic sceneの合計選択枚数を通常は少量に抑えつつ、他の有用候補が足りない場合だけ超過を許す上限。Video Set selectorではBlog Image Type Soft Coverageに置き換え、併用しない。
_Avoid_: Video Set selection rule, per-scene cinematic quota, hard reject, exact quota

**Recurring Gameplay Pattern**:
戦闘UI、探索画面、パズル盤面など、ゲーム中に頻繁に表示される通常playの画面構造。同じ構図でも状態や進行の違いがブログ上の説明価値になるため、複数のvariantを選ぶ余地がある。
_Avoid_: duplicate image, cinematic scene, static menu

**Variant Expansion**:
recurring gameplay pattern で、同じ variant group から複数の画像を選ぶこと。要求選択枚数が多いほど強まり、同じ画面構造の中にある状態差や進行差を拾うために使う。
_Avoid_: duplicate flooding, one-representative-only selection, manual expansion mode

**Visual Near-Duplicate**:
Video Set selectorで使う正規化済み視覚特徴のcosine similarityが0.995を超えるRepresentative Frameの組。要求枚数が不足しても同時には選択しない。
_Avoid_: recurring gameplay variant, same scene, temporal neighbor

**Variant Group**:
同じ scene の中で、見た目や構図が近くブログ上の役割が重複する画像のまとまり。最終選択では原則として各 variant group から代表画像を1枚だけ選ぶが、recurring gameplay pattern では variant expansion の対象になる。
_Avoid_: scene, duplicate file

**Ollama Stage Failure**:
Scene CatalogまたはCandidate Annotationが、同じsemantic入力による初回と1回の再試行後もtransport、schema、domain validationを完了できなかった状態。`other`への分類とは区別し、fallbackや失敗Candidateの除外で処理を継続せず、最終選定とoutput公開を中止する。
_Avoid_: other scene, silent exclusion, catalog fallback, partial output

**Resumable Run**:
中断された画像選択を、再利用可能なCompleted Stageから後で続ける実行。Video Setや設定が変わった場合は、一致するStage Fingerprintの成果物だけを再利用する。
_Avoid_: fresh run, output overwrite
