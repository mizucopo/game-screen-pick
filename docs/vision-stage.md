# Video Set Vision Stage

この文書は、公開前の動画入力selectorが共有Scene CatalogとCandidate Annotationを生成する内部契約を説明します。installed public CLIはIssue #190までscreenshot入力のままです。

## Module seam

`VideoSetVisionProcessor`は次の入力を一度に受け取り、model transport、strict validation、retry、Moment単位cacheを呼び出し側から隠します。

- 一つのVideo Set
- Neutral Image Analysisからlocalに選ばれた1〜24件のScene Catalog Representative Set
- Representative Setを導出したFrame Candidate Extraction Stage fingerprint
- 決定的なlocal順を持つSelection ShortlistのCandidate Annotation request
- Effective Configurationとrun中にfreeze済みのResolved Models

Scene Catalog Representative Setは要求画像枚数から独立します。Selection Shortlistは決定的selectorが不足時に追加batchを受けて拡張し、batch sizeと実model capacityの受け入れはIssue #189が所有します。各Candidate Momentの1〜3件のFrame Candidateは、Neutral Image AnalysisのQuality ScoreとFrame Candidate IDで一つのRepresentative Frameへ先に確定します。複数Momentが同じRepresentative Frameを共有する場合は、品質と見た目の多様性で決定したshortlist順の最初のMomentだけを注釈対象にし、後続の一意なRepresentative Frameを持つMomentは保持します。Candidate Annotation requestにはその1件、versioned policyで選ばれた近傍Context Cue、Video Set Progress、Selection Intentを明示し、VisionRuntimeが暗黙に候補やCueを削りません。

## Ollama operation

Ollamaの`/api/chat`を次の8種類だけに使います。

1. `build-scene-catalog`: Video Set共有の3〜8 sceneを一回生成します。各sceneにScene Kind（`combat`、`exploration`、`interface`、`event`、`other`）を付けます。`other`を必ず1件含め、そのScene Kindは`other`、Scene Selection Roleは`ordinary`です。
2. `annotate-candidate`: Selection ShortlistのCandidate Momentごとに独立して実行し、先に確定した一つのRepresentative Frameの意味観測を主推論で返します。画像は対応するFrame Candidate IDとその画像だけに対する直接観測条件を持つmessageで送り、同じMoment内で切り替わった別画面の内容を混ぜません。総合分類の指示は画像messageの後に一度だけ送ります。
3. 戦闘有無専用確認: 主推論が掲載価値ありの非戦闘としたScene Kind `combat`のgameplay、または`recurring_gameplay`のactionに条件付きで実行します。敵・boss固有のstatus UIまたは対戦する本体から戦闘かを確認します。Scene Kind `combat`では戦闘を確認できなければ掲載価値を保持しません。それ以外の`recurring_gameplay`では戦闘可視性専用確認との交差確認へ進みます。
4. 戦闘有無の独立再確認: 最初の戦闘有無専用確認が非戦闘を返した場合だけ、先の回答を参照しない別promptで同じ画素を観測し直します。二回とも非戦闘だったScene Kind `combat`のgameplayは掲載価値を下げます。それ以外の`recurring_gameplay` actionは、戦闘の見落としを検出するため戦闘可視性専用確認へ進みます。
5. 戦闘可視性専用確認: 主推論が掲載可能な戦闘とした場合、または戦闘有無確認の対象になった`recurring_gameplay`のactionに実行します。同じRepresentative Frame一枚だけを入力し、音声、Context Cue、前後場面、主推論の説明文を渡しません。戦闘と確認済みの場合は不明瞭なら直ちに掲載価値を下げます。
6. 戦闘可視性の独立再確認: 戦闘と確認済みで最初の戦闘可視性専用確認が掲載可能を返した場合、または二回の戦闘有無確認がともに非戦闘だった場合に、先の回答を参照しない別promptで同じ画素を観測し直します。前者は二回とも敵本体が明瞭で構図内に収まる場合だけ掲載価値を保持します。後者は二回とも敵本体が不在でエフェクトだけではない場合、または二回とも掲載可能な戦闘として一致する場合だけ掲載価値を保持します。
7. 戦闘構図の外周strip監査: 二回の戦闘可視性確認がともに掲載可能だった場合だけ、元画像の上端・下端・左端・右端から外周30%をlocalで切り出し、4枚を一度に渡す専用schemaで監査します。どれか一辺で敵本体が判別され、その主要な輪郭が元画像の実際の外端へ到達する場合は掲載価値を下げます。
8. 掲載境界専用確認: 主推論が掲載価値ありとした非戦闘の地図、またはScene Selection Roleが`cinematic`の場面だけに条件付きで実行します。同じRepresentative Frame一枚だけを入力し、音声、Context Cue、前後場面、主推論の説明文を渡しません。

Scene Catalog promptはScene Kindと`ordinary`、`cinematic`、`recurring_gameplay`の意味を明示し、同じplay画面を一時的な敵や発光だけで別sceneへ分割しないよう要求します。Scene Kindは複数sceneで重複できますが、Scene Slugはcatalog内で一意とし、同じScene Kindのsceneには`battle`・`boss-battle`、`shop`・`map`のように視覚的・説明上の役割を区別するslugを要求します。Candidate Annotation promptは、総合分類より先にInterface Kind、会話eventの大きな人物立ち絵・胸像の有無、黒帯・HUDのない固定camera・人物配置から分かるCinematic Event Presentationの有無、画面内に実在する台詞文字の有無とDialogue Text Presentation、具体的な動作・判別可能な人物または敵の有無、戦闘かどうか、player本体・攻撃相手本体それぞれの`clear`・`partial`・`absent`、一時的な光・爆発・煙だけが主内容かを直接観測させます。その後、画面内容、Explanation Value、Screen Text Kind、主対象の視認性、一時的な遮蔽、Context Cue Relevance、Spoiler Riskの全境界を評価させます。Dialogue Text Presentationは`none`、`dialogue_box`、`speech_bubble`、`subtitle_overlay`、`other`のいずれかとし、画面内台詞文字の真偽値と一致させます。音声やContext Cueに会話文があっても、画像内で文字を読めなければ`none`です。手紙・手記・日誌・記録を読む画面は`document`とし、文書本文を台詞にしません。戦闘HUDだけを`other_interface`とせず、会話eventの大きな人物立ち絵・胸像と画面隅の小さな常設HUD portraitを区別します。通常の戦闘・探索HUDをCinematic Event Presentationにしません。人物portrait、空の台詞欄、説明文、目的表示、tutorial文、menu項目を台詞にしません。Portrait、HUD、文字、影、発光、移動軌跡を人物・player・攻撃相手の本体に数えません。各frame内に実在する情報だけを評価し、別frameの台詞やContext Cueを画面内情報として補いません。Context Cueが存在するだけでは`strong`にせず、進行位置だけではSpoiler Riskを上げません。戦闘有無専用確認は、戦闘の有無と`enemy_status_ui`・`opposing_bodies`・`both`の根拠だけをstrict schemaで返します。戦闘可視性専用確認は、エフェクトの画面占有率、最大の前景要素、player本体と攻撃相手本体の可視性、攻撃相手本体が画面内へ収まる構図、エフェクトの本体への重なり、エフェクトだけのframeか、という七つの直接観測だけをstrict schemaで返します。許可方向の独立再確認は同じschemaと画像を使いますが、先の回答を推測せず画素を最初から観測し直す専用promptを使います。二回とも掲載可能な場合は、元画像から生成した上端・下端・左端・右端の外周30% stripを一度に渡し、各stripの敵本体の有無と元画像の実際の外端への到達だけを専用strict schemaで監査します。掲載境界専用確認は、一時的な遷移effectの有無・種類・画面占有率、上下の黒帯、event用の人物配置、画面内台詞文字、人物の具体的な動作、主内容の可読性という八つの直接観測だけをstrict schemaで返します。

Scene Slug、Scene Display Name、Scene Descriptionは、後続のどのCandidate Frameにも再利用できる分類にします。一部のRepresentative Frameからのみ推測した町・ダンジョン・固有人物・物語上の結果は断定しません。Candidate Annotationは、選択したScene Catalog entryの表示名と説明を対象画像の画素だけで裏づけられるかをScene Catalog Matchで返します。大分類だけが合う場合や、音声・Context Cue・Video Set Progressで補わなければ合わない場合は`false`です。

modelはRepresentative Frame、Blog Image Type、eligible/selected flagを返しません。Representative Frameは推論前にlocalで確定済みです。local処理は具体的なInterface Kindを曖昧な画面内容分類より優先します。ただし、具体的な動作が見えるframeの`other_interface`は戦闘HUDなどの誤認として画面内容を上書きしません。大きなevent人物立ち絵またはCinematic Event Presentationと画面内台詞文字を持つ会話eventも、汎用的な`other_interface`より優先します。台詞のない`event_dialogue`、動作のないaction分類、台詞も動作もない会話eventの大きな人物立ち絵またはCinematic Event Presentationを静止場面へ補正してから、Blog Image Type、公開用要約、選択理由を決定的に導出します。`document`、`tutorial_help`、台詞も動作もない`event_setup`、`save`、人物も敵も判別できない`shop`、攻撃相手本体が`clear`でない戦闘、一時的な光・爆発・煙だけが主内容のframe、主対象不在、深刻な一時遮蔽はExplanation Valueを`none`へ正規化します。主推論が掲載可能とした戦闘では戦闘可視性専用確認を優先し、掲載可能という観測には独立再確認との一致と外周strip監査の通過を要求します。Scene Kind `combat`のgameplayを主推論が非戦闘とした場合は戦闘有無専用確認を二回行い、どちらでも戦闘を確認できなければ「戦闘sceneとして説明できないframe」として`none`へ正規化します。それ以外で主推論が非戦闘とした掲載可能な`recurring_gameplay`のactionは、戦闘有無専用確認の結果にかかわらず戦闘可視性専用確認でも交差確認します。二回とも戦闘有無が否定された場合は可視性を二回確認し、両方で敵本体が不在か、両方で掲載可能な戦闘として一致して外周strip監査も通る場合だけ掲載価値を保持します。非戦闘の地図、`cinematic` scene、または画素から上下両端の太い暗色帯が検知されたframeでは掲載境界専用確認の結果を優先し、一時的な遷移effectがあるframe、または上下の黒帯とevent用の人物配置があり画面内台詞も具体的な動作もないframeを`none`へ正規化します。画素検知した黒帯はmodelの黒帯見落としより優先します。いずれの専用確認も主推論の分類や説明を上書きしません。

Scene Catalog Matchが`false`なCandidate AnnotationはScene Slugを`other`へ正規化し、具体的なScene Display Nameを公開用要約へ連結せず、画像内容のラベルだけを公開します。Scene Kind `other`の表示名と説明は汎用的な分類の逃げ先へ正規化します。元のScene KindとScene Selection Roleは戦闘確認や掲載境界確認の対象判定に保持し、説明の正規化で品質検査を迂回させません。

各推論attemptの直前と応答受領直後に`/api/version`と`/api/tags`でOllama server versionとconfigured tagのlocal完全digestを再確認します。Model LifecycleでfreezeしたRuntime IdentityまたはResolved Model Identityと異なる場合は、別runtime／digestの結果を同じfingerprintへ保存せず停止します。この確認はtagの更新や再解決を行いません。

各operationはJSON Schema object全体、`stream=false`、`think=false`、`temperature=0`、`seed=0`を送ります。固定seedはScene Catalog、主注釈、全専用確認で共通し、generation optionsとしてStage fingerprintにも含めます。JSON Schema検証後にも、Representative Frame IDが一度だけ返されたこと、Scene SlugとContext Cue IDが入力集合へ属すること、Context Cueの有無とRelevanceが整合すること、画面内台詞文字の真偽値とDialogue Text Presentationが一致することをlocalで検査します。公開自由文の改行・連続空白は内容順を保った一行へ正規化します。絶対pathまたはendpoint形式が残るfieldはScene Catalogとenumから組み立てた決定的な説明へ置換します。local生成したannotation summary、Representative Frameの選択理由、modelが返したspoiler evidenceに正規化後3文字以上の入力Context Cue本文が逐語再出力された場合も、該当fieldだけを同じ非逐語説明へ置換します。その説明もCueと一致する場合は明示的な省略記号へ置換し、安全化前の自由文はcacheへ保存しません。1〜2文字の一般語は独立生成との区別がつかないため引用判定から除外します。

OllamaのCandidate Annotation responseはframeごとに次の意味情報だけを返します。

- Scene Slug
- Scene Catalog Match
- 画面内容
- Interface Kind
- 会話eventの大きな人物立ち絵・胸像の有無
- Cinematic Event Presentationの有無
- 画面内に実在する台詞文字の有無とDialogue Text Presentation
- 具体的な動作の有無
- 判別可能な人物または敵の有無
- 戦闘かどうか
- player本体の可視性（`clear`、`partial`、`absent`）
- 攻撃相手本体の可視性（`clear`、`partial`、`absent`）
- 一時的な光・爆発・煙だけが主内容か
- Explanation Value
- Screen Text Kind
- 主対象の視認性
- 一時的な遮蔽
- Spoiler Riskと引用を含まないevidence summary

response全体にはContext Cue Relevanceと参照Cue IDも含めます。Candidate Annotation artifactはlocalに決めたRepresentative Frame、Blog Image Type、annotation summary、Representative Frameの選択理由を従来どおり保持します。

Quality Score、model confidence、final score、soft coverage、eligible/selected flag、生成した逐語的画面テキスト、reasoning traceはschemaに含めません。最終採否は[Video Set最終選定Stage](selection-stage.md)の決定的selectorが所有し、Explanation Valueが`none`の候補を要求枚数の穴埋めに使いません。

## Retryとfailure

各Ollama operationは同じsemantic入力で初回と一回のretryだけを行います。Candidate Annotation Stageは主推論に加えて、関係修復、戦闘有無と戦闘可視性の二段階、それぞれの許可方向に対する独立再確認、戦闘構図の外周strip監査、または掲載境界専用確認を条件付きで含むため、Stage全体のdiagnosticsは合計1〜15 attemptになります。timeout、connection failure、HTTP 408/429/5xx、空・打ち切り応答、schema/domain validation failureがretry対象です。このHTTP分類は推論前の`/api/tags`確認にも適用します。429の`Retry-After`は秒数とHTTP-dateの両形式を解釈して最大30秒まで尊重し、その他は1秒待ちます。

response/schema/domain validation retryではstable validation codeを追加し、raw responseを次promptへ戻しません。Scene Kind `other`のsceneは自由なslugが返っても、分類の逃げ先として正確な`other`、汎用表示名と汎用説明へ決定的に正規化します。Scene Catalogのdomain違反ではScene Kindの重複を許しつつScene Slugを一意にし、`other`のkind・role関係を満たす修正指示を再提示します。再試行でも非`other`のScene Slugだけが重複した場合は、入力順に`-2`、`-3`のsuffixを付けて決定的に一意化します。Scene Kind `other`が複数ある場合やその他のdomain違反は補正せず失敗します。Context Cueを持つCinematic Event Presentationまたは大きなevent portraitで画面内台詞文字ありと返された場合は、同じ画像とsemantic入力を使う主推論の一回のretryで、音声やContext Cueを根拠にせずDialogue Text Presentationを再確認します。再確認でも台詞文字ありなら有効な会話画面として保持し、文字表示なしなら静止eventへ正規化します。

Candidate Annotation主推論のresponse全体がschemaに適合し、Context Cue参照またはSpoiler Evidenceの関係だけに違反した場合は、通常のCandidate Annotationを再実行せずCandidate Annotation Relationship Repairへ進みます。Context Cue Relevance、各frameのSpoiler Risk、分類、観測、公開文を凍結し、違反した`supporting_context_cue_ids`または`spoiler_evidence`だけを動的なstrict schemaで一度修復します。Context Cue参照とSpoiler Evidenceが同時に違反していても同じ一回の修復に含めます。修復推論は`num_predict=1024`、Spoiler Evidenceは最大160文字とし、返された従属fieldだけを元のresponseへ合成した後にCandidate Annotation全体を再検証します。打ち切り、schema違反、domain違反はそれぞれ`candidate_annotation_relationship_repair_response_truncated`、`candidate_annotation_relationship_repair_schema_invalid`、`candidate_annotation_relationship_repair_domain_invalid`としてfatalにし、三回目の推論、分類の変更、決定的fallbackは行いません。修復prompt、raw response、Context Cue本文はdiagnosticsまたはcacheへ保存しません。

主推論またはそのretryの最終結果が掲載可能な非戦闘のScene Kind `combat` gameplay、または`recurring_gameplay` actionなら、戦闘有無専用確認を実行します。敵・boss固有の名前とHP・status bar、または対戦するplayer・相手本体を直接観測し、敵本体が画面外やエフェクト内でもstatus UIがあれば戦闘とします。最初の結果が非戦闘なら別promptで独立再確認します。Scene Kind `combat`で二回とも戦闘を確認できなければExplanation Valueを`none`へ正規化します。それ以外の`recurring_gameplay` actionは、結果にかかわらず次の戦闘可視性専用確認へ進み、戦闘有無の見落としを異なる観測軸で検出します。

主推論が掲載可能な戦闘とした場合、または戦闘有無専用確認で戦闘と確認された場合、またはScene Kind `combat`以外で戦闘有無専用確認の対象になった`recurring_gameplay` actionには、戦闘可視性専用確認を実行します。専用確認は主推論の分類・音声・Context Cueを根拠にせず、Representative Frameの画素だけを小さなschemaで観測します。専用確認自体のtransport、schema、domain failureにも一回だけretryします。戦闘と確認済みの場合は、player本体が`absent`、攻撃相手本体が`partial`・`absent`、本体の主要部が画面端で切れる・エフェクト等で隠れる・不在、またはエフェクトだけなら即座にExplanation Valueを`none`へ正規化します。最初の確認が掲載可能なら別promptで独立再確認し、二回ともplayer本体が`clear`または`partial`、攻撃相手本体が`clear`、構図が`complete`、かつエフェクトだけではない場合に外周strip監査へ進みます。外周strip監査は元画像の上端・下端・左端・右端からそれぞれ30%を決定的に切り出して一度に渡し、どれか一辺で敵本体が判別され、その主要な輪郭が元画像の実際の外端へ到達する場合はExplanation Valueを`none`へ正規化します。HUD、敵名、HP bar、光、effect、影、背景、診断用の内側crop境界は敵本体の外端到達に数えません。Scene Kind `combat`以外で二回の戦闘有無確認がともに非戦闘だった場合は戦闘可視性を必ず二回確認し、両方で敵本体が不在かつエフェクトだけではない場合、または両方で掲載可能な戦闘として一致して外周strip監査も通る場合だけ元のExplanation Valueを保持します。片方だけが敵本体を検出した場合は不一致を許可しません。戦闘可視性専用確認が返す全fieldと外周strip監査の4辺・真偽値関係はstrict responseを保つために検証しますが、主推論の分類や説明を上書きしません。

主推論で戦闘が直接観測された事実、または戦闘専用確認の後に二回の可視性確認でplayer本体と攻撃相手本体が一致して確認された事実は、自由文のScene名ではなく`combat_action`としてCandidate Annotation artifactへ保存します。敵status UIだけでは`combat_action`へ昇格しません。最終selectorは、掲載可能な`normal_gameplay`かつSpoiler Risk `none`の`combat_action`だけを通常戦闘coverageへ数えます。固有boss戦はSpoiler Risk境界により通常戦闘へ数えません。

主推論またはそのretryの最終結果が掲載可能な非戦闘の地図、`cinematic` scene、または画素から上下両端の太い暗色帯が検知されたframeなら、独立した掲載境界専用確認を実行します。黒帯のlocal検知は上下各4%以上の連続した暗色帯と、その間の可視内容を要求し、片側だけの暗色帯、細い境界線、暗転を対象にしません。この専用確認もRepresentative Frameの画素だけを小さなschemaで観測し、transport、schema、domain failureには一回だけretryします。画素検知した黒帯はmodelの黒帯見落としより優先します。一時的な遷移effectがある場合、または上下の黒帯とevent用の人物配置があり画面内台詞も具体的な動作もない場合はExplanation Valueを`none`へ正規化します。地図の雲、cursor、選択marker、常設UIは遷移effectにしません。

Cue逐語一致は再推論へ依存せずfield単位で決定的に安全化し、diagnosticsへ`candidate_annotation_verbatim_context_redacted`を記録します。公開前には同じ安全化をVideo Set全体のContext Cueに対して再適用し、Candidateに未提示のCueと偶然一致する生成文もreportへ公開しません。model出力が存在しないtransport retryでは元のpromptを変更しません。Representative Frame、Context Cue、Catalogを削る、別frameへ差し替える、`other`へfallbackする、失敗したCandidateを黙って除外する処理は行いません。model不在と408/429以外のHTTP 4xxは即時fatalです。

## Completed Stage cache

Scene CatalogはVideo Setごとに一つ、Candidate AnnotationはCandidate Momentごとに一つのatomic Completed Stageです。外部model request自体より細かい安全なcheckpointは作りません。同じStage Fingerprintの推論はfingerprint lock内で一度だけ実行し、並行workerも最初に確定したartifactを復元します。Vision batch境界とpublisher開始時にはInput Lock内のpath・size・`mtime_ns`を検査し、snapshot一致を通るまでoutputを公開しません。一つのAnnotationが最終失敗した場合、最終選定とoutput公開へ進みませんが、それ以前に完了したCatalogとAnnotationは次回runで再利用されます。Ollama runtimeまたはrole固有model identityの変更は該当するVision Stageとdownstreamだけを失効させ、Video Identity、Video Stage、STT checkpointを削除しません。

`combat_action`を持たない旧Candidate Annotation artifactは現行schemaとして復元せず、該当Annotationとdownstreamだけを再計算します。Video Identity、Video Stage、Context Cue、無関係なmodel Stageは再利用し、動画のwhole-file SHA-256やframe抽出をやり直しません。

fingerprintには、Representative Frame IDと画像SHA-256、Context Cue ID・正確な範囲・本文SHA-256、Cue選択policy、Scene Catalog fingerprint、Video Set Progress、Selection Intent、Resolved Model Identity、Ollama runtime identity、generation option、主推論・Candidate Annotation Relationship Repair・戦闘有無専用確認・戦闘有無独立再確認・戦闘可視性専用確認・戦闘可視性独立再確認・戦闘構図外周strip監査・掲載境界専用確認それぞれのprompt/schema/stage version、Relationship Repairの`num_predict`とSpoiler Evidence最大長、外周strip生成version、上下黒帯検知version、retry versionを含めます。現行より古いversioned Candidate Annotation Stage Contractをmanifestから認識したcache entryはcache準備時に削除し、現行versionの設定違いまたは認識できないentryは削除しません。target acceptanceでは旧artifactを正式evidenceへ流用せず、suite resetによって全cacheを削除してからcold実行します。

次の値はmodel contentを変えないため含めません。

- 要求画像枚数
- spoiler sensitivityと決定的penalty
- Blog Image Type Soft Coverage
- output pathとreport表示形式
- model更新診断と`models.auto_upgrade`（実行identityが同じ場合）

cache artifactには検証済みCatalog／Annotationと、canonical形式を検証済みのmodel/runtime identity、version、試行回数、stable validation code、画像・Cue件数、wall時間、token件数、done reasonだけを保存します。prompt body、raw response、chain of thought、Context Cue本文、absolute path、credentialは保存しません。

## 検証

```bash
uv run task test
```

fake VisionRuntime goldenで推論前後のmodel/runtime identity変更、schema invalid、domain invalid、transport failure、打ち切り応答、retry成功・失敗、関係だけが不正なCandidate Annotationの従属field限定修復・凍結field保持・修復failure code、戦闘有無専用確認・戦闘有無独立再確認・戦闘可視性専用確認・戦闘可視性独立再確認・戦闘構図外周strip監査・掲載境界専用確認と集約attempt数、秒数／HTTP-dateのRetry-After、Context Cue有無、Cue逐語一致fieldの決定的な安全化、未安全化raw textのcache拒否、canonical diagnostic identity、major spoiler evidence、warm cache、同一fingerprintの並行処理、途中失敗後のMoment単位再開を検証します。
