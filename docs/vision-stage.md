# Video Set Vision Stage

この文書は、公開前の動画入力selectorが共有Scene CatalogとCandidate Annotationを生成する内部契約を説明します。installed public CLIはIssue #190までscreenshot入力のままです。

## Module seam

`VideoSetVisionProcessor`は次の入力を一度に受け取り、model transport、strict validation、retry、Moment単位cacheを呼び出し側から隠します。

- 一つのVideo Set
- Neutral Image Analysisからlocalに選ばれた1〜24件のScene Catalog Representative Set
- Representative Setを導出したFrame Candidate Extraction Stage fingerprint
- 決定的なlocal順を持つSelection ShortlistのCandidate Annotation request
- Effective Configurationとrun中にfreeze済みのResolved Models

Scene Catalog Representative Setは要求画像枚数から独立します。Selection Shortlistは決定的selectorが不足時に追加batchを受けて拡張し、batch sizeと実model capacityの受け入れはIssue #189が所有します。各Candidate Momentの1〜3件のFrame Candidateは、Neutral Image AnalysisのQuality ScoreとFrame Candidate IDで一つのRepresentative Frameへ先に確定します。Candidate Annotation requestにはその1件、versioned policyで選ばれた近傍Context Cue、Video Set Progress、Selection Intentを明示し、VisionRuntimeが暗黙に候補やCueを削りません。

## Ollama operation

Ollamaの`/api/chat`を次の3種類だけに使います。

1. `build-scene-catalog`: Video Set共有の3〜8 sceneを一回生成します。`other`を必ず1件含め、そのScene Selection Roleは`ordinary`です。
2. `annotate-candidate`: Selection ShortlistのCandidate Momentごとに独立して実行し、先に確定した一つのRepresentative Frameの意味観測を主推論で返します。画像は対応するFrame Candidate IDとその画像だけに対する直接観測条件を持つmessageで送り、同じMoment内で切り替わった別画面の内容を混ぜません。総合分類の指示は画像messageの後に一度だけ送ります。
3. 戦闘可視性専用確認: 主推論が攻撃相手本体`clear`、エフェクトだけではない、掲載価値ありとした戦闘だけに条件付きで実行します。同じRepresentative Frame一枚だけを入力し、音声、Context Cue、前後場面、主推論の説明文を渡しません。

Scene Catalog promptは`ordinary`、`cinematic`、`recurring_gameplay`の意味を明示し、同じplay画面を一時的な敵や発光だけで別sceneへ分割しないよう要求します。Candidate Annotation promptは、総合分類より先にInterface Kind、会話eventの大きな人物立ち絵・胸像の有無、黒帯・HUDのない固定camera・人物配置から分かるCinematic Event Presentationの有無、画面内に実在する台詞文字の有無とDialogue Text Presentation、具体的な動作・判別可能な人物または敵の有無、戦闘かどうか、player本体・攻撃相手本体それぞれの`clear`・`partial`・`absent`、一時的な光・爆発・煙だけが主内容かを直接観測させます。その後、画面内容、Explanation Value、Screen Text Kind、主対象の視認性、一時的な遮蔽、Context Cue Relevance、Spoiler Riskの全境界を評価させます。Dialogue Text Presentationは`none`、`dialogue_box`、`speech_bubble`、`subtitle_overlay`、`other`のいずれかとし、画面内台詞文字の真偽値と一致させます。音声やContext Cueに会話文があっても、画像内で文字を読めなければ`none`です。手紙・手記・日誌・記録を読む画面は`document`とし、文書本文を台詞にしません。戦闘HUDだけを`other_interface`とせず、会話eventの大きな人物立ち絵・胸像と画面隅の小さな常設HUD portraitを区別します。通常の戦闘・探索HUDをCinematic Event Presentationにしません。人物portrait、空の台詞欄、説明文、目的表示、tutorial文、menu項目を台詞にしません。Portrait、HUD、文字、影、発光、移動軌跡を人物・player・攻撃相手の本体に数えません。各frame内に実在する情報だけを評価し、別frameの台詞やContext Cueを画面内情報として補いません。Context Cueが存在するだけでは`strong`にせず、進行位置だけではSpoiler Riskを上げません。戦闘可視性専用確認は、エフェクトの画面占有率、最大の前景要素、player本体と攻撃相手本体の可視性、エフェクトの本体への重なり、エフェクトだけのframeか、という六つの直接観測だけをstrict schemaで返します。

modelはRepresentative Frame、Blog Image Type、eligible/selected flagを返しません。Representative Frameは推論前にlocalで確定済みです。local処理は具体的なInterface Kindを曖昧な画面内容分類より優先します。ただし、具体的な動作が見えるframeの`other_interface`は戦闘HUDなどの誤認として画面内容を上書きしません。台詞のない`event_dialogue`、動作のないaction分類、台詞も動作もない会話eventの大きな人物立ち絵またはCinematic Event Presentationを静止場面へ補正してから、Blog Image Type、公開用要約、選択理由を決定的に導出します。`document`、`tutorial_help`、台詞も動作もない`event_setup`、`save`、人物も敵も判別できない`shop`、攻撃相手本体が`clear`でない戦闘、一時的な光・爆発・煙だけが主内容のframe、主対象不在、深刻な一時遮蔽はExplanation Valueを`none`へ正規化します。主推論が掲載可能とした戦闘だけは専用確認の攻撃相手本体可視性とエフェクトだけかという結果を優先し、それ以外の正規化を目的とする追加推論は行いません。

各推論attemptの直前と応答受領直後に`/api/version`と`/api/tags`でOllama server versionとconfigured tagのlocal完全digestを再確認します。Model LifecycleでfreezeしたRuntime IdentityまたはResolved Model Identityと異なる場合は、別runtime／digestの結果を同じfingerprintへ保存せず停止します。この確認はtagの更新や再解決を行いません。

各operationはJSON Schema object全体、`stream=false`、`think=false`、`temperature=0`を送ります。JSON Schema検証後にも、Representative Frame IDが一度だけ返されたこと、Scene SlugとContext Cue IDが入力集合へ属すること、Context Cueの有無とRelevanceが整合すること、画面内台詞文字の真偽値とDialogue Text Presentationが一致することをlocalで検査します。公開自由文の改行・連続空白は内容順を保った一行へ正規化します。絶対pathまたはendpoint形式が残るfieldはScene Catalogとenumから組み立てた決定的な説明へ置換します。local生成したannotation summary、Representative Frameの選択理由、modelが返したspoiler evidenceに正規化後3文字以上の入力Context Cue本文が逐語再出力された場合も、該当fieldだけを同じ非逐語説明へ置換します。その説明もCueと一致する場合は明示的な省略記号へ置換し、安全化前の自由文はcacheへ保存しません。1〜2文字の一般語は独立生成との区別がつかないため引用判定から除外します。

OllamaのCandidate Annotation responseはframeごとに次の意味情報だけを返します。

- Scene Slug
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

各Ollama operationは同じsemantic入力で初回と一回のretryだけを行います。Candidate Annotation Stageは主推論に加えて戦闘可視性専用確認を条件付きで含むため、Stage全体のdiagnosticsは合計1〜4 attemptになります。timeout、connection failure、HTTP 408/429/5xx、空・打ち切り応答、schema/domain validation failureがretry対象です。このHTTP分類は推論前の`/api/tags`確認にも適用します。429の`Retry-After`は秒数とHTTP-dateの両形式を解釈して最大30秒まで尊重し、その他は1秒待ちます。

response/schema/domain validation retryではstable validation codeを追加し、raw responseを次promptへ戻しません。Candidate Annotation主推論の関係違反では、Context Cue参照とSpoiler Evidenceの条件をstable validation codeから決まる修正指示として再提示します。Context Cueを持つCinematic Event Presentationまたは大きなevent portraitで画面内台詞文字ありと返された場合は、同じ画像とsemantic入力を使う主推論の一回のretryで、音声やContext Cueを根拠にせずDialogue Text Presentationを再確認します。再確認でも台詞文字ありなら有効な会話画面として保持し、文字表示なしなら静止eventへ正規化します。

主推論またはそのretryの最終結果が掲載可能な戦闘なら、独立した戦闘可視性専用確認を実行します。専用確認は主推論の分類・音声・Context Cueを根拠にせず、Representative Frameの画素だけを小さなschemaで観測します。専用確認自体のtransport、schema、domain failureにも一回だけretryし、攻撃相手本体が`partial`・`absent`またはエフェクトだけならExplanation Valueを`none`へ正規化します。専用確認が返す他の四fieldはstrict responseを保つために検証しますが、主推論の分類や説明を上書きしません。

Cue逐語一致は再推論へ依存せずfield単位で決定的に安全化し、diagnosticsへ`candidate_annotation_verbatim_context_redacted`を記録します。公開前には同じ安全化をVideo Set全体のContext Cueに対して再適用し、Candidateに未提示のCueと偶然一致する生成文もreportへ公開しません。model出力が存在しないtransport retryでは元のpromptを変更しません。Representative Frame、Context Cue、Catalogを削る、別frameへ差し替える、`other`へfallbackする、失敗したCandidateを黙って除外する処理は行いません。model不在と408/429以外のHTTP 4xxは即時fatalです。

## Completed Stage cache

Scene CatalogはVideo Setごとに一つ、Candidate AnnotationはCandidate Momentごとに一つのatomic Completed Stageです。同じStage Fingerprintの推論はfingerprint lock内で一度だけ実行し、並行workerも最初に確定したartifactを復元します。Vision batch境界とpublisher開始時にはInput Lock内のpath・device・inode・size・mtime・ctimeを検査し、snapshot一致を通るまでoutputを公開しません。一つのAnnotationが最終失敗した場合、最終選定とoutput公開へ進みませんが、それ以前に完了したCatalogとAnnotationは次回runで再利用されます。

fingerprintには、Representative Frame IDと画像SHA-256、Context Cue ID・正確な範囲・本文SHA-256、Cue選択policy、Scene Catalog fingerprint、Video Set Progress、Selection Intent、Resolved Model Identity、Ollama runtime identity、generation option、主推論と戦闘可視性専用確認それぞれのprompt/schema/stage version、retry versionを含めます。

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

fake VisionRuntime goldenで推論前後のmodel/runtime identity変更、schema invalid、domain invalid、transport failure、打ち切り応答、retry成功・失敗、戦闘可視性専用確認と集約attempt数、秒数／HTTP-dateのRetry-After、Context Cue有無、Cue逐語一致fieldの決定的な安全化、未安全化raw textのcache拒否、canonical diagnostic identity、major spoiler evidence、warm cache、同一fingerprintの並行処理、途中失敗後のMoment単位再開を検証します。
