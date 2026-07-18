# Video Set Vision Stage

この文書は、公開前の動画入力selectorが共有Scene CatalogとCandidate Annotationを生成する内部契約を説明します。installed public CLIはIssue #190までscreenshot入力のままです。

## Module seam

`VideoSetVisionProcessor`は次の入力を一度に受け取り、model transport、strict validation、retry、Moment単位cacheを呼び出し側から隠します。

- 一つのVideo Set
- Neutral Image Analysisからlocalに選ばれた1〜24件のScene Catalog Representative Set
- Representative Setを導出したFrame Candidate Extraction Stage fingerprint
- 決定的なlocal順を持つSelection ShortlistのCandidate Annotation request
- Effective Configurationとrun中にfreeze済みのResolved Models

Scene Catalog Representative Setは要求画像枚数から独立します。Selection Shortlistは決定的selectorが不足時に追加batchを受けて拡張し、batch sizeと実model capacityの受け入れはIssue #189が所有します。Candidate Annotation requestには1〜3件のFrame Candidate、versioned policyで選ばれた近傍Context Cue、Video Set Progress、Selection Intentを明示し、VisionRuntimeが暗黙に候補やCueを削りません。

## Ollama operation

Ollamaの`/api/chat`を次の2種類だけに使います。

1. `build-scene-catalog`: Video Set共有の3〜8 sceneを一回生成します。`other`を必ず1件含め、そのScene Selection Roleは`ordinary`です。
2. `annotate-candidate`: Selection ShortlistのCandidate Momentごとに独立して実行し、入力frame別の意味観測を一回の推論で返します。各画像は対応するFrame Candidate IDだけを持つ個別messageで送り、別画像の内容を混ぜません。

Scene Catalog promptは`ordinary`、`cinematic`、`recurring_gameplay`の意味を明示し、同じplay画面を一時的な敵や発光だけで別sceneへ分割しないよう要求します。Candidate Annotation promptは画面内容、Explanation Value、Screen Text Kind、主対象の視認性、一時的な遮蔽、Context Cue Relevance、Spoiler Riskの全enum境界を明示します。各frame内に実在する情報だけを評価し、別frameの台詞やContext Cueを画面内情報として補いません。Context Cueが存在するだけでは`strong`にせず、進行位置だけではSpoiler Riskを上げません。

modelはRepresentative Frame、Blog Image Type、eligible/selected flagを返しません。local処理がframe別観測のExplanation Value、画面内容、主対象の視認性、一時的な遮蔽、Neutral Image Analysisの順でRepresentative Frameを決め、Blog Image Type、公開用要約、選択理由を決定的に導出します。著しく画質・情報量・視認性が低いframeは明瞭なpeerより優先しません。`tutorial_help`、台詞も動作もない`event_setup`、主対象不在、深刻な一時遮蔽はExplanation Valueを`none`へ正規化します。追加推論は行いません。

各推論attemptの直前と応答受領直後に`/api/version`と`/api/tags`でOllama server versionとconfigured tagのlocal完全digestを再確認します。Model LifecycleでfreezeしたRuntime IdentityまたはResolved Model Identityと異なる場合は、別runtime／digestの結果を同じfingerprintへ保存せず停止します。この確認はtagの更新や再解決を行いません。

両方ともJSON Schema object全体、`stream=false`、`think=false`、`temperature=0`を送ります。JSON Schema検証後にも、全Frame Candidate IDが入力順に一度ずつ返されたこと、Scene SlugとContext Cue IDが入力集合へ属すること、Context Cueの有無とRelevanceが整合することをlocalで検査します。local生成したannotation summary、Representative Frameの選択理由、modelが返したspoiler evidenceに正規化後3文字以上の入力Context Cue本文が逐語再出力された場合は、該当fieldだけをScene Catalogとenumから組み立てた非逐語説明へ置換します。その説明もCueと一致する場合は明示的な省略記号へ置換し、安全化前の自由文はcacheへ保存しません。1〜2文字の一般語は独立生成との区別がつかないため引用判定から除外します。

OllamaのCandidate Annotation responseはframeごとに次の意味情報だけを返します。

- Scene Slug
- 画面内容
- Explanation Value
- Screen Text Kind
- 主対象の視認性
- 一時的な遮蔽
- Spoiler Riskと引用を含まないevidence summary

response全体にはContext Cue Relevanceと参照Cue IDも含めます。Candidate Annotation artifactはlocalに決めたRepresentative Frame、Blog Image Type、annotation summary、Representative Frameの選択理由を従来どおり保持します。

Quality Score、model confidence、final score、soft coverage、eligible/selected flag、生成した逐語的画面テキスト、reasoning traceはschemaに含めません。最終採否は[Video Set最終選定Stage](selection-stage.md)の決定的selectorが所有し、Explanation Valueが`none`の候補を要求枚数の穴埋めに使いません。

## Retryとfailure

同じsemantic入力で初回と一回のretryだけを行います。timeout、connection failure、HTTP 408/429/5xx、空・打ち切り応答、schema/domain validation failureがretry対象です。このHTTP分類は推論前の`/api/tags`確認にも適用します。429の`Retry-After`は秒数とHTTP-dateの両形式を解釈して最大30秒まで尊重し、その他は1秒待ちます。

response/schema/domain validation retryではstable validation codeを追加し、raw responseを次promptへ戻しません。Candidate Annotationの関係違反では、Context Cue参照とSpoiler Evidenceの条件をstable validation codeから決まる修正指示として再提示します。Cue逐語一致は再推論へ依存せずfield単位で決定的に安全化し、diagnosticsへ`candidate_annotation_verbatim_context_redacted`を記録します。公開前には同じ安全化をVideo Set全体のContext Cueに対して再適用し、Candidateに未提示のCueと偶然一致する生成文もreportへ公開しません。model出力が存在しないtransport retryでは元のpromptを変更しません。画像、Context Cue、Catalogを削る、代表画像を減らす、`other`へfallbackする、失敗したCandidateを黙って除外する処理は行いません。model不在と408/429以外のHTTP 4xxは即時fatalです。

## Completed Stage cache

Scene CatalogはVideo Setごとに一つ、Candidate AnnotationはCandidate Momentごとに一つのatomic Completed Stageです。同じStage Fingerprintの推論はfingerprint lock内で一度だけ実行し、並行workerも最初に確定したartifactを復元します。Vision batch境界とpublisher開始時にはInput Lock内のpath・device・inode・size・mtime・ctimeを検査し、snapshot一致を通るまでoutputを公開しません。一つのAnnotationが最終失敗した場合、最終選定とoutput公開へ進みませんが、それ以前に完了したCatalogとAnnotationは次回runで再利用されます。

fingerprintには、順序付きFrame Candidate IDと画像SHA-256、Context Cue ID・正確な範囲・本文SHA-256、Cue選択policy、Scene Catalog fingerprint、Video Set Progress、Selection Intent、Resolved Model Identity、Ollama runtime identity、generation option、prompt/schema/stage/retry versionを含めます。

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

fake VisionRuntime goldenで推論前後のmodel/runtime identity変更、schema invalid、domain invalid、transport failure、打ち切り応答、retry成功・失敗、秒数／HTTP-dateのRetry-After、Context Cue有無、Cue逐語一致fieldの決定的な安全化、未安全化raw textのcache拒否、canonical diagnostic identity、major spoiler evidence、warm cache、同一fingerprintの並行処理、途中失敗後のMoment単位再開を検証します。
