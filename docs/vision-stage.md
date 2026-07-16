# Video Set Vision Stage

この文書は、公開前の動画入力selectorが共有Scene CatalogとCandidate Annotationを生成する内部契約を説明します。installed public CLIはIssue #190までscreenshot入力のままです。

## Module seam

`VideoSetVisionProcessor`は次の入力を一度に受け取り、model transport、strict validation、retry、Moment単位cacheを呼び出し側から隠します。

- 一つのVideo Set
- Neutral Image Analysisからlocalに選ばれた1〜24件のScene Catalog Representative Set
- Representative Setを導出したFrame Candidate Extraction Stage fingerprint
- 決定的なlocal順を持つSelection ShortlistのCandidate Annotation request
- Effective Configurationとrun中にfreeze済みのResolved Models

Scene Catalog Representative Setは要求画像枚数から独立します。Selection Shortlistの初期sizeと不足時の拡張は、決定的selectorとcapacity acceptanceを実装する後続Issue #186・#189が所有します。Candidate Annotation requestには1〜3件のFrame Candidate、versioned policyで選ばれた近傍Context Cue、Video Set Progress、Selection Intentを明示し、VisionRuntimeが暗黙に候補やCueを削りません。

## Ollama operation

Ollamaの`/api/chat`を次の2種類だけに使います。

1. `build-scene-catalog`: Video Set共有の3〜8 sceneを一回生成します。`other`を必ず1件含め、そのScene Selection Roleは`ordinary`です。
2. `annotate-candidate`: Selection ShortlistのCandidate Momentごとに独立して実行し、入力frameの一つをRepresentative Frameとして返します。

両方ともJSON Schema object全体、`stream=false`、`think=false`、`temperature=0`を送ります。JSON Schema検証後にも、Scene Slug、Frame Candidate ID、Context Cue IDが入力集合へ属することをlocalで検査します。

Candidate Annotation v1は次の意味情報だけを返します。

- Scene Slug
- Blog Image Type
- Explanation Value
- Screen Text Kind
- Context Cue Relevanceと参照Cue ID
- Spoiler Riskと引用を含まないevidence summary
- annotation summaryとRepresentative Frameの選択理由

Quality Score、model confidence、final score、soft coverage、eligible/selected flag、生成した逐語的画面テキスト、reasoning traceはschemaに含めません。最終採否はIssue #186の決定的selectorが所有します。

## Retryとfailure

同じsemantic入力で初回と一回のretryだけを行います。timeout、connection failure、HTTP 408/429/5xx、空・打ち切り応答、schema/domain validation failureがretry対象です。429の`Retry-After`は最大30秒まで尊重し、その他は1秒待ちます。

validation retryではstable validation codeだけを追加し、raw responseを次promptへ戻しません。画像、Context Cue、Catalogを削る、代表画像を減らす、`other`へfallbackする、失敗したCandidateを黙って除外する処理は行いません。model不在と408/429以外のHTTP 4xxは即時fatalです。

## Completed Stage cache

Scene CatalogはVideo Setごとに一つ、Candidate AnnotationはCandidate Momentごとに一つのatomic Completed Stageです。一つのAnnotationが最終失敗した場合、最終選定とoutput公開へ進みませんが、それ以前に完了したCatalogとAnnotationは次回runで再利用されます。

fingerprintには、順序付きFrame Candidate IDと画像SHA-256、Context Cue ID・正確な範囲・本文SHA-256、Cue選択policy、Scene Catalog fingerprint、Video Set Progress、Selection Intent、Resolved Model Identity、Ollama runtime identity、generation option、prompt/schema/stage/retry versionを含めます。

次の値はmodel contentを変えないため含めません。

- 要求画像枚数
- spoiler sensitivityと決定的penalty
- Blog Image Type Soft Coverage
- output pathとreport表示形式
- model更新診断と`models.auto_upgrade`（実行identityが同じ場合）

cache artifactには検証済みCatalog／Annotationと、identity、version、試行回数、stable validation code、画像・Cue件数、wall時間、token件数、done reasonだけを保存します。prompt body、raw response、chain of thought、Context Cue本文、absolute path、credentialは保存しません。

## 検証

```bash
uv run task test
```

fake VisionRuntime goldenでschema invalid、domain invalid、transport failure、retry成功・失敗、Context Cue有無、major spoiler evidence、warm cache、途中失敗後のMoment単位再開を検証します。
