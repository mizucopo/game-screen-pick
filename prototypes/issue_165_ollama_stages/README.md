# PROTOTYPE — 段階別Ollama評価の責務とschema

このprototypeは本番実装ではありません。Issue #165の次の問いへ、人間が具体例を見て判断した設計と、その根拠を再現するための資料です。

> `segment_labeling`、`candidate_scoring`、`frame_refinement`を独立したOllama段階にするべきか。各判断のowner、schema、失敗・再試行、cache、診断をどう分けるか。

## 実行

```bash
uv run python prototypes/issue_165_ollama_stages/prototype.py
```

`[n]`と`[p]`でcaseを切り替え、`[q]`で終了します。全caseをJSONで表示する場合は次を実行します。

```bash
uv run python prototypes/issue_165_ollama_stages/prototype.py --all
```

## 確定した結論

3つを独立したOllama段階にはしない。Ollamaを使う操作は、Video Set Stage内の次の2種類に限定する。

1. **Scene Catalog**: 要求出力枚数に依存しないVideo Set横断の代表frame最大24枚から、共有Scene Catalogを1回作る。
2. **Candidate Annotation**: local Selection Shortlist内のCandidate Momentごとに、1〜3枚の有効なFrame Candidateと近傍Context Cueを1回で評価する。

`segment_labeling`は設けない。Candidate Momentの発見と密度制限はVideo Stageのlocal処理で行う。Video Stageでsceneを付けると、後段のVideo Set Stageだけが所有する共有Scene Catalogへの逆依存になる。

`frame_refinement`はVideo Stageのlocal処理とし、native frame抽出、無効frame除外、source-local deduplication、Neutral Image Analysisまでを所有する。Ollama Candidate Annotationは、その処理を通った1〜3枚からブログ上の意味を最も表すRepresentative Frameを選ぶが、画質点を再計算しない。

`candidate_scoring`はOllama段階にしない。Ollamaはscene、blog image type、説明価値、Context Cue関連度、spoiler riskという意味情報だけを返し、最終点、soft coverage、diversity、spoiler penaltyは決定的なVideo Set selectorが計算する。具体的な順位規則はIssue #167の責務とする。

## 判断の唯一のowner

| 判断 | owner | Ollama出力に含めるか |
|---|---|---:|
| Candidate Moment発見・密度 | Video Stageのlocal処理 | いいえ |
| Frame Refinement | Video Stageのlocal処理 | いいえ |
| 画像品質 | Video StageのNeutral Image Analysis | いいえ |
| Context Cue抽出 | Video Stageのsubtitle/STT処理 | いいえ |
| Scene CatalogとScene Selection Role | Video Set StageのScene Catalog | はい |
| Representative Frame | Video Set StageのCandidate Annotation | はい |
| Scene | Video Set StageのCandidate Annotation | はい |
| Blog Image Type | Video Set StageのCandidate Annotation | はい |
| Explanation Value | Video Set StageのCandidate Annotation | はい |
| 画面内テキストのrole | Video Set StageのCandidate Annotation | はい |
| Context Cue Relevance | Video Set StageのCandidate Annotation | はい |
| Spoiler Risk | Video Set StageのCandidate Annotation | はい |
| 最終点・soft coverage・diversity・spoiler penalty | 決定的なVideo Set selector | いいえ |

音声・内蔵字幕はIssue #166で確定した専用経路でContext Cueへ変換し、Ollamaにはtextとしてだけ渡す。Scene CatalogとCandidate Annotationはどちらもvision入力を必要とするため、責務上は別vision modelや別text modelを要求しない。stageごとのmodel fingerprintは持つため、将来の実測で別modelが必要になってもcache境界は保てる。

## Candidate Annotation v1

入力は次の通り。

- Candidate Moment ID
- local検査済みの1〜3件のFrame Candidate IDと画像
- usableな近傍Context CueのID、正確なVideo Time範囲、text
- 共有Scene Catalog
- Selection Intent
- Video Orderから導出したVideo Set内の相対進行位置

構造化出力は次の形とする。実際のJSON Schema全体は`prototype.py`の3番目のcaseで確認できる。

```json
{
  "representative_frame_id": "frame-618_5",
  "scene_slug": "conversation",
  "blog_image_type": "event",
  "explanation_value": "high",
  "annotation_summary": "依頼の背景が説明される会話",
  "frame_choice_reason": "事情が画面内の台詞として最も具体的に示される",
  "screen_text_kind": "dialogue",
  "context_relevance": "strong",
  "supporting_context_cue_ids": ["cue-618_5"],
  "spoiler_risk": "none",
  "spoiler_evidence": ""
}
```

`quality_score`、model自身の`confidence`、最終点、`eligible`、最終採否は含めない。sceneが不明ならScene Catalogの`other`を選ぶが、`other`を処理失敗時のfallbackには使わない。

画面内テキストは`screen_text_kind`だけを保持する。実機probeでは、modelが選択frame以外の近傍frameやContext Cueを混ぜて文章を生成したため、生成された転記を正確な引用として公開しない。元のContext CueとFrame Candidateを根拠として保持し、Candidate Annotationは参照したCue IDだけを返す。

## Structured Outputsとdomain validation

Ollamaの`/api/chat`へ、`format: "json"`ではなくJSON Schema object全体、`stream: false`、`think: false`、`temperature: 0`を送る。Ollamaはvision入力にも同じstructured-output contractを提供している。

- [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama Chat API](https://docs.ollama.com/api/chat)
- [Ollama Model Details API](https://docs.ollama.com/api-reference/show-model-details)

JSON Schema検証後に、入力Frame Candidate ID、Scene Catalog slug、Context Cue IDとの所属をlocalで再検証する。現行parserのように応答中のJSONらしい部分を正規表現で救済しない。

## 失敗・再試行

- 最大2回（初回 + 1回）とする。
- connection reset、timeout、HTTP 408/429/5xx、空・打ち切り応答、schema/domain validation failureだけを1回再試行する。
- 429は`Retry-After`を最大30秒まで尊重し、それ以外は1秒待つ。
- validation再試行では安定したerror codeだけをversioned repair instructionへ加え、raw responseをpromptへ戻さない。
- 画像、Cue、Catalogを削る、代表画像を半減する、`other`へfallbackする、失敗Candidateを黙って除外する再試行は行わない。
- model不存在、vision capability不足、不正設定、408/429以外のHTTP 4xxは即時fatalとする。

Scene CatalogはVideo Setにつき1つのatomicなCompleted Stageとする。Candidate AnnotationはCandidate Momentごとに独立したCompleted Stageとする。1件が最終失敗したら最終選定とoutput公開は中止するが、すでに完了した別Candidate Annotationは再開時に再利用できる。

## prompt/schema versionとcache key

Scene CatalogのStage Fingerprintには、Video Setと上流Neutral Image Analysis、要求出力枚数に依存せずlocalに選ぶ順序付き代表Frame Candidate ID/content hash、Selection Intent/Scene Hint、model名/digest、Ollama version、generation options、prompt/schema/stage/retry policy versionを含める。

Candidate AnnotationのStage Fingerprintには、Video SetとCandidate Moment、順序付きFrame Candidate ID/content hash、渡したContext Cue ID/text/正確な範囲、Cue選択policy、Scene Catalog fingerprint、Video Order由来の進行位置、Selection Intent、model名/digest、Ollama version、generation options、prompt/schema/stage/retry policy versionを含める。

要求出力枚数、最終score weight、spoiler sensitivityとpenalty、blog image typeのsoft coverage、output path、report表示形式はCandidate Annotationの結果を変えないため含めない。要求出力枚数が増えてSelection Shortlistへ新しいCandidate Momentが入った場合は、そのAnnotationだけを追加し、既存のScene CatalogとCandidate Annotationを再利用する。

## reportに残す診断

- model名、model digest、Ollama version
- prompt、schema、stage contract、retry policyのversion
- request fingerprint、cache hit、試行回数、validation error code
- 画像・Cue件数、各duration、prompt/eval token数、done reason
- Candidate Moment、入力Frame Candidate、Representative Frame、Scene、Blog Image Type、Explanation Value、参照Cue ID、Context Cue Relevance、Spoiler Risk

絶対path、modelのreasoning trace、生成した画面内テキストの逐語引用は公開reportへ含めない。公開reportの具体的なschemaはIssue #168で確定する。

## Candidate Moment Density

既定値は**毎分2件**とする。これは平均30秒に1件の上限であり、採用ノルマではない。提供された50時間40分26.481秒のVideo Setでは最大6,081 Candidate Moment、1件最大3枚なら最大18,243 Frame Candidateになる。

| 毎分上限 | Candidate Moment上限 | Frame Candidate最大 |
|---:|---:|---:|
| 1 | 3,041 | 9,123 |
| 2 | 6,081 | 18,243 |
| 4 | 12,162 | 36,486 |

Ollama Candidate Annotationは全Candidate Momentではなくlocal Selection Shortlistだけに行う。したがって密度はlocal coverage、refinement、cache容量を調整し、Ollama call数を直接決めない。実測した最終schemaのwarm serial latencyは1件2.176〜2.366秒で、500件なら約18.1〜19.7分になる。3つの独立Ollama段階に分けると、この候補向け時間が少なくとも倍になり、さらにsegment labelingの費用が加わる。

## Target実機probe

Windows 11 Pro / WSL2 Ubuntu / RTX 5090、Ollama 0.31.2、`qwen3-vl:8b-instruct`（digest `0533d743...`）で確認した。

- 提供動画の604.5秒、611.5秒、618.5秒の3枚と3件のContext Cueを1回のrequestへ渡した。
- JSON Schemaは受理され、3回とも618.5秒のframe、conversation、event、high、strong、spoiler noneという同じ判断になった。
- 最終schemaのwarm実測は2.176〜2.366秒、prompt eval 4,747 token、output 291 tokenだった。
- 10,015秒の1枚、Context Cue未提供のcaseでは`context_relevance: unavailable`となり、2.340秒だった。
- WSL2からWindows Ollamaへは検証時のgateway経由で接続できた。gateway値は運用configへ固定せず、host解決はIssue #169で扱う。

## 人間による確認結果

2026-07-14に、次の3点すべてが承認された。

1. 3つの独立Ollama段階を廃止し、Scene Catalog + Candidate Annotationへ集約する。
2. Candidate Moment Densityの既定値を毎分2件にする。
3. 生成した画面内テキスト全文を公開値にせず、roleと参照Cue IDだけを保持する。
