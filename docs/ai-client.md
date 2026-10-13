# 共通 AI 接続契約（#362 先行基盤）

`ai::OpenAiClient` は Game Context・一次／二次画像評価・動画理解に共有する同期 JSON 呼出境界です。
`JsonRequest` の prompt・JSON Schema・順序付き content を `POST <base_url>/chat/completions` へ送り、
アプリで schema を検証してから caller の用途検証を呼びます。custom service prefix と percent encoding を保持します。
用途別 prompt、Brave、runtime 管理、CLI の選定接続は #342 の範囲です。現在の通常 CLI は未接続として停止します。

## 設定と capability の区別

[設定例](../config.example.toml) の `ai.protocol` は client 使用時に必須です。
`reasoning_format = "reasoning_effort"` は top-level の明示レベルを送り、`chat_template` は
`none` を `enable_thinking=false`、他レベルを `enable_thinking=true` と同名 `reasoning_effort` にします。
backend/model/template がその形式と値を受理することを確認して `supported_inference_levels` に列挙してください。
client は列挙されないレベルを拒否し、server default や別 provider の値へ置換しません。

`json_schema=true` は `response_format.json_schema` を送ります。false でも schema は prompt に含めます。
両者とも local JSON Schema 検証と caller の用途検証を行います。server の grammar 制約を仮定しません。
local schema の外部 HTTP/file 参照解決は無効です。不正 schema は通信前に失敗します。
tools は未使用で、予期しない tool response は拒否します。Brave の固定検索を自律 tool loop にしません。

`max_output_tokens` は1〜65536、`max_request_bytes` は1KiB〜256MiB、
`max_response_bytes` は1KiB〜16MiB、`max_attempts` は1〜3です。送信限界は base64 化後の全 JSON に適用します。
`ai.timeout_seconds` は1〜3600秒で、header/body・redirect・修正試行を含む一呼出の期限です。
通信／HTTP失敗・拒否・打切り・tool要求は再試行しません。完了応答の JSON/schema/用途契約違反だけに
安全な分類と修正指示を付けて有限再試行します。生応答や validator のエラーを修正 prompt／例外へ転載しません。

`ai.protocol.media` の不在は画像・動画非対応です。`vision_revision` に encoder/projector と処理条件を記録し、
画像数（0〜64）・画素（最大200MP）・辺（最大32768）・動画bytes（最大128MiB）・秒数（最大3600）を明示します。
PNG/JPEG は実 decode と寸法検証を行い、縮小せず inline data URL へ送ります。
動画は1 request に1本、MP4/MOV の inline bytes を受理します。`ftyp` box の通常／extended header・
宣言サイズ・brand領域の完全性と、宣言bytes/秒数を検証します。動画全体の decode を保証する検査ではありません。
実 frame・PTS・duration と MIME の対応は、#342 が既存の `media::ChunkArtifact` と probe 結果から作り、
用途検証で source と実時間範囲を照合します。client 自身は動画を再 probe/transcode しません。
時刻付き画像列への代替や1枚への置換は実装していません。

Strata は動画宣言も `Content::Video` も通信前に拒否します。
2026-10-13 確認の [Strata a0116b2 の公式説明](https://github.com/Niko1221/Strata/blob/a0116b2782a5b82622358c04a06496485d12b2d6/docs/DETAILS.md)
は画像に vision 設定を要し、動画は未対応としています。同版の JSON Schema は prompt と生成後検証で、grammar 制約ではありません。
[vLLM の公式 multimodal 仕様](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)
の `image_url`／`video_url` 形式を使いますが、個々の model・server の対応を別途検証する必要があります。

## caller・cache・出力保護

`Session::prepare` で出力/cache の保護と排他を確立し、cache-miss callback 内から client を呼びます。
constructor 自身は通信／server 起動／`/models` 取得を行いません。start/stop command や自動 fallback はありません。
`AiConfig::model_conditions()` と `cache_conditions(prompt_revision, schema_revision, media_revision, context)` を
組で使用します。認証は除外し、backend・URL・model・推論レベル・revision と、protocol／vision／上限／encoding の
digest を既存 cache 条件へ渡します。model 重み・server version・template 等の変更は `cache_revision` を更新してください。
prompt/schema/media 処理自体の変更も各 revision を更新します。認証だけの変更は推論 cache を無効化しません。
この helper の計算は環境変数解決や通信を行いません。warm 時は client を呼ぶ必要がありません。

caller は構造検証に続けて `selection::validate_assessments` による全表示 ID の一致・重複／欠落・score と理由、
semantic event の source・実時間範囲・event品質、Game Context の同定・情報充足を検証します。
画像／動画を理解したと証明できる用途上の検証と実 model 試験は #342/#343 の責務で、JSONの形だけで成功を宣言しません。
未検証応答を cache／report／成果物へ保存せず、既存の publish/checkpoint 境界を保ちます。

HTTP は proxy 自動利用を無効化し、認証を bearer header だけへ渡します。307/308 の POST を保持する redirect は
同一 scheme/host/effective port で3回までです。別 origin、userinfo/query/fragment、他の redirect status を拒否します。
HTTP／schema／用途検証エラーや server body を診断へ転載せず、応答に認証値が含まれた場合も拒否します。
API key を serialization に含めず、config の Debug では redact します。

## 検証状況と後続確認

`tests/ai_contract.rs` は loopback HTTP の wire/mock 契約、`tests/ai_settings.rs` は設定と cache 条件を確認します。
既存の media／選定／cache／出力保護試験と共に `sh tests/check.sh` で実行します。
合成画像・動画の bytes が request に保持されても、mock が画像を読んだ／動画を理解した証明にはなりません。

| 機能 | Strata | vLLM | 実証の条件 |
| --- | --- | --- | --- |
| text・推論形式・local JSON/schema | mock 契約 | mock 契約 | 実 server/model ごとに明示レベルと出力を確認 |
| JSON Schema request | mock 契約、serverは生成後検証方式 | mock 契約 | server versionと機能設定を記録 |
| 画像・表示 ID・before/center/after | inline wire/mock 契約 | inline wire/mock 契約 | 合成 sheet の ID・候補・順序を実 model で判読 |
| 動画・順序・PTS・event | 未対応として拒否 | inline wire/mock 契約 | 合成動画の順序・元時刻・eventを実 model で確認 |
| tools＋JSON | 未使用 | 未使用 | 必要な用途ができた場合だけ追加契約を検証 |

実サーバー試験はまだ行っていません。#342 接続後、各backendについて server version/commit、model revision、
vision encoder/projector、推論形式、解像度・画像／動画frame数・token/context上限、合成素材と応答、
実 model の画像／動画の理解結果を記録します。`/models` の一覧だけで対応済みにせず、両動画経路の確認前に
全フロー切替完了としません。Strata動画未対応が続く場合は停止条件として明示し、必要な対応方針を別途確認します。
実モデルへの接続先・model・認証・vision構成の情報が未提供のため、現段階は #362 の先行基盤までです。
