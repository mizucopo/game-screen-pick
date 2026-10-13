# Brave 検索と Game Context

`--game-title` の Context 生成は固定の Brave Web Search を使い、取得した根拠を共通 OpenAI 互換 client に渡す。
ゲームの同定、ジャンル、進行構造、代表的な画面、選定上の優先事項、参照 URL を JSON にまとめる。
同名作品の混同、曖昧な同定、未解決の矛盾、検索根拠にない参照 URL は成功扱いしない。
JSON と mock の合格は、実モデルの画像・動画理解や生成内容の事実性を証明しない。

## 適用契約の確認

2026-10-13 に確認した [Brave 公開料金ページ](https://brave.com/search/api/) は Search を $5 / 1,000 requests、月 $5 の credit として説明している。
この公開説明から、利用アカウントの無料枠超過時に必ず無課金で停止するとは判断しない。
同ページの FAQ は API 結果の保存には保存権を明示した plan が必要としている。
一般の [Search API Terms of Use](https://api-dashboard.search.brave.com/terms-of-service) は 2026-09-01 更新で、一時利用以外の保存、派生物、モデルの作成・評価・改善等に制約がある。
ここから通常推論が一律禁止されると解釈するものではないが、個別の Order Form / 適用契約による権利を確認せず、公開説明だけで許可済みと扱わない。

本対応では利用者のアカウント契約を取得・検証していない。以下の宣言は利用者が適用契約を確認した記録であり、CLI が契約内容や契約締結を検証した証明ではない。
契約本文・キー・非公開の個人情報を設定に貼らず、ローカルの確認記録 ID や必要な範囲の参照を使う。

| 設定 | 意味 |
| --- | --- |
| `quota_stop_confirmed` + `quota_contract_reference` | 対象アカウントの無料枠上限で provider が停止し、paid overage を発生させないことを確認した宣言と参照 |
| `context_generation_confirmed` + `context_contract_reference` | 検索根拠を選択 backend に渡して Game Context を生成する用途の権利を確認した宣言と参照 |
| `generated_context_persistence_allowed` + `storage_contract_reference` | 生成文・引用 URL と下流の cache / report に含まれる生成 Context を保存できることを確認した宣言と参照 |

保存は既定で `false`。保存権の確認がない場合、検索・生成を一時的に行うための権利宣言だけでは保存を有効にできない。
未確認の quota または生成用途の宣言は、タイトル指定の実行を外部操作前に停止する。
Context 直接指定は Brave を利用しないので、未確認の Brave 設定を残していても利用できる。
本 CLI は契約の変更、paid plan の申込、billing 設定変更をしない。

## 設定

`[brave]` は任意の section。未知の key、文字列の boolean / 数値、範囲外値を拒否する。
キーは `api_key`、省略時は `GAME_SCREEN_PICK_BRAVE_API_KEY`。Debug、cache identity、診断にはキーを含めない。
以下は設定名を示す未確認の例であり、そのまま検索を許可する設定ではない。

```toml
[brave]
timeout_seconds = 30
max_response_bytes = 1048576
result_count = 5
country = "US"
search_lang = "en"
quota_stop_confirmed = false
# quota_contract_reference = "reviewed-applicable-account-contract-id"
context_generation_confirmed = false
# context_contract_reference = "reviewed-context-generation-rights-id"
generated_context_persistence_allowed = false
# storage_contract_reference = "reviewed-generated-and-downstream-storage-rights-id"
```

`timeout_seconds` は 1–300、`max_response_bytes` は 1 KiB–2 MiB、`result_count` は 1–20。
`country` は大文字 2 文字、`search_lang` は小文字の言語識別子。
言語識別子の対応は [Brave Web Search の公式 documentation](https://api-dashboard.search.brave.com/app/documentation/web-search/get-started) と対象アカウントの API を確認する。
unsupported language などの API error を別値に置き換えて retry しない。

endpoint は `https://api.search.brave.com/res/v1/web/search` に固定。
自動テストのため、同じ path を持つ literal loopback IP の HTTP(S) endpoint だけを例外として受け付ける。
`localhost`、他社・別 remote host、credentials / query / fragment を含む endpoint は受け付けない。
HTTP client は proxy を無効化し、remote authentication は公式 HTTPS endpoint のみに送る。

## HTTP と保存の境界

検索は `"ゲーム名" video game genre gameplay` と `"ゲーム名" video game progression screenshots representative scenes` の固定 2 query。
`safesearch=moderate`、`spellcheck=false`、`extra_snippets=true`、`text_decorations=false` を固定し、pagination・再試行・別 provider・paid fallback はない。
同一 origin の redirect だけ最大 3 回まで許可し、別 origin にキーを送らない。
2 query と redirect 全体でひとつの deadline を共有し、headers 待ち・body 待ちも含める。
response body は Content-Length の有無にかかわらず設定した上限で読み止める。

401 / 403、429、その他の HTTP / API error、空の検索結果、quota / rate window の残数 0 または壊れた残数 header はその場で停止する。
rate header や本プロセスの query 数は他のアプリによるアカウント利用や課金残高を表さないため、無料枠の無課金停止保証として代用しない。
raw response / title / snippet は一時メモリだけに置き、ファイルへ保存する経路を持たない。
キーが API response に混入した場合も、Context 生成前に拒否する。

生成 Context の cache identity は固定検索 revision、query 条件、endpoint、制限、適用契約参照、保存宣言と共通 client の model / inference level / protocol / prompt / schema revision を含める。
キー変更は cache identity に含めない。契約の内容・適用アカウント・権利が変わる場合は契約参照を更新する。
保存を許可した生成文だけを共通 `Session::generated_context` に保存し、raw 検索結果は保存しない。
