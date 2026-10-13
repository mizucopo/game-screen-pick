# Game Screen Pick

ゲーム録画からブログ用画像を選ぶ Rust CLI。設定・入力検証、元動画の full-resolution frame／前後 context 抽出、二段階画像評価と選定を実装しています。
Rust library は品質判定・多様性選定・二段階評価の候補/context sheet・full-resolution JPEG と最終一覧/report の作成まで実装しています。
Rust library の cache・完了batch/chunkの再開・所有権検証と安全な成果物公開も実装しています。
Rust library に [共通 OpenAI 互換 client](docs/ai-client.md) と local JSON/schema 検証・有限 retry を実装しています。
通常 CLI は共通 client で一次／二次画像評価と semantic 動画理解を呼び、cache と安全な公開へ接続します。
Game Context は直接指定、または契約確認済みの [Brave 固定検索](docs/brave-search.md) と明示 model で生成します。
任意 runtime 管理は自己が起動した foreground session だけに限定します。実 model の能力・成果物品質は未確認です。

開発に Rust 1.88 以上（stable）、FFmpeg／FFprobe、jq が必要です。日本語描画 font は binary に含めます。

```sh
cargo build --locked
sh tests/check.sh
cargo run --locked -- --help
```

`config.example.toml` を `config/local.toml` にコピーし、接続先・model・推論レベルを明示します。
`ai.protocol` に推論形式・対応レベル・受信／送信上限を宣言します。未設定の vision/media は通信前に拒否します。
実際の backend/model の画像・動画能力と推論レベル対応は個別検証が必要です。mock 成功は実能力を保証しません。
認証は非空の `ai.api_key` を優先し、なければ `GAME_SCREEN_PICK_API_KEY` を使う設計です。
設定や秘密情報を含む `config/` と runtime cache は Git へ登録しません。

設定と入力を外部 command・通信・書込なしで確認できます。

```sh
cargo run --locked -- validate --config config/local.toml --count 2 \
  --game-context "探索・戦闘・会話の見やすい場面を選ぶ" recordings output
```

`--game-context` または `--game-title` の一方だけを指定します。枚数は 1〜999。
入力 directory 直下の通常動画を名前順に扱い、symlink は追跡しません。
出力は入力・設定・cacheと重ならない空、新規、または内容を検証できるRust所有の directory を指定します。
cacheは出力の親にある `.game-screen-pick-cache` を使い、`--cache-directory` で変更できます。
`validate` は所有権まで確認しますが、lock・cache作成・復旧は通常commandが行います。

抽出の確認は、既存の親 directory の下に新しい出力 directory を指定して行います。

```sh
cargo run --locked -- extract --at 0.5 --context-seconds 0.25 \
  "recordings/ゲーム 録画.mkv" frame-inspection
```

`before.png`／`frame.png`／`after.png` と `extraction.json` を出力します。
context を省略した場合は中央の画像のみです。時刻は選んだ映像 stream の最初の有効 PTS を0とする秒数。
要求時刻以降の最初の frame を選び、最終端点では最後の frame を使います。
JSON に source・stream・要求／実時刻・表示寸法を記録します。既存出力は上書きしません。

通常の選定 command は次の形です。画像能力を宣言した起動済み endpoint を指定します。

```sh
cargo run --locked -- --config config/local.toml --count 2 \
  --game-context "探索・戦闘・会話の見やすい場面を選ぶ" recordings output
```

同じ条件の正常な warm run は検索・推論・runtime 操作を行いません。`--cache-only` は欠落時に停止してこれらを禁止します。
生成 context の保存を許可していない場合、次回の自動生成は再検索が必要で、cache-only は停止します。
Strata の動画入力、vision 未設定、不適合な推論レベル、未確認の Brave 契約は理由付きで停止します。
成果物 E2E と実録画／実 model の Human Review は #343、配布検証は #344 の担当です。

詳細は [設定と抽出](docs/technical-reference.md)、[受入基準](docs/acceptance.md)、
[開発・リリース方針](CONTRIBUTING.md) を参照してください。
