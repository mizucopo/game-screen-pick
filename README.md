# Game Screen Pick

ゲーム録画からブログ用画像を選ぶ Rust CLI。現在は設定・入力検証と、元動画の full-resolution frame／前後 context 抽出を実装しています。
Rust library は品質判定・多様性選定・二段階評価の候補/context sheet・full-resolution JPEG と最終一覧/report の作成まで実装しています。
Rust library の cache・完了batch/chunkの再開・所有権検証と安全な成果物公開も実装しています。
Rust library に [共通 OpenAI 互換 client](docs/ai-client.md) と local JSON/schema 検証・有限 retry を実装しています。
用途別 prompt・Brave・runtime と CLI への接続は #342 で行います。通常の選定 command は事前確認・排他取得後、理由を示して終了 code 1 で停止します。

開発に Rust 1.88 以上（stable）、FFmpeg／FFprobe、jq が必要です。日本語描画 font は binary に含めます。

```sh
cargo build --locked
sh tests/check.sh
cargo run --locked -- --help
```

`config.example.toml` を `config/local.toml` にコピーし、接続先・model・推論レベルを明示します。
`ai.protocol` に推論形式・対応レベル・受信／送信上限を宣言します。未設定の vision/media は通信前に拒否します。
実際の backend/model の画像・動画能力と推論レベル対応は、用途接続後に個別検証します。mock 成功は実能力を保証しません。
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

通常の選定 command は次の形です。現段階では入力検証後に未実装として停止し、完成済み画像セットを作りません。

```sh
cargo run --locked -- --config config/local.toml --count 2 \
  --game-context "探索・戦闘・会話の見やすい場面を選ぶ" recordings output
```

詳細は [設定と抽出](docs/technical-reference.md)、[受入基準](docs/acceptance.md)、
[開発・リリース方針](CONTRIBUTING.md) を参照してください。
