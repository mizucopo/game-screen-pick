# Game Screen Pick

ゲーム録画からブログ用画像を選ぶ Rust CLI。現在は設定・入力検証と、元動画の full-resolution frame／前後 context 抽出を実装しています。
画像評価・選定・contact sheet・cache・AI 接続・成果物公開は未実装です。通常の選定 command は理由を示して終了 code 1 で停止します。

開発に Rust stable、FFmpeg／FFprobe、jq が必要です。

```sh
cargo build --locked
sh tests/check.sh
cargo run --locked -- --help
```

`config.example.toml` を `config/local.toml` にコピーし、接続先・model・推論レベルを明示します。
実際の backend/model の画像・動画能力と推論レベル対応は、AI 接続の実装時に検証します。
認証は非空の `ai.api_key` を優先し、なければ `GAME_SCREEN_PICK_API_KEY` を使う設計です。
設定や秘密情報を含む `config/` と runtime cache は Git へ登録しません。

設定と入力を外部 command・通信・書込なしで確認できます。

```sh
cargo run --locked -- validate --config config/local.toml --count 2 \
  --game-context "探索・戦闘・会話の見やすい場面を選ぶ" recordings output
```

`--game-context` または `--game-title` の一方だけを指定します。枚数は 1〜999。
入力 directory 直下の通常動画を名前順に扱い、symlink は追跡しません。
出力は入力・設定と重ならない空または新しい directory を指定します。

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
