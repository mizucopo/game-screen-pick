# Game Screen Pick

ゲーム録画からブログ用の原寸 JPEG を選び、一覧画像と選定根拠の `report.json` を作る Rust CLI。
等間隔の画像候補評価と semantic 動画理解に対応し、完了した評価の cache と安全な再開・出力保護を備えます。
実 backend/model の能力、実録画の選定品質、Brave の適用契約と live 接続、Human Review は未検証です。
合成素材と固定応答の成功でこれらを確認済みとは扱いません。[検証状況](docs/verification.md) を参照してください。

## 配布 binary の配置

対象は Apple Silicon macOS の `aarch64-apple-darwin` と Linux x86_64 GNU の `x86_64-unknown-linux-gnu` です。
FFmpeg／FFprobe と、画像・動画能力を確認した推論 endpoint を別途用意します。日本語 font は binary に含まれます。
本体の利用に Rust toolchain、Python、uv、venv、OpenCV は不要です。
OS／dynamic library と FFmpeg の条件・検証範囲は [配布検証](docs/release.md#配布対象と検証) と archive 内の `DEPENDENCIES.txt` で確認してください。

[GitHub Releases](https://github.com/mizucopo/game-screen-pick/releases) から、Rust archive が公開された version を選びます。
Rust 版の実公開は、integration 向け PR の完了後に承認を受ける別工程です。公開前の確認には PR の Actions artifact を使います。
空の download directory で `RELEASE_VERSION` を選んだ version に置き換え、対象 target を一つ指定します。

```sh
version=RELEASE_VERSION
target=aarch64-apple-darwin # Linux: x86_64-unknown-linux-gnu
archive="game-screen-pick-${version}-${target}.tar.gz"
base="https://github.com/mizucopo/game-screen-pick/releases/download/${version}"
curl --fail --location --remote-name "$base/$archive"
curl --fail --location --remote-name "$base/$archive.sha256"
```

checksum を確認してから展開します。macOS は次を使います。

```sh
shasum -a 256 -c "$archive.sha256"
```

Linux は `sha256sum -c "$archive.sha256"` を使います。検証失敗時は展開・配置せず再取得してください。
成功後、実行 file を利用者の PATH に配置します。

```sh
tar -xzf "$archive"
bundle="${archive%.tar.gz}"
mkdir -p "$HOME/.local/bin"
install -m 755 "$bundle/bin/game-screen-pick" "$HOME/.local/bin/game-screen-pick"
export PATH="$HOME/.local/bin:$PATH"
game-screen-pick --version
game-screen-pick --help
ffmpeg -version
ffprobe -version
```

新しい shell でも使う場合は `~/.local/bin` を shell の PATH 設定へ追加します。
archive は設定例、利用文書、license、build・依存情報も含みます。

## 設定して実行

新しい Rust 設定を作り、接続先・model・推論レベルと `ai.protocol` の能力・上限を明示します。
古い設定は流用せず、手元の設定・録画・出力・cache を保持してください。
以下の `cp -n` は既存 file を上書きしません。`config/rust.toml` がある場合は別の新規 file 名を選び、後の command にも指定します。

```sh
mkdir -p config recordings
cp -n "$bundle/config.example.toml" config/rust.toml
```

[設定例](config.example.toml) の vision は無効のため、接続する backend/model の画像能力を確認して `ai.protocol.media` を設定します。
`recordings` へ利用許可のある手元の動画を置きます。
動画理解を選ぶ場合は動画能力も実測してください。非対応の Strata 動画経路、未設定の vision、不適合な推論レベルは理由付きで停止します。
認証は非空の `ai.api_key`、なければ `GAME_SCREEN_PICK_API_KEY` を使います。設定・秘密情報・runtime cache は Git に登録しません。
起動済み endpoint には `[runtime]` が不要です。任意の起動／停止は自己が開始した foreground session だけを管理します。

まず設定と入力を外部 command・通信・書込なしで確認し、同じ引数で実行します。

```sh
game-screen-pick validate --config config/rust.toml --count 2 \
  --cache-directory .game-screen-pick-cache-rust \
  --game-context "探索・戦闘・会話の見やすい場面を選ぶ" recordings output-rust
game-screen-pick --config config/rust.toml --count 2 \
  --cache-directory .game-screen-pick-cache-rust \
  --game-context "探索・戦闘・会話の見やすい場面を選ぶ" recordings output-rust
```

枚数は 1〜999。入力直下の通常動画を名前順に扱い、symlink は追跡しません。
出力は入力・設定・cache と重ならない、新規・空・所有権を検証できる Rust 成果物 directory を指定します。
未所有・編集済み file は保持して拒否します。その場合は既存出力を削除せず、別の空の出力先を指定してください。
cache は新しい Rust namespace を使い、旧 cache を読み込みません。既定では出力の親の `.game-screen-pick-cache`、変更は上記の `--cache-directory` です。

`--game-context` と `--game-title` は一方だけ指定します。直接 context は検索しません。
Game Title からの生成は [Brave の適用契約](docs/brave-search.md) と認証を確認後に設定してください。
同じ条件の正常な warm run は検索・推論・runtime 操作を行いません。`--cache-only` は必要な cache がなければ外部操作せず停止します。
完成済みの所有成果物は、cache 消失後も入力・実効条件が一致すれば再利用します。
生成 context の保存を許可していない場合、再計算に必要な context がなければ再検索が必要です。

## 更新

新 version の archive と checksum を新しい空 directory へ取得し、上記と同じ検証・展開・配置を行います。
設定例を既存設定へ上書きせず、必要な設定差分を別の新しい Rust 設定で確認してください。
録画・既存出力・cache は保持し、新しい空出力先で `validate` と実行を確認します。
model 重み・server 設定が変わった場合は `ai.cache_revision` を更新します。
未検証の OS・target や下流 package manager への配布を、この archive の確認から対応済みとは扱いません。

## 抽出と開発

推論を使わず元 frame と前後 context だけ確認できます。既存の親 directory の下に新規出力先を指定します。

```sh
game-screen-pick extract --at 0.5 --context-seconds 0.25 \
  "recordings/ゲーム 録画.mkv" frame-inspection
```

`before.png`／`frame.png`／`after.png` と `extraction.json` を作ります。context 省略時は中央だけです。
時刻は選んだ映像 stream の最初の有効 PTS を0とする秒数。要求時刻以降の最初の frame、最終端点は最後の frame を使い、既存出力は上書きしません。

開発には Rust 1.88 以上（stable）、FFmpeg／FFprobe、jq が必要です。

```sh
cargo build --locked
sh tests/check.sh
cargo run --locked -- --help
```

詳細は [設定・抽出・選定](docs/technical-reference.md)、[AI 接続](docs/ai-client.md)、[受入基準](docs/acceptance.md)、
[配布・公開手順](docs/release.md)、[開発方針](CONTRIBUTING.md) を参照してください。
