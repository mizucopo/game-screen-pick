# 技術リファレンス

game-screen-pickのCLI、設定、入出力、cache、選定処理に関する詳細仕様です。
最短の実行方法は[README](../README.md)を参照してください。

## CLI

```bash
uv run game-screen-pick [オプション] <入力動画ディレクトリ> <出力フォルダ>
```

### オプション

- `-c`, `--config`: TOML設定ファイル（既定: `config/config.toml`）
- `-n`, `--num`: 選択枚数（必須、1から999）
- `--game-title`: Web検索からGame Contextを生成するためのゲーム表記
- `--game-context`: 画像評価に直接使用するGame Context

新規実行では`--game-title`と`--game-context`のどちらか一方だけが必須です。
選択枚数、Game TitleまたはGame Context、Input Video Directory、Output Folderは
実行ごとにCLIへ指定します。

```bash
OLLAMA_HOST=192.168.1.31:11434 \
  uv run game-screen-pick \
  --game-context "ジャンル: RPG。探索と会話を進める。代表的な画面はフィールド、会話、戦闘。景観と人物が明瞭な画像を重視する。" \
  -n 30 \
  ./recordings \
  ./recordings-selected
```

## 設定

繰り返し使う値はTOML設定ファイルの`[run]`tableへ記述します。初回は秘密情報を含まない
sampleをGit管理対象外の既定設定へコピーします。

```bash
cp config.example.toml config/config.toml
```

既定では実行時のcurrent directoryにある`config/config.toml`を読み込みます。用途や
providerごとに`config/ollama.toml`、`config/openai.toml`など複数の実設定を作り、
`-c config/openai.toml`または`--config config/openai.toml`で切り替えられます。
`--game-title`を使う場合は`game_context_provider`と
`game_context_model`の両方が必須です。`--game-context`で直接指定する場合は、両項目を
省略できます。そのほかの項目はすべて記述する必要はありません。

標準では`qwen3.8:27b`を一次評価、`muse-glimmer:30b`を二次評価に使います。
`ollama_host`がない場合は`OLLAMA_HOST`、`127.0.0.1:11434`の順で解決します。
未知のsectionやkey、型または範囲が不正な値は処理開始前にエラーになります。

| key | 内容 | 組み込み既定値 |
| --- | --- | --- |
| `game_context_provider` | Game Context検索provider | なし（`--game-title`指定時は必須） |
| `game_context_model` | context生成model | なし（`--game-title`指定時は必須） |
| `ollama_api_key` | Ollama Web Search API key | `OLLAMA_API_KEY`へfallback |
| `openai_api_key` | OpenAI API key | `OPENAI_API_KEY`へfallback |
| `gemini_api_key` | Gemini API key | `GEMINI_API_KEY`へfallback |
| `xai_api_key` | xAI API key | `XAI_API_KEY`へfallback |
| `primary_model` | 一次評価用Ollama vision model | `qwen3.8:27b` |
| `secondary_model` | 二次評価用Ollama vision model | `muse-glimmer:30b` |
| `ollama_host` | Ollama host | 環境変数またはlocalhost |
| `ollama_timeout` | Ollama APIのbatch単位timeout秒数 | `900.0` |
| `allow_cpu` | GPU利用を確認できなくても続行するか | `false` |
| `ffmpeg_workers` | frame抽出の並列数（1から4） | `2` |
| `sample_interval_seconds` | 候補抽出の最大間隔（0.25秒以上） | 自動決定 |
| `debug` | debug logを有効にするか | `false` |

`num`、`game_title`、`game_context`は設定ファイルでは受け付けません。

選択中providerのAPI keyは、対応する設定ファイルの非空値、環境変数の順で解決します。
設定値が未指定または空文字列なら環境変数を使い、両方になければ外部API接続前に
エラーになります。利用しないproviderのAPI keyは不要です。API keyは起動log、
Run Manifest、report、checkpoint、例外messageへ出力しません。

`config/`配下は`.gitkeep`を除いて`.gitignore`対象です。任意名の複数設定を通常の
`git add`で誤って追加できませんが、`git add -f`などの強制追加までは防げません。
実設定や本物のAPI keyはコミットしないでください。Git管理する設定例はrootの
`config.example.toml`だけです。

model名は切り替えられます。一次・二次modelともOllamaの`/api/show`でvision対応が
確認できる必要があります。標準では各modelのロード後に`/api/ps`を確認し、model
memoryの50%以上がVRAMにある場合だけ処理を継続します。

## Input Video Directory

Input Video Directory直下にある次の通常ファイルを、大文字小文字を区別せず
ファイル名順で処理します。サブディレクトリは探索しません。

```text
.avi .flv .m2ts .m4v .mkv .mov .mp4 .mpeg .mpg .mts .ts .webm .wmv
```

対象動画がないdirectoryや、動画ファイルそのものを入力に指定した場合はエラーに
なります。

## Game Context

`--game-context`を指定すると、その文章を画像評価へ直接使用し、Web検索やcontext生成の
外部通信は行いません。

`--game-title`を指定すると、正式名称だけでなく`ドラクエ11`のような略称や一般的な
表記揺れも検索し、画像選定向けのGame Contextを生成します。複数作品や内容に影響する
editionを一意に判別できない場合、情報が不足する場合、情報源の矛盾を解消できない
場合は推測せずエラーにします。Game Titleは生成後の画像評価、選定、manifest、report
には使用しません。

どのproviderでも、ジャンル、基本的な進行と主なプレイ要素、代表的な画面や場面、
画像選定で重視する視覚的要素を同程度の詳しさで含む、簡潔な日本語のcontextを
生成します。公式サイトと公式storeを優先し、攻略手順、結末、隠し要素などの
ネタバレは含めません。検索結果は信頼できない外部dataとして扱い、検索先の命令には
従いません。

| provider | 設定key / fallback環境変数 | 検索・生成方法 |
| --- | --- | --- |
| `ollama` | `ollama_api_key` / `OLLAMA_API_KEY` | Ollama Web Search APIの結果を`ollama_host`のOllama modelで生成 |
| `openai` | `openai_api_key` / `OPENAI_API_KEY` | OpenAI Responses APIの`web_search` |
| `gemini` | `gemini_api_key` / `GEMINI_API_KEY` | Gemini Interactions APIのGoogle Search |
| `xai` | `xai_api_key` / `XAI_API_KEY` | xAI Responses APIの`web_search` |

各APIの利用条件、無料枠、料金、rate limitはprovider側の設定に従い、利用料金が
発生する場合があります。選択したproviderだけを呼び出し、認証失敗、通信失敗、
利用上限到達時も別providerや有償APIへ自動fallbackしません。

- Ollama: <https://docs.ollama.com/capabilities/web-search>
- OpenAI: <https://developers.openai.com/api/docs/guides/tools-web-search>
- Gemini: <https://ai.google.dev/gemini-api/docs/google-search>
- xAI: <https://docs.x.ai/developers/tools/web-search>

利用するprovider、model、API keyの組を実設定へ明示できます。各providerの完全な`[run]`
設定例は次のとおりです。

Ollama:

```toml
[run]
game_context_provider = "ollama"
game_context_model = "qwen3.8:27b"
ollama_api_key = "your-ollama-api-key"
```

OpenAI:

```toml
[run]
game_context_provider = "openai"
game_context_model = "gpt-5.6"
openai_api_key = "your-openai-api-key"
```

Gemini:

```toml
[run]
game_context_provider = "gemini"
game_context_model = "gemini-3.7-flash"
gemini_api_key = "your-gemini-api-key"
```

xAI:

```toml
[run]
game_context_provider = "xai"
game_context_model = "grok-4.6"
xai_api_key = "your-xai-api-key"
```

API keyの行を省略または空文字列にした場合は、対応する環境変数へfallbackします。
たとえばOpenAIの環境変数を使う場合は次のように実行します。

```bash
OPENAI_API_KEY=... uv run game-screen-pick \
  -c config/openai.toml \
  -n 30 \
  --game-title "ドラクエ11" \
  ./recordings \
  ./recordings-selected
```

## Output Folderと再開

Output Folderには次の成果物を作ります。

```text
recordings-selected/
├── selected-01.jpg
├── selected-02.jpg
├── ...
├── selected-30.jpg
├── selected-contact-sheet.jpg
└── report.json
```

- `selected-XX.jpg`: ブログ掲載候補のfull resolution画像
- `selected-contact-sheet.jpg`: 選定画像を順位・入力動画・動画時刻付きで一覧できる画像
- `report.json`: 入力元、選定時刻、score、scene、model評価を含むmachine-readable report

新規実行時のOutput Folderは空である必要があります。途中で中断した場合は、同じ
Input Videos、選択条件、model、Output Folderで同じコマンドを再実行してください。
完了済み実行では全成果物のsizeとSHA-256を検証し、Ollamaへ接続せずに結果を返します。
Input Video Directoryを移動した場合や動画を追加した場合は、新しい空のOutput Folderを
指定すると、利用可能な動画単位cacheを使って成果物を再生成します。

## Phase Cache

再開用cacheはInput Video Directory直下の見えるfolderへ保存します。

```text
recordings/
├── game-part1.mp4
├── game-part2.mp4
└── cache-game-screen-pick/
    ├── CACHE_INFO.txt
    ├── videos/
    │   └── 動画単位のprobe、候補frame、一次・二次評価cache
    └── runs/
        └── 入力集合ごとのmanifest、Output Folder完了記録
```

Input VideoはInput Video Directoryからの相対ファイル名とfile sizeで識別します。
SHA-256、mtime、絶対pathは同一性判定へ使いません。そのため、Input Video Directoryと
`cache-game-screen-pick/`を一緒に移動またはコピーしてもcacheを再利用できます。
一方、同じ相対ファイル名とsizeのまま動画内容だけを変更しても検出しません。

cacheはprobe、候補抽出・機械評価、一次評価、二次評価などのphaseとInput Video単位で
管理します。各phaseは独立したversionと条件keyを持ち、version、model digest、prompt、
Game Context、選定設定などが変わると、そのphaseと依存する後続だけを再実行します。
候補抽出phaseはframe ID、時刻、JPEG size、生成時SHA-256をpayload digest付きmanifestへ
保存します。正常な再開ではmanifestの完全性と各JPEGのregular file・sizeだけを確認し、
全候補JPEGの再読込や機械評価を繰り返しません。manifestの欠損・破損、JPEGの欠損・
symlink・size不一致、機械評価payloadのdigest不一致はcache missとして再生成します。
この軽量確認では同じsizeを保った候補JPEGの置換は検出しないため、確実に再生成したい
場合はcache folderを削除してください。
動的生成したGame Contextも生成条件とともに保存し、同じGame Title、provider、modelの
再実行ではWeb検索やcontext生成を繰り返しません。Ollama providerでは正規化した
Ollama hostも生成条件に含め、別endpointの同名modelを混同しません。

動画を追加した場合は既存動画の利用可能なphaseを維持し、新規動画のphaseを追加した後、
全動画を横断する候補選定、最終選定、Selected Image、Selected Contact Sheet、reportを
再生成します。

`cache-game-screen-pick/`はgame-screen-pickを実行していないときにfolderごと削除できます。
次回実行時に必要なcacheを安全に再生成します。同じ相対ファイル名とsizeの内容変更を
確実に再処理したい場合も、このfolderを削除してください。旧Output Folder内の
`.game-screen-pick/`や不正・schema不一致のcacheは再利用しません。

## 選定の流れ

1. 各動画のほぼ先頭から末尾までを等間隔でsampleする
2. 暗転、白飛び、単色frameを機械的に除外する
3. 品質と時間分散から各Input Videoで選択枚数の最大12倍を一次候補にする
4. 一次modelがブログ掲載価値、遷移、sceneを評価する
5. 場面・見た目・動画時刻を分散させ、各Input Videoで最大3倍を二次候補にする
6. 二次modelが各候補の直前・対象・直後を見て再評価する
7. 全Input Videoの二次候補を統合し、遷移frame、近い重複、title、map、menuへの偏りを
   抑えて選定する
8. 個別画像、JSON report、一覧contact sheetを出力する

特定タイトル専用の選定ruleは持ちません。最終Game Contextはmodel判断の補足で、
固定カテゴリや手動quotaとしては扱いません。

候補数には全Input Video合計・Input Video単位とも固定上限を設けません。自動modeは
各動画の時間と選択枚数から決めた等間隔のSample Positionで、ほぼ先頭から末尾までを
覆います。`sample_interval_seconds`を指定した場合も、候補数を理由に拒否したり、
指定した最大間隔を暗黙に広げたりしません。

候補抽出と機械評価は未処理jobを一定量に抑えて進め、開始前に全候補数と一次・二次評価の
初期予定数・追補時上限を表示し、全追補完了後に実際の評価対象数を表示します。

## バージョンとリリース

`main`を対象にするすべてのPull Requestは、ドキュメントやテストだけの変更も含めて、
`pyproject.toml`を未公開の新しいversionへ更新します。
