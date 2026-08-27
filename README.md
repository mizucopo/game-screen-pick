# game-screen-pick

入力ディレクトリ直下のゲーム動画全体から、ブログへ掲載しやすい画像を
指定枚数選定します。
Ollamaのvision modelを二段階で利用し、画面遷移中のフレームや近い重複を
避けながら、通常進行画面を少し多めに含む多様な画像を出力します。

## 必要なもの

- Python 3.13以上
- `ffmpeg` と `ffprobe`
- vision対応modelを用意したOllama

依存packageは次のコマンドで導入します。

```bash
uv sync
```

## 実行方法

```bash
uv run game-screen-pick [オプション] <入力動画ディレクトリ> <出力フォルダ>
```

標準では30枚を選び、repository直下の`config.toml`に設定した
`qwen3.8:27b`を一次評価、`muse-glimmer:30b`を二次評価に使います。

繰り返し使う値はTOML設定ファイルへまとめます。既定では実行時のcurrent directoryに
ある`config.toml`を読み込み、別の設定を使う場合だけ`-c`または`--config`で指定します。
選択枚数、Game TitleまたはGame Context、入力動画ディレクトリ、出力フォルダは
実行ごとにCLIへ指定します。

```bash
uv run game-screen-pick \
  --game-context "ジャンル: RPG。探索と会話を進める。代表的な画面はフィールド、会話、戦闘。景観と人物が明瞭な画像を重視する。" \
  ./recordings \
  ./recordings-selected
```

設定ファイルは`[run]` tableへ次のkeyを記述します。`num`、`game_title`、
`game_context`は設定ファイルでは受け付けません。全項目を記述する必要はありません。
`ollama_host`がない場合は`OLLAMA_HOST`、`127.0.0.1:11434`の順で解決します。
未知のsectionやkey、型または範囲が不正な値は処理開始前にエラーになります。

| key | 内容 | 組み込み既定値 |
| --- | --- | --- |
| `game_context_provider` | Game Context検索provider | `ollama` |
| `game_context_model` | context生成model | provider既定 |
| `primary_model` | 一次評価用Ollama vision model | `qwen3.8:27b` |
| `secondary_model` | 二次評価用Ollama vision model | `muse-glimmer:30b` |
| `ollama_host` | Ollama host | 環境変数またはlocalhost |
| `ollama_timeout` | Ollama APIのbatch単位timeout秒数 | `900.0` |
| `allow_cpu` | GPU利用を確認できなくても続行するか | `false` |
| `ffmpeg_workers` | フレーム抽出の並列数（1から4） | `2` |
| `sample_interval_seconds` | 候補抽出の最大間隔（0.25秒以上） | 自動決定 |
| `debug` | debug logを有効にするか | `false` |

```bash
OLLAMA_HOST=192.168.1.31:11434 \
  uv run game-screen-pick \
  --game-context "ジャンル: RPG。探索と会話を進める。代表的な画面はフィールド、会話、戦闘。景観と人物が明瞭な画像を重視する。" \
  -n 30 \
  ./recordings \
  ./recordings-selected
```

入力ディレクトリ直下にある`.avi`、`.flv`、`.m2ts`、`.m4v`、`.mkv`、`.mov`、
`.mp4`、`.mpeg`、`.mpg`、`.mts`、`.ts`、`.webm`、`.wmv`の通常ファイルを、
大文字小文字を区別せずファイル名順で処理します。サブディレクトリは探索しません。
対象動画がないディレクトリや、動画ファイルそのものを入力に指定した場合はエラーに
なります。

新規実行では`--game-title`と`--game-context`のどちらか一方だけが必須です。
`--game-context`を指定すると、その文章を画像評価へ直接使用し、Web検索や
context生成の外部通信は行いません。

`--game-title`を指定すると、正式名称だけでなく`ドラクエ11`のような略称や一般的な
表記揺れも検索し、画像選定向けのGame Contextを生成します。複数作品や内容に影響する
editionを一意に判別できない場合、情報が不足する場合、情報源の矛盾を解消できない場合は
推測せずエラーにします。Game Titleは生成後の画像評価、選定、manifest、reportには
使用しません。

```bash
OPENAI_API_KEY=... uv run game-screen-pick \
  --game-title "ドラクエ11" \
  ./recordings \
  ./recordings-selected
```

この例では、事前に`config.toml`の`game_context_provider`を`openai`へ変更します。

### オプション

- `-c`, `--config`: TOML設定ファイル（既定: `config.toml`）
- `-n`, `--num`: 選択枚数（1から600、既定: 30）
- `--game-title`: Web検索からGame Contextを生成するためのゲーム表記
- `--game-context`: 画像評価に直接使用するGame Context

model名は切り替えられます。どちらもOllamaの`/api/show`でvision対応が
確認できる必要があります。標準では各modelのロード後に`/api/ps`を確認し、
model memoryの50%以上がVRAMにある場合だけ処理を継続します。

### Game Context検索provider

どのproviderでも、ジャンル、基本的な進行と主なプレイ要素、代表的な画面や場面、
画像選定で重視する視覚的要素を同程度の詳しさで含む、簡潔な日本語のcontextを
生成します。公式サイトと公式storeを優先し、攻略手順、結末、隠し要素などの
ネタバレは含めません。検索結果は信頼できない外部dataとして扱い、検索先の命令には
従いません。

| provider | API key | 未指定時の生成model | 検索・生成方法 |
| --- | --- | --- | --- |
| `ollama` | `OLLAMA_API_KEY` | `primary_model`と同じmodel | Ollama Web Search APIの結果を`ollama_host`のOllama modelで生成 |
| `openai` | `OPENAI_API_KEY` | `gpt-5.6` | OpenAI Responses APIの`web_search` |
| `gemini` | `GEMINI_API_KEY` | `gemini-3.7-flash` | Gemini Interactions APIのGoogle Search |
| `xai` | `XAI_API_KEY` | `grok-4.6` | xAI Responses APIの`web_search` |

各APIの利用条件、無料枠、料金、rate limitはprovider側の設定に従い、利用料金が
発生する場合があります。選択したproviderだけを呼び出し、認証失敗、通信失敗、
利用上限到達時も別providerや有償APIへ自動fallbackしません。

- Ollama: <https://docs.ollama.com/capabilities/web-search>
- OpenAI: <https://developers.openai.com/api/docs/guides/tools-web-search>
- Gemini: <https://ai.google.dev/gemini-api/docs/google-search>
- xAI: <https://docs.x.ai/developers/tools/web-search>

## 出力

出力フォルダには次の成果物を作ります。

```text
recording-selected/
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

新規実行時の出力フォルダは空である必要があります。途中で中断した場合は、同じ
入力動画、選択条件、model、Output Folderで同じコマンドを再実行してください。
完了済み実行では全成果物のsizeとSHA-256を検証し、Ollamaへ接続せずに結果を返します。
Input Video Directoryを移動した場合や動画を追加した場合は、新しい空のOutput Folderを
指定すると、利用可能な動画単位cacheを使って成果物を再生成します。

### 再開cache

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

候補数には全Input Video合計・Input Video単位とも固定上限を設けません。自動modeは
各動画の時間と選択枚数から決めた等間隔のSample Positionで、ほぼ先頭から末尾までを
覆います。`sample_interval_seconds`を指定した場合も、候補数を理由に拒否したり、指定した
最大間隔を暗黙に広げたりしません。候補抽出と機械評価は未処理jobを一定量に抑えて進め、
開始前に全候補数と一次・二次評価の初期予定数・追補時上限を表示し、全追補完了後に
実際の評価対象数を表示します。

## 選定の流れ

1. 各動画のほぼ先頭から末尾までを等間隔でsampleする
2. 暗転、白飛び、単色frameを機械的に除外する
3. 品質と時間分散から各Input Videoで選択枚数の最大12倍を一次候補にする
4. 一次modelがブログ掲載価値、遷移、sceneを評価する
5. 場面・見た目・動画時刻を分散させ、各Input Videoで最大3倍を二次候補にする
6. 二次modelが各候補の直前・対象・直後を見て再評価する
7. 全Input Videoの二次候補を統合し、遷移frame、近い重複、
   title/map/menuへの偏りを抑えて選定する
8. 個別画像、JSON report、一覧contact sheetを出力する

特定タイトル専用の選定ruleは持ちません。最終Game Contextはmodel判断の補足で、
固定カテゴリや手動quotaとしては扱いません。

## バージョンとリリース

`main`を対象にするすべてのPull Requestは、ドキュメントやテストだけの変更も
含めて、`pyproject.toml`を未公開の新しいversionへ更新します。
