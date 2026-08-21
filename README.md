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

標準では30枚を選び、一次評価に`qwen3.8:27b`、二次評価に
`muse-glimmer:30b`を使います。

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
OPENAI_API_KEY=... \
  uv run game-screen-pick \
  --game-title "ドラクエ11" \
  --game-context-provider openai \
  ./recordings \
  ./recordings-selected
```

### オプション

- `-n`, `--num`: 選択枚数（1から600、既定: 30）
- `--game-title`: Web検索からGame Contextを生成するためのゲーム表記
- `--game-context`: 画像評価に直接使用するGame Context
- `--game-context-provider`: `--game-title`指定時の検索provider。`ollama`、`openai`、`gemini`、`xai`から選択（既定: `ollama`）
- `--game-context-model`: context生成model。未指定時はprovider既定
- `--primary-model`: 一次評価用Ollama vision model
- `--secondary-model`: 遷移確認を含む二次評価用Ollama vision model
- `--ollama-host`: Ollama host。CLI、`OLLAMA_HOST`、localhostの順で解決
- `--ollama-timeout`: Ollama APIのbatch単位timeout秒数（既定: 900）
- `--ffmpeg-workers`: フレーム抽出の並列数（1から4、既定: 2）
- `--sample-interval-seconds`: 候補抽出の最大間隔（0.25秒以上）。通常は自動設定を推奨。候補が4,000件を超える指定は拒否
- `--allow-cpu`: GPU利用を確認できなくても続行する
- `--debug`: debug logを有効化する

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
| `ollama` | `OLLAMA_API_KEY` | `--primary-model`と同じmodel | Ollama Web Search APIの結果を`--ollama-host`のOllama modelで生成 |
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
├── report.json
└── .game-screen-pick/
    └── 再開用のmanifest、候補画像、評価cache
```

- `selected-XX.jpg`: ブログ掲載候補のfull resolution画像
- `selected-contact-sheet.jpg`: 選定画像を順位・入力動画・動画時刻付きで一覧できる画像
- `report.json`: 入力元、選定時刻、score、scene、model評価を含むmachine-readable report

新規実行時の出力フォルダは空である必要があります。途中で中断した場合は、
同じ入力ディレクトリ内容・選択条件・modelで同じコマンドを再実行してください。
抽出済みframeと完了済みOllama batchを再利用します。対象動画の追加、削除、改名、
内容変更などで条件が変わった既存フォルダは上書きせず、新しい出力フォルダを要求します。
完了済み実行では全成果物のsizeとSHA-256を検証し、Ollamaへ接続せずに結果を返します。
動的生成したGame Context、provider、modelはmanifestへ保存します。再開時は保存済みの
Game Contextを使用し、Web検索やcontext生成を繰り返しません。従来manifestに保存済みの
Game Contextも再利用します。

候補数は全入力動画の合計で4,000件までです。指定した抽出間隔で上限を超える
場合は、`--sample-interval-seconds`を広げてください。

## 選定の流れ

1. 各動画のほぼ先頭から末尾までを等間隔でsampleする
2. 暗転、白飛び、単色frameを機械的に除外する
3. 品質と時間分散から選択枚数の最大12倍を一次候補にする
4. 一次modelがブログ掲載価値、遷移、sceneを評価する
5. 入力動画・場面・見た目・動画時刻を分散させ、最大3倍を二次候補にする
6. 二次modelが各候補の直前・対象・直後を見て再評価する
7. 遷移frameを除外し、近い重複とtitle/map/menuへの偏りを抑えて選定する
8. 個別画像、JSON report、一覧contact sheetを出力する

特定タイトル専用の選定ruleは持ちません。最終Game Contextはmodel判断の補足で、
固定カテゴリや手動quotaとしては扱いません。

## バージョンとリリース

`main`を対象にするすべてのPull Requestは、ドキュメントやテストだけの変更も
含めて、`pyproject.toml`を未公開の新しいversionへ更新します。
