# game-screen-pick

1本のゲーム動画全体から、ブログへ掲載しやすい画像を指定枚数選定します。
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
uv run game-screen-pick [オプション] <入力動画> <出力フォルダ>
```

標準では30枚を選び、一次評価に`qwen3.8:27b`、二次評価に
`muse-glimmer:30b`を使います。

```bash
OLLAMA_HOST=192.168.1.31:11434 \
  uv run game-screen-pick \
  --game-title "冒険家エリオットの千年物語" \
  -n 30 \
  ./recording.mp4 \
  ./recording-selected
```

`--game-title`を省略すると、動画ファイル名から末尾の`Part 7`や`#02`を
除いた文字列をゲームタイトルとして使います。日付だけのファイル名など、
推測できない名前では明示してください。

### オプション

- `-n`, `--num`: 選択枚数（1から600、既定: 30）
- `--game-title`: ゲームタイトル。未指定時はファイル名から推測
- `--game-context`: ゲーム内容やブログ掲載意図の任意補足
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
- `selected-contact-sheet.jpg`: 選定画像を順位・動画時刻付きで一覧できる画像
- `report.json`: 選定時刻、score、scene、model評価を含むmachine-readable report

新規実行時の出力フォルダは空である必要があります。途中で中断した場合は、
同じ動画・選択条件・modelで同じコマンドを再実行してください。抽出済みframeと
完了済みOllama batchを再利用します。条件が異なる既存フォルダは上書きせず、
新しい出力フォルダを要求します。完了済み実行では全成果物のsizeとSHA-256を
検証し、Ollamaへ接続せずに結果を返します。

## 選定の流れ

1. 動画のほぼ先頭から末尾までを等間隔でsampleする
2. 暗転、白飛び、単色frameを機械的に除外する
3. 品質と時間分散から選択枚数の最大12倍を一次候補にする
4. 一次modelがブログ掲載価値、遷移、sceneを評価する
5. 場面・見た目・動画時刻を分散させ、最大3倍を二次候補にする
6. 二次modelが各候補の直前・対象・直後を見て再評価する
7. 遷移frameを除外し、近い重複とtitle/map/menuへの偏りを抑えて選定する
8. 個別画像、JSON report、一覧contact sheetを出力する

特定タイトル専用の選定ruleは持ちません。`--game-context`はmodel判断の補足で、
固定カテゴリや手動quotaとしては扱いません。

## バージョンとリリース

`main`を対象にするすべてのPull Requestは、ドキュメントやテストだけの変更も
含めて、`pyproject.toml`を未公開の新しいversionへ更新します。
