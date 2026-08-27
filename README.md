# game-screen-pick

ゲーム録画全体から、ブログへ掲載しやすい画像を指定枚数選びます。
Ollamaのvision modelで画面遷移中のframeや近い重複を避けながら、
通常進行画面を少し多めに含む多様な画像を出力します。

## 必要なもの

- Python 3.13以上
- uv
- `ffmpeg`と`ffprobe`
- vision対応modelを用意したOllama

## はじめかた

依存packageを導入します。

```bash
uv sync
```

ゲームタイトルと、動画を置いたInput Video Directory、空のOutput Folderを指定して
実行します。`--game-title`を使う場合は、事前に環境変数`OLLAMA_API_KEY`を
設定してください。標準では30枚を選びます。

```bash
uv run game-screen-pick \
  --game-title "ドラクエ11" \
  ./recordings \
  ./recordings-selected
```

Web検索でGame Contextを生成する代わりに、直接指定することもできます。

```bash
uv run game-screen-pick \
  --game-context "ジャンル: RPG。探索と会話を進める。代表的な画面はフィールド、会話、戦闘。景観と人物が明瞭な画像を重視する。" \
  ./recordings \
  ./recordings-selected
```

modelやproviderなど、繰り返し使う値はcurrent directoryの`config.toml`で変更できます。

## 出力

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
- `selected-contact-sheet.jpg`: 順位、入力動画、動画時刻をまとめた確認用画像
- `report.json`: 入力元、選定時刻、score、scene、model評価を含むreport

中断後は同じコマンドで再開できます。Input Video Directoryを移動した場合や動画を
追加した場合も、利用可能な動画単位cacheを再利用できます。

## 詳しい仕様

CLI option、設定項目、対応動画、Game Context provider、cache、選定処理、
release規約は[技術リファレンス](docs/technical-reference.md)を参照してください。
