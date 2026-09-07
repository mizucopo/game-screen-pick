# game-screen-pick

ゲーム録画全体から、ブログへ掲載しやすい画像を指定枚数選びます。
vLLMによる動画理解、またはOllamaによる静止画評価を使い、画面遷移中のframeや近い重複を避けながら、
通常進行画面を少し多めに含む多様な画像を出力します。

## 必要なもの

- Python 3.13以上
- uv
- `ffmpeg`と`ffprobe`
- 動画・画像入力に対応するmodelを提供するvLLMサーバー、またはvision対応modelを用意したOllama

## はじめかた

依存packageを導入します。

```bash
uv sync
cp config.example.toml config/config.toml
```

`config/config.toml`へ利用するGame Context provider、model、API keyを設定します。
`config/`配下の実設定はGit管理対象外です。用途別の設定は
`config/openai.toml`など別名で保存し、`-c config/openai.toml`で切り替えられます。

選択枚数、Game Title、動画を置いたInput Video Directory、空のOutput Folderを
指定して実行します。

```bash
uv run game-screen-pick \
  -c config/config.toml \
  -n 30 \
  --game-title "ドラクエ11" \
  ./recordings \
  ./recordings-selected
```

Web検索でGame Contextを生成する代わりに、直接指定することもできます。

```bash
uv run game-screen-pick \
  -n 30 \
  --game-context "ジャンル: RPG。探索と会話を進める。代表的な画面はフィールド、会話、戦闘。景観と人物が明瞭な画像を重視する。" \
  ./recordings \
  ./recordings-selected
```

modelやproviderなど、繰り返し使う値はcurrent directoryの
`config/config.toml`で変更できます。API keyは設定ファイルの非空値を優先し、未指定・
空文字列の場合だけproviderに対応する環境変数を使います。実設定はコミットしないで
ください（`.gitignore`は`git add -f`による強制追加までは防ぎません）。

## vLLMで動画を理解して選ぶ

起動済みのvLLMサーバーの接続先と、動画・画像入力に対応するmodel名を設定します。
次を `config/semantic.toml` として保存し、model名を実際の提供名へ置き換えてください。

```toml
[run]
selection_method = "semantic_video"
vllm_base_url = "http://127.0.0.1:8000/v1"
vllm_model = "YOUR_SERVED_VIDEO_MODEL"
vllm_cache_revision = "1"
# vllm_api_key = "your-key"  # 未指定時はVLLM_API_KEY
```

```bash
uv run game-screen-pick -c config/semantic.toml -n 30 \
  --game-context "探索とボス戦を中心とするRPG。景観、人物、戦況が伝わる画像を重視する。" \
  ./recordings ./recordings-selected
```

全編を短い重複区間に分け、vLLMへ動画として渡して重要場面と遷移区間を解析します。
重要場面の周辺を元動画から抽出し、同じvLLM modelで二段階の画像評価を行います。
動画から得た説明と重要度は候補の優先順位・評価に使い、選定根拠をreportへ保存します。
Ollamaによる既存方式は `selection_method = "sampled_frames"`（既定値）で使えます。

解析済み区間は中断後や選択枚数の変更時にも再利用します。サーバーの重み・量子化・
processor・vLLM設定を変更した場合は `vllm_cache_revision` を変更してください。
起動・停止も管理する場合は、同じ `[run]` に任意設定を追加します。

```toml
ollama_host = "http://127.0.0.1:11434"
ollama_unload_before_vllm = true
vllm_start_command = ["/opt/game-screen-pick/start-vllm", "{model}"]
vllm_stop_command = ["/opt/game-screen-pick/stop-vllm"]
```

アンロードは `true` の場合だけ、既存の `ollama_host` に対して実行します。
未設定・`false` ならアンロード用の通信もしません。起動・停止コマンドも未設定なら
実行せず、起動済みサーバーを使用します。コマンドは両方指定する必要があります。

上のscriptは利用環境に用意する起動・停止用の実行ファイルです。起動側はサーバーを
バックグラウンドで起動して終了し、停止側はそのサーバーとworkerを終了させてください。
`{model}` は `vllm_model` に置換します。Dockerやsystemctlの制御コマンドも設定でき、
リモートの場合は `ssh` を明示します。

```toml
vllm_base_url = "http://gpu-host:8000/v1"
ollama_host = "http://gpu-host:11434"
vllm_start_command = ["ssh", "-o", "BatchMode=yes", "gpu-host", "/opt/game-screen-pick/start-vllm", "{model_shell}"]
vllm_stop_command = ["ssh", "-o", "BatchMode=yes", "gpu-host", "/opt/game-screen-pick/stop-vllm"]
```

SSHのremote shellへ渡すモデル引数には、引用付きの `{model_shell}` を使います。
この例はリモート側がPOSIX shellの環境を対象にしています。

推論が必要になった時だけ、Ollamaの全ロード済みモデルを解放・確認してからvLLMを
起動し、処理後に停止します。cacheだけで完了する場合は操作しません。
モデル導入とGPU配置は運用側で設定します。詳しい設定と制限は
[技術リファレンス](docs/technical-reference.md#動画理解による選定)を参照してください。

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
