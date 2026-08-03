# game-screen-pick
ゲームスクリーンショットから、ブログで使いやすい画像をOllamaの画像分類で選択するAIツールです。

## 動画入力版の設計と内部実装（公開前）

複数のゲーム録画を一つのVideo Setとして扱う次期interfaceは、次の一つのcommandへ置き換える設計です。探索・content identity・Input Lock・Completed Stage cache、system FFmpeg/ffprobeを閉じ込めるMediaRuntimeに加え、GPU/CPU余力のrolling metricに応じたVideo Scanの動的並列制御、動画単位のscan、exact timeline、Candidate Moment、native frame refinement、model-free Neutral Image Analysis、embedded subtitle/audio STTからのContext Cue収集、Ollama/Hugging Faceのmodel lifecycleとrun単位のidentity freeze、共有Scene Catalog、一枚ずつ独立したCandidate Annotationと条件付きCombat Representative Fallback、soft coverage・通常戦闘とイベントの条件付き最低coverage・spoiler・動画横断diversity・分類揺れをまたぐSemantic Duplicate Groupを扱う決定的な最終selector、固定WebP、`game-screen-pick/report@2.1.0`、gallery-first Markdown、atomic publisher、structured progress、RunFailure、動画identity・15分scan partition・refinement group・subtitle stream・PCM range・STT chunk・Candidate Annotation frame・公開画像1枚まで細分化した再開まで内部実装済みです。同じ意味入力からの再開では、選択ID・順序・公開画像bytesを変えず、atomic公開済みoutputも完全検証後にそのまま再利用します。real runtimeを通すtarget acceptance harnessは、RTX 5090でParallelism Baselineと自動並列化の成果物・resource・wall timeも比較します。再測定は`parallelism-baseline`、`fresh-processing`、`cache-reuse`の実行単位でresetできます。public CLI切り替えはIssue #190で行うため、次のproduction commandはまだ実行できません。

通常戦闘の最低coverageは、Candidate AnnotationのCombat Encounter Kindが`ordinary`で、一般敵の群れ・編成または通常遭遇を示す積極的なCombat Encounter Basisがある候補だけを対象にします。Primary Representative Frameが戦闘を示す一方でExplanation Value `none`になった場合だけ、同じCandidate Momentの残り最大2枚を別request・別contextで独立評価し、説明価値、画面内容、敵と主対象の視認性、遮蔽、Neutral品質からRepresentative Frameを決定します。独立requestは`ollama.max_parallel_requests`まで並列化し、一部失敗時は成功済みframeを保存して再開時に未完了分だけを処理します。主推論が戦闘とした候補も画像一枚だけのCombat Encounter Verificationで種別と根拠を再確認し、主推論の`ordinary`判定だけでは最低coverageへ数えません。主要戦闘の根拠がないことや、敵名・HP barだけでは`ordinary`にしません。通常・主要どちらの積極的根拠もない戦闘は`uncertain`となり、`major`とともに最低coverageの対象外です。Spoiler Riskは物語上のネタバレ評価として独立して扱います。詳細は[共有Scene CatalogとCandidate Annotation Stage](docs/vision-stage.md)と[決定的なVideo Set最終選定Stage](docs/selection-stage.md)を参照してください。

Combat Representative Fallbackでは、説明価値があっても非戦闘と確定した代替frameを戦闘Representativeにしません。またSelection Shortlist内の全Primaryを先に予約し、代替Frame Candidate IDをshortlist順で一度だけ割り当てるため、複数Momentが同じ出力frameへ集約されません。

```bash
game-screen-pick \
  --config ./video-selection.toml \
  --image-count 100 \
  <VIDEO_INPUT_FOLDER> \
  <OUTPUT_FOLDER>
```

最低runtimeはPython 3.13、FFmpeg/ffprobe 6.1.1、Ollama server 0.31.2、faster-whisper 1.2.1、CTranslate2 4.8.1です。Windows 11 + WSL2のtarget acceptanceでは、Windowsの非loopback addressを指定した明示URLでWindows native Ollamaへ接続し、Windows/WSLのserverを自動切替しません。

- [動画入力とCLI](docs/video-input.md)
- [Pipelineの処理順・checkpoint・安全な再開](docs/pipeline-resume.md)
- [動画単位のscan、timeline、Frame Candidate、Context Cue cache](docs/video-stage.md)
- [共有Scene CatalogとCandidate Annotation Stage](docs/vision-stage.md)
- [決定的なVideo Set最終選定Stage](docs/selection-stage.md)
- [TOML、優先順位、model更新](docs/configuration.md)
- [runtime、cache、進捗、エラー、WSL2運用](docs/operations.md)
- [選択画像とreport](docs/report.md)
- [移行、統合検証、性能受け入れ](docs/migration-acceptance.md)
- [supported targetでのacceptance実行](docs/target-acceptance.md)
- [完全な設定例](docs/examples/video-selection.toml)

## 現在の実装について

以下は現在利用できるscreenshot入力版の説明です。動画入力版は後方互換性を持たない置き換えとして別途実装します。

## インストール

```bash
uv sync
```

## 使用方法

### 実行方法

```bash
uv run game-screen-pick [オプション] <入力フォルダ> <出力フォルダ>
```

### オプション

- `-n <数値>`, `--num <数値>`: 選択枚数
- `-s <数値>`, `--similarity <数値>`: 類似度しきい値（0.0〜1.0、デフォルト: 0.72）
- `-r`, `--recursive`: サブフォルダも検索
- `--config <パス>`: TOML設定ファイル
- `--ollama-model <文字列>`: Ollamaの画像分類モデル名。未指定の場合はエラー
- `--ollama-host <URLまたはhost>`: Ollamaホスト。`OLLAMA_HOST` より優先。`192.168.1.31` のようにschemeとportを省略した場合は `http://192.168.1.31:11434` として扱う
- `--ollama-timeout <秒>`: Ollama APIタイムアウト秒数（デフォルト: 60）
- `--ollama-max-workers <数値>`: Ollama分類の並列ワーカー数（デフォルト: 1）
- `--ollama-scene-hint <文字列>`: Ollama scene catalog 作成時に渡す補助情報
- `--reset-cache`: 既存キャッシュを削除してから実行
- `--max-dim <数値>`: 画像リサイズ時の長辺の最大ピクセル数
- `--max-memory-gb <数値>`: チャンク処理時のメモリ予算GB
- `--batch-size <数値>`: CLIP推論のバッチサイズ
- `--result-max-workers <数値>`: 結果構築の並列ワーカー数

### 使用例

```bash
# スクリーンショットから15枚選択して出力フォルダにコピー
uv run game-screen-pick --ollama-model gemma4 -n 15 ./screenshots ./output

# 設定ファイルを使用
uv run game-screen-pick --config ./picker.toml ./screenshots ./output

# アドベンチャーゲーム向けの補助ヒント
uv run game-screen-pick --ollama-model gemma4 \
  --ollama-scene-hint "アドベンチャーゲーム。会話差分が多く、表情や背景の違いを重視したい" \
  ./screenshots ./output

# パズルゲーム向けの補助ヒント
uv run game-screen-pick --ollama-model gemma4 \
  --ollama-scene-hint "パズルゲーム。盤面の状態が似やすいので、進行や結果が分かる画像を優先したい" \
  ./screenshots ./output

# RPG向けの補助ヒント
uv run game-screen-pick --ollama-model gemma4 \
  --ollama-scene-hint "RPG。戦闘、探索、会話、メニューが混在している" \
  ./screenshots ./output

# 既存キャッシュを削除して最初から実行
uv run game-screen-pick --ollama-model gemma4 --reset-cache ./screenshots ./output
```

## 処理の流れ

現在の実装は次の流れです。

1. 入力画像をすべて解析し、CLIP特徴・結合特徴・画質メトリクスを作る
2. 解析結果をもとに `content filter` を実施し、暗転・白飛び・単色・遷移フレームを厳格に明示的な reject reason 付きで除外する
3. 残った blog candidate から画質と見た目の多様性で Selection Shortlist を作る
4. Selection Shortlist から高品質・多様性・頻出patternを含む代表画像を最大24枚選び、Ollamaでその実行用の scene catalog を作る
5. scene catalog は3〜8個の scene で構成され、必ず `other` と scene selection role を含む
6. Selection Shortlist の各画像を scene catalog のいずれかへ分類し、分類失敗した画像は最終選択対象から外す
7. 同じ scene 内で見た目や構図が近い画像を variant group にまとめ、recurring gameplayでは要求枚数に応じてvariantを広げる
8. scene selection role、scene ごとの自動配分、画質、分類信頼度、類似度除外を組み合わせて最終出力を決める
9. 選定結果を copy / console / JSON report 共通の出力recordへ変換する
10. `OutputPlanner` が scene slug別連番とreport用 `output_path` をcopyなしで計画する
11. 計画済みの出力先へ画像をコピーし、同じrecordから表示と `<出力フォルダ>/report.json` のJSONレポートを生成する

出力フォルダが存在する場合は、処理開始前に空である必要があります。既存ファイル、既存フォルダ、`report.json` などが1件でもある場合は失敗します。
JSONレポートは常に `<出力フォルダ>/report.json` へ出力されます。
選択画像は常に `battle0001.ext` や `conversation0001.ext` のように、scene slug と scene 内連番で出力されます。

### コンソールログ

通常実行では、入力画像の検索件数、中立解析キャッシュ確認中の処理件数、キャッシュのhit/miss件数、未cache画像の解析開始、解析チャンク数、画像読み込み、CLIP特徴抽出の開始・完了がコンソールに出力されます。
CLIPモデルの初回ロード前にも、画像読み込みやチャンク準備の進捗が出るため、処理が進んでいるかを確認できます。
各ログ行の先頭には日時と直前ログ行からの経過秒が出力されます。形式は `2026-06-06 13:45:12.345 (+1.234s): 入力画像: 120件` です。

### scene の意味

- `scene_slug`: `battle`、`conversation`、`menu` など、ファイル名やJSONキーに使う英語slug
- `scene_display_name`: `戦闘`、`会話`、`メニュー` など、人が読む日本語名
- `scene_description`: ブログ画像選択に役立つ短い説明文
- `scene_selection_role`: `ordinary`、`cinematic`、`recurring_gameplay` のいずれか。`cinematic` は合計で控えめに、`recurring_gameplay` は通常プレイの状態差を拾いやすく扱います
- `variant_group`: 同じscene内でブログ上の役割が重複する差分画像のまとまり
- `semantic_duplicate_group`: Scene SlugやVariant Groupの分類揺れをまたいで同じtitle、主要戦闘遭遇、近接した同役割画面を示す候補のまとまり。次期Video Set selectorでは要求枚数不足時も代表1枚だけを選びます

### 類似度フィルタリング

- 類似度判定は Ollama の scene 分類後に実行します
- 既に選ばれた画像と似すぎる候補は、scene をまたいでも除外します
- `--similarity` はこの除外判定の基準値で、高いほど緩く、低いほど厳しくなります
- `cinematic` roleのsceneは、通常は選択要求枚数の10%（最低1枚）までに抑えます。他の有用候補が足りない場合は補充として超過できます
- `recurring_gameplay` roleのsceneでは、戦闘UI、探索画面、パズル盤面など頻繁に表示される通常プレイ画面の状態差を拾うため、要求枚数が多いほど同じvariant groupからも複数枚選びやすくなります
- ただし、recurring gameplayでも類似度判定は残るため、ほぼ同一の連番フレームだけで埋まることは避けます

### Ollama分類キャッシュ

- 分類結果は入力画像のあるフォルダ配下の `.game-screen-pick/cache/ollama-scenes.json` に保存されます
- キャッシュキーには画像パス、更新時刻、サイズ、モデル名、scene selection roleを含むscene catalog が含まれます
- コンソールには scene catalog 作成開始・完了、画像分類の対象件数、分類済み件数、成功・失敗件数が出力されます
- JSONレポートには Selection Shortlist 外になった件数として `rejected_by_selection_shortlist` が出力され、各候補には `scene_selection_role` が含まれます
- scene catalog 応答が不正な場合は、代表画像数を減らして再試行します
- scene catalog 作成が最終的に失敗した場合は、`fallback` sceneで選定を継続し、console / JSON report に `ollama_catalog_fallback_used` と `ollama_catalog_fallback_reason` を出力します
- `ollama_classification_failed` は、scene catalog 作成後に個別画像を catalog 内の scene へ分類できなかった件数です。catalog 作成失敗による `fallback` とは別に集計されます

### 処理再開

- 中立解析結果は入力フォルダ配下の `.game-screen-pick/cache/neutral-analysis/` に保存されます
- Ctrl+Cで中断した場合は、同じコマンドを再実行すると処理済みの中立解析結果とOllama分類キャッシュを再利用します
- 中立解析キャッシュは画像パス、更新時刻、サイズ、解析設定をもとに再利用可否を判定します
- `--ollama-scene-hint` を変更して再実行した場合、中立解析結果は再利用されますが、scene catalog 作成とOllama分類は新しいヒントに基づいて実行されます
- 古いOllama分類キャッシュは削除されず、scene catalog ごとに別のキャッシュとして保持されます
- 最初から実行したい場合は `--reset-cache` を指定します。入力フォルダとそのサブフォルダ配下の `.game-screen-pick/cache/` を削除してから実行し、その実行中に新しいキャッシュを保存します

## 設定ファイル

```toml
[ollama]
model = "gemma4"
host = "http://localhost:11434"
timeout = 60
max_workers = 1

[thresholds]
similarity = 0.72
```

Ollama host の優先順位は `--ollama-host`、`OLLAMA_HOST`、`[ollama].host`、`http://localhost:11434` です。`192.168.1.31` のようにschemeとportを省略したhostは `http://192.168.1.31:11434` として扱われます。

## 性能チューニング

- `--max-dim`: 小さいほど高速ですが、精度が下がる可能性があります
- `--max-memory-gb`: 大きいほどチャンクサイズが増え、GPU利用率が上がりやすくなります
- `--batch-size`: 大きいほど高速ですが、VRAM消費量が増えます
- `--result-max-workers`: CPU並列度を調整します
- Ollama分類は全blog candidateではなく、画質と見た目の多様性で絞った Selection Shortlist にだけ実行されます。Selection Shortlist は選択枚数の10倍または500件の大きい方を基本に、通常は最大2000件まで自動調整されます。ただし、選択枚数が2000件を超える場合は、Ollama分類失敗に備えて要求枚数より少し多めに確保されます
- scene catalog作成に使う代表画像は最大24枚のまま、高品質画像、見た目の多様な画像、頻出する通常プレイpatternが混ざるように選ばれます
- Ollamaの `/api/chat` には常に `think=false` を送信します。scene分類では最終JSONだけを使うため、thinking対応モデルでは推論trace生成を抑えて速度を優先します

## 関連ドキュメント

- [ADR index](docs/adr/README.md)
- [ADR 0004: Select Video Set Blog Images Deterministically](docs/adr/0004-select-video-set-blog-images-deterministically.md)（動画入力向け設計、内部移行中）
- [ADR 0005: Publish Video Selection Artifacts Atomically](docs/adr/0005-publish-video-selection-artifacts-atomically.md)（動画入力向け設計、内部移行中）
- [ADR 0006: Expose Video Selection Through CLI and Versioned Config](docs/adr/0006-expose-video-selection-through-cli-and-versioned-config.md)（動画入力向け設計、内部移行中）
- [ADR 0007: Migrate to the Video Set Selector Through a Gated Cutover](docs/adr/0007-migrate-to-video-set-selector-through-gated-cutover.md)（動画入力向け移行設計、内部移行中）
