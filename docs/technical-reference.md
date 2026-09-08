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
| `selection_method` | 候補発見方式（`sampled_frames` / `semantic_video`） | `sampled_frames` |
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
設定値が未指定または空文字列なら環境変数を使います。生成済みcheckpointを再利用する
場合はAPI keyがなくても再開でき、生成が必要なcache missで両方になければ外部API接続前に
エラーになります。利用しないproviderのAPI keyは不要です。API keyは起動log、
Run Manifest、report、checkpoint、例外messageへ出力しません。

`config/`配下は`.gitkeep`を除いて`.gitignore`対象です。任意名の複数設定を通常の
`git add`で誤って追加できませんが、`git add -f`などの強制追加までは防げません。
実設定や本物のAPI keyはコミットしないでください。Git管理する設定例はrootの
`config.example.toml`だけです。

`sampled_frames`ではmodel名を切り替えられます。一次・二次modelともOllamaの`/api/show`でvision対応が
確認できる必要があります。標準では各modelのロード後に`/api/ps`を確認し、model
memoryの50%以上がVRAMにある場合だけ処理を継続します。

## 動画理解による選定

`selection_method = "semantic_video"`では、vLLMサーバーの動画・画像対応modelを
使用します。動画理解と一次・二次画像評価は同じmodelへ送信します。
`primary_model`、`secondary_model`、`allow_cpu`は画像選定に使いません。
`ollama_timeout`はGame Context生成と、有効にしたOllamaモデル解放に適用します。
`ollama_host`はOllamaでのGame Context生成と、明示的に有効にしたモデル解放に使用します。
`sample_interval_seconds`の指定はエラーになります。

| key | 内容 | 既定値 |
| --- | --- | --- |
| `vllm_base_url` | HTTP(S)のOpenAI互換API接続先 | `http://127.0.0.1:8000/v1` |
| `vllm_model` | サーバーが提供する動画・画像対応model名 | 必須 |
| `vllm_api_key` | API認証。logやcacheへ保存しない | `VLLM_API_KEY` |
| `vllm_timeout` | API requestごとのtimeout秒数 | `900.0` |
| `vllm_cache_revision` | 重み・量子化・processor・サーバー設定変更時に更新するcache識別子 | `"1"` |
| `semantic_chunk_seconds` | 解析する区間の秒数（1から120） | `30.0` |
| `semantic_overlap_seconds` | 隣接区間の重複秒数。区間の前進幅は1秒以上 | `2.0` |
| `ollama_unload_before_vllm` | `true`のときだけ既存Ollama接続先の全ロード済みmodelを解放 | `false` |
| `vllm_start_command` | 起動制御commandの文字列配列。`{model}`を`vllm_model`へ置換 | 未設定 |
| `vllm_stop_command` | 停止制御commandの文字列配列。startと両方指定が必要 | 未設定 |
| `vllm_command_timeout` | 起動・停止commandそれぞれの実行期限（秒） | `60.0` |
| `vllm_startup_timeout` | サーバーの状態・起動準備の確認期限（秒） | `900.0` |
| `vllm_shutdown_timeout` | 停止command成功後、接続終了の確認期限（秒） | `60.0` |

起動停止の設定がなければ既存サーバーへそのまま接続します。アンロードが未設定・
`false`ならOllamaの状態確認も行いません。有効な起動停止・解放設定は`semantic_video`
だけで使用でき、`sampled_frames`との組合せは操作前に設定エラーになります。

最初の動画解析または画像評価のcache missで、設定された操作だけを順に実行します。
Ollamaは既存`ollama_host`、`ollama_api_key`（未指定時は対応する環境変数）を使い、
`/api/ps`で列挙した各modelへ`/api/generate`の`keep_alive: 0`を送り、一覧が空になるまで
`ollama_timeout`以内で確認します。失敗時はvLLMを起動しません。
[Ollama API](https://docs.ollama.com/api/ps)、[解放方法](https://docs.ollama.com/faq#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately)。

commandはshellを介さずargvとして実行します。ローカルのDocker、systemctl、管理script、
または明示的な`ssh` commandを使用します。ローカルの引数にはそのままの`{model}`、
SSHのremote POSIX shellへ渡す一つの引数には引用付きの`{model_shell}`を使います。
`{model_shell}`にさらに引用符を重ねないでください。remote shellがPOSIX以外の場合は
対応するlauncherを用意します。`~`・環境変数・pipeをアプリ側では暗黙に展開しません。
制御commandの標準入出力は破棄するため、serviceの診断logは起動先で保存してください。
起動commandはバックグラウンドのサーバーを開始して終了する形式とし、停止commandは
対応するworkerも終了させます。フォアグラウンドで動き続ける`vllm serve`を直接指定せず、
制御scriptなどから起動してください。

起動前に接続先が既に応答していれば、Ollama解放やcommand実行をせずエラーにします。
OllamaでGame Contextを新規生成する場合も、生成前にこの確認を行います。起動後は
`/health`と`/v1/models`のmodel名で準備を確認します。停止確認は接続拒否を条件とするため、
停止中も応答を返すproxyではなく、管理するサーバーの直接endpointを指定してください。
停止中の接続リセットは停止未確認として期限まで再確認します。状態確認・解放の期限は
HTTP headerとbodyの受信中も適用します。OSのDNS名前解決時間は別途OSの設定に依存します。
各profileは実行中に排他的に使用してください。入力・出力のlockは別入力や別端末からの
同一GPU使用を排他せず、ほかのアプリからのOllama再ロードも制御しません。

一度でも起動を試みたrunは、正常終了・起動失敗・timeout・例外・通常の中断のいずれでも
停止commandを試みます。停止失敗はエラーとして報告し、処理本体のエラーと生成済み成果物を
保持します。Ollama server自体は停止せず、modelの自動再ロードもしません。
強制killや端末切断時の保証は管理script/service側で用意してください。

映像streamの先頭から末尾までを重複付きで分割し、各区間を低解像度の短いMP4として
`video_url`へ送ります。既定は1 fps・最大幅512 px、1区間の送信前サイズ上限は16 MiBです。
音声は解析しません。動画理解から重要区間、概算時刻、説明、重要度、遷移の除外区間を
求め、概算時刻の前後1秒以内を0.5秒刻みで候補にします。候補は重要区間内に限定し、
重複をまとめ、除外区間を取り除きます。最終frame時刻が取得できない場合は、平均frame rateの
1frame分と0.05秒のうち長い方を終端余白にします。短い場面の検出精度はmodelと動画のsamplingに依存します。

重要度と機械的品質を等しい重みで一次候補の優先順位へ反映し、動画から得た場面の説明を
両段階の画像評価にも渡します。最終画像は元動画の同じ時刻からfull resolutionで抽出し、
評価候補との見た目の一致を確認します。reportの各画像に`semantic_provenance`、動画ごとに
解析結果を保存します。候補不足や解析失敗はエラーになり、別方式へ自動切替しません。

動画解析はInput Videoと区間ごとに検証済み結果をatomicに保存します。選択枚数の変更は
動画解析を無効にせず、途中の区間だけが欠損・破損していても後続の正常な解析を再利用します。
cacheの条件には動画identity、probe結果、区間、処理条件、接続先、model、cache revision、
Game Context、prompt/schema versionを含めます。結果の変更も後続の抽出・評価へ伝わります。
全cache hit時はvLLMへ通信しません。

`/v1/models`では提供model名を確認しますが、重みの実体やGPU配置までは検証しません。
reportのvLLM `digest`は設定から作ったcache fingerprintです。同じmodel名のままサーバーを
変更する場合は必ず`vllm_cache_revision`を更新してください。commandとtimeoutは推論結果の
cache条件に含めず、全cache hit時は起動停止・解放も行いません。vLLMの導入、model対応と
context長の設定、GPU管理、および起動停止script/serviceの用意は運用側で行います。
API契約は[vLLM Multimodal Inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)と
[Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)を参照してください。

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

## sampled_framesの選定の流れ

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
