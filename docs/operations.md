# 動画入力の運用

> [!IMPORTANT]
> Video Set探索・identity、Input Lock、Completed Stage cache、Legacy Cache削除とFFmpeg MediaRuntimeの内部基盤は実装済みです。MediaRuntimeを使うVideo Stage、進捗表示、public CLIは後続Issueで接続し、installed CLIはIssue #190までscreenshot入力のままです。

## 最低runtime

| Component | Project floor | 追加検査 |
|---|---:|---|
| Python | 3.13 | project dependencyのimport |
| FFmpeg / ffprobe | 6.1.1、同一build | 対象demuxer・decoder・filter、JSON probe、実動画stream |
| Ollama server | 0.31.2 | version、vision、context、structured output、model load |
| faster-whisper | 1.2.1 | configured modelのload |
| CTranslate2 | 4.8.1 | configured device / compute typeの初期化 |

新しいversionは許可しますが、実際のtool/runtime versionは関係するStage Fingerprintとprovenanceへ記録します。version番号だけで能力を推測せず、処理開始前に必要なoperationを検査します。

## FFmpeg MediaRuntime

MediaRuntimeはPATH上のsystem `ffmpeg` / `ffprobe`だけを使い、binaryをbundleしません。preflightでは両toolが6.1.1以上かつ同一buildであることに加え、Matroska/MP4 demux、AV1/AAC/text subtitle decode、frame/audio filter、ffprobe JSON出力の能力を検査します。tool不在、最低version未満、build不一致、能力不足はそれぞれstable reason codeへ変換されます。

後段のVideo Stageへはsubprocess command、終了code、stderrではなく、次の意味結果だけを返します。

- containerとordered stream metadataのprobe
- source PTS/time base付きのstreaming RGB24 frame scan
- 指定source PTSの単一frame artifact
- source PTSと連続sample位置を持つmono signed 16-bit PCM
- 元packet PTS/time baseと本文を持つembedded text subtitle

real-runtime testは実行時に`lavfi`、synthetic tone、repository所有の短い字幕だけからCFR、VFR、multiple stream、破損packet fixtureを生成します。binary mediaはrepositoryへ保存しません。通常suiteはFFmpegを起動せず、real suiteだけを次で実行します。

```bash
uv run task test-ffmpeg
```

PRでは通常quality checkと別のUbuntu 24.04 jobとして実行されます。

## preflightとcache reset

処理前に、CLI/TOML、入力・出力path、Video Set snapshot、cache書き込み、同一inputの非待機lock、外部tool、stream、model解決と能力を検査します。異常時はcache処理やOutput Folder公開を始めません。

`--reset-cache`は上記の安全なpreflightとlock取得が成功した後に、`<VIDEO_INPUT_FOLDER>/.game-screen-pick/cache/`全体だけを削除します。Output Folder、Ollama model store、Hugging Face model cacheには触れません。Stage単位またはVideo単位の手動reset、自動削除、保持期限、容量上限はv1に含めません。

## processing cache基盤

Input Lockは`<VIDEO_INPUT_FOLDER>/.game-screen-pick/input.lock`でVideo Input Folder単位に取得します。待機queueは作らず、同じinputの別runが保持中なら即時に失敗します。lockはVideo Set snapshotの非破壊検査後から、cache準備、全Processing Stage、Output Folder公開の終了まで保持します。各Stageの前と公開直前にもsnapshotを再検査し、入力が変化したrunを確定しません。

processing cacheはcontent-addressedな次のnamespaceを使います。

```text
<VIDEO_INPUT_FOLDER>/.game-screen-pick/cache/
├── videos/<VIDEO_FINGERPRINT>/<STAGE>/<STAGE_FINGERPRINT>/
└── video-sets/<VIDEO_SET_FINGERPRINT>/<STAGE>/<STAGE_FINGERPRINT>/
```

各Stage folderには`artifact.json`と`manifest.json`を置きます。manifestはschema、Stageとsubjectの完全fingerprint、上流fingerprint、Stage固有の正規化済み入力、artifactの相対path・byte数・SHA-256、timezone付き完了日時を保持し、absolute pathを含みません。artifactを書いた後にmanifestを作り、temporary directoryをrenameして一括公開します。manifestまたはartifactが欠ける、hashやmetadataが一致しない、symlinkであるなどのpartial・破損Stageはcache hitにせず再計算します。

通常実行ではcacheの書込検査とInput Lock取得後に、認識済みLegacy Cacheの`neutral-analysis/`と`ollama-scenes.json`だけを自動削除します。削除件数と内容byte数をstructured diagnosticへ記録し、削除失敗はfatalです。新しい`videos/`、`video-sets/`、未知のentry、Output Folder、model storeは保持します。Legacy Cacheを変換または再利用する互換layerはありません。

## 自動再開

通常実行は常に再開可能です。`--resume`はありません。

- 完了manifestと成果物がatomicに確定したCompleted Stageだけを再利用します。
- 中断・失敗したStageの部分成果物は再利用せず、そのStageから再実行します。
- 同じVideoのVideo StageはpathやVideo Orderが変わっても再利用できます。
- Videoの追加・削除・並べ替えでは再利用可能なVideo Stageを残し、Video Set Stageだけを新しいVideo Set Fingerprintで再実行します。
- model identity、prompt、schema、policy、Stage固有設定が変わった場合は影響するStageだけを再計算します。

同じVideo Input Folderの同時実行は即時エラーです。異なるinput folderは並行実行できます。

## 進捗表示

全体の根拠のないpercentは表示せず、現在のProcessing Stageに対する観測可能な進捗を表示します。

- `Stage i/N`とStage名
- Video Order、総動画数、正規化済み相対path
- 処理済み件数と現在判明している総件数
- cache hit、miss、recompute
- Stage経過時間
- 信頼できるsampleがある場合だけStage ETA
- model downloadのartifact、bytes、percent
- Stage開始、完了、warning、再試行、cache再利用などのevent

TTYでは更新型表示、redirect/CIでは一行event logにします。進捗、warning、errorはstderrへ出し、v1ではstdoutにmachine-readable reportを流しません。60秒以上沈黙し得る外部処理でもStage eventまたはheartbeatを出します。

## 終了codeとエラー表示

| Exit | 意味 |
|---:|---|
| 0 | 成功。Selection Shortfallのwarning付き成功も含む |
| 1 | preflight、外部tool、model、Processing Stage、公開などの運用失敗 |
| 2 | CLIまたはTOMLのusage/validation error |
| 130 | Ctrl+C |

エラーはstable reason code、秘密を含まない観測値、修復方法、再実行時に再利用できるcacheを示します。通常はstack traceを表示しません。`--debug`時だけ安全化済みstack traceを加えますが、credential、環境変数一覧、絶対path、prompt本文、raw model response、Context Cue本文は出しません。

fatal errorではOutput Folderを公開しません。Selection Shortfallだけはexit 0の`completed_with_warnings`として選べたsubsetと理由をatomicに公開します。

## Windows 11 + WSL2 reference runtime

検証済みreferenceは、WSL2 UbuntuからWindows native Ollamaへ明示URLで接続する構成です。Windows OllamaとWSL内Ollamaを自動探索したり、実行途中で切り替えたりしません。

mirrored networkingではlocalhost、標準NATではWindows host IPを`--ollama-host`またはTOMLへ明示します。host IPはWSL restartで変わり得るため組み込み値にしません。

Ollamaを`0.0.0.0:11434`へbindするとWSLから接続しやすくなる一方、他interfaceにも公開され得ます。可能ならmirrored networking + localhostを使い、NATで外部bindが必要ならWindows Firewallとnetwork profileでWSL networkだけに到達元を制限してください。接続先を自動発見しないことは、誤ったserver/model storeを使わないための契約でもあります。

既定modelの根拠と実測値は[Issue #169 runtime/model research](research/issue-169-runtime-model-contract.md)および[24-image probe](../prototypes/issue_169_runtime_contract/README.md)を参照してください。
