# 動画入力の運用

> [!IMPORTANT]
> Video Set探索からScene Catalog、Candidate Annotation、決定的selector、canonical publication、structured progress、target acceptanceまで内部実装済みです。public CLIはIssue #190で接続し、installed CLIはそれまでscreenshot入力のままです。target acceptanceは[専用手順](target-acceptance.md)から実行します。

## 最低runtime

| Component | Project floor | 追加検査 |
|---|---:|---|
| Python | 3.13 | project dependencyのimport |
| FFmpeg / ffprobe | 6.1.1、同一build | 対象demuxer・decoder・encoder・muxer・filter、JSON probe、実動画stream |
| Ollama server | 0.31.2 | version、vision、context、structured output、model load |
| faster-whisper | 1.2.1 | configured modelのload |
| CTranslate2 | 4.8.1 | configured device / compute typeの初期化 |
| CUDA user-space libraries | CUDA 12 cuBLAS 12.8.4.1、CUDA Runtime 12.8.90、cuDNN 9.10.2.21 | faster-whisperによる実推論 |

新しいversionは許可しますが、実際に使用したtool/runtime versionは関係するStage Fingerprintとprovenanceへ記録します。subtitle・audio streamがなくSTTを呼び出さなかったrunではSpeech Runtime Identityを意味結果へ含めません。version番号だけで能力を推測せず、処理開始前に必要なoperationを検査します。

Linux x86_64ではfaster-whisper / CTranslate2が要求するCUDA 12 cuBLASとcuDNN 9をproject dependencyとして導入します。Torchは同じCUDA 12 namespaceを共有する2.9.1以上2.11未満へ制約し、CUDA 13 wheelとの混在を避けます。lock済みversionは`uv.lock`を正とします。

## ModelRuntime

ModelRuntimeはScene Catalog、Candidate Annotation、Speech to Textの3 roleをmodel依存Stageより前に解決します。同じOllama tagを複数roleが共有するときは、省略tagと`:latest`も同じselectorとして扱い、local identityの解決とpullをdistinct tagごとに一度だけ行います。roleごとのcontext requirementをすべて検証してから同じpost-pull identityをfreezeします。

Ollama adapterは`/api/version`、`/api/tags`、`/api/pull`、`/api/show`、固定JSON Schemaを渡す最小`/api/chat` probeを使い、最低server version、完全manifest digest、vision、要求context length、structured outputを検査します。このcapability probeは`keep_alive = 0`を指定し、検証用にloadしたmodelをtimed phase前に解放します。`/api/tags`のbare 64桁digestは境界内で`sha256:`付きcanonical identityへ正規化します。全共有roleのcapability検証後にmutable tagを再確認し、検証中も同じdigestを指していたartifactだけをfreezeします。Hugging Face adapterはlocal `refs/main`をnetworkなしで確認し、更新時は`model_info(..., revision = "main")`の完全commit SHAを一度解決してから、そのSHAのimmutable snapshotだけを取得します。identity一致とfaster-whisper local-only loadの検証が終わったsnapshotだけを`refs/main`へatomicに記録し、次のofflineまたは`auto_upgrade = false` runから再利用できます。検証またはref公開に失敗しても、以前の検証済みrefは置き換えません。local refやsnapshotがpartialならlocal候補にせず、online同期による修復を妨げません。

同期後artifactがpartial、identity不一致、load不能、capability不足なら、更新前artifactへ戻さずfatalです。offline、timeout、registry/Hub障害、token・権限不足、repo不在などで同期selector自体が利用不能な場合に限り、同期前に利用可能なlocal artifactがあったことを条件にlocal storeを再解決し、全共有roleの再検査へ合格したartifactを`update_status = "unavailable"`として使用します。別modelへのfallbackは行いません。

run内の解決結果はroleごとに設定名、canonical名、更新前identity、更新status、実行identity、runtime identityを分離します。runtime identityはstore kindと検証済みclient/server versionから構築し、自由形式の値を受け入れません。model storeの絶対pathとtokenは内部のload境界だけで使い、Stage入力、provenance、warning、errorへ含めません。fingerprintとCompleted Stage artifactにはrole固有の設定名、実行identity、runtime identityだけを渡し、更新前identityと更新statusは現在runの`report.json`へ記録します。このため、同じidentityへのno-op pullや一時offlineはcacheを無効化せず、同じfingerprintのartifactへ古いrun診断を固定しません。あるroleのidentity変更も、無関係なroleのsemantic inputを変えません。

## FFmpeg MediaRuntime

MediaRuntimeはPATH上のsystem `ffmpeg` / `ffprobe`だけを使い、binaryをbundleしません。preflightでは両toolが6.1.1以上かつ同一buildであることに加え、Matroska/MP4 demux、AV1/AAC/text subtitle decode、PPM/PCM/SRT encode・mux、frame/audio filter、ffprobe JSON出力の能力を検査します。tool不在、最低version未満、build不一致、能力不足はそれぞれstable reason codeへ変換されます。

Frame Candidate Extractionはmerge済みの各refinement PTS rangeへ1秒前からinput seekし、range終端直後までにdecodeを制限します。全Video Sourceを2回目も先頭からdecodeせず、`select`で半開rangeのexact source PTSだけを返します。

後段のVideo Stageへはsubprocess command、終了code、stderrではなく、次の意味結果だけを返します。

- containerとordered stream metadataのprobe
- source PTS/time base付きのstreaming RGB24 frame scan
- 指定source PTSの単一frame artifact
- source PTSと連続sample位置を持つmono signed 16-bit PCM
- 元packet PTS/time baseと本文を持つembedded text subtitle

PCM timestampはaudio streamの開始PTSを原点とし、resample後の連続sample indexから生成します。container packet PTSの量子化ずれをchunk境界へ持ち込まず、`async=0`のままsampleを挿入・削除しません。

faster-whisperが0.01秒へ量子化した個別word tokenは、startとendが同一点でも推測で延長せず保持します。Context Cueはgap policyでまとめたtoken列全体に正の時間幅を要求します。group全体が同一点の場合は時間を補間せず、`asr_zero_duration`の低信頼diagnosticとしてprivate cacheへ隔離します。PCM範囲外、時刻逆転、sample grid不一致は`timestamp_drift`です。

real-runtime testは実行時に`lavfi`、synthetic tone、repository所有の短い字幕だけからCFR、VFR、AV1/AAC、multiple stream、破損packet fixtureを生成します。binary mediaはrepositoryへ保存しません。通常suiteはFFmpegを起動せず、real suiteだけを次で実行します。

```bash
uv run task test-ffmpeg
```

PRでは通常quality checkと別のUbuntu 24.04 jobとして実行されます。

## preflightとcache reset

処理前に、CLI/TOML、入力・出力path、Video Set snapshot、cache書き込み、同一inputの非待機lock、外部tool、stream、model解決と能力を検査します。異常時はcache処理やOutput Folder公開を始めません。不存在・動画なし・Duplicate Video・不正なOutput Folderなど、実行前に利用者が修正できる入力不備はexit 2と`fix_configuration`で返します。fingerprint計算中を含む実行時のsnapshot変更やI/O障害はoperation failureとしてexit 1にし、usage errorへ分類しません。

`--reset-cache`は上記の安全なpreflightとlock取得が成功した後に、`<VIDEO_INPUT_FOLDER>/.game-screen-pick/cache/`と`<VIDEO_INPUT_FOLDER>/.game-screen-pick/video-identities/`を削除します。Output Folder、Ollama model store、Hugging Face model cacheには触れません。Stage単位またはVideo単位の手動reset、自動削除、保持期限、容量上限はv1に含めません。

## processing cache基盤

Input Lockは`<VIDEO_INPUT_FOLDER>/.game-screen-pick/input.lock`でVideo Input Folder単位に取得します。待機queueは作らず、同じinputの別runが保持中なら即時に失敗します。lockはVideo Set discovery前に取得し、model・runtime・media preflight、identity discovery、cache準備、全Processing Stage、Output Folder公開の終了まで保持します。

cache missのVideo Identityはstat-content-statでwhole-file SHA-256を計算し、動画1本が確定するたびに専用cacheへatomic保存します。engine version、入力rootと相対pathから作るprivacy-safeなlogical source key、size、`mtime_ns`が一致する場合はidentityを再利用します。device、inode、ctimeは判定に使いません。lock取得後、media probe、各Stage、Vision batch、publisher前後でpath・size・mtimeを再検査します。同じsize・mtimeへ意図的に内容を書き換えた場合は検知できないため、入力管理者がmtimeを正しく更新する契約です。これにより1 TiB級Video SetをStageごと、process再起動ごと、acceptanceのprocessing cache切替ごとに再hashしません。

Video Identity cacheとprocessing cacheは寿命を分離した次のnamespaceを使います。

```text
<VIDEO_INPUT_FOLDER>/.game-screen-pick/
├── input.lock
├── video-identities/<LOGICAL_SOURCE_KEY>.json
└── cache/
    ├── work-units/<SUBJECT>/<OPERATION>/<WORK_UNIT_FINGERPRINT>/
    ├── videos/<VIDEO_FINGERPRINT>/<STAGE>/<STAGE_FINGERPRINT>/
    └── video-sets/<VIDEO_SET_FINGERPRINT>/<STAGE>/<STAGE_FINGERPRINT>/
```

Video Identity entryはengine version、privacy-safeなlogical source key、size、mtime、whole-file SHA-256を保持し、absolute/relative pathやvideo名を含みません。各Completed StageとDurable Work Unitには`artifact.json`と`manifest.json`を置きます。manifestはschema、subject・operation・engine・semantic inputのfingerprint、artifactの相対path・byte数・SHA-256、timezone付き完了日時を保持し、absolute pathを含みません。artifactとmanifestをfsyncした後にtemporary directoryをrenameし、parent directoryもfsyncして一括公開します。manifestまたはartifactが欠ける、hashやmetadataが一致しない、symlinkであるなどのpartial・破損entryはcache hitにせず、その最小単位だけを再計算します。

通常実行ではcacheの書込検査とInput Lock取得後に、認識済みLegacy Cacheの`neutral-analysis/`、`ollama-scenes.json`、旧processing cache内の`video-identities/`だけを自動削除します。削除件数と内容byte数をstructured diagnosticへ記録し、削除失敗はfatalです。新しい`videos/`、`video-sets/`、`work-units/`、独立した`video-identities/`、未知のentry、Output Folder、model storeは保持します。Legacy Cacheを変換または再利用する互換layerはありません。

## 自動再開

通常実行は常に再開可能です。`--resume`はありません。

- 完了manifestと成果物がatomicに確定したCompleted Stageだけを再利用します。
- Stage内では、動画1本のidentity、15分のVideo Scan partition、Refinement Window Group、Embedded Subtitle stream、PCM sample range、STT chunk、選択WebP 1枚をDurable Work Unitとして個別に再利用します。Candidate AnnotationはFrame Candidate一枚ごとのCompleted Stageとして再利用し、異なるMomentの主評価またはCombat Representative Fallbackの一部だけが失敗しても、同じ並列batchで成功したframeを保持します。
- 中断・失敗した最小Work Unitの未確定成果物は再利用しません。認識可能なtemporary entryだけを削除し、そのWork Unitから再実行します。健全な兄弟Work Unit、Completed Stage、未知のdirectoryは削除しません。
- 同じVideoのVideo StageはpathやVideo Orderが変わっても再利用できます。
- Videoの追加・削除・並べ替えでは再利用可能なVideo Stageを残し、Video Set Stageだけを新しいVideo Set Fingerprintで再実行します。
- model identity、prompt、schema、policy、Stage固有設定が変わった場合は影響するStageだけを再計算します。
- Ollama server/modelの変更はVideo Identity、Video Scan、Frame Candidate、STTを失効させません。STT modelの変更もVideo ScanとFrame Candidateを失効させません。
- atomic rename済みの完成Canonical Outputは全artifactとsemantic digestを検証し、一致時はbyte変更なしで再利用します。不完全または異なる既存outputを黙って上書きしません。
- Permission、mount、transient I/O access failureはlocal corruptionとして扱いません。読めないCompleted Stage、Work Unit、Canonical Outputを削除・上書きせず、access failureを返します。

同じ意味入力から再開した場合、選択Candidate ID、選択順、公開WebP bytes、canonical reportの意味内容を中断なしの実行と一致させます。attempt時刻、経過時間、resource sample、cache hit/recompute件数は運用診断であり、この出力不変条件には含めません。詳細な処理順と失効表は[Pipelineと安全な再開](pipeline-resume.md)を参照してください。

同じVideo Input Folderの同時実行は即時エラーです。異なるinput folderは並行実行できます。

## 進捗表示

全体の根拠のないpercentは表示せず、現在のProcessing Stageに対する観測可能な進捗をrenderer非依存の`ProgressEvent`として発行します。1回のrunではProcessing Stageを直列に扱い、`run_started`、Stageの開始・進行・完了、最後の`run_completed`、`run_failed`、`run_interrupted`の順序を守ります。

- atomicなProcessing Stage候補ごとのStage番号とStage名。総Stage数が判明した場合だけ`Stage i/N`とする
- Video Order、総動画数、正規化済み相対path
- 処理済み件数と現在判明している総件数
- cache lookupのhit/missと、実処理のreuse/recompute
- Stage経過時間
- 信頼できるsampleがある場合だけStage ETA
- model downloadのartifact、bytes、percent
- Stage開始、完了、warning、再試行、cache再利用などのevent

Stage ETAは同じrun内の`Stage種別 × work-unit種別 × reuse/recompute`が同じsampleだけから求めます。各sampleはcache lookup前のStage開始から、recomputeではartifactとmanifestのatomic completion、reuseでは検証済みcacheの復元完了までのcurrent-run実時間を記録し、0秒のsampleは除外します。残りのreuse/recompute件数が別々に判明し、各系列に5件以上のsampleがあり、Stage開始から30秒以上経過した場合だけ表示します。新しいsampleで予測が50%を超えて変動した系列は破棄し、5件の新しいsampleが集まるまで`estimating`へ戻します。runをまたぐ実績や観測済みcache hit率から今後のcache結果を推測しません。

TTYでは更新型表示、redirect/CIでは一行event logにします。`stderr.isatty()`で自動選択し、v1では強制切替optionを設けません。relative pathの制御文字をescapeし、line rendererの1 event 1行を維持します。進捗、warning、errorはstderrへ出し、stdoutにmachine-readable reportを流しません。60秒以上沈黙し得る外部処理は開始eventを直ちに発行し、その後30秒ごとにelapsedだけのheartbeatを出します。

同じCLI process内では共有GPU coordinatorがOllamaとSTTのGPU-heavy処理を直列化します。別々に起動したCLI process間のGPU排他は行いません。

## 終了codeとエラー表示

| Exit | 意味 |
|---:|---|
| 0 | 成功。Selection Shortfallまたはmodel更新不能のwarning付き成功も含む |
| 1 | preflight、外部tool、model、Processing Stage、公開などの運用失敗 |
| 2 | CLIまたはTOMLのusage/validation error |
| 130 | Ctrl+C |

最外周のrun controllerがStageの型付き例外を`RunFailure`へ正規化し、stable reason code、allowlistで許可した安全な観測値、修復方法、再実行時に再利用できるcacheを示します。未知の例外は`internal_error`とし、元の例外は内部causeとしてだけ保持します。通常はstack traceを表示しません。`--debug`時だけ安全化済みstack traceを加えますが、credential、環境変数一覧、絶対path、prompt本文、raw model response、Context Cue本文は出しません。

Ctrl+Cはfailureではなく`run_interrupted`、reason `user_interrupt`、exit 130として扱います。並列Video Scanでは未開始のscanを取り消してから実行中のscanを終了し、割り込み後に新しいscanを開始しません。処理中だった最小Work Unitだけは次回再計算し、それ以前にatomic確定したpartition、group、chunk、画像は再利用します。

fatal errorではOutput Folderを公開しません。Selection Shortfallと、検証済みlocal modelを使えたmodel更新不能はexit 0の`completed_with_warnings`として理由をatomicに公開します。後者は`model_update_unavailable`と対象roleを`report.json`へ記録し、model storeのpathやtokenは含めません。

## Windows 11 + WSL2 reference runtime

検証済みreferenceは、WSL2 UbuntuからWindows native Ollamaへ明示URLで接続する構成です。Windows OllamaとWSL内Ollamaを自動探索したり、実行途中で切り替えたりしません。

mirrored networkingではlocalhost、標準NATではWindows host IPを`--ollama-host`またはTOMLへ明示します。host IPはWSL restartで変わり得るため組み込み値にしません。

Ollamaを`0.0.0.0:11434`へbindするとWSLから接続しやすくなる一方、他interfaceにも公開され得ます。可能ならmirrored networking + localhostを使い、NATで外部bindが必要ならWindows Firewallとnetwork profileでWSL networkだけに到達元を制限してください。接続先を自動発見しないことは、誤ったserver/model storeを使わないための契約でもあります。

既定modelの根拠と実測値は[Issue #169 runtime/model research](research/issue-169-runtime-model-contract.md)および[24-image probe](../prototypes/issue_169_runtime_contract/README.md)を参照してください。
