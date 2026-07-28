# Target acceptance

この手順はIssue #189の内部Video Set pipelineを、supported target上のreal
FFmpeg、Ollama、faster-whisper/CUDAで検証する。installed `game-screen-pick`
CLIはIssue #190までscreenshot入力版のままであり、`acceptance-target`は開発・release
判定専用である。

## Supported target

- Windows 11 Pro host
- WSL2 Ubuntu 24.04内のPython 3.13以上
- system FFmpeg / ffprobe 6.1.1以上
- NVIDIA GeForce RTX 5090
- Windowsの非loopback addressをhostに持つ明示URLで接続するWindows native Ollama

host alias、WSL gateway、実際のmedia pathはrepositoryへ保存しない。別構成で動作しても、
v2.0のfull-runtime合格を示すrecordはこのtargetでだけ生成する。
preflightは設定hostの解決先がWindows interfaceであり、そのportのWindows側listenerを
`ollama.exe`が所有することをmodel解決の前後で検証する。`localhost`やloopback addressは
WSL内Ollamaとの区別を信頼できる形で固定できないため、通常実行で到達できてもtarget
acceptanceでは受理しない。検証後のstateにはdeployment種別とprocess名だけを保存し、
host、IP、process pathは保存しない。

## Private profile

[`examples/target-acceptance.toml`](examples/target-acceptance.toml)をrepository外へ
copyし、target上の実値へ置き換える。実値入りprofileはcommit、Issue/PR artifact、
`acceptance.json`へ添付しない。

profile schemaはstrictで、次だけを持つ。

- `input_root`: full suiteのVideo Set rootとrelease intervalの基準root
- `configuration_path`: 通常のVideo Selection TOML
- `artifact_root`: suite state、run output、private worksheetを置くprivate root
- `release_suite`: 合計duration、境界tolerance、relative source、start/end、scenario role
- `full_scale_suite`: video count、合計duration、duration tolerance

`artifact_root`は`input_root`自身またはその配下に置かない。full suiteのrecursive source
discoveryへ生成済み匿名inputが混入する構成はprofile読込時に拒否される。

Ollama host、model、STT device、選択枚数などをprofileへ複製しない。これらは
`configuration_path`のTOMLを通常どおり読み、明示CLI、TOML、`OLLAMA_HOST`、組み込み
既定値の優先順位で解決する。harnessが最優先で差し替えるのは、匿名化したsuite用
Video Input Folderと固定3比較run・cold/warm別Output Folderだけである。

## 実行

30分release suiteとfull suiteは必ず明示する。`--suite`を省略するとexit 2になり、
50時間40分の処理を暗黙に開始しない。

```bash
uv run task acceptance-target \
  --profile /absolute/private/target-acceptance.toml \
  --suite release

uv run task acceptance-target \
  --profile /absolute/private/target-acceptance.toml \
  --suite full
```

modelの更新確認、download、capability検証とResolved Model Identityのfreezeは最初のrun
timerより前に行う。releaseはclean cacheのcold、同じVideo Set・設定・model identity・
processing cacheを使うexact warmの順に実行する。fullは固定3 workerのcold比較runを
先に実行し、そのcacheを完全に削除したauto cold、auto coldのcacheを使うexact warmの
2 Acceptance Phaseを順に実行する。固定3比較を含めた合計3 runは同じResolved Model
Identityを使い、実artifactの性能値だけを除いたcanonical content digest、
resource予算、autoのpeak workerが3を超えたこと、Video Scan wall timeの改善を自動gateで
比較する。性能予算超過は処理を途中でkillせず、完了後のgate failureにする。run durationは
atomic publicationまたはoperation failureで確定し、その後のresource monitor停止時間を
含めない。中断再開された比較runでは各Acceptance Run AttemptのVideo Scan wall秒を合算し、再開後に
cache hitした残作業だけを短い実行として比較しない。

新規suiteではmaterialize後かつmodel実行より前に、materialize済みsuite inputの合計byteと
artifact filesystemの空き容量を測る。persistent cache 64 GiBとtemporary/staging 96 GiBの
合計160 GiB未満なら長時間処理を開始せずpreflight failureにする。開始時の測定値はdurable
stateとprivacy-safe recordへ保存する。persistent cacheは独立した64 GiB budgetだけへ計上し、
temporary workとoutput stagingの96 GiB peakへ二重計上しない。確定済みoutputはpeakから除く。
background disk sampleが一度でも失敗した場合はerror件数を記録し、後続sampleと停止に成功しても
resource samplingを不完全として扱う。

coldのVideo Identity cache missではwhole-file SHA-256を一度計算する。exact warmはcoldで
確定したpath非依存identityをdevice、inode、size、mtime、ctime一致時だけ再利用し、1 TiB級
full Video Setを再hashしない。fullのauto coldに使う独立Video Scanは
`video_scan.workers = "auto"`を要求し、NVDECとresource余力がある24 logical CPU targetでは
保守的な3 workerから開始し、rolling判断で最大6 workerまで利用する。
開始時点でCPU・Decoder・memory・GPU・VRAM・disk pressureを検知した場合は1 workerへ抑制する。
CPU・Decoder・memory・VRAM・diskの直近3 sampleと、
disk throughput・1 stream処理速度のtrendを使い、pressure時はscan完了境界で1 workerずつ
減らし、active scanを止めず未開始taskの投入だけを抑制する。
Video Order上の対象scanが確定した時点で、そのVideoの
candidate extractionとcontext collectionを後続Videoのscanと重ねて開始する。background
scan待機中もactive Stageとheartbeatを通知し、通常のscan失敗でも一次障害を保持したまま
待機中workerをcancelする。
scanのprocess登録とcancellation要求は同じlockで直列化し、cancel後に新しいdecoderを開始しない。
新規scan artifactは対象sourceのcontent snapshotを確定直前に再検証し、scan中に変更された
bytesを元のVideo Fingerprint配下へ保存しない。
full suiteの匿名symlinkとduration probeがすべて完了した後もsource stat snapshotを再検証し、
materialize中に置換・変更されたsourceを古いsuite identityへ結び付けない。
candidate extractionのCPU時間は所有threadと、そのStageが起動したrange decoder・proxy
encoder subprocessだけを合算し、並列中の後続Video Scanを二重計上しない。
Context Collection完了後はSpeech Runtime Identityを保持してSTT modelを明示closeし、
GPU資源を解放してからOllama Vision推論を開始する。
model capability probeは`keep_alive = 0`でOllama modelを解放してから最初のrunを開始する。
Context Collection中にもOllama modelが常駐している場合、STT peakはsystem使用量からその
`size_vram`を除いた非Ollama使用量として保守的に計上する。

release intervalは全streamをFFmpeg stream copyした`scenario-001.mkv`形式の匿名clipに
変換する。source metadataとchapterは引き継がず、FFmpegのbitexact format flagを使うため、
同じ入力、区間、tool identityから再生成したclipは同じwhole-file fingerprintになる。
ffprobeの実測開始、終了、durationがprofileの許容差を超える場合はpipeline前にexit 2に
なる。ffprobeのcontainer差は、stream durationがある場合はstream timing、ないMatroskaでは
非0 startを含むformat end、それ以外ではformat elapsed durationから、absolute endと経過
durationへ正規化する。各release区間の境界だけでなく、全clipの正規化済み実測duration合計も
profileの期待合計と同じtolerance内であることを検証する。full suiteはこの正規化済み経過
durationを合算する。
release/full双方のmaterialization manifestは、その生成・duration probeに使った
FFmpeg/ffprobe versionとbuild capability identityへ固定する。tool identityが変わった
materializationは再利用せず、`--reset-suite`後に現在toolで作り直す。
確定済みrelease inputの再利用時はmanifest記載clipと対応videoの完全一致も検証する。
materialize時間はrun予算に含めない。

## Durable resumeとreset

Comparison Runとphaseの完了はsuite別の`acceptance-state.json`へatomicに確定する。中断後に同じcommandを
実行すると、同じprofile、suite、設定、source snapshot、Resolved Model Identity、target
identity、commitを検証し、未完了runだけを続行する。driver、FFmpeg、kernelなどtarget
probeの値が変わったstateは混在させない。completed coldを再実行してwarmへ戻したり、
completed Comparison Runやcold/warmを再実行したりしない。
初回probeの`visible_ram_bytes`実測値はstateとAcceptance Recordへそのまま保持する一方、
durable resumeではWSL2の起動ごとに生じるpage単位のRAM accounting差だけを吸収するため、
保存値との差が1 MiB以内なら同じtargetとして扱う。1 MiBを超える差、field欠落、非整数値は
target identity不一致とし、`visible_ram_bytes`以外のtarget fieldは完全一致を要求する。
設定file外の`OLLAMA_HOST`を含む実効endpointも、URLを公開しないdigestとしてsuite identityへ
固定する。

完了済みphaseからhuman reviewを再開する場合も、現在のsourceをmaterializeし直してsuite
fingerprintを照合し、Resolved Model Identityを再解決してからrecordを確定する。入力または
modelが変わっていれば既存の完了stateを流用しない。同じ実行identityに対する
`Model Update Status`や更新前identityの違いはrun別診断であり、再利用可否を変えない。
worksheet未生成から再開するときはcold reportをphase digestと照合し、selection artifactを
Completed Stage manifest、artifact hash、semantic fingerprintで再検証する。
worksheet生成済みのhuman review待ちから再開するときも、cold/warm双方のcanonical reportを
phase確定時のfile hashと照合し、selected画像のpath、byte数、SHA-256を再検証する。phase確定後に
reportまたは画像が削除・置換されていればreviewを集計しない。worksheet生成とstate確定の間で
中断した場合は、review記入欄を除くcandidate bindingが同じ既存worksheetだけを再利用する。

user interruptや計測済みoperation failureの未完了runはCompleted Stage cacheを保持する。
fullの固定3比較が中断された場合も固定3 cacheから再開し、固定3が完了した後だけそのcacheを
一度削除する。auto cold開始後は固定3 cache削除済みのdurable flagを保持するため、auto coldの
中断・再開でauto cacheを再削除しない。
再開後のrun recordでは、それ以前の試行を含む経過時間、cache/recompute count、Stage時間、
storage/GPU aggregateを累積または保守的な最大値として集計するため、再開後の短い試行だけで
性能を判定しない。user interruptで詳細計測を確定できなかった場合も経過時間を試行へ残し、
resource計測を不完全とする。Completed Stage cacheから再開できるが、そのsuiteは不完全な
性能・resource根拠では合格しない。process強制終了などでinterrupt handler自体を通らず
active runが残った場合は、安全な試行境界を復元できないため`--reset-suite`を要求する。

identityを変えた場合や意図的にcoldからやり直す場合だけ、対象suiteを明示的にresetする。

```bash
uv run task acceptance-target \
  --profile /absolute/private/target-acceptance.toml \
  --suite release \
  --reset-suite
```

`--reset-suite`は選んだsuiteのstate、run output、worksheet、
processing cacheを破棄する。
suite rootを完全に削除できない場合はpartial resetのまま続行せず失敗する。
releaseとfullのartifactは混在しない。

## Human review

cold/warmと自動gateが完了すると、次のprivate worksheetが生成される。

```text
<artifact_root>/target-acceptance/<suite>/review-worksheet.json
```

target上のphase outputとmediaを確認し、各selected entryのpending値をworksheet記載の
stable enumへ置き換える。`reviewer`とtimezone-aware ISO 8601形式の`completed_at`も記入する。
selected/rejected candidate ID、selected output relative path、rejection reasonはcold phaseの
候補集合へdigestで固定されるため、追加・削除・書き換えない。`--human-review`で別fileを
渡す場合も、このimmutable集合が一致しなければ集計されない。

- `visual_quality`: `pass|broken|black|white|transition|near_duplicate`
- `blog_usable`: `yes|no`
- `annotation_consistency`: `consistent|contradictory`
- `context_overrode_visual_invalidity`: `yes|no`
- suite check `spoiler_monotonicity`: `pass|fail`

その後、phaseを再実行せずに同じworksheetを集計する。

```bash
uv run task acceptance-target \
  --profile /absolute/private/target-acceptance.toml \
  --suite release \
  --human-review /absolute/private/review-worksheet.json
```

0枚選択はproduction pipelineでは正常な`completed_with_warnings`だが、human gateの
`invalid_visual_selected_zero`と利用可能率を満たさないためacceptanceは不合格になる。

## Exit code

| Exit | 意味 |
|---:|---|
| 0 | 必要なphase、parallelism比較、performance/resource/privacy、human qualityの全gateが合格 |
| 1 | pipeline operationまたはperformance/resource/privacy/quality gateが不合格 |
| 2 | CLI、profile、configuration、target preflightが不正 |
| 3 | 必要なphaseと自動gateは合格したがhuman reviewが未完了 |
| 130 | 計測済みuser interrupt。completed runとCompleted Stage cacheを保持し、再開時に試行計測を累積 |

## Artifactとprivacy

`artifact_root`配下にはsuiteごとにrun output、durable state、private worksheet、
`acceptance.json`を置く。recordにはcommit、target/runtime、Resolved Model Identity、
実Faster Whisper adapter/backendのSpeech Runtime Identity、path非依存Video Set fingerprint、
run/cache/storage/GPU aggregate、固定3対autoのprivacy-safeな比較、gate aggregateだけを含める。canonical reportの各Stageも
実semantic inputと推論診断からtool/model/contract参照、設定、validation、token数を記録する。
performance比較用configurationには設定file digest、URLを含まないendpoint identity、
privacy-safeな全実効performance設定を保存する。
cold/warmの結果一致digestには、選定・棄却・near miss・Context Cue・警告を含むcanonical
reportの全semantic resultと、公開WebPのpath、SHA-256、寸法、byte数を含める。run ID、
timestamp、Stageのcache・retry・token・duration診断、および同じexecution/runtime identityに
付随するmodelの`local_identity_before_update`と`update_status`だけを除き、利用者に見える結果が
異なるrunを一致扱いしない。
absolute path、video名、media、raw Context Cue、prompt、model response、credential、個別human
判定は含めない。

releaseのtemporary clipとprocessing cacheはcold/warmおよびrecord生成後、合格・不合格・
review pendingのいずれでも削除する。完全に削除できない場合はpassingまたはreview pendingを
確定せず、`release_cleanup_failed`で受入不合格にする。phase output、state、worksheet、
recordは保持する。

合格時は同じsuite directoryの`baseline/baseline.json`と`baseline/baseline.md`へ、source
commitを除いた正規化baselineを生成する。再評価がpendingまたは不合格なら、以前のpassing
baselineを削除する。ただし新しいworksheet、candidate集合、acceptance record、privacyを
検証できるまでは既存のpassing baselineを保持する。通常runではtarget artifactに留める。Issue #190の
cutoverまたはperformance contract変更時だけprivacy検査済みの両fileをreviewし、専用PRで
repositoryへ取り込む。実値入りprofileとprivate worksheetは一緒にcopyしない。
