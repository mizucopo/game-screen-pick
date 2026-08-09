# Target acceptance

この手順はIssue #189の内部Video Set pipelineを、supported target上のreal
FFmpeg、Ollama、faster-whisper/CUDAで検証する。installed `game-screen-pick`
CLIはIssue #190までscreenshot入力版のままであり、`acceptance-target`は開発・release
判定専用である。

## Supported target

- Windows 11 Pro host
- WSL2 Ubuntu 24.04内のPython 3.13以上
- system FFmpeg / ffprobe 6.1.1以上
- NVIDIA GPUとしてNVIDIA GeForce RTX 5090が1台だけ搭載されたhost
- Windowsの非loopback addressをhostに持つ明示URLで接続するWindows native Ollama

host alias、WSL gateway、実際のmedia pathはrepositoryへ保存しない。別構成で動作しても、
v2.0のfull-runtime合格を示すrecordはこのtargetでだけ生成する。
preflightは`nvidia-smi`がちょうど1台のRTX 5090だけを返すことを要求する。複数NVIDIA
GPU構成では、FFmpeg、faster-whisper、Windows native Ollama、resource samplerが同じGPUを
使うことを一意に証明できないため、RTX 5090を含んでいても受理しない。
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
さらに`input_root`、`configuration_path`、private profile自身を、
`<artifact_root>/target-acceptance/<suite>`自身またはその配下に置かない。
`--reset-suite`とrelease work cleanupが利用者のsourceを削除し得る配置は、削除前に拒否される。

Ollama host、model、STT device、選択枚数などをprofileへ複製しない。これらは
`configuration_path`のTOMLを通常どおり読み、明示CLI、TOML、`OLLAMA_HOST`、組み込み
既定値の優先順位で解決する。harnessが最優先で差し替えるのは、匿名化したsuite用
Video Input FolderとParallelism Baseline・Fresh Processing・Cache Reuse別Output Folder
だけである。

supported RTX 5090 targetでは、`configuration_path`が指すprivate TOMLへ
`ollama.max_parallel_requests = 2`を明示する。これはCombat Representative Fallbackの
最大2枚を別request・別conversation contextで同時評価するtarget profileであり、repositoryの
組み込み既定値`1`は変更しない。Candidate Annotation frameごとのCompleted Stageにより、
片方が失敗しても成功済みframeを保持し、次回は未完了frameだけを再実行する。

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

同じ`artifact_root`とsuiteを指定したcommandは一つだけ実行できる。suite単位の非待機lockは
profileと安全なsuite rootを解決した後、state、active attempt、journal、cache、resetを
読み書きする前に取得する。後発commandは先発の証拠を変更せず明示的に拒否される。
releaseとfullは別々に排他される。lock fileはprocess終了後も残り得るが、fileの存在や保存PIDを
実行中判定には使わず、OS lockが解放されていれば同じcommandで直ちに再開できる。

modelの更新確認、download、capability検証とResolved Model Identityのfreezeは最初のrun
timerより前に行う。利用者向けのrun名と役割は次のとおり。

| run名 | 役割 | suite |
|---|---|---|
| Parallelism Baseline (`parallelism-baseline`) | 固定3 workerで自動並列化の比較基準を測る | fullのみ |
| Fresh Processing (`fresh-processing`) | processing cacheなしで本処理を測る | release / full |
| Cache Reuse (`cache-reuse`) | Fresh Processingと同じcacheで再利用性能と結果一致を測る | release / full |

releaseはFresh Processing、Cache Reuseの順に実行する。fullはParallelism Baselineを
先に実行し、そのcacheを完全に削除したFresh Processing、Fresh Processingのcacheを使う
Cache Reuseの順に実行する。Parallelism Baselineを含めた合計3 runは同じResolved Model
Identityを使い、実artifactの性能値だけを除いたcanonical content digest、
resource予算、autoのpeak workerが3を超えたこと、Video Scan wall timeの改善を自動gateで
比較する。性能予算超過は処理を途中でkillせず、完了後のgate failureにする。run durationは
atomic publicationまたはoperation failureで確定し、その後のresource monitor停止時間を
含めない。中断再開された比較runでは各Acceptance Run AttemptのVideo Scan wall秒を合算し、再開後に
cache hitした残作業だけを短い実行として比較しない。

新規suiteと未完了suiteの各起動ではmaterialize後かつmodel実行より前に、materialize済みsuite
inputの合計byteとartifact filesystemの現在の空き容量を測り直す。persistent cache 64 GiBと
temporary/staging 96 GiBの合計160 GiB未満なら、再開時も長時間処理を開始せずpreflight
failureにする。最新の測定値はdurable stateとprivacy-safe recordへ保存する。persistent
cacheは独立した64 GiB budgetだけへ計上し、temporary workとoutput stagingの96 GiB peakへ
二重計上しない。確定済みoutputはpeakから除く。
background disk sampleが一度でも失敗した場合はerror件数を記録し、後続sampleと停止に成功しても
resource samplingを不完全として扱う。
GPU sampleはsystem GPU memoryを継続測定する一方、process GPU memoryはrun開始時の
baselineとして一度だけ取得する。各`nvidia-smi` queryには2秒のtimeoutを設け、同じGPU
sample内で一時的な失敗を一度だけ即時再試行する。再試行も失敗したsample、または停止timeout
内に終了しないbackground probeが一つでもあれば、error件数を記録してresource samplingを
不完全として扱う。

Fresh ProcessingのVideo Identity cache missではwhole-file SHA-256を動画1本ずつatomicに
確定する。Cache Reuse、Parallelism BaselineからFresh Processingへのprocessing cache切替、
process再起動は、engine version、
privacy-safeなlogical source key、size、mtimeが一致するidentityを再利用し、1 TiB級
full Video Setを再hashしない。device、inode、ctimeはidentity cacheの再利用判定に使わない。
fullのFresh Processingに使う独立Video Scanは
`video_scan.workers = "auto"`かつ`video_scan.auto_max_workers >= 4`を要求する。
さらにbackend、targetのlogical CPU数、実scenario数、設定上限、rolling判断を開始できる
完了数、増加後のworkerを満たす未完了scan数から、本番controllerと同じ到達可能な最大worker数を
算出し、4 worker未満なら長時間runの前に拒否する。24 logical CPUのNVDEC targetで
保守的な3 workerから開始する場合、最初に4 workerへ増やすには最低8 scenarioが必要になる。
NVDECとresource余力がある24 logical CPU targetでは保守的な3 workerから開始し、
rolling判断で最大6 workerまで利用する。
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
releaseはintervalごと、fullはsource symlinkとdurationごとにcheckpointをatomic確定する。
各checkpoint自身が確定時のMedia Runtime Identityを持つ。同じidentityで中断した場合、
完了済みunitを保持して未完了unitだけを続行する。materialization contextが欠損・破損しても
contextだけを再構築し、identityとartifactを検証できる完了済みunitは保持する。
未完成materializationのFFmpeg/ffprobe versionまたはbuild capability identityが変わった
場合は、安定した順序で旧identityのunitだけを現在toolへ置き換える。全unitが現在identityへ
揃うまで終端manifestを公開しないため、旧・新toolの混在はpipeline outputへ到達しない。
置換途中で再び中断しても現在identityへ置換済みのunitを再利用する。pipelineの
Video IdentityやProcessing Stage cacheを全削除せず、`--reset-suite`も要求しない。
各置換は検証済みartifactとpending checkpointを先にdurable化し、固定名artifactのatomic
切替後にpendingをcheckpointへ昇格する。artifact切替とcheckpoint確定の間でprocess、
WSL2、Windowsが停止しても、次回はpendingからそのunitを確定できる。置換前の失敗では
旧artifactと旧checkpointを変更しない。

全unitとmanifestまで完成したmaterializationは終端成果物として扱う。後からtool identityが
変わっても再probe・再生成せず、sourceのsize・mtime、匿名file集合、symlink target、
release clipのwhole-file SHA-256、duration descriptorを再検証して同じdescriptorを返す。
終端manifest自体が欠損・破損していても、全unit checkpointが同じ記録済みruntimeで健全なら、
現在のFFmpeg/ffprobeをprobeする前にmanifestを再構築する。
materialize時間はrun予算に含めない。

## Durable resumeとreset

Comparison Runとphaseの完了はsuite別の`acceptance-state.json`へatomicに確定する。中断後に
同じcommandを実行すると、profile、suite、materialize済みsourceのsuite fingerprintを
検証し、未完了runだけを続行する。完了済みFresh ProcessingやCache Reuse、
Parallelism Baselineを暗黙に再実行しない。ただしParallelism Baseline完了後かつ
Fresh Processing開始前に
Video Scan Comparison Contextが変わった場合だけ、旧Parallelism Baseline outputと
processing cacheを
破棄してParallelism Baselineを現在contextで再測定する。共有Video Identity cacheは保持する。

Video Scan Comparison Contextはsource revision、実効configuration/model/endpoint identity、
target runtimeから作り、boot単位で微小に変わるvisible RAMは除外する。Parallelism Baselineと
Fresh Processingの全attemptが同じcontextの場合だけwall time改善gateを評価する。
Fresh Processingが一度でも開始された
後にcontextが変わった場合は、新旧環境の性能証拠を混在させず、追加runを始める前に
`--reset-run parallelism-baseline`を要求する。

releaseではFresh Processing完了後かつCache Reuse開始前にAcceptance Evidence Contextが
変わった場合、Fresh Processing以降だけを自動的に破棄し、現在contextで再測定する。
Cache Reuseが一度でも開始された後にcontextが変わった場合は証拠を混在させず、
`--reset-run fresh-processing`を要求する。

未完了suiteでは、現在のcommit、実効設定、Ollama endpoint、Resolved Model Identity、
runtime/target probeを新しいexecution contextとして記録する。以前のcontextとの差は
privacy-safeなhistoryへ残すが、suite全体の再開拒否やcache全削除の理由にはしない。
各Completed StageとDurable Work Unitのsemantic fingerprintが、FFmpeg、STT、Ollama、
設定、algorithmの変更を影響範囲へ局所化する。たとえばOllama versionまたはmodel identityが
変わっても、Video Identity、Video Scan、Frame Candidate、STT checkpointは保持する。
新しいcontextで再計算されたStage以降だけを新しい依存関係へ接続する。

完了済みrunからhuman reviewまたはfinalizationを再開する場合は、記録済みsourceの
size・mtime・suffix snapshot fingerprintを現在値と照合する。releaseのprivate clipが
合格確定時にcleanup済みなら再materializeしない。保存済みFresh Processing／Cache Reuseの
reportと選択画像は確定時のhash、size、semantic evidenceで再検証する。この経路では現在の
commit、GPU/driver/kernel、
Ollama deployment、server version、Resolved Model Identityをprobeまたは再解決しない。
完了済み成果物の意味内容と記録済みprovenanceを、後から更新されたruntimeで置き換えない。
worksheet未生成から再開するときはFresh Processing reportをphase digestと照合し、
selection artifactを
Completed Stage manifest、artifact hash、semantic fingerprintで再検証する。
この検証に合格すれば`worksheet_ready`が未確定でもmaterialize、Git dirty検査、target
preflight、Ollama接続、model更新・解決、phase workloadを実行せずworksheetだけを復旧する。
retained report、selected画像、selection artifact、candidate bindingのいずれかが欠落または
改変されていれば、別の結果を作らず明示的に停止する。
worksheet生成済みのhuman review待ちから再開するときも、Fresh Processing／Cache Reuse双方の
canonical reportを
phase確定時のfile hashと照合し、`report.md`の決定的projection、selected画像のpath、byte数、
SHA-256を再検証する。phase確定後にJSON、Markdownまたは画像が削除・置換されていればreviewを
集計しない。worksheet生成とstate確定の間で中断した場合は、review記入欄を除くcandidate
bindingが同じ既存worksheetだけを再利用する。

user interruptや計測済みoperation failureの未完了runはCompleted Stage cacheを保持する。
fullのParallelism Baselineが中断された場合も同じcacheから再開し、完了した後だけそのcacheを
一度削除する。Fresh Processing開始後は基準測定cache削除済みのdurable flagを保持するため、
中断・再開でFresh Processingのcacheを再削除しない。
Select Images Completed Stageのatomic確定後、request index保存前に中断した場合は、
Stage manifest内のrequest fingerprintとartifact integrityから一意な完了Stageを回復し、
indexを再構築してrecomputeではなくcache reuseとして記録する。
Canonical Outputのatomic rename後、phase record確定前に中断した場合は、既存folderのschema、
JSON・Markdown、選択画像hash、layout、privacyを再検証する。新attemptが作るsemantic reportと
一致すれば既存outputを一byteも変更せず再利用し、不完全なsuite-owned outputだけを除去する。
異なるexecution contextで意味結果が変わる場合も、完成済みoutputを黙って上書きしない。
再開後のrun recordでは、それ以前の試行を含む経過時間、cache/recompute count、Stage時間、
storage/GPU aggregateを累積または保守的な最大値として集計するため、再開後の短い試行だけで
性能を判定しない。user interruptで詳細計測を確定できなかった場合も経過時間を試行へ残し、
resource計測を不完全とする。Completed Stage cacheから再開できるが、そのsuiteは不完全な
性能・resource根拠では合格しない。process強制終了などでinterrupt handler自体を通らず
active run markerが残った場合は、次回起動時に経過時間を保守的な`process_abandoned`
attemptとして確定する。active attempt journalのexecution context、cache resolutionと、
kill後も残ったVideo Identity、Durable Work Unit、Completed Stage manifestを照合して
確定直前の作業量も回復し、markerを消して新しいattemptから自動再開する。`--reset-suite`や
最初からの再処理は要求しない。
回復対象はactive markerの種別keyと保存run名の両方でexecution planから一意に解決する。
phaseとcomparisonが同時にactiveな場合、または保存run名がplanに存在しない場合は、
attempt、state、journalを推測で変更せずfail-fastする。

identityを変えた場合や意図的に特定runからやり直す場合だけ、対象runまたはsuiteを
明示的にresetする。

```bash
uv run task acceptance-target \
  --profile /absolute/private/target-acceptance.toml \
  --suite full \
  --reset-run parallelism-baseline

uv run task acceptance-target \
  --profile /absolute/private/target-acceptance.toml \
  --suite release \
  --reset-run fresh-processing

uv run task acceptance-target \
  --profile /absolute/private/target-acceptance.toml \
  --suite release \
  --reset-run cache-reuse

uv run task acceptance-target \
  --profile /absolute/private/target-acceptance.toml \
  --suite release \
  --reset-suite
```

`--reset-run`は指定runと、それに依存する後続runだけを再測定する。

| 指定値 | 破棄して再測定するrun | processing cache |
|---|---|---|
| `parallelism-baseline` | Parallelism Baseline、Fresh Processing、Cache Reuse | 破棄 |
| `fresh-processing` | Fresh Processing、Cache Reuse | 破棄 |
| `cache-reuse` | Cache Reuseのみ | Fresh Processingのcacheを保持 |

`parallelism-baseline`はfull suiteだけで利用できる。`cache-reuse`に必要なFresh Processingの
cacheが既にない場合は、空cacheを再利用測定として扱わず`fresh-processing`を要求する。
いずれもmaterialized inputとsuite間で共有するVideo Identity cacheは保持する。
suite rootから各output、work、materialized input、processing cacheまでの既存階層を
recursive deleteより前にすべて`lstat`し、symbolic link、通常directory以外、suite外参照を
一つでも検出したら全削除対象を変更せず拒否する。このpreflightはmaterializeより前にも行うため、
外部symlink先へ再生成fileを書き込まない。`--reset-run fresh-processing`、
`--reset-suite`、Parallelism BaselineからFresh Processingへ移るcache解放に同じ境界を使う。
symlink構成を安全な通常directoryへ戻すまで、reset種別を変えて処理を続けない。
削除失敗時もstateを変更せず追加runを開始しない。`--reset-suite`との同時指定は拒否する。

`--reset-suite`は選んだsuiteのstate、run output、worksheet、
processing cacheを破棄する。release/full suiteの親にある共有Video Identity cacheは保持し、
同じmaterialize済み動画のSHA-256を再計算しない。
suite rootを完全に削除できない場合はpartial resetのまま続行せず失敗する。
input root、通常設定、private profileがsuite root内にある場合も、sourceを削除する前に失敗する。
releaseとfullのartifactは混在しない。

## Human review

Fresh Processing、Cache Reuseと自動gateが完了すると、次のprivate worksheetが生成される。

```text
<artifact_root>/target-acceptance/<suite>/review-worksheet.json
```

target上のphase outputとmediaを確認し、各selected entryのpending値をworksheet記載の
stable enumへ置き換える。`reviewer`とtimezone-aware ISO 8601形式の`completed_at`も記入する。
selected/rejected candidate ID、selected output relative path、rejection reasonは
Fresh Processingの
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
STT実行時だけ実adapter/backendのSpeech Runtime Identity、path非依存Video Set fingerprint、
run/cache/storage/GPU aggregate、Parallelism Baseline対自動並列化のprivacy-safeな比較、
gate aggregateだけを含める。canonical reportの各Stageも
実semantic inputと推論診断からtool/model/contract参照、設定、validation、token数を記録する。
STTを呼び出さなかったphaseでは`runtime.speech_to_text`を`null`として明示し、
Fresh Processing／Cache Reuseの
両方が未使用ならSpeech Runtime consistency gateを合格とする。Acceptance Recordとbaselineの
schemaはこのnullable契約を追加した`1.2.0`とする。
performance比較用configurationには設定file digest、URLを含まないendpoint identity、
privacy-safeな全実効performance設定を保存する。
Fresh Processing／Cache Reuseの結果一致digestには、選定・棄却・near miss・Context Cue・
警告を含むcanonical reportの全semantic resultと、公開WebPのpath、SHA-256、寸法、byte数を
含める。run ID、
timestamp、Stageのcache・retry・token・duration診断、および同じexecution/runtime identityに
付随するmodelの`local_identity_before_update`と`update_status`だけを除き、利用者に見える結果が
異なるrunを一致扱いしない。
absolute path、video名、media、raw Context Cue、prompt、model response、credential、個別human
判定は含めない。

releaseのtemporary clipとprocessing cacheは、再測定可能性が残るhuman review pendingまたは
自動gate不合格の間はprivate artifact rootへ保持する。human reviewを含む全gateの合格確定時に
削除し、完全に削除できない場合はpassingを確定せず`release_cleanup_failed`で受入不合格にする。
privacy gate自体が不合格の場合は直ちにcleanupを試みる。保持中のprivate workを明示的に
破棄する場合は`--reset-suite`を使う。phase output、state、worksheet、recordは保持する。

合格時は同じsuite directoryの`baseline/baseline.json`と`baseline/baseline.md`へ、source
commitを除いた正規化baselineを生成する。再評価がpendingまたは不合格なら、以前のpassing
baselineを削除する。ただし新しいworksheet、candidate集合、acceptance record、privacyを
検証できるまでは既存のpassing baselineを保持する。canonical artifact更新前にstateを
`finalizing`へ遷移し、JSONとMarkdownをatomicかつdurableに公開してからacceptance record、
最後に`passed` stateを確定する。途中のIO失敗やuser interruptは`failed` stateへ残し、
baselineのない`passed`を公開しない。通常runではtarget artifactに留める。Issue #190の
cutoverまたはperformance contract変更時だけprivacy検査済みの両fileをreviewし、専用PRで
repositoryへ取り込む。実値入りprofileとprivate worksheetは一緒にcopyしない。
