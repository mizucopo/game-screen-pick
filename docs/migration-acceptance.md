# 動画入力v2.0の移行・統合検証・性能受け入れ戦略

## 状態と目的

この文書はIssue #170で確定した、screenshot入力版からVideo Set入力版への
実装順序、test境界、target acceptance、性能予算、human quality gate、公開切替を
定義する。ここに記載するcommandとmoduleは設計であり、実装Issueが完了するまで
現在のscreenshot CLIからは利用できない。

移行の原則は「内部は段階導入、公開切替は一括」である。Issues IMP-01〜IMP-12は
常にgreenな内部sliceとして順番に導入するが、public CLI、READMEの既定導線、package
versionは変えない。IMP-13だけが次の公開状態をatomicに切り替える。

| Public state | CLI input | Package | Legacy code | ADR 0001〜0003 |
|---|---|---:|---|---|
| IMP-13より前 | screenshot folder | 1.5.2 | present | superseded historyとして保持 |
| IMP-13完了後 | Video Set folder | 2.0.0 | absent | superseded historyとして保持 |

途中状態を公開してはならない。video CLIのpreview flag、screenshot互換subcommand、旧option
alias、compatibility modeは設けない。

## 実装sliceと依存順

一つの実装Issueを一つのreview可能なPRにする。各PRは前段を取り込み、該当するgateを
満たしてからmergeする。番号はこの文書内の安定した追跡IDであり、作成後のGitHub Issue
番号を併記する。

| ID | Slice | 依存 | PR gate | Cutover前に必要 |
|---|---|---|---|---|
| [IMP-01 / #178](https://github.com/mizucopo/game-screen-pick/issues/178) | fake walking skeletonとProcessing Stage実行基盤 | なし | fake E2E | yes |
| [IMP-02 / #179](https://github.com/mizucopo/game-screen-pick/issues/179) | versioned TOML、Effective Configuration、内部CLI adapter | IMP-01 | config contract | yes |
| [IMP-03 / #180](https://github.com/mizucopo/game-screen-pick/issues/180) | Video Set探索・identity・lock・cache・legacy cache削除 | IMP-02 | real filesystem + fault injection | yes |
| [IMP-04 / #181](https://github.com/mizucopo/game-screen-pick/issues/181) | FFmpeg MediaRuntimeと合成fixture、required CI | IMP-03 | `test-ffmpeg` | yes |
| [IMP-05 / #182](https://github.com/mizucopo/game-screen-pick/issues/182) | heartbeat/scene signal/timeline/refinement/Neutral Image Analysis | IMP-04 | media and timeline scenarios | yes |
| [IMP-06 / #183](https://github.com/mizucopo/game-screen-pick/issues/183) | subtitle/audio Context CueとSpeechRuntime | IMP-05 | context contract | yes |
| [IMP-07 / #184](https://github.com/mizucopo/game-screen-pick/issues/184) | model lifecycleとModelRuntime | IMP-06 | identity/update/cache contract | yes |
| [IMP-08 / #185](https://github.com/mizucopo/game-screen-pick/issues/185) | Scene Catalog/Candidate AnnotationとVisionRuntime | IMP-07 | schema and retry contract | yes |
| [IMP-09 / #186](https://github.com/mizucopo/game-screen-pick/issues/186) | deterministic Video Set selector | IMP-08 | normalized selection golden | yes |
| [IMP-10 / #187](https://github.com/mizucopo/game-screen-pick/issues/187) | WebP、Canonical Selection Report、atomic publication | IMP-09 | artifact golden + publication faults | yes |
| [IMP-11 / #188](https://github.com/mizucopo/game-screen-pick/issues/188) | structured progress、reason code、中断・再開 | IMP-10 | interruption/ETA matrix | yes |
| [IMP-12 / #189](https://github.com/mizucopo/game-screen-pick/issues/189) | target acceptanceと性能検証harness | IMP-11 | target records + budgets + human gate | yes |
| [IMP-13 / #190](https://github.com/mizucopo/game-screen-pick/issues/190) | public video CLI cutover、legacy削除、README、2.0.0 | IMP-12 |全cutover checklist | atomic transition |

IMP-13より前の内部adapterやacceptance harnessはtest/開発用の入口であり、installed
console scriptのpublic surfaceへ追加しない。内部PRではpackage versionを上げない。

## Moduleとtest seam

pipeline全体の概念的なinterfaceは次の一つとする。

```text
run(EffectiveConfiguration) -> RunOutcome
```

Processing Stageは再開とcacheを所有する内部seamであり、外部toolの呼び出し詳細を
domainへ漏らさない。testでは`subprocess`やHTTP clientを各所でpatchせず、次の深い
moduleをfakeへ置換する。

| Module | 所有するinterface | 所有しないもの |
|---|---|---|
| `MediaRuntime` | probe、scan、frame、audio、embedded subtitle | timeline domain、selection |
| `ModelRuntime` | update/download、Resolved Model Identity、capability | prompt意味評価、selection |
| `VisionRuntime` | Scene Catalog、Candidate Annotation | HTTP/Ollama lifecycle、final score |
| `SpeechRuntime` | speech-to-text、word timestamp | audio抽出、Context Cue選定 |
| `RunObserver` | clock、ProgressEvent、runtime metrics | business decision、renderer文字列 |

filesystemは実際の一時directoryを使う局所的な置換境界とし、汎用repository interfaceを
追加しない。fake E2Eでもdiscovery、cache、manifest、staging、renameは`tmp_path`上の
real filesystemで動かす。

## Test ladder

実装後の標準commandを次の三層に分ける。

| Command | 内容 | 外部依存 | Gate |
|---|---|---|---|
| `uv run task test` | unit、contract、fake E2E | binary/network/GPUなし | 全PR必須 |
| `uv run task test-ffmpeg` | 生成fixtureとreal FFmpeg/ffprobe | system FFmpeg | 別のrequired PR check |
| `uv run task acceptance-target --profile PATH --suite release\|full` | real Ollama/STT、性能、human review | target PC | cutover/release/該当変更時必須 |

real Ollamaとreal STTは通常のPR gateにしない。次の場合にtarget acceptanceを必須にする。

- IMP-13のpublic cutover前と2.0.0 release前。
- model、prompt、structured schema、runtime floor、STT policyを変えるとき。
- 抽出、cache、parallelism、GPU stage schedulingの性能特性を変えるとき。
- acceptance/performance contract自体を変えるとき。

### 合成video fixture

binary videoはrepositoryへcommitしない。repo-owned scriptがFFmpeg `lavfi`を使い、test
実行時に短い合法なfixtureを生成する。映像は色、test pattern、移動図形、fade、black/
white frameを組み合わせ、音声はsynthetic tone/silence、字幕はrepository所有の短い文字列
だけを使う。CFR、VFR、audioなし、subtitleなし、embedded text subtitle、複数stream、破損・
decoder errorを生成できることをgenerator contractにする。

assertionはcontainer byteやencoder hashではなく、stream、PTS、正規化Video Time、候補区間、
reason code、artifact schemaなどの意味を検証する。

### 必須シナリオ

- 複数Videoの自然順、rename/move後の同一identity、duplicate Videoのfail-fast。
- CFRとVFRのsource PTS/time base、最初の表示frameを0とするVideo Time。
- heartbeatとscene signal、refinement、境界のfirst/middle/last candidate。
- audio/subtitleあり、audioなし、subtitleなし、優先subtitle、STT fallback。
- corrupt packet、decoder error、FFmpeg/ffprobe不一致、unsupported stream。
- Scene Catalog/Candidate Annotationのschema/domain failureと一回だけのretry。
- Video横断selection、spoiler感度、soft coverage、visual/temporal diversity、shortfall。
- empty/absent Output Folder、staging validation、atomic rename、cross-artifact一致。
- cold、exact warm、部分的fingerprint変更、model identity更新、Ctrl+C再開。

## Fault injectionと再開

faultはdurable commit pointの直前と直後へ注入する。少なくとも各Video Stage manifest、
first/middle/lastのVideoとannotation、Scene Catalog、selection、output staging/renameを覆う。
rename失敗、disk full、permission denied、不正manifest、artifact欠落、Ctrl+C、operation errorを
区別する。

検証する不変条件は次の通り。

- 完了manifestと成果物がatomicに確定したCompleted Stageだけを再利用する。
- recognized partial/in-progress StageはInput Lock取得後に削除して再計算し、Completed Stageと未知のdirectoryは削除しない。fatal runはOutput Folderを公開しない。
- Ctrl+Cはexit 130、operation errorはexit 1であり、どちらも完了済み上流Stageを保持する。
- 同じVideo内のfirst/middle/lastおよび複数Videoの一部失敗で、成功済み独立Video Stageを
  再計算しない。
- publication renameの前後どちらで失敗しても、partial final folderを観測させない。

## Structured progressとETA

applicationはrenderer非依存の`ProgressEvent`を発行し、TTY rendererとline rendererを
adapterとして分ける。eventはStage、Video Order/安全なrelative path、処理済み/既知総数、
cache hit/miss/recompute、elapsed、任意のETA、severity、reason codeを持つ。credential、hostの
absolute path、raw Context Cue、prompt、model responseを含めない。domain testは日本語の表示
文字列ではなくeventを検証する。

一回のrunでactiveなProcessing Stageは一つだけとし、`run_started`からStageの開始・進行・完了を
繰り返して、一つのterminal eventで終了する。Stage番号はatomicなCompleted Stage候補ごとに増やし、
総Stage数は判明した場合だけeventと表示へ含める。cache lookupのhit/missと実処理の
reuse/recomputeは別の観測値とし、miss後に実処理したunitはmissとrecomputeの双方へ数える。

ETAは次の全部を満たすときだけ表示する。

- 残りのreuse/recompute件数が別々に判明している。
- 同じrun内の`Stage種別 × work-unit種別 × reuse/recompute`ごとに少なくとも5 unitのsampleがある。
- Stage開始から実時間で30秒以上経過している。
- runをまたぐ実績や観測済みcache hit率から今後のcache結果を推測していない。

安定したfake workloadでは最終実績の20%以内へ収束させる。新しいsampleで予測が50%を
超えて変動した系列はresetし、新しい5 sampleが集まるまでETAを隠して`estimating`とする。
根拠のない0%、0秒、全runの擬似percentageは表示しない。

TTY/line rendererは`stderr.isatty()`で自動選択し、line rendererはrelative path内の制御文字を
escapeして1 event 1行を守る。外部処理は開始eventを直ちに発行し、完了まで30秒ごとにelapsed
だけのheartbeatを発行する。同じCLI process内ではOllamaとSTTのGPU-heavy処理を共有coordinatorで
直列化し、Context Collection後はSTT modelをcloseしてからOllama推論へ進む。別process間の
GPU排他は行わない。

最外周の内部run controllerだけがStage例外を安全な`RunFailure`へ正規化し、terminal event、
resume guidance、exit 1/2/130を決める。Ctrl+Cは`run_interrupted / user_interrupt / 130`とし、
public CLIへの接続はIMP-13まで行わない。

## Target acceptance profileと記録

full-runtimeのv2.0 support targetは次に限定する。

- Windows 11 Pro。
- WSL2 Ubuntu 24.04内でPython applicationとsystem FFmpegを実行。
- Windowsの非loopback addressを指定した明示URLでWindows native Ollamaへ接続し、
  Windows側の`ollama.exe`によるlistener所有をpreflightで検証。
- NVIDIA GPUとして1台だけ搭載されたGeForce RTX 5090上のCUDA STT。

Ubuntu CIはunit/fake/FFmpeg integrationを保証する。native Linux、macOS、direct Windowsは
動く可能性があっても、v2.0のfull E2E保証対象ではない。Macからの`ssh winpc`は任意の
orchestrationにすぎず、repositoryやproduction commandへhost alias、gateway、target media
pathをhard-codeしない。
target preflightは`nvidia-smi`がちょうど1台のRTX 5090だけを返すことを要求する。
複数NVIDIA GPU構成では、FFmpeg、faster-whisper、Windows native Ollama、resource samplerを
記録対象GPUへ一意に固定できないため、RTX 5090が含まれていても受理しない。

target-onlyのuntracked profileはinput root、通常設定を指すconfiguration path、private
artifact rootと、relative video path、start/end Video Time、scenario roleだけを保持する。repositoryには
[`docs/examples/target-acceptance.toml`](examples/target-acceptance.toml)のschema/templateだけを
置く。生成したtemporary clipはrun終了後に削除する。実行、durable resume、private worksheet、
終了code、baseline承認は[Target acceptance](target-acceptance.md)を参照する。

各runはversioned `acceptance.json`を生成し、release/Issue artifactとして保管する。次を
記録する。

- commit、acceptance schema、Effective Configurationの安全な要約。
- OS/WSL、CPU、RAM、GPU/driver/CUDA、FFmpeg、Ollama、STT runtime。
- 実際に解決された完全なmodel identityと更新結果。
- pathを含まないVideo Set fingerprint、対象duration、scenario count。
- Stage時間、cold/warm、cache hit/miss/recompute、cache byte。
- Ollama/STTのprocess baseline、model `size_vram`、global GPU peak。
- quality gateの集計とhuman reviewer判定。

absolute path、video/audio/subtitle/image、raw Context Cue、prompt、model response、credentialは
記録しない。cutover時と性能contract変更時だけ、正規化したbaseline JSONとMarkdown summaryを
repositoryへcommitする。通常runのrecordはartifactに留める。

## 性能予算

model download/update時間はすべてのrun予算から除外する。cold timerはResolved Model
Identityをfreezeした後からatomic publicationまでを測る。予算超過はacceptance failureであり、
runtimeを途中killするtimeoutではない。

### 30分release suite

target動画から代表scenarioを固定し、合計約30分のintervalとして実行する。

| Metric | Budget |
|---|---:|
| clean processing cacheのcold | 20分以内 |
| 同一Video Set/config/model identity、空の別Output Folderへのwarm | 3分以内 |
| warmのunexpected Stage recompute | 0 |
| warm result | run固有診断を除くcanonical reportの全semantic resultがcoldと同一 |

### 50時間40分full-scale suite

12 videos、合計50時間40分の全体を、reference PCを他用途で使わず実行する。IMP-13前と、
抽出/cache/parallelismの性能へ影響する変更時だけ必須とする。

設計時に確認したtarget inputは1,124,448,879,219 bytes、対象driveの空きは
5,976,873,041,920 bytesだった。これは固定identityや将来の合格値にはせず、各runのpreflightで
現在のinput byteと必要空き容量を改めて記録する。

| Metric | Budget |
|---|---:|
| cold | 24時間以内 |
| exact warm | 30分以内 |
| warmのunexpected Stage recompute | 0 |

### Resource budget

| Resource | Budget | Accounting |
|---|---:|---|
| clean default profileのpersistent processing cache | 64 GiB以下 | model store/outputを除外 |
| temporary/stagingを含むpeak追加容量 | 96 GiB以下 | persistent cache/model store/outputを除外 |
| Ollama global GPU peak | 18 GiB以下 | 100% GPU、CPU offload不可 |
| STT non-Ollama global GPU peak | 8 GiB以下 | system使用量から同時常駐Ollama `size_vram`を除外 |

旧fingerprint artifactの併存はclean profile予算から除くが、runはcache root全体の容量と警告を
記録する。容量予算はacceptance gateであり、runtimeの強制quotaではない。OllamaとSTTの
GPU-heavy Stageは重ねない。GPU recordはprocess baseline、model `size_vram`、system全体の
peakを分ける。Ollama `/api/ps`のmodel `size`と`size_vram`も比較し、coldでmodelが観測され、
全量がGPU residentである場合だけ自動gateを合格させる。停止timeout内にbackground GPU
probeまたはdisk samplerが終了しない場合もsampling incompleteとして不合格にする。
process GPU baselineはrun開始時に一度だけ取得し、継続sampleではsystem GPU memoryだけを
`nvidia-smi`から取得する。各queryは2秒でtimeoutし、GPU sampleは一時的な失敗を同じsample
内で一度だけ即時再試行する。再試行も失敗したsampleはsampling incompleteとして不合格にする。
model capability probeは`keep_alive = 0`でtimed phase前にOllama modelを解放する。

既存prototypeの参考値は、#163の全scan約14時間見込み、heartbeat proxy約17 GB、#165の
500 annotations約18〜20分見込み、#166の600秒STT 4.641秒/peak 5,196 MiB、#169の24-image
Ollama global peak 14,629 MiBである。これはbudgetの代わりではなく、最初のbaseline比較材料
とする。

## Result consistencyとhuman quality gate

fake E2Eのgoldenはselected ID/order、reason code、JSON、Markdownの正規化結果を完全一致
させる。real cold runではschema、enum、reference、count、Stage、privacyを検証する。同じcacheの
warm runではnormalized selected ID/orderとcache済みmodel contentをcoldと完全一致させる。
processing cacheをresetしたcold runではmodel wordingやslugの差を許す。model identityが
変わったときはraw response snapshotを更新せず、human acceptanceで新baselineを承認する。

30分suiteとfull-scale suiteの両方で次を満たす。

- selectedにbroken、black、white、Transition Frame、Visual Near-Duplicateが0件。
- 90%以上がそのままブログ画像として利用可能。
- 明確なsummary/Scene/Blog Image Type矛盾が10%未満。
- Context Cueだけを根拠に視覚的に不適格なframeを採用しない。
- 同じ候補集合でSpoiler Sensitivityを上げたとき、Major Spoiler Signalを持つselected件数が
  増えない。
- rejected candidateはfree textだけでなくstable enum reason codeを持つ。

human判定はcandidate IDとenum reasonで集計し、raw subtitleやmodel responseをartifactへ
転記しない。

## Legacy削除とpublic cutover

IMP-13では次を同じPRで行う。

- screenshot固有のCLI request/config/orchestrationを削除する。
- path-based neutral-analysis cache、`ollama-scenes.json`と旧cache adapterを削除する。
- 旧scene classification/score、Cinematic Soft Cap、旧`OutputRecord`/reportを削除する。
- legacy test、fake、option、dead codeを削除する。
- READMEをvideo-only quickstartへ切り替え、package versionを2.0.0にする。
- release noteへ旧TOML非互換、video-only、legacy cache自動削除を記載する。

CLIP、画質metrics、content filter、vector operationなど中立なalgorithm kernelは、旧domain typeを
残さず`FrameCandidate`/`NeutralImageAnalysis`へ適合する場合だけ再利用できる。

cache lock取得後、認識できたlegacy entry（少なくとも`neutral-analysis/`と
`ollama-scenes.json`）だけを自動削除し、新しい`videos/`と`video-sets/`は保持する。削除件数と
byteを記録し、削除失敗はfatalにする。`--reset-cache`だけはcache root全体を削除する。legacy
cacheは単なる再生成可能cacheなので、retentionやmigration compatibilityは設けない。

public cutoverの最終gateは次の全部である。

1. IMP-01〜IMP-12がmerge済みでrequired checksがgreen。
2. 追跡マトリクスにorphan requirementがない。
3. 30分cold/warm、full-scale cold/warm、cache/GPU budgetを満たす。
4. human quality gateを満たし、acceptance artifactが保存済み。
5. IMP-13のlegacy grep、CLI/help、README、version、package smoke testがgreen。
6. screenshot/videoの混在状態がなく、一度のmergeでpublic stateが切り替わる。

report schemaはpackage versionと独立し、`game-screen-pick/report@1.0.0`から開始する。

## Traceability matrix

`Implementation`欄は実装Issue作成後にGitHub番号を併記する。この表がacceptanceのsource of
truthであり、実装Issue、test、cutover checklistのどれにも紐付かない要件を残してはならない。

| Requirement | Source | Observable outcome | Layer / fixture / fault | Implementation | Cutover |
|---|---|---|---|---|---|
| MIG-001 | #170, ADR 0007 | internal PR中はscreenshot、最後だけvideo/2.0.0へatomic切替 | gate prototype + CLI smoke | IMP-01, IMP-13 | yes |
| MIG-002 | #170 | 1 Issue = 1 PR、13 sliceが依存順でgreen | required checks | IMP-01〜13 | yes |
| MIG-003 | #170 | screenshot compatibility surfaceとdead legacyが残らない | legacy grep + package smoke | IMP-13 | yes |
| CFG-001 | #169, ADR 0006 | CLI > TOML > env > default、unknown設定はfail-fast | unit/contract | IMP-02 | yes |
| INP-001 | #162 |自然順Video Set、content identity、duplicate拒否 | fake E2E + real fs | IMP-03 | yes |
| INP-002 | #162 | lock後だけcache mutation、partial Stage非再利用 | fault matrix | IMP-03, IMP-11 | yes |
| CACHE-001 | #162, #169 | Stage Fingerprint一致だけ再利用しwarm recompute 0 | fake/target warm | IMP-03, IMP-12 | yes |
| CACHE-002 | #170 | recognized legacyだけ自動削除し、失敗はfatal | temp legacy tree + permission fault | IMP-03, IMP-13 | yes |
| MED-001 | #163 | heartbeat/scene signal/refinementの意味結果 | generated CFR/VFR | IMP-04, IMP-05 | yes |
| MED-002 | #163, #164 | source PTS/time baseから正確なVideo Timeを得る | real FFmpeg VFR | IMP-04, IMP-05 | yes |
| MED-003 | #163 | corrupt/decoder/tool不整合をreason-coded failureにする | corrupt fixture | IMP-04, IMP-11 | yes |
| CTX-001 | #166 | embedded text subtitle優先、audio STT fallback | generated streams | IMP-06 | yes |
| CTX-002 | #166 | stream不在は正常、処理失敗はfatal | no-audio/no-subtitle/fault | IMP-06 | yes |
| CTX-003 | #166 | Context Cue単独で候補生成・frame適格化しない | domain/fake E2E | IMP-06, IMP-09 | yes |
| MOD-001 | #169, ADR 0006 | 実行時の完全model identityをfreeze/provenance化 | fake registry/runtime | IMP-07 | yes |
| MOD-002 | #169 | auto_upgrade既定true、offline local reuse、no fallback | contract matrix | IMP-07 | yes |
| VIS-001 | #165 | Scene CatalogとAnnotationのstrict schema/retry/fatal | fake VisionRuntime | IMP-08 | yes |
| VIS-002 | #165 | model意味評価とdeterministic selectorを分離 | contract/fake E2E | IMP-08, IMP-09 | yes |
| SEL-001 | #167, ADR 0004 | selected ID/order/reasonが決定的 | normalized golden | IMP-09 | yes |
| SEL-002 | #167 | soft coverage、diversity、shortfall契約を満たす | candidate matrices | IMP-09 | yes |
| SEL-003 | #167 | spoiler感度を上げてもmajor spoiler選択が増えない | metamorphic test | IMP-09, IMP-12 | yes |
| PUB-001 | #168, ADR 0005 | WebP/JSON/Markdownが同じcanonical objectから一致 | artifact golden | IMP-10 | yes |
| PUB-002 | #168 | 一回のdirectory renameだけで公開 | rename/disk/permission faults | IMP-10 | yes |
| PUB-003 | #168 | fatal時Output Folderなし、shortfallだけwarning success | fake E2E + faults | IMP-10 | yes |
| OBS-001 | #169, #170 | structured eventからTTY/lineを描画 | observer contract | IMP-11 | yes |
| OBS-002 | #170 | ETAは5 sample/30秒/stability条件下だけ表示 | fake clock/workload | IMP-11 | yes |
| RES-001 | #162, #170 | 各durable pointでcompletedだけ再利用 | first/middle/last faults | IMP-03〜11 | yes |
| TEST-001 | #170 | `test`はexternal-free、`test-ffmpeg`はrequired | CI configuration | IMP-01, IMP-04 | yes |
| ACC-001 | #170 | supported WSL2 targetでreal runtimeを検証 | untracked target profile | IMP-12 | yes |
| ACC-002 | #168〜170 | versioned acceptance recordにpath/raw dataがない | schema/privacy contract | IMP-12 | yes |
| PERF-001 | #170 | 30分cold ≤20m、warm ≤3m、recompute 0 | target release suite | IMP-12 | yes |
| PERF-002 | #170 | full cold ≤24h、warm ≤30m、recompute 0 | 12-video full suite | IMP-12 | yes |
| PERF-003 | #170 | cache ≤64 GiB、peak追加≤96 GiB | target metrics | IMP-12 | yes |
| PERF-004 | #166, #169, #170 | Ollama≤18 GiB、STT≤8 GiB、GPU stage非重複 | system GPU metrics | IMP-11, IMP-12 | yes |
| QUAL-001 | #163〜168, #170 | invalid/near-duplicate selected 0、usable ≥90% | human review | IMP-12 | yes |
| QUAL-002 | #165, #167, #170 | semantic contradiction <10%、stable reject enum | human + report query | IMP-08, IMP-12 | yes |
| REL-001 | #169, #170 | video-only README/help/package 2.0.0 | final smoke/release checklist | IMP-13 | yes |

## Definition of done

Issue #170の設計完了は、この文書、ADR 0007、prototype、target profile template、13件の
ready-for-agent実装Issueが相互に参照し、上のtraceability matrixにorphanがないこととする。
production実装とpublic cutoverはIMP-01〜IMP-13の責務であり、Issue #170では行わない。
