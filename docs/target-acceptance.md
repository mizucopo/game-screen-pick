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
- 明示URLで接続するWindows native Ollama

host alias、WSL gateway、実際のmedia pathはrepositoryへ保存しない。別構成で動作しても、
v2.0のfull-runtime合格を示すrecordはこのtargetでだけ生成する。

## Private profile

[`examples/target-acceptance.toml`](examples/target-acceptance.toml)をrepository外へ
copyし、target上の実値へ置き換える。実値入りprofileはcommit、Issue/PR artifact、
`acceptance.json`へ添付しない。

profile schemaはstrictで、次だけを持つ。

- `input_root`: full suiteのVideo Set rootとrelease intervalの基準root
- `configuration_path`: 通常のVideo Selection TOML
- `artifact_root`: suite state、phase output、private worksheetを置くprivate root
- `release_suite`: 合計duration、境界tolerance、relative source、start/end、scenario role
- `full_scale_suite`: video count、合計duration、duration tolerance

Ollama host、model、STT device、選択枚数などをprofileへ複製しない。これらは
`configuration_path`のTOMLを通常どおり読み、明示CLI、TOML、`OLLAMA_HOST`、組み込み
既定値の優先順位で解決する。harnessが最優先で差し替えるのは、匿名化したsuite用
Video Input Folderとcold/warm別Output Folderだけである。

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

modelの更新確認、download、capability検証とResolved Model Identityのfreezeはcold timer
より前に行う。一回の起動でclean cacheのcold、同じVideo Set・設定・model identity・
processing cacheを使うexact warmの順に実行する。性能予算超過は処理を途中でkillせず、
完了後のgate failureにする。phase durationはatomic publicationまたはoperation failureで
確定し、その後のresource monitor停止時間を含めない。

新規suiteではmaterialize後かつmodel実行より前に、materialize済みsuite inputの合計byteと
artifact filesystemの空き容量を測る。persistent cache 64 GiBとtemporary/staging 96 GiBの
合計160 GiB未満なら長時間処理を開始せずpreflight failureにする。開始時の測定値はdurable
stateとprivacy-safe recordへ保存する。

coldのVideo Identity cache missではwhole-file SHA-256を一度計算する。exact warmはcoldで
確定したpath非依存identityをdevice、inode、size、mtime、ctime一致時だけ再利用し、1 TiB級
full Video Setを再hashしない。fullの独立Video Scanはlogical CPU 8個につき1 worker、
最大3 workerで並列実行する。Video Order上の対象scanが確定した時点で、そのVideoの
candidate extractionとcontext collectionを後続Videoのscanと重ねて開始する。background
scan待機中もactive Stageとheartbeatを通知し、通常のscan失敗でも一次障害を保持したまま
待機中workerをcancelする。
scanのprocess登録とcancellation要求は同じlockで直列化し、cancel後に新しいdecoderを開始しない。
新規scan artifactは対象sourceのcontent snapshotを確定直前に再検証し、scan中に変更された
bytesを元のVideo Fingerprint配下へ保存しない。
candidate extractionのCPU時間は所有threadと、そのStageが起動したrange decoder・proxy
encoder subprocessだけを合算し、並列中の後続Video Scanを二重計上しない。
Context Collection完了後はSpeech Runtime Identityを保持してSTT modelを明示closeし、
GPU資源を解放してからOllama Vision推論を開始する。

release intervalは全streamをFFmpeg stream copyした`scenario-001.mkv`形式の匿名clipに
変換する。source metadataとchapterは引き継がず、FFmpegのbitexact format flagを使うため、
同じ入力、区間、tool identityから再生成したclipは同じwhole-file fingerprintになる。
ffprobeの実測開始、終了、durationがprofileの許容差を超える場合はpipeline前にexit 2に
なる。ffprobeのcontainer差は、stream durationがある場合はstream timing、ないMatroskaでは
非0 startを含むformat end、それ以外ではformat elapsed durationから、absolute endと経過
durationへ正規化する。各release区間の境界だけでなく、全clipの正規化済み実測duration合計も
profileの期待合計と同じtolerance内であることを検証する。full suiteはこの正規化済み経過
durationを合算する。
確定済みrelease inputの再利用時はmanifest記載clipと対応videoの完全一致も検証する。
materialize時間はphase予算に含めない。

## Durable resumeとreset

phase完了はsuite別の`acceptance-state.json`へatomicに確定する。中断後に同じcommandを
実行すると、同じprofile、suite、設定、source snapshot、Resolved Model Identity、target
identity、commitを検証し、未完了phaseだけを続行する。driver、FFmpeg、kernelなどtarget
probeの値が変わったstateは混在させない。completed coldを再実行してwarmへ戻したり、
completed cold/warmを再実行したりしない。
設定file外の`OLLAMA_HOST`を含む実効endpointも、URLを公開しないdigestとしてsuite identityへ
固定する。

完了済みphaseからhuman reviewを再開する場合も、現在のsourceをmaterializeし直してsuite
fingerprintを照合し、Resolved Model Identityを再解決してからrecordを確定する。入力または
modelが変わっていれば既存の完了stateを流用しない。同じ実行identityに対する
`Model Update Status`や更新前identityの違いはrun別診断であり、再利用可否を変えない。
worksheet未生成から再開するときはcold reportをphase digestと照合し、selection artifactを
Completed Stage manifest、artifact hash、semantic fingerprintで再検証する。

user interruptや計測済みoperation failureの未完了phaseはCompleted Stage cacheを保持する。
再開後のphase recordでは、それ以前の試行を含む経過時間、cache/recompute count、Stage時間、
storage/GPU aggregateを累積または保守的な最大値として集計するため、再開後の短い試行だけで
性能を判定しない。user interruptで詳細計測を確定できなかった場合も経過時間を試行へ残し、
resource計測を不完全とする。Completed Stage cacheから再開できるが、そのsuiteは不完全な
性能・resource根拠では合格しない。process強制終了などでinterrupt handler自体を通らず
active phaseが残った場合は、安全な試行境界を復元できないため`--reset-suite`を要求する。

identityを変えた場合や意図的にcoldからやり直す場合だけ、対象suiteを明示的にresetする。

```bash
uv run task acceptance-target \
  --profile /absolute/private/target-acceptance.toml \
  --suite release \
  --reset-suite
```

`--reset-suite`は選んだsuiteのstate、phase output、worksheet、processing cacheを破棄する。
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
| 0 | cold/warm、performance/resource/privacy、human qualityの全gateが合格 |
| 1 | pipeline operationまたはperformance/resource/privacy/quality gateが不合格 |
| 2 | CLI、profile、configuration、target preflightが不正 |
| 3 | cold/warmと自動gateは合格したがhuman reviewが未完了 |
| 130 | 計測済みuser interrupt。completed phaseとCompleted Stage cacheを保持し、再開時に試行計測を累積 |

## Artifactとprivacy

`artifact_root`配下にはsuiteごとにphase output、durable state、private worksheet、
`acceptance.json`を置く。recordにはcommit、target/runtime、Resolved Model Identity、
実Faster Whisper adapter/backendのSpeech Runtime Identity、path非依存Video Set fingerprint、
phase/cache/storage/GPU aggregate、gate aggregateだけを含める。canonical reportの各Stageも
実semantic inputと推論診断からtool/model/contract参照、設定、validation、token数を記録する。
performance比較用configurationには設定file digest、URLを含まないendpoint identity、
privacy-safeな全実効performance設定を保存する。
cold/warmの結果一致digestには選定判断とmodel identityに加え、公開WebPのSHA-256、寸法、
byte数を含め、同じ候補から異なる画像artifactが公開されたrunを一致扱いしない。
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
