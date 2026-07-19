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
完了後のgate failureにする。

coldのVideo Identity cache missではwhole-file SHA-256を一度計算する。exact warmはcoldで
確定したpath非依存identityをdevice、inode、size、mtime、ctime一致時だけ再利用し、1 TiB級
full Video Setを再hashしない。fullの独立Video Scanは最大2 workerで先行確定される。

release intervalは全streamをFFmpeg stream copyした`scenario-001.mkv`形式の匿名clipに
変換する。source metadataとchapterは引き継がず、FFmpegのbitexact format flagを使うため、
同じ入力、区間、tool identityから再生成したclipは同じwhole-file fingerprintになる。
ffprobeの実測開始、終了、durationがprofileの許容差を超える場合はpipeline前にexit 2に
なる。materialize時間はphase予算に含めない。

## Durable resumeとreset

phase完了はsuite別の`acceptance-state.json`へatomicに確定する。中断後に同じcommandを
実行すると、同じprofile、suite、設定、source snapshot、Resolved Model Identity、commitを
検証し、未完了phaseだけを続行する。completed coldを再実行してwarmへ戻したり、completed
cold/warmを再実行したりしない。

identityを変えた場合や意図的にcoldからやり直す場合だけ、対象suiteを明示的にresetする。

```bash
uv run task acceptance-target \
  --profile /absolute/private/target-acceptance.toml \
  --suite release \
  --reset-suite
```

`--reset-suite`は選んだsuiteのstate、phase output、worksheet、processing cacheを破棄する。
releaseとfullのartifactは混在しない。

## Human review

cold/warmと自動gateが完了すると、次のprivate worksheetが生成される。

```text
<artifact_root>/target-acceptance/<suite>/review-worksheet.json
```

target上のphase outputとmediaを確認し、各selected entryのpending値をworksheet記載の
stable enumへ置き換える。`reviewer`と`completed_at`も記入する。

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
| 130 | user interrupt。completed phaseとCompleted Stage cacheを保持 |

## Artifactとprivacy

`artifact_root`配下にはsuiteごとにphase output、durable state、private worksheet、
`acceptance.json`を置く。recordにはcommit、target/runtime、Resolved Model Identity、
path非依存Video Set fingerprint、phase/cache/storage/GPU aggregate、gate aggregateだけを
含める。absolute path、video名、media、raw Context Cue、prompt、model response、credential、
個別human判定は含めない。

releaseのtemporary clipとprocessing cacheはcold/warmおよびrecord生成後、合格・不合格・
review pendingのいずれでも削除する。phase output、state、worksheet、recordは保持する。

合格時は同じsuite directoryの`baseline/baseline.json`と`baseline/baseline.md`へ、source
commitを除いた正規化baselineを生成する。通常runではtarget artifactに留める。Issue #190の
cutoverまたはperformance contract変更時だけprivacy検査済みの両fileをreviewし、専用PRで
repositoryへ取り込む。実値入りprofileとprivate worksheetは一緒にcopyしない。
