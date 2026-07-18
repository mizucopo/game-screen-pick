# 動画単位のVideo Stage

この文書は、公開前の動画入力selectorが一つのVideo Sourceから再利用可能なFrame CandidateとContext Cueを作る内部契約を説明します。現在のscreenshot入力CLIからはまだ呼び出されません。

## Processing Stage

Video Set内の全sourceをVideo Order順にprobeした後、独立した`scan-video`をlogical CPU数に応じて最大2 workerで先行確定します。`extract-frame-candidates`と`collect-context`、結果順、progress通知はVideo Order順です。各Stage境界ではpath・device・inode・size・mtime・ctime snapshotを検査し、内容のwhole-file SHA-256はVideo Identity cache miss時だけ計算します。

1. `scan-video`
   - `attached_pic`を除外し、default disposition、stream indexの順でPrimary Video Streamを決めます。
   - 一回のnative decodeを、1秒heartbeat、320px scene signal、全frame timingへ分岐します。
   - heartbeat/scene proxyは1件ずつRGB decode・測定して解放し、全proxyのRGBを同時保持しません。
   - Heartbeat Proxy、scene signalの時刻、exact timeline、scan metricをatomicに確定します。
2. `extract-frame-candidates`
   - timeline順の単調windowでscene近傍のheartbeat品質を参照し、density windowごとに最大1件のCandidate Momentを発見します。
   - Moment前後のnative frameだけをrange scanで取り出します。独立rangeは入力順を保ったまま、logical CPU数に応じて最大4 workerで並列decodeします。
   - 同時に保持するdecode結果はworker数以下へ制限します。重なるMoment windowを一つのRefinement Window Groupとして順次処理し、選抜proxyを書いた時点でそのgroupのRGB frameを解放します。
   - group内でmodel-free Neutral Image Analysis、無効frame除外、Moment内deduplication、多様性選抜を行います。
   - Frame Candidate Proxyと抽出metricをatomicに確定します。
3. `collect-context`
   - embedded text subtitleを優先し、選ばれなかった場合だけaudio STTを実行します。forced subtitleの場合はsubtitleとSTTの両方を保持します。
   - subtitle packet PTSまたは16 kHz PCM sample gridをexact Video Timeへ対応付け、Context Cueとsource別outcomeをatomicに確定します。
   - Frame CandidateやCandidate Momentを入力にせず、Context Cueだけで候補を生成したり、視覚的に不適格なframeを適格化したりしません。

Video Stage cacheはVideo Fingerprintごとに`<VIDEO_INPUT_FOLDER>/.game-screen-pick/cache/videos/`へ保存されます。動画のrenameやVideo Order変更はfingerprintに含まれないため、同じ内容の動画では再利用されます。

## Exact timeline

Video Timeは整数source PTSとstream time baseから導出し、最初の表示frameを0とします。Video Durationは最終表示frameの`PTS + duration_ts`を優先し、取得できない場合だけstreamのexactな`start_pts + duration_ts`を使います。float秒、平均fps、frame間隔からは推測しません。

Timeline Segmentは0、scene signalのVideo Time、Video Durationを境界とする半開区間です。scene境界は後側のsegmentに属し、全segmentが`[0, Video Duration)`をgapやoverlapなく覆います。

## ProxyとNeutral Image Analysis

- Heartbeat Proxy: 長辺960px以下、FFmpeg MJPEG `q:v=3`、source metadataなし
- scene signal画像: 長辺320px以下。Scan Proxy Analysis後に削除し、Completed Stageには時刻と解析結果だけを残す
- Frame Candidate Proxy: 長辺960px以下、FFmpeg MJPEG `q:v=3`、source metadataなし
- 元解像度frame: cacheしない。公開時にexact PTSから再抽出する

Neutral Image AnalysisはOpenCV/NumPyの画質metricsとL2正規化済みHSV・輝度・edge特徴だけを使い、CLIPやHugging Face modelをloadしません。stable reject reasonは次の6種類です。

- `blackout`
- `whiteout`
- `single_tone`
- `blur`
- `fade_transition`
- `temporal_transition`

絶対的に無効な露出・単色・ぼけを先に除外し、暗いゲーム画面にはRefinement Window Group内の相対分布を使います。`temporal_transition`は同一stream・time baseで、前frameの`PTS + duration_ts`が次frameのPTSと一致する3つのnative frameだけに適用します。離れたrangeのsampleを前後frameとして比較しません。

## Context CueとSpeechRuntime

subtitle/audio streamは明示index、設定言語、default dispositionの順で一意に選びます。同順位が複数ある場合はstream indexで推測せずfatalにします。選ばれたbitmap subtitle、存在しない明示index、decode失敗でも別streamやaudioへsilent fallbackしません。track自体がない場合は正常な`absent`、選択・decodeされたtrackにeventがない場合は正常な`no_context / no_subtitle_events` outcomeとして扱います。後者がnon-forced subtitleならaudio STTへfallbackしません。

non-forced text subtitleが選ばれるとSTTは実行しません。forced subtitleではaudioも選び、両sourceのCueとprovenanceを残します。同じ正規化本文かつ時間範囲が重なるCueは、一回の発話ごとに各source最大1件の非推移的なequivalence groupへまとめ、後続annotationにはsource PTSを持つsubtitleを代表として一度だけ渡します。別時刻の同文subtitleは長いSTT Cueを介して畳みません。

各Cueのprivate cache provenanceにはcodec、stream languageとdisposition、観測source PTS/time baseを保持します。STT Cueと低reliability診断にはさらにPCM chunk sample範囲、Speech Runtime Identity、freeze済みResolved Model Identity、device、compute typeを保持し、採用Cueにはsegmentの未校正average log probability、no-speech probability、word probabilityも保存します。これらはcache再利用とtraceabilityのための値であり、公開reportへraw textやbackend scoreを出しません。

audioはMediaRuntimeがmono signed 16-bit、16 kHzの連続PCM sample gridとしてstreaming decodeします。設定したchunkとoverlapへ分け、overlap中央の半開境界で一つのCue所有者を決めます。観測PCM origin、chunk連続性、word timestampが1 sampleの許容範囲を超える場合はclipせず`timestamp_drift`にします。SpeechRuntimeはmodel lifecycleが解決・freezeしたlocal artifactだけをloadし、backend非依存のword sample位置を返します。Context Stageが1.5秒のword gap、Video Time対応付け、reliability policy、Cue IDを所有します。

STT結果は平均log probabilityが-0.8以上かつ句読点・空白以外が3文字以上のword groupだけをContext Cueにします。VADで発話がなかった`vad_no_speech`、VAD通過後に文字がなかった`asr_no_speech`、文字はあったが全件不採用の`low_reliability`を区別します。不採用のraw transcriptとbackend値はprivate processing cacheの診断だけに保存し、annotation、進捗、error、public reportへ渡しません。`--reset-cache`では他のprocessing cacheとともに削除されます。

## Cache再利用

`scan-video`はVideo Fingerprint、Primary Video Stream、Media Runtime Identity、decode backend、heartbeat/scene設定、proxy・scan・timeline algorithmだけで識別されます。Media Runtime IdentityはFFmpeg/ffprobe versionと、正規化済みbuild情報・検証済みcapability一覧から実行時に導出した完全SHA-256を持つため、同じversionの別buildでもcacheを再利用しません。raw build文字列や手入力hashは保存しません。density、refinement半径、最大Frame Candidate数、Neutral Image Analysisが変わっても再利用されます。

`extract-frame-candidates`は上流scan fingerprint、density、refinement半径、最大Frame Candidate数、Neutral Analysis・reject・dedupe・ID・proxy algorithmで識別されます。

`collect-context`はVideo Fingerprint、scanのexact timeline digest、選択streamと選択・抽出policy、Media Runtime Identity、Cue生成policyで識別されます。STTを実行した場合だけSpeech Runtime Identity、実行時にfreezeしたResolved Model Identity、device、compute type、beam、VAD、chunk、overlapを加えます。設定上のmodel alias、model更新時刻、`models.auto_upgrade`は、Resolved Model Identityが同じならfingerprintへ含めません。subtitleだけを使う場合はSTT関連identityと設定をすべて無視します。

次のdownstream値は`scan-video`と`extract-frame-candidates`のfingerprintに入りません。Context StageもFrame Candidate設定や最終選定設定には依存しません。

- 選択画像枚数
- Selection Intent、scene hint、spoiler sensitivity、similarity threshold
- Ollama、STT、model lifecycle設定
- source path、filename、Video Order

Completed Stage manifestは`artifact.json`を含む全artifactの相対path、byte数、SHA-256を記録します。JPEGが1件でも欠損・破損した場合はStage全体を再利用せず、上流の健全なStageを残して再計算します。

## Metric

`scan-video`はexact duration、wall/CPU秒、処理速度、decode backend、decode pass数、heartbeat件数・容量・最大/p95 gap、scene signal件数、segment件数を記録します。

`extract-frame-candidates`はwall/CPU秒、density上限と実Moment数、native frame件数、reason別reject件数、dedupe件数、0-frame Moment件数、Frame Candidate件数・容量を記録します。両Stageのwall時間はartifact構築まで、CPU時間はcurrent processのPython/OpenCV/NumPyとFFmpeg child processの合計です。cache hitでは初回計算時のmetricを復元し、metric値はStage Fingerprintに含めません。

`collect-context`は選択・試行したsourceごとに`available`、`absent`、`no_context`、`no_speech`、`low_reliability`とstable reason、Cue件数、除外件数を記録します。ambiguous stream、unsupported subtitle、decode/STT失敗、timestamp drift、一部chunkの`chunk_failed`はfatalであり、先に成功したCueを含むpartial manifestを公開しません。

## 検証

```bash
uv run task test
uv run task test-ffmpeg
```

生成FFmpeg fixtureでCFR/VFR、first/middle/last、scene境界、single decode、metadataなしMJPEG、exact range refinementを検証します。
