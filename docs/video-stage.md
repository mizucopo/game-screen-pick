# 動画単位のVideo Stage

この文書は、公開前の動画入力selectorが一つのVideo Sourceから再利用可能なFrame CandidateとContext Cueを作る内部契約を説明します。現在のscreenshot入力CLIからはまだ呼び出されません。

## Processing Stage

Video Set内の全sourceをVideo Order順にprobeした後、独立した`scan-video`をbounded並列実行します。既定の`video_scan.workers = "auto"`では、CPU decodeはlogical CPU 8個につき1 workerの保守的上限を使います。NVDECはCPU・memory・NVIDIA Decoder・GPU・VRAM・diskの初期sampleにpressureがあれば1 worker、正常時は同じ保守値から開始します。その後は各resourceと1 streamあたりの処理速度をrolling windowで観測し、余力があればlogical CPU 4個につき1 worker、既定で最大6 workerまで使います。利用率は直近3 sample、disk throughputと処理速度のtrendは直近2 sample対その前の2 sampleで判断します。並列数はscan完了境界で1ずつ変更し、active scanを停止せず未開始taskの投入だけを調整します。resource sample、disk観測、またはstream速度trendを取得できない場合は増加せず、安全側のworker数を維持します。

Video Order上の対象scanが確定した時点で、後続Videoのscanを続けながら、そのVideoの`extract-frame-candidates`と`collect-context`を開始します。downstream、結果、progress通知はVideo Order順であり、後続scanの完了順には依存しません。worker数と変更履歴はStage Fingerprintやcache identityへ含めず、privacy-safeなrun provenanceへだけ記録します。各Stage境界ではpath・size・`mtime_ns` snapshotを検査し、内容のwhole-file SHA-256はVideo Identity cache miss時だけ計算します。device、inode、ctimeは再利用判定に使いません。Video IdentityのSHA-256はdisk/CPU処理であり、NVDECへ移せません。

一つのVideo Sourceの`extract-frame-candidates`では、互いに離れたRefinement Window Groupをbounded並列実行します。worker数はGroup数、最大4、実行中の後続Video Scanが予約しているlogical CPUを除いた残り容量（4 logical CPUにつき1 worker）、現在利用可能なmemoryから求めた上限の最小値です。CPU decodeのactive scanは1件あたり8 logical CPU、NVDEC scanは4 logical CPUを予約済みとして扱うため、scanのCPU予算を無視してRefinementを追加で最大4件走らせません。これはFFmpegのsoftware range decodeとOpenCV/NumPyによるmodel-free解析を行うCPU・disk処理であり、OllamaのGPU推論ではありません。worker数はpublic設定やsemantic identityに加えず、同じ入力を1 workerまたは4 workerで処理しても結果をPTS range順へ戻します。処理全体のresource対応は[Pipeline処理フローと計算資源](processing-flow.md)を参照してください。

Ctrl+Cでは未開始の`scan-video`を先に取り消し、その後で実行中のscanへ終了を要求します。Refinement Window Groupの並列処理中はqueued Groupを取り消し、既に実行中のGroupだけを短いatomic境界まで完了または失敗させます。割り込み後に待機中のscanやGroupを新しく開始しません。Completed Stageが未確定でも、atomicに確定済みのscan partition、Refinement Window Group、Embedded Subtitle stream、PCM sample range、STT chunkは次回runで再利用し、実行中または未開始だった最小Work Unitだけを再計算します。

1. `scan-video`
   - `attached_pic`を除外し、default disposition、stream indexの順でPrimary Video Streamを決めます。
   - exact stream timingを15分の固定PTS partitionへ分け、各partitionの一回のnative decodeを、1秒heartbeat、320px scene signal、全frame timingへ分岐します。streamの`duration_ts`がないcontainerでは、ffprobeのcontainer durationを有理数のままstream tickへ切り上げ、完全な15分区間の境界だけを決めます。15分未満の端数は独立partitionにせず、最後のpartitionを直前の境界からEOFまで開きます。過大なcontainer tailで空partitionへ到達した場合は、その空結果を確定して同じ開始PTSからEOFまでを確認します。後半frameがなければ後続境界を止め、timestamp gap後にframeがあれば最終partitionとして保持します。
   - 各partitionをDurable Work Unitとしてatomicに確定します。cold runと再開runが同じpartition境界を使うため、再開の有無でtimeline、scene signal、Candidate IDを変えません。
   - heartbeat/scene proxyは1件ずつRGB decode・測定して解放し、全proxyのRGBを同時保持しません。
   - partitionをPTS順に集約し、Heartbeat Proxy、scene signalの時刻、exact timeline、scan metricをCompleted Stageとしてatomicに確定します。
2. `extract-frame-candidates`
   - timeline順の単調windowでscene近傍のheartbeat品質を参照し、density windowごとに最大1件のCandidate Momentを発見します。
   - Moment前後のnative frameだけをrange scanで取り出します。重なるMoment windowを一つのRefinement Window Groupとし、各Groupは一つの独立range decode、Neutral Image Analysis、proxy encodeを所有します。
   - Group数、active Video Scanの予約分を除いたlogical CPU数、最大4のsafe cap、available memoryからworker数を自動決定します。memory上限は長辺960px、240 frame/秒、保持frameあたり4 byte/pixelで各PTS rangeを保守的に見積もり、その時点のavailable memoryの4分の1へ同時Groupが収まる値です。残る4分の3はOpenCV/NumPyの解析用一時領域、runtime、他processの余裕として予約します。available memoryまたはsource寸法を取得できない場合は1 workerへ抑制します。
   - 同時に保持するGroupのRGB decode結果をworker数以下へ制限し、選抜proxyを書いた時点でそのGroupのRGB frameを解放します。resource値と決定worker数は実行時制御だけに使い、Completed Stage Fingerprint、Durable Work Unit key、成果物へ含めません。
   - 各Refinement Window Groupのproxyと解析結果を別々のDurable Work Unitとしてatomicに確定し、並列完了順にかかわらずPTS順に親Stageへ集約します。cache hit Groupはdecodeせず、破損・未完了Groupだけを再計算します。
   - group内でmodel-free Neutral Image Analysis、無効frame除外、Moment内deduplication、多様性選抜を行います。最初に最高Qualityのframeを保持し、残りは選択済みframeとの最小時間距離を最優先、最小視覚距離とQualityを後続条件として、Refinement Window全体へ決定的に分散させます。
   - Frame Candidate Proxyと抽出metricをatomicに確定します。
3. `collect-context`
   - embedded text subtitleを優先し、選ばれなかった場合だけaudio STTを実行します。forced subtitleの場合はsubtitleとSTTの両方を保持します。
   - embedded text subtitleは選択stream全体を一つのDurable Work Unitとして確定し、再開時に再抽出しません。
   - audioを16 kHz mono PCMへ変換するsample rangeと、overlapを含むPCM chunkごとのSpeech Recognition Resultを別々のDurable Work Unitとしてatomicに確定します。STT modelだけが変わった場合はPCMを保持し、未完了または失効した認識chunkだけを再推論します。
   - subtitle packet PTSまたは16 kHz PCM sample gridをexact Video Timeへ対応付け、Context Cueとsource別outcomeをatomicに確定します。
   - Frame CandidateやCandidate Momentを入力にせず、Context Cueだけで候補を生成したり、視覚的に不適格なframeを適格化したりしません。

Completed Video StageはVideo Fingerprintごとに`<VIDEO_INPUT_FOLDER>/.game-screen-pick/cache/videos/`、Stage内の最小checkpointは`cache/work-units/`へ保存されます。動画のrenameやVideo Order変更はStage fingerprintに含まれないため、同じ内容の動画では再利用されます。詳細は[Pipelineと安全な再開](pipeline-resume.md)を参照してください。

## Exact timeline

Video Timeは整数source PTSとstream time baseから導出し、最初の表示frameを0とします。Video Durationは最終表示frameの`PTS + duration_ts`を優先し、取得できない場合だけstreamのexactな`start_pts + duration_ts`を使います。container durationはpartition開始点を不足させないためのhintに限り、Video Durationやframe時刻には使いません。float秒、平均fps、frame間隔からは推測しません。

Timeline Segmentは0、scene signalのVideo Time、Video Durationを境界とする半開区間です。scene境界は後側のsegmentに属し、全segmentが`[0, Video Duration)`をgapやoverlapなく覆います。

## ProxyとNeutral Image Analysis

- Heartbeat Proxy: 長辺960px以下、FFmpeg MJPEG `q:v=3`、source metadataなし
- scene signal画像: 長辺320px以下。Scan Proxy Analysis後に削除し、Completed Stageには時刻と解析結果だけを残す
- Frame Candidate Proxy: 長辺960px以下、FFmpeg MJPEG `q:v=3`、source metadataなし
- 元解像度RGB frame: cacheしない。公開時にexact PTSから再抽出し、選択画像1枚ごとの固定WebPだけをDurable Work Unitとして保存する

Neutral Image AnalysisはOpenCV/NumPyの画質metricsとHSV・輝度・edge特徴だけを使い、各特徴成分を個別にL2正規化して等しく結合します。解像度や勾配量のscale差で一成分が他成分を消失させず、CLIPやHugging Face modelもloadしません。stable reject reasonは次の6種類です。

- `blackout`
- `whiteout`
- `single_tone`
- `blur`
- `fade_transition`
- `temporal_transition`

絶対的に無効な露出・単色・ぼけを先に除外します。`whiteout`は純白一色に限らず、中央の主対象を覆う大きな連結白領域、画面全体の4%以上25%未満かつ中央領域の15%以上を占める3.5%以上の連結白領域、画面の大半を覆う低情報の淡い白も対象にします。これにより背景が見える局所的な白い発光も除外します。明るくても罫線や区画が判別できるmenuは、明るさだけを理由に除外しません。暗いゲーム画面にはRefinement Window Group内の相対分布を使います。`temporal_transition`は同一stream・time baseで、前frameの`PTS + duration_ts`が次frameのPTSと一致するnative frameだけに適用します。短い3-frame dipに加え、0.25秒以内に淡い連結明領域が画面の一部から60%以上へ拡大または縮小する3枚以上のframeを遷移として除外します。領域が静止した明るいmenuは除外しません。離れたrangeのsampleを前後frameとして比較しません。

## Context CueとSpeechRuntime

subtitle/audio streamは明示index、設定言語、default dispositionの順で一意に選びます。同順位が複数ある場合はstream indexで推測せずfatalにします。選ばれたbitmap subtitle、存在しない明示index、decode失敗でも別streamやaudioへsilent fallbackしません。track自体がない場合は正常な`absent`、選択・decodeされたtrackにeventがない場合は正常な`no_context / no_subtitle_events` outcomeとして扱います。後者がnon-forced subtitleならaudio STTへfallbackしません。

non-forced text subtitleが選ばれるとSTTは実行しません。forced subtitleではaudioも選び、両sourceのCueとprovenanceを残します。同じ正規化本文かつ時間範囲が重なるCueは、一回の発話ごとに各source最大1件の非推移的なequivalence groupへまとめ、後続annotationにはsource PTSを持つsubtitleを代表として一度だけ渡します。別時刻の同文subtitleは長いSTT Cueを介して畳みません。

各Cueのprivate cache provenanceにはcodec、stream languageとdisposition、観測source PTS/time baseを保持します。STT Cueと低reliability診断にはさらにPCM chunk sample範囲、Speech Runtime Identity、freeze済みResolved Model Identity、device、compute typeを保持し、採用Cueにはsegmentの未校正average log probability、no-speech probability、word probabilityも保存します。これらはcache再利用とtraceabilityのための値であり、公開reportへraw textやbackend scoreを出しません。

audioはMediaRuntimeがmono signed 16-bit、16 kHzの連続PCM sample gridとしてstreaming decodeします。設定したchunkとoverlapへ分け、overlap中央の半開境界で一つのCue所有者を決めます。観測PCM origin、chunk連続性、範囲外または逆転したword timestampが1 sampleの許容範囲を超える場合はclipせず`timestamp_drift`にします。SpeechRuntimeはmodel lifecycleが解決・freezeしたlocal artifactだけをloadし、backend非依存のword sample位置を返します。Context Stageが1.5秒のword gap、Video Time対応付け、reliability policy、Cue IDを所有します。

STT結果は正の時間幅を持ち、平均log probabilityが-0.8以上かつ句読点・空白以外が3文字以上のword groupだけをContext Cueにします。group全体のstartとendが同一点なら時刻を推測で延長せず、幅0の`asr_zero_duration`診断へ隔離します。VADで発話がなかった`vad_no_speech`、VAD通過後に文字がなかった`asr_no_speech`、文字はあったが全件不採用の`low_reliability`を区別します。不採用のraw transcriptとbackend値はprivate processing cacheの診断だけに保存し、annotation、進捗、error、public reportへ渡しません。`--reset-cache`では他のprocessing cacheとともに削除されます。

## Cache再利用

`scan-video`はVideo Fingerprint、Primary Video Stream、Media Runtime Identity、decode backend、heartbeat/scene設定、proxy・scan・timeline algorithmだけで識別されます。Media Runtime IdentityはFFmpeg/ffprobe versionと、正規化済みbuild情報・検証済みcapability一覧から実行時に導出した完全SHA-256を持つため、同じversionの別buildでもcacheを再利用しません。raw build文字列や手入力hashは保存しません。density、refinement半径、最大Frame Candidate数、Neutral Image Analysisが変わっても再利用されます。

`extract-frame-candidates`は上流scan fingerprint、density、refinement半径、最大Frame Candidate数、Neutral Analysis・reject・dedupe・ID・proxy algorithmで識別されます。

`collect-context`はVideo Fingerprint、scanのexact timeline digest、選択streamと選択・抽出policy、Media Runtime Identity、Cue生成policyで識別されます。STTを実行した場合だけSpeech Runtime Identity、実行時にfreezeしたResolved Model Identity、device、compute type、beam、VAD、chunk、overlapを加えます。設定上のmodel alias、model更新時刻、`models.auto_upgrade`は、Resolved Model Identityが同じならfingerprintへ含めません。subtitleだけを使う場合はSTT関連identityと設定をすべて無視します。

次のdownstream値は`scan-video`と`extract-frame-candidates`のfingerprintに入りません。Context StageもFrame Candidate設定や最終選定設定には依存しません。

- 選択画像枚数
- Selection Intent、scene hint、spoiler sensitivity、similarity threshold
- Ollama、STT、model lifecycle設定
- source path、filename、Video Order

Completed Stage manifestは`artifact.json`を含む全artifactの相対path、byte数、SHA-256を記録します。JPEGが1件でも欠損・破損した場合は親Stageを直接再利用しません。健全なscan partitionまたはRefinement Window Groupが残っていればそれらから親Stageを再集約し、破損した最小Work Unitだけを再計算します。

## Metric

`scan-video`はexact duration、wall/CPU秒、処理速度、decode backend、decode partition数、heartbeat件数・容量・最大/p95 gap、scene signal件数、segment件数を記録します。

`extract-frame-candidates`はwall/CPU秒、density上限と実Moment数、native frame件数、reason別reject件数、dedupe件数、0-frame Moment件数、Frame Candidate件数・容量を記録します。両Stageのwall時間はartifact構築まで、CPU時間はcurrent processのPython/OpenCV/NumPyとFFmpeg child processの合計です。cache hitでは初回計算時のmetricを復元し、metric値はStage Fingerprintに含めません。

`collect-context`は選択・試行したsourceごとに`available`、`absent`、`no_context`、`no_speech`、`low_reliability`とstable reason、Cue件数、除外件数を記録します。ambiguous stream、unsupported subtitle、decode/STT失敗、PCM gridまたは範囲外・逆転時刻のtimestamp drift、一部chunkの`chunk_failed`はfatalであり、先に成功したCueを含むpartial Completed Stage manifestを公開しません。ただし検証済みSpeech Recognition chunk checkpointは保持し、次回runでは失敗chunkだけを再実行します。

## 検証

```bash
uv run task test
uv run task test-ffmpeg
```

生成FFmpeg fixtureでCFR/VFR、first/middle/last、scene境界、single decode、metadataなしMJPEG、exact range refinementを検証します。
