# 動画単位のVideo Stage

この文書は、公開前の動画入力selectorが一つのVideo Sourceから再利用可能なFrame Candidateを作る内部契約を説明します。現在のscreenshot入力CLIからはまだ呼び出されません。

## Processing Stage

Video Set内の動画はVideo Order順に直列処理されます。各Video Identityには、次の2つのCompleted Stageが作られます。

1. `scan-video`
   - `attached_pic`を除外し、default disposition、stream indexの順でPrimary Video Streamを決めます。
   - 一回のnative decodeを、1秒heartbeat、320px scene signal、全frame timingへ分岐します。
   - Heartbeat Proxy、scene signalの時刻、exact timeline、scan metricをatomicに確定します。
2. `extract-frame-candidates`
   - density windowごとに最大1件のCandidate Momentを発見します。
   - Moment前後のnative frameだけを一回のrange scanで取り出します。
   - 重なるMoment windowを一つのRefinement Window Groupとして順次処理し、選抜proxyを書いた時点でそのgroupのRGB frameを解放します。
   - group内でmodel-free Neutral Image Analysis、無効frame除外、Moment内deduplication、多様性選抜を行います。
   - Frame Candidate Proxyと抽出metricをatomicに確定します。

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

## Cache再利用

`scan-video`はVideo Fingerprint、Primary Video Stream、Media Runtime Identity、decode backend、heartbeat/scene設定、proxy・scan・timeline algorithmだけで識別されます。Media Runtime IdentityはFFmpeg/ffprobe versionと、正規化済みbuild情報・検証済みcapability一覧から実行時に導出した完全SHA-256を持つため、同じversionの別buildでもcacheを再利用しません。raw build文字列や手入力hashは保存しません。density、refinement半径、最大Frame Candidate数、Neutral Image Analysisが変わっても再利用されます。

`extract-frame-candidates`は上流scan fingerprint、density、refinement半径、最大Frame Candidate数、Neutral Analysis・reject・dedupe・ID・proxy algorithmで識別されます。

次のdownstream値はどちらのfingerprintにも入りません。

- 選択画像枚数
- Selection Intent、scene hint、spoiler sensitivity、similarity threshold
- Ollama、STT、model lifecycle設定
- source path、filename、Video Order

Completed Stage manifestは`artifact.json`を含む全artifactの相対path、byte数、SHA-256を記録します。JPEGが1件でも欠損・破損した場合はStage全体を再利用せず、上流の健全なStageを残して再計算します。

## Metric

`scan-video`はexact duration、wall/CPU秒、処理速度、decode backend、decode pass数、heartbeat件数・容量・最大/p95 gap、scene signal件数、segment件数を記録します。

`extract-frame-candidates`はwall/CPU秒、density上限と実Moment数、native frame件数、reason別reject件数、dedupe件数、0-frame Moment件数、Frame Candidate件数・容量を記録します。両Stageのwall時間はartifact構築まで、CPU時間はcurrent processのPython/OpenCV/NumPyとFFmpeg child processの合計です。cache hitでは初回計算時のmetricを復元し、metric値はStage Fingerprintに含めません。

## 検証

```bash
uv run task test
uv run task test-ffmpeg
```

生成FFmpeg fixtureでCFR/VFR、first/middle/last、scene境界、single decode、metadataなしMJPEG、exact range refinementを検証します。
