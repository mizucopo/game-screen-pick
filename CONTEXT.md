# Game Screen Pick

ゲーム動画から、ブログで使いやすい画像を選び出すための文脈。

## Language

**Video Set**:
1回のブログ画像選定でまとめて扱う、1本以上の順序を持つ入力動画の集合。実行中は構成・順序・各Videoの内容が変わらないsnapshotとして扱い、保存場所である input folderや選定設定とは区別する。
_Avoid_: input folder, video folder, run, file list

**Video Input Folder**:
Video Setを発見し、そのcacheを保持するためにユーザーが指定するルートフォルダ。Video Setそのもののidentityではない。
_Avoid_: Video Set, Output Folder, cache key

**Input Lock**:
一つのVideo Input Folderに対する同時実行を即時拒否し、非破壊validation完了後からprocessing cacheの変更とrun終了までを保護する非待機排他境界。異なるVideo Input Folderの実行は互いに妨げない。
_Avoid_: Stage lock, waiting queue, global lock, cache artifact

**Video Set Snapshot Validation**:
Video Identity確定後に、Video Set全体のpath順、file size、`mtime_ns`を発見時snapshotへ照合する不変性検査。Input Lock取得後、各Video Sourceのmedia probe前、Work UnitとCompleted Stageの確定前、Output Folder公開前に実行し、Stageごとのwhole-file再hash、device・inode・ctimeの照合とは区別する。同じsize・mtimeへ意図的に内容を書き換えた場合は検知しない運用上のtrade-offを持つ。
_Avoid_: cache artifact validation, full-set hash per Video Stage, inode validation, Input Lock

**Video Set Fingerprint**:
Video Setの構成とVideo Orderをcacheやreportで参照するための、順序付きVideo Fingerprint列から導出される安定した識別子。
_Avoid_: input path hash, unordered file set, global setting hash

**Video Order**:
Video Set 内のゲーム進行順。入力ルートからの相対パスの自然順で決まり、更新日時やファイル列挙順には依存しない。
_Avoid_: filesystem order, mtime order, discovery order

**Video Source**:
Video Fingerprintで識別される、一つのVideo Identityとその有効なpresentation timelineの組。Video Input Folder内のpathやVideo Orderとは独立する。
_Avoid_: input file path, ordered Video Set member, path-based identity

**Video Identity**:
動画内容によって決まる、個々の Video の安定した同一性。ファイル名、配置、更新日時が変わっても内容が同じなら維持され、内容が変われば新しい identity になる。
_Avoid_: file path, filename, mtime, Video Order

**Video Identity Cache Entry**:
一つのlogical sourceについて、identity engine version、入力rootと相対pathから導出したprivacy-safe key、file size、`mtime_ns`、whole-file SHA-256をatomicに保持するlookup hint。動画1本のhash確定直後に保存し、processing cacheのFresh Processing/reset境界から寿命を分離する。Video Identityそのものではなく、raw path、video名、device、inode、ctimeを保持しない。
_Avoid_: Video Identity, stat-only identity, processing Stage, absolute path record

**Duplicate Video**:
一つのVideo Set内で、複数のpathが同じVideo Identityを指している入力不整合。
_Avoid_: similar video, repeated scene, same filename

**Video Fingerprint**:
Video Identity をcacheやreportで参照するための、動画内容から導出される安定した識別子。
_Avoid_: path hash, file stat, stage setting hash

**Processing Stage**:
再開可能な画像選定を構成する、入力と再利用可能な成果物の境界が明示された処理単位。
_Avoid_: arbitrary function, progress message, whole run

**Progress Event**:
一回のrunとProcessing Stageの開始、観測可能な進行、cache利用、完了、中断、失敗をrenderer非依存の型付き値で表す通知。表示文、外部toolの生出力、raw Context Cue、prompt、model responseは含めない。
_Avoid_: log line, renderer message, model trace, exception text

**Comparable Work Series**:
一回のrun内でStage種別、work-unit種別、reuseまたはrecomputeの処理方法が等しく、Stage ETAのsampleを共有できる作業系列。runをまたぐ実績や、今後の処理方法が未確定なunitを推定へ混ぜない。
_Avoid_: whole-run average, persisted benchmark, cache hit ratio forecast

**Run Failure**:
runを終了させる失敗をstable reason code、安全な観測値、修復方法、Completed Stageの再利用案内、終了codeで表した利用者向け結果。元の例外や外部toolの生出力そのものではない。
_Avoid_: traceback, raw exception, Progress Event message, partial success

**Stage Resource Metric**:
Completed Stageを初回計算するときの処理開始からartifact構築までを対象にしたwall時間とCPU時間。CPU時間はcurrent processとchild processの合計で、cache再利用時は初回に保存された値を復元する。
_Avoid_: FFmpeg-only metric, cache lookup duration, current-run reuse overhead

**Migration Gate**:
Video Set selectorのpublic cutoverに必要な実装PR、test、target性能、human quality、traceabilityの全証拠を一つの判定として扱う境界。不足時はscreenshot CLI、package version、legacy codeの公開状態を一切変えない。
_Avoid_: feature flag, partial rollout, runtime timeout, individual PR check

**Public Cutover**:
Migration Gate通過後の一つのPRで、public CLIをVideo Set入力へ切り替え、packageを2.0.0にし、screenshot固有codeを削除するatomicな公開変更。
_Avoid_: internal adapter, preview mode, compatibility period, Processing Stage

**Acceptance Profile**:
supported target上の実videoからrelease suiteとfull-scale suiteの対象を指定するtarget-onlyのuntracked設定。repositoryにはschema templateだけを置き、実pathやvideo名を記録しない。
_Avoid_: public TOML, Effective Configuration, committed fixture, production default

**Acceptance Record**:
一回のtarget acceptanceで得たcommit、runtime/model identity、pathなしのVideo Set fingerprint、Stage時間、resource、cache、quality判定をversioned JSONとして保存する証拠。media、absolute path、raw text、prompt、model responseを含めない。
_Avoid_: Canonical Selection Report, runtime log, baseline snapshot, raw benchmark output

**Acceptance Run**:
Acceptance PhaseまたはAcceptance Comparison Runを総称する論理的な完了・集約単位。一つ以上のAcceptance Run Attemptを持ち、再開をまたぐ全試行を集約して完了recordと予算判定を確定する。
_Avoid_: Acceptance Run Attempt, one process lifetime, Processing Stage, production run

**Acceptance Phase**:
target acceptanceでcacheなしの本処理を測るFresh Processing、または同じcacheを使うCache Reuseの論理的な性能判定単位。中断後に再開された場合は複数のAcceptance Run Attemptを持ち、全attemptの経過時間と作業量、保守的なresource peakをまとめて予算判定する。
_Avoid_: Acceptance Comparison Run, Acceptance Run Attempt, Processing Stage, one process lifetime

**Acceptance Comparison Run**:
full target acceptanceで固定workerと自動並列化を比較するParallelism Baseline。fresh processing cache上で固定3 workerを使う独立runであり、Fresh Processingと同じpipelineを実行するがAcceptance Phaseではない。完了後にcacheを削除してFresh Processingを開始する。
_Avoid_: Acceptance Phase, benchmark fixture, Cache Reuse, production default

**Acceptance Run Reset**:
target acceptanceの論理runと、その結果に依存する後続runだけを破棄して再測定する明示操作。利用者向け対象は`parallelism-baseline`、`fresh-processing`、`cache-reuse`であり、materialized inputと共有Video Identity cacheは保持する。`fresh-processing`はprocessing cacheも破棄し、`cache-reuse`は本処理cacheが残っている場合だけそれを保持して再測定する。
_Avoid_: suite全体reset, Processing Stage reset, cache migration, automatic Stage invalidation

**Video Scan Comparison Context**:
Parallelism BaselineとFresh Processingのwall timeを同一条件の証拠として比較するために共有する、source revision・実効設定/model identity・target runtimeを正規化したprivacy-safe identity。bootごとのvisible RAM差は含めず、両runの全attemptで一致するものだけを同一比較として扱う。
_Avoid_: Acceptance Run execution context, raw host snapshot, Stage artifact digest

**Acceptance Run Attempt**:
一つのAcceptance PhaseまたはAcceptance Comparison Runについて、開始から正常終了、中断、またはoperation failureまで連続して計測された実行区間。Completed Stageを再利用してrunを再開しても、それ以前のattemptを性能証拠から除外しない。process強制終了時もactive markerとAcceptance Attempt Journalから`process_abandoned`として閉じる。
_Avoid_: Acceptance Run, Acceptance Phase, Acceptance Comparison Run, retry inside model inference, Processing Stage

**Acceptance Attempt Journal**:
activeなAcceptance Run Attemptのexecution context、cache・reuse・recompute件数、Stage aggregate、Work Unitごとのresolutionを、Progress Event境界でatomicに更新するprivacy-safeな回復記録。process強制終了後はCompleted Stage、Durable Work Unit、Video Identityの確定manifestと照合してkill直前の作業量を回復する。resource samplerの未確定sampleを捏造せず、attempt完了後に削除する。
_Avoid_: Acceptance Record, pipeline checkpoint, raw progress log, resource sample database

**Legacy Cache**:
旧screenshot selectorが作成した認識可能なprocessing cache entry、旧processing cache内の`video-identities/`、またはmanifestから現行より古いversioned Candidate Annotation Stage Contractだと識別できるCompleted Stage。cache lock取得後に自動削除し、変換・保持・互換利用は行わない。現行の独立Video Identity cache、現行contractの設定違いによるStage Fingerprint不一致、認識できない`videos/`・`video-sets/`・`work-units/` entryは含まない。
_Avoid_: old model store, fingerprint mismatch alone, user output, unknown directory

**Stage Fingerprint**:
Processing Stage の成果物に影響する上流成果物と、そのStage固有の設定・versionだけから導出される識別子。
_Avoid_: global config hash, Video Fingerprint, unrelated downstream setting

**Effective Configuration**:
明示CLI、明示TOML、公開環境変数、組み込み既定値の順で項目ごとに解決し、型・範囲・相互制約を検証した1回の実行設定。設定sourceはprovenanceに残すが、無関係な全項目を一つのStage Fingerprintへ混ぜない。
_Avoid_: raw TOML, environment dump, global config hash, CLI defaults applied before precedence

**Model Runtime**:
Effective Configurationの3 model roleをdistinctなstore/nameごとに一度だけlocal解決・同期し、全共有roleの完全性とcapabilityを検証してResolved Modelをrun単位でfreezeするruntime境界。promptの意味評価や選定を所有せず、model storeのpath、token、external error detailを境界外へ出さない。
_Avoid_: Vision Runtime, Speech Runtime, model store, inference client, model fallback

**Resolved Model Identity**:
configured model名から実行時に解決し、完全性とload能力を検証して1 run内でfreezeする、Ollamaの完全manifest digestまたはHugging Faceの完全commit SHA。model依存Stageのfingerprintとprovenanceへ保持し、TOMLへ手入力するhashとはしない。
_Avoid_: model tag, configured model name, truncated report value, expected digest

**Resolved Model**:
一つのmodel roleについて、設定名、canonical名、完全でload可能な更新前identity、Model Update Status、freeze済み実行identity、runtime identity、非公開のlocal artifact locationを分離して保持するrun内結果。model依存Stageへはrole固有の設定名、実行identity、runtime identityだけを渡し、locationや更新診断をfingerprintへ含めない。
_Avoid_: Resolved Model Identity, configured model, model store entry, global model bundle

**Model Update Status**:
一つのdistinct modelにModel Upgrade Policyを適用したrun別診断。`not_requested`、`unchanged`、`updated`、`bootstrapped`、`unavailable`のいずれかで、共有tagを使うroleへ同じ結果を割り当てる。実行identityと分離してprovenanceへ残し、それ自体ではStage Fingerprintを変えない。
_Avoid_: Resolved Model Identity, model version, progress state, cache invalidation reason

**Model Runtime Identity**:
model store adapterが実行時に検証したstore kindとclientまたはserver versionのprivacy-safeなcanonical identity。Resolved Model Identityとは分離し、関係するmodel依存Stageのfingerprintとprovenanceへ保持する。path、host、token、自由形式のexternal version detailは含めない。
_Avoid_: Resolved Model Identity, model store path, endpoint, credential, raw version response

**Speech Runtime**:
ModelRuntimeからrun単位で渡された、解決済みでload可能なSTT model artifactとfreeze済みResolved Model Identityを使い、設定されたdevice、compute type、beamなどのprofileでspeech-to-textとword timestampを実行するruntime境界。Context Cueを直接返さず、backend非依存のSpeech Recognition Resultを返す。faster-whisperは既定adapterだが、固定revisionの解決、download、更新確認、run中のmodel freezeは所有しない。
_Avoid_: ModelRuntime, audio extraction, stream selection, Context Cue policy, self-downloading STT backend

**Speech Runtime Identity**:
Speech Runtime adapterとSTT backend、CTranslate2、必要なGPU runtimeのversion・capabilityから実行時に導出するcanonical identity。Resolved Model Identity、device・compute typeなどのoperation設定、host名、GPU serial、model pathとは分離し、STTを実行したContext Collection Stageのfingerprintとprovenanceへ保持する。
_Avoid_: Resolved Model Identity, configured device profile, host identity, model store path

**Speech Recognition Result**:
Speech Runtimeが返す、word本文、chunk内の開始・終了PCM sample位置、source segmentとの対応、未校正のbackend diagnosticsからなるinfra-level結果。faster-whisper固有型とbinary float秒を境界の外へ漏らさず、Context Collection Stageがword grouping、global Video Time変換、chunk overlap所有権、reliability、Context Cue IDを決める。
_Avoid_: Context Cue, backend object, calibrated confidence, global Video Time

**Rejected Speech Diagnostic**:
word grouping後に低reliabilityとしてContext Cueへ採用しなかったSTT文字列、正確な時刻範囲、未校正backend値を保持する非公開のContext Collection Stage artifact。processing cache内だけに保存し、画像評価、progress、error、Human Selection Report、machine-readable reportへ渡さず、`--reset-cache`で削除する。
_Avoid_: Context Cue, public report evidence, progress detail, permanent transcript archive

**Media Runtime Identity**:
system FFmpegとffprobeのversion、および正規化したbuild情報と検証済みcapability一覧から実行時に導出する完全SHA-256の組。raw build文字列やTOMLへ手入力するhashではなく、同じversionでもbuildまたはcapabilityが変われば別identityになる。
_Avoid_: version-only identity, raw version output, configured runtime hash

**Model Upgrade Policy**:
全model roleへ適用する`auto_upgrade`設定とbootstrap規則。既定では処理前に更新を試み、更新不能でも完全でload可能なlocal modelがあればwarning付きで使い、別modelへのfallbackやpartial downloadの利用は行わない。実際のcache互換性は設定値でなくResolved Model Identityが決める。
_Avoid_: model fallback, cache reset, model identity, notification-only update check

**Completed Stage**:
成果物と完了manifestがatomicに確定し、再利用できる Processing Stage。完了manifestのない部分成果物は含まない。
_Avoid_: partial cache, in-progress stage, progress checkpoint

**Durable Work Unit**:
長時間のProcessing Stageを構成する最小の再計算可能単位。engine version、stable key、semantic inputからfingerprintを作り、artifactとmanifestの全fileをfsyncした後にatomic renameで確定する。Video Identity 1本、Video Scan partition、Refinement Window Group、Embedded Subtitle stream、PCM sample range、Speech Recognition chunk、Selected Image WebPが該当し、親Completed Stageが未確定または破損しても健全な兄弟unitを再利用する。integrityだけでなくdomain schema・件数・参照も復元時に検証し、不正なfingerprintだけを修復する。
_Avoid_: progress sample, arbitrary loop iteration, partial Stage, mutable scratch file

**Resume Output Invariance**:
同じsemantic inputから中断後に再開したrunが、中断なしのrunと同じ選択Candidate ID、選択順、公開WebP bytes、canonical reportの意味内容を返す契約。attempt時刻、経過時間、resource sample、cache hit/recompute件数などの運用診断は含めない。初回runとresume runで同じ固定partitionと安定集約順を使い、Video ScanとRefinement Window Groupのworker数・開始順・完了順をsemantic identityへ混ぜない。atomic rename済みの完成Canonical Outputは自己検証とsemantic digest一致後にbyte変更なしで再利用する。
_Avoid_: bit-identical operational telemetry, cache performance equality, unvalidated output reuse

**Recognized Partial Stage**:
Completed StageまたはDurable Work Unitとして確定する前に中断・失敗したことをcache構造から安全に識別できる専用temporary成果物。再利用せず、Input Lock取得後にその対象だけを削除して再計算する。確定済み兄弟Work Unitは削除しない。
_Avoid_: Completed Stage, Durable Work Unit, Legacy Cache, unknown directory

**Video Stage**:
一つのVideo Identityだけを対象とし、Video Setの構成やVideo Orderから独立して再利用できる Processing Stage。同一動画内で完結する時間構造、候補密度、frame refinement、Neutral Image Analysis、Context Cue収集を所有し、scan partition、Refinement Window Group、Speech Recognition chunkをDurable Work Unitとして確定する。複数Videoでは独立したscanをbounded workerで並列実行し、Video Order上の対象scanが確定した時点で後続Videoのscanを続けながら、そのVideoのcandidate extractionとcontext collectionを順序どおり進める。一つのVideo Source内では互いに独立したRefinement Window Groupだけをlogical CPU容量とsafe capでbounded並列実行し、結果をPTS range順へ戻す。中断時は未開始scanとRefinement Window Groupを取り消し、実行中の最小atomic unitを完了または失敗させた後に新しい兄弟unitを開始しない。resultとprogressはVideo Order順に確定し、Video Order、worker数、taskの開始順・完了順はfingerprintへ含めない。
_Avoid_: Video Set Stage, cross-video selection, whole-run stage

**Video Scan Stage**:
一つのVideo Sourceをexact stream timingから15分の固定PTS partitionへ分け、各partitionを一度decodeしてDurable Work Unitとして確定し、heartbeat proxy、scene signal metadata、exact timeline、scan metricを安定順にCompleted Stageへ集約するVideo Stage。streamの`duration_ts`がない場合だけ、ffprobeのcontainer durationを有理数としてstream tickへ切り上げ、partition開始点の個数を決めるhintにする。Video Durationやframe時刻には使わない。最後のpartitionは開始PTSからEOFまでを対象にする。FingerprintにはVideo Fingerprint、選択video stream、partition duration hint、Media Runtime Identity、decode backend、heartbeat/scene設定、partition・proxy・scan・timeline algorithm versionだけを含める。metricにはexact duration、wall/CPU時間、処理速度、decode backend、partition数、heartbeat件数/bytes/gap、scene signal件数、Timeline Segment数を残すがfingerprintへ含めない。後続の密度、refinement、最大Frame Candidate数、Neutral Image Analysisだけが変わった場合も再利用できる。
_Avoid_: Frame Refinement, Candidate Annotation, Video Set Stage

**Primary Video Stream**:
Video SourceからVideo Scan Stageが選ぶ一つの表示映像stream。`attached_pic`など静止coverを除外し、default dispositionを優先して、同順位は最小stream indexで決める。stream選択をpublic configにせず、選択結果のindex、codec、time base、寸法をscan fingerprintへ残す。
_Avoid_: audio stream, cover art, user-selected stream index

**Heartbeat Proxy**:
Video Scan Stageがnative heartbeatごとに永続化する、長辺960px、FFmpeg MJPEG `q:v=3`、metadataなしのcache画像。Scan Proxy Analysisでは1件ずつRGB decode・測定して解放し、全proxyのdecoded RGBを同時保持しない。scene signal用の一時320px画像、Frame Candidate Proxy、公開画像とは区別し、pathをidentityにしない。
_Avoid_: scene signal image, Frame Candidate, selected output

**Frame Candidate Extraction Stage**:
Video Scan Stageを上流にして、Candidate Moment Density、Frame Refinement、Neutral Image AnalysisをCompleted Stageとして確定するVideo Stage。merge済みRefinement Window Groupごとにrange decode、解析、proxyをDurable Work Unitとしてatomicに確定する。独立GroupはVideo Scanと共有するlogical CPU、available memory、最大4のresource policyでbounded並列実行する。Refinement実行中は選択worker分のCPUを予約し、後続scanのadmissionも残り容量へ制限する。memory見積もりには最低240fpsと、Video Scanで全native frameから実測した最小PTS差・同一PTS最大frame数による上限の大きい方を使う。旧cacheなどで実測hintを取得不能な場合はcacheを失効させず逐次実行する。完了順にかかわらずPTS順に親Stageへ集約する。FingerprintにはVideo Fingerprint、上流Stage Fingerprint、density、refinement半径、最大Frame Candidate数、Neutral Analysis/reject/dedupe/ID/proxyのalgorithm versionだけを含める。metricにはwall/CPU時間、density上限/実Moment数、refinement frame数、reason別reject、dedupe、0-frame Moment、Frame Candidate件数/bytesを残すがfingerprintへ含めない。worker数・resource値・frame timing hint・完了順、heartbeat/scene設定やdecode結果を独自に作り直さない。
_Avoid_: full video scan, Context Cue extraction, final selection

**Context Collection Stage**:
Frame Candidate Extraction Stageの後に実行し、一つのVideo SourceからContext CueをCompleted Stageとして確定する3番目のVideo Stage。Video Scan Stageのexact timelineとVideo Durationだけを時間基準として使い、Frame Candidate、Candidate Moment、候補密度、refinementの成果物や設定には依存しない。STTを使う場合はoverlapを含むPCM chunkごとのSpeech Recognition ResultをDurable Work Unitとして確定し、sample順に集約する。FingerprintにはVideo Fingerprint、exact timeline digestとcontract version、選択stream metadata、stream選択・抽出policyと関連設定、Media Runtime Identity、Cue生成policy versionを含め、STTを実行した場合だけSpeech Runtime Identity、Resolved Model Identity、STT・chunk・VAD設定を加える。model更新時刻と`auto_upgrade`は実行identityが同じなら含めない。Context Cueは後続のVideo Set Stageで集約され、Candidate Momentを生成せずframeの適格性も変更しない。
_Avoid_: Video Set context collection, candidate-generating subtitle stage, candidate-dependent context cache

**Collected Context**:
Video Set Stageへ集約されたContext Cueと、STTが実際に実行された場合だけ存在するSpeech Runtime Identityの組。STT未実行と、STTを実行したがCueが0件だった結果を区別し、Context fingerprintへSTT model・runtime・profileを条件付きで加える判断を運ぶ。
_Avoid_: Context Cue, Context Stage Result, unconditional STT dependency, raw transcript

**Context Source Outcome**:
Context Collection Stageが実際に選択・試行したsource kindごとに残すstatusと安定reason code。usable cueがある`available`、track不在の`absent`、選択・decodeされたsourceからusable eventが得られない`no_context`、発話のない`no_speech`、文字列はあるが全件不採用の`low_reliability`を正常結果として区別する。non-forced subtitleの`no_context / no_subtitle_events`ではaudio STTへfallbackしない。ambiguous・unsupported・decode・STT・timestamp・partial chunk failureはfatalであり、一方のsourceだけ成功してもCompleted Stageをpublishしない。
_Avoid_: empty-context fallback, partial success, no-speech reliability failure, free-text-only status

**Video Set Stage**:
順序付きのVideo Setと各Video Stageの成果物を入力にして、Scene Catalog、Candidate Annotation、動画横断の比較と多様性、最終選定を所有する Processing Stage。各Video Sourceからの最低採用数は持たない。
_Avoid_: Video Stage, per-video processing, per-video selection quota

**Video Time**:
一つのVideo Sourceのpresentation timeline上の正確な位置。source PTSとtime baseから最初の表示可能frameを0として導出され、float秒やframe indexとは区別する。
_Avoid_: frame index, float timestamp, wall-clock time

**Video Duration**:
一つのVideo Sourceの有効なpresentation timelineが持つ、0から終端までの正確で正の時間長。終端は最終表示frameの`PTS + duration_ts`を優先し、取得できない場合だけvideo streamの`start_pts + duration_ts`を使って、最初の表示frame PTSからの有理数として導出する。frame間隔、平均fps、containerのfloat秒からは推測せず、正確な終端を得られなければfail-fastする。
_Avoid_: container float duration, inferred frame duration, last frame index, wall-clock duration

**Timeline Segment**:
一つのVideo Sourceのtimeline全体をgapや重複なく覆う、順序付きの半開区間。境界は0、scene signalの正確なVideo Time、Video Durationから作り、scene signal時刻は後側segmentの開始点とする。heartbeatはsegmentを分割しない。各Candidate Momentはanchor時刻によって必ず一つのTimeline Segmentに属する。
_Avoid_: overlapping window, scene, refinement window

**Timeline Segment ID**:
algorithm名、Video Fingerprint、開始・終了Video Timeの既約分数をcanonical JSONにしてSHA-256化した、`seg_`と64桁digestからなる安定識別子。pathやVideo Orderを含めず、表示時だけ短縮できる。
_Avoid_: segment index, display ID, source path

**Candidate Moment**:
一つのVideo Source内で、ブログに有用なframeがanchor Video Timeの周辺に存在すると判断された時間上の候補。複数の検出根拠をまとめ、refinement後に有効なFrame Candidateがない状態も保持する。
_Avoid_: extracted image, Frame Candidate, scene event

**Candidate Moment ID**:
algorithm名、Video Fingerprint、anchor Video Timeの既約分数をcanonical JSONにしてSHA-256化した、`mom_`と64桁digestからなる安定識別子。path、Video Order、検出根拠を含めず、表示時だけ短縮できる。
_Avoid_: candidate index, evidence hash, display ID

**Context Cue**:
一つのVideo SourceのVideo Time区間に対応付けられた、内蔵text subtitleまたは音声の文字起こしから得る文脈テキスト。視覚的なCandidate Momentへの加点根拠に限り、単独ではCandidate Momentを生成せずframeの採否も決めない。
_Avoid_: external subtitle, raw ASR segment, independent candidate, prompt text

**Context Language**:
Context Cue抽出の対象言語。設定されたBCP 47相当tagとstream metadataは同じprimary languageとして扱い、stream選択とSpeech Runtimeへ一貫して適用する。
_Avoid_: raw backend language, stream-only language policy

**Context Cue ID**:
algorithm名、Video Fingerprint、source kind、stream index、正確な開始・終了Video Time、保存textのSHA-256をcanonical JSON化して導出する、`cue_`と64桁digestからなる安定識別子。source path、Video Order、model名、runtime identity、平文textを含めない。
_Avoid_: sequential cue number, plaintext-derived display ID, model-specific cue ID

**Context Cue Equivalence Group**:
forced embedded subtitleとSTTを併用したとき、同じ一回の発話を表すContext Cueを関連付ける非推移的なpair。同じ正規化本文と重なるVideo Timeを持つ各source kind最大1件で構成し、両方のCueとprovenanceをcacheに保持しながらCandidate Annotationへはembedded subtitleだけを渡す。
_Avoid_: cue deletion, fuzzy semantic deduplication, duplicate annotation input, source provenance loss

**Report Context Evidence**:
Context Cueが選定へどう関係したかを公開成果物で追跡する、Cue ID、source kind、正確なVideo Time範囲、reliability、Context Cue Relevanceの組。Context Cue本文は処理cacheだけに保持し、Human Selection Report、machine-readable report、採用理由、要約では引用しない。
_Avoid_: subtitle quotation, ASR transcript, raw Context Cue text, model reasoning trace

**Candidate Moment Density**:
Video Durationに比例して、一つのVideo Sourceが保持できるCandidate Moment数を定める上限率。`60秒 / density_per_minute`幅の半開区間ごとに最大1件を残し、既定値は毎分2件、つまり30秒ごとに最大1件とする。この密度区間はTimeline Segmentや最終選定の時間quotaではない。heartbeat proxyとscene signalから得たanchorをScan Proxy Analysisによってrefinement前に絞り、同点はscene signalの有無、区間中央への近さ、早いVideo Timeの順で解消する。scene signal自体は画質への加点や予約枠にしない。適格なanchorがない区間は0件とし、refinement後に有効なFrame CandidateがなかったCandidate Momentも診断対象として保持する。
_Avoid_: fixed per-video count, per-video selection quota, requested-output multiplier

**Scan Proxy Analysis**:
Candidate Momentの密度選抜だけに使う一時的な中立画質評価。heartbeat anchorは自身のproxy、scene signal anchorは一時320px画像とrefinement範囲内のheartbeat proxyにある有効画像のうち最高画質を使う。scene近傍のheartbeat品質はtimeline順の単調windowで参照し、sceneごとに全heartbeatを再走査しない。これにより短い画面と、白飛びなど無効なscene signal瞬間の前後を拾う。永続的なFrame Candidate評価であるNeutral Image Analysisとは区別する。
_Avoid_: Neutral Image Analysis, Candidate Annotation, scene importance

**Frame Candidate**:
Candidate Moment周辺のrefinementで有効と判断された、一つのVideo Source上の正確なsource frame。同じframeを複数のCandidate Momentが参照でき、proxy画像や出力画像とは区別する。
_Avoid_: Candidate Moment, cached proxy, output image

**Frame Candidate Proxy**:
Candidate Annotationとcache再利用のためにFrame Candidateごとに永続化する、長辺960px、FFmpeg MJPEG `q:v=3`、metadataなしの画像。公開時には#187が同じexact PTSから元解像度frameを再抽出してWebP quality 95を作る。
_Avoid_: Heartbeat Proxy, original-resolution frame, selected output

**Frame Refinement**:
Candidate Momentのanchor前後にあるnative frameを対象に、Content Reject Reason判定、Source-Local Frame Deduplication、最大Frame Candidate数への選抜を行うVideo Stage処理。重なるrefinement windowは一つのRefinement Window Groupとして扱い、最初に最もQuality Scoreが高いframeを選び、残りは選択済みframeとの最小時間距離、最小視覚距離、Quality Score、anchorへの近さ、早いVideo Timeの順で最大件数まで選ぶ。高Qualityの一時的なeffectが一時点へ集中してもfallback候補を同じ瞬間だけで埋めず、Refinement Window内の有効frameへ時間的に分散させる。Candidate Momentから参照する最終順序はVideo Time順とする。
_Avoid_: fixed-fps conversion, Candidate Annotation, Representative Frame selection

**Refinement Window Group**:
一つのVideo Source内で互いに重なるCandidate Momentのrefinement windowから作る最大の連続時間範囲。Neutral Image Analysisの相対分布と前後関係はこの範囲内だけで共有し、離れた範囲のsampleを隣接frameとして扱わない。離れたGroupとはsemantic stateを共有せず、一つのDurable Work Unitとして独立して並列実行できる。
_Avoid_: whole-video refinement, Timeline Segment, density window, arbitrary frame batch

**Source-Local Frame Deduplication**:
一つのCandidate Momentのrefinement内で、知覚的に同じnative frameを一つへまとめるVideo Stageの処理。Quality Score順に64×36 grayscale署名を比較し、すでに残したframeとの平均絶対画素差が2.0以下なら除外する。閾値は設定値でなくversioned algorithm contractとする。同一PTSのFrame CandidateはVideo Source内で一つだけ作って複数Momentから共有するが、離れたMoment間の似たframeは削除しない。動画全体・動画横断の視覚的重複抑制とは区別する。
_Avoid_: cross-video diversity, global near-duplicate removal, exact-frame duplication

**Frame Candidate ID**:
algorithm名、Video Fingerprint、frame自身のVideo Timeの既約分数をcanonical JSONにしてversion付きSHA-256 derivationで作る、`frm_`と64桁digestからなる安定識別子。出力先や選択順が変わっても同じFrame Candidateを指し、表示時だけ短縮できる。
_Avoid_: output filename, selection index, source path

**Primary Representative Frame**:
Selection Shortlist内の一つのCandidate Momentが参照する1から3件のFrame Candidateから、Candidate Annotationより前にNeutral Image AnalysisのQuality Score、Frame Candidate IDの順で最初に一つへ確定したframe。戦闘を示すが説明価値を持たない場合に限り、同じCandidate Moment内のCombat Representative Fallback対象になり得る。
_Avoid_: Representative Frame, model-selected output, selected output

**Representative Frame**:
Candidate Momentを最終的に代表する一つのFrame Candidate。通常はPrimary Representative Frameと一致し、Combat Representative Fallbackが成立した場合だけ同じCandidate Moment内の代替frameになるが、最終採用を意味しない。
_Avoid_: Primary Representative Frame, selected output, Frame Refinement

**Combat Representative Fallback**:
戦闘を示すPrimary Representative Frameが説明価値を持たないCandidate Momentで、同じMoment内の別Frame Candidateを独立評価し、戦闘を示して説明価値を持つ結果だけをRepresentative Frame候補として扱う境界。非戦闘カテゴリや別Candidate Momentのframeを補充せず、通常戦闘coverageのために不適格な画像を採用する処理とも区別する。
_Avoid_: multi-frame annotation, cross-moment substitution, ordinary-combat quota

**Candidate Frame Observation**:
Candidate Annotationの主Ollama推論が、一つのFrame Candidateだけを対象に返すstrict enum中心の意味観測。Scene Slug、Scene Catalog Match、画面内容、画面全体の主用途を表すInterface Kind、会話eventの大きな人物立ち絵・胸像の有無、黒帯・HUDのない固定camera・人物配置から分かるCinematic Event Presentationの有無、画面内に実在する台詞文字の有無とDialogue Text Presentation、具体的な動作・判別可能な人物または敵の有無、Combat Encounter KindとCombat Encounter Basis、Combat Subject Evidence、player本体・攻撃相手本体それぞれの`clear`・`partial`・`absent`、一時的な光・爆発・煙だけが主内容か、Explanation Value、Screen Text Kind、主対象の視認性、一時的な遮蔽、Spoiler Riskとevidenceを持つ。Dialogue Text Presentationは`none`、`dialogue_box`、`speech_bubble`、`subtitle_overlay`、`other`のいずれかで、画面内台詞文字の有無と必ず一致する。音声やContext Cueの会話文を画面内台詞にしない。手紙・手記・日誌・記録を読む画面はInterface Kind `document`として観測する。戦闘HUDだけではInterface Kindを`other_interface`にせず、人物portrait、空の台詞欄、説明文、目的表示、menu項目を台詞として扱わない。会話eventの大きな人物立ち絵・胸像と、画面隅の小さな常設HUD portraitを区別する。通常の戦闘・探索HUDをCinematic Event Presentationにしない。Portrait、HUD、文字、影、発光、移動軌跡を人物・player・攻撃相手の本体として数えない。上下両端の太い暗色帯は画素から決定的に補助検知し、Scene Catalogや主推論がeventを`other`へ誤分類しても掲載境界監査を省略しない。Representative Frame、最終score、適格性、最終採否は決めない。
_Avoid_: Candidate Annotation artifact, Representative Frame, model-selected output

**Representative Frame Evidence**:
独立評価済みのFrame CandidateをRepresentative Frameとして比較するためにCandidate Frame Observationから持ち越す、正規化済みの画面内容、主対象と攻撃相手本体の視認性、一時的遮蔽。自由文、推論順、worker数、最終scoreや採否を含まず、既存artifactに存在しない場合もPrimary Representative Frameのcache再利用を妨げない。
_Avoid_: Candidate Frame Observation, Explanation Value, Neutral Image Analysis

**Cross-Video Diversity**:
Video Set全体で、視覚的に重複するframeや進行上の一部へ偏ったframeを最終選定から抑える性質。特定のVideo Sourceへ採用枠を保証するものではない。
_Avoid_: per-video quota, source-local deduplication, equal allocation

**Video Set Progress**:
Video Orderに従って各Video Durationを連結したVideo Set全体におけるCandidate Momentの進行位置。先行するVideo Durationの合計と現在のVideo TimeをVideo Set全体の長さで正規化した0以上1未満の値で、単独では候補の有用性やSpoiler Riskを表さない。
_Avoid_: per-video position, story importance, spoiler score, selection quota

**Temporal Diversity Penalty**:
要求枚数を`N`としたとき、選択済み候補との最短Video Set Progress距離が`1/N`未満の候補へ最大0.08を線形に適用するsoft penalty。進行位置そのものへの減点や時間帯ごとの採用枠ではない。
_Avoid_: late-video penalty, timeline bucket quota, per-video quota

**Scene Catalog Representative Set**:
Video Set全体のNeutral Image Analysisから、品質、見た目の多様性、頻出patternを表すFrame Candidateを最大24件選んだScene Catalog専用の入力集合。Frame Candidate IDは集合内で一意とし、Selection Shortlistと要求出力枚数から独立するため、要求枚数の変更だけではScene Catalogを変えない。
_Avoid_: Selection Shortlist, selected output, per-video representatives

**Candidate Annotation**:
Selection Shortlist内の一つのCandidate Momentについて、Primary Representative Frame、共有Scene Catalog、近傍Context Cue、Selection Intent、Video Set内の進行位置を入力にし、主Ollama推論でID付きCandidate Frame Observationを評価するVideo Set Stage。Primary Representative Frameが戦闘を示す一方でExplanation Valueが`none`になった場合だけ、同じCandidate Moment内の残り最大2件を一枚ずつ独立評価し、すべての成功した観測から決定的にRepresentative Frameを確定する。推論の失敗は画像の不適格性とみなさずCandidate Momentを未確定のままにし、全frameが`none`の場合は代替frameを強制採用しない。各観測はScene Slug、Scene Catalog Match、画面内容、Interface Kind、会話eventの大きな人物立ち絵・胸像の有無、Cinematic Event Presentationの有無、画面内に実在する台詞文字の有無とDialogue Text Presentation、動作・人物または敵の有無、Combat Encounter KindとCombat Encounter Basis、Combat Subject Evidence、player・攻撃相手それぞれの本体可視性、一時的な光・爆発・煙だけが主内容か、Explanation Value、Screen Text Kind、主対象の視認性、一時的な遮蔽、Spoiler Riskを持つ。音声やContext Cueの会話文は画面内台詞文字とScene Catalog Matchの根拠に使わない。Blog Image Type、公開用要約と理由は観測からlocalに決定し、Scene Catalog MatchがfalseならScene Slugを`other`へ正規化し、具体的なScene Display Nameを要約に使わない。具体的なInterface Kindは曖昧な画面内容分類より優先する一方、動作が見えるframeの`other_interface`は戦闘HUDなどの誤認として上書きに使わない。大きなevent人物立ち絵またはCinematic Event Presentationと画面内台詞文字を持つ会話eventも、汎用的な`other_interface`より優先する。台詞のない`event_dialogue`、動作のないaction分類、台詞も動作もない会話eventの大きな人物立ち絵またはCinematic Event Presentationを静止場面へ補正し、`document`、`tutorial_help`、台詞も動作もない`event_setup`、`save`、人物も敵も判別できない`shop`、攻撃相手本体が`clear`でない戦闘、一時的な光・爆発・煙だけが主内容のframe、主対象不在、深刻な一時遮蔽はExplanation Valueを`none`に正規化する。主推論が掲載可能とした戦闘と、非戦闘とした掲載可能なScene Kind `combat`のgameplayまたは`recurring_gameplay` actionにはCombat Encounter Verificationを追加する。主推論の戦闘種別はそのまま採用せず、専用確認とCombat Visibility Verificationを通ったCombat Encounter KindとCombat Encounter Basisだけを保存する。最初の非戦闘判定は独立再確認し、Scene Kind `combat`で二回とも戦闘を確認できなければExplanation Valueを`none`にする。それ以外の`recurring_gameplay` actionは戦闘有無の結果にかかわらずCombat Visibility Verificationへ進め、二回の敵本体観測がともに敵不在、または掲載可能な戦闘として一致してCombat Visibility Edge Auditも通る場合だけ元のExplanation Valueを保持する。掲載価値ありとした非戦闘の地図または`cinematic` sceneにはPublication Boundary Verificationを追加し、一時的な遷移effectと、台詞も動作もないevent導入の直接観測を適格性境界に優先する。最終score、soft coverage、最終採否は決めない。
Candidate Annotation artifactのSemantic Annotation SummaryにはScene Display Nameを保持して選定時の意味識別子にし、Context Cue逐語一致による伏字時も検証済みScene Slugを非公開fallbackとして保持する。公開前のcopyでは最終的なCombat Encounter Kindと直接観測した画面内容からPublication Annotation Summaryを生成して置換し、戦闘種別をgameplay説明へ反映するのは画像内容もgameplayの場合だけとする。画面内容分類にかかわらず、画面内台詞文字だけで人物・event構図・動作のいずれもない会話風frameと、敵が`weak`または`recognizable`でも直接的な戦闘関係が見えないframeはExplanation Valueを`none`にする。
異なるCandidate Momentは各画像のrequestとconversation contextを共有せず設定上限まで並列評価できるが、各Moment内ではPrimary Representative Frameの成功後だけ条件付きfallbackへ進み、完了順やworker数をAnnotation順、Representative Frame、Stage Fingerprintへ混ぜない。
Combat Representative FallbackのRepresentative候補は戦闘を示し、Explanation Valueが`none`ではない観測だけに限定し、説明価値のある非戦闘frameで戦闘Momentを置換しない。
主推論が戦闘としたものの戦闘可視性を通らないframeはExplanation Valueを`none`にし、主推論の未検証な特定種別を残さず`uncertain`と`ambiguous`へ正規化する。主推論が`not_combat`としたframeを専用確認で戦闘とした後に可視性を確認できなかった場合は、確認できない戦闘actionを保存せず`not_combat`と`none`へ戻す。主推論の戦闘誤判定から非戦闘へ訂正された地図または`cinematic` sceneにもPublication Boundary Verificationを適用する。
_Avoid_: Candidate Scoring, Frame Refinement, Neutral Image Analysis, final selection

**Semantic Annotation Summary**:
Candidate Annotation artifactが選定前に保持する非公開の意味識別文。Scene Catalog MatchがtrueならScene Display Nameと直接観測した画面内容ラベルを組み合わせ、`recurring_gameplay`の異なるsceneや有用なaction variantをSemantic Duplicate Groupで区別する。Context Cue逐語一致とprivacy検査は受け、逐語一致時は検証済みScene Slugを非公開fallbackとして保持するが、公開reportへそのまま投影しない。
_Avoid_: Publication Annotation Summary, final selection reason, raw model text

**Publication Annotation Summary**:
最終選定後の公開前安全化で、検証済みCombat Encounter KindとRepresentative Frame Evidenceだけから生成する有限の画像説明。Semantic Annotation Summary、Scene Display Name、推定固有名を入力にせず、「通常戦闘の具体的なプレイ」「通常プレイ画面」「画面内テキストのあるイベント」など、画像と矛盾しない汎用表現へ選定結果のcopyだけを置換する。Title Semanticsは汎用的なRepresentative Frame Evidenceより優先する。戦闘種別を具体的なplay説明へ反映するのはRepresentative Frame Evidenceもgameplayの場合だけとし、event画像の意味を上書きしない。最終selectorの入力には戻さない。
_Avoid_: semantic deduplication discriminator, scene identity, inferred proper name

**Combat Encounter Kind**:
Representative Frameに戦闘が見えるかと、その戦闘が通常戦闘か主要戦闘かを画像内根拠だけで表す`not_combat`、`ordinary`、`major`、`uncertain`の分類。`not_combat`は戦闘が見えず、残る3値は戦闘が見える。`ordinary`は一般敵の群れ・編成や通常遭遇の提示など、通常戦闘を示す積極的なCombat Encounter Basisがある戦闘だけを表す。`major`はboss専用表示、特別な演出や構図、通常敵と明確に異なる相手など主要戦闘の直接根拠がある戦闘を表す。戦闘は見えるがどちらの積極的根拠もない場合、または根拠が競合する場合は`uncertain`とする。敵名やHP・status barが見えることだけでは`ordinary`にも`major`にもせず、物語上のネタバレを表すSpoiler Riskとは独立する。
_Avoid_: combat_action, Spoiler Risk, Scene Kind, boss name inference

**Combat Encounter Basis**:
Combat Encounter Kindを支持する画像内の積極的根拠を表す`none`、`ordinary_opponent_presentation`、`ordinary_encounter_presentation`、`major_opponent_presentation`、`major_encounter_presentation`、`ambiguous`の分類。`not_combat`は`none`、`ordinary`は二つの`ordinary_*`、`major`は二つの`major_*`、`uncertain`は`ambiguous`だけと組み合わせる。一般敵の群れ・編成または通常遭遇だと直接分かる提示を`ordinary_*`とし、主要な相手の外見または専用演出を`major_*`とする。敵名、HP・status bar、主要戦闘の根拠がないことだけでは`ordinary_*`にならない。
_Avoid_: confidence score, enemy name inference, absence of major evidence, Spoiler Evidence

**Candidate Annotation Relationship Repair**:
Candidate Annotationの主推論がschemaには適合するがContext Cue参照またはSpoiler Evidenceの関係だけに違反したとき、分類と他の観測を凍結し、違反した従属fieldだけを一度修復してAnnotation全体を再検証する境界。
_Avoid_: second Candidate Annotation, classification retry, deterministic fallback, silent candidate drop

**Combat Encounter Verification**:
Candidate Annotationの主推論が掲載可能とした戦闘、または掲載可能な非戦闘としたScene Kind `combat`のgameplayまたは`recurring_gameplay` actionのRepresentative Frame一枚だけに対し、音声、Context Cue、前後場面、主推論の説明文を与えず実行する条件付きOllama推論。敵・boss固有の名前とHP・status bar、または戦うplayer・相手本体から戦闘の有無を確認し、Combat Encounter KindとCombat Encounter Basisを必ず整合する組として返す。主推論の戦闘種別はそのまま採用しない。敵名やHP・status barだけでは通常戦闘にも主要戦闘にもせず、両方の積極的根拠がなければ`uncertain`とする。敵本体が画面外・画面端・エフェクト内でも敵status UIがあれば戦闘とし、Combat Visibility Verificationへ進める。player自身の通常HUD、portrait、操作button、minimapだけでは戦闘にしない。最初の確認が非戦闘なら、先の回答を推測しない別promptで同じ画素を独立再確認する。Scene Kind `combat`で二回とも非戦闘なら、戦闘sceneとして説明できないframeとしてExplanation Valueを`none`にする。それ以外の`recurring_gameplay` actionではCombat Visibility Verificationへ進め、敵本体の直接観測との不一致を検出する。
_Avoid_: combat visibility, contextual combat classification, final selection

**Combat Visibility Verification**:
Combat Encounter Verificationで戦闘と確認されたframe、またはScene Kind `combat`以外で同確認の対象になった`recurring_gameplay` actionのRepresentative Frame一枚だけに対し、音声、Context Cue、前後場面、主推論の説明文を与えず実行する条件付きOllama推論。エフェクトの画面占有率、最大の前景要素、player本体と攻撃相手本体の可視性、攻撃相手本体が画面内へ収まる構図、Opponent Presentation、Combat Interaction Visibility、エフェクトの本体への重なり、エフェクトだけのframeかをstrict enumで観測する。見下ろし型・遠景・非人型の小型敵はgameplay spriteの輪郭全体で可視性と構図を判定し、複数敵では最も明瞭かつ完全に収まる一体を基準にする。別の敵の端欠け、player付近のeffect、damage number、色づいた地面は基準にした敵本体の遮蔽として扱わない。戦闘と確認済みの場合、player本体が`absent`、攻撃相手本体が`partial`・`absent`、構図が`edge_cropped`・`occluded`・`absent`、Opponent Presentationが`weak`・`absent`、または`recognizable`でもCombat Interaction Visibilityが`direct`でない場合はExplanation Valueを`none`に下げる。最初の確認が掲載可能なら、先の回答を推測しない別promptで同じ画素を独立再確認し、二回とも同じ掲載境界を満たす場合だけCombat Visibility Edge Auditへ進む。Scene Kind `combat`以外で戦闘有無が二回とも否定された場合は可視性を必ず二回確認し、両方で敵本体が不在かつエフェクトだけではない場合、または両方で掲載可能な戦闘として一致してCombat Visibility Edge Auditも通る場合だけ元のExplanation Valueを保持する。Combat Encounter Verificationの戦闘種別と根拠は、player本体と攻撃相手本体の可視性および掲載可読性が一致しCombat Visibility Edge Auditも通るまで確定しない。Scene Slug、画面内容、Spoiler Riskは変更しない。公開用要約は検証済みのCombat Encounter Kindから有限表現へ正規化する。
主推論が戦闘としたframeで戦闘としての可視性または外周strip監査を通らない場合は、主推論の`ordinary`または`major`を残さず`uncertain`と`ambiguous`へ正規化する。主推論が`not_combat`としたframeで専用確認後の可視性を確認できない場合は、`not_combat`と`none`へ戻す。
攻撃相手本体の可視性、構図、Opponent Presentationは不在状態を一致させ、相手不在でCombat Interaction Visibilityが`none`以外、またはplayer不在で`direct`とする応答をschema相関違反として再試行する。これにより非戦闘交差確認の「二回とも敵不在」経路へ矛盾した戦闘signalを流さない。
_Avoid_: second Candidate Annotation, contextual combat classification, final selection

**Combat Visibility Correlation Repair**:
Combat Visibility Verificationの再観測後もplayer本体不在と直接戦闘が同時に返された場合に、独立観測を保持して従属するCombat Interaction Visibilityだけを安全側へ整合させ、Candidate Annotation全体を再検証する境界。掲載価値を上げず、候補をdropしない。
_Avoid_: schema relaxation, publication promotion, silent candidate drop

**Opponent Presentation**:
Combat Visibility Verificationが攻撃相手本体のブログ画像内での見せ方を表す`prominent`、`recognizable`、`weak`、`absent`の直接観測。主要な被写体として一目で識別できる相手を`prominent`、小さくても輪郭・色・姿勢から単体で識別できる相手を`recognizable`、HUDや名前を手掛かりに探さなければ識別しにくい相手を`weak`とする。敵の物語上の重要度やCombat Encounter Kindは含めない。
_Avoid_: boss importance, screen coverage threshold, enemy name, model confidence

**Combat Interaction Visibility**:
Combat Visibility Verificationがplayerと基準にした攻撃相手の直接的な戦闘関係を表す`direct`、`indirect`、`none`の直接観測。攻撃姿勢、弾道、接触、命中effectなどが両者を視覚的に結び付ける場合だけ`direct`とし、敵名、HP bar、戦闘HUDだけでは`direct`にしない。
_Avoid_: combat existence, Encounter Basis, audio cue, inferred action

**Combat Visibility Edge Audit**:
二回のCombat Visibility Verificationがともに掲載可能としたRepresentative Frame一枚だけに対し、元画像と、上端・下端・左端・右端それぞれの外周30%をlocalで切り出した4枚を一度に渡して実行する最終の条件付きOllama推論。元画像で最も明瞭かつ完全に収まる攻撃相手一体を選び、その同じ本体だけを各stripで追跡して、主要な輪郭が元画像の実際の外端へ到達するかを専用strict schemaで直接観測する。別の攻撃相手、敵名、HP bar、光、攻撃effect、影、背景、診断用の内側crop境界を選んだ敵本体の外端到達に数えない。どれか一辺で選んだ敵本体の存在と実際の外端への到達がともに確認された場合はExplanation Valueを`none`に下げる。二回の可視性確認と外周strip監査のすべてを通った場合だけ掲載価値を保持する。Scene Slug、画面内容、Spoiler Risk、説明文は変更しない。
_Avoid_: generic effect threshold, second Candidate Annotation, final selection

**Publication Boundary Verification**:
Candidate Annotationの主推論が掲載可能とした非戦闘の地図、Scene Selection Roleが`cinematic`の場面、または画素から上下両端の太い暗色帯が検知されたRepresentative Frame一枚だけに対し、音声、Context Cue、前後場面、主推論の説明文を与えず実行する条件付きOllama推論。一時的な遷移effectの有無・種類・画面占有率、上下の黒帯、event用の人物配置、画面内台詞文字、人物の具体的な動作、主内容の可読性をstrict enumで観測する。画素検知した黒帯はmodelの見落としより優先する。一時的な遷移effectがある場合、または上下の黒帯とevent用の人物配置があり画面内台詞も具体的な動作もない場合はExplanation Valueを`none`に下げるが、Scene Slug、画面内容、Spoiler Risk、説明文を変更しない。地図の雲、cursor、選択marker、常設UIは遷移effectにしない。
主推論が戦闘と誤判定した対象も、Combat Encounter VerificationとCombat Visibility Verificationで非戦闘へ訂正された後に同じ確認を実行する。
_Avoid_: second Candidate Annotation, contextual event classification, final selection

**Scene**:
ブログ用の画像選択で使う、画像内容を表すカテゴリ。ゲームジャンルや入力画像群に応じて決まる。
_Avoid_: play/event density bucket, fixed category

**Scene Slug**:
scene を表す小文字英数字の安定名。出力ファイル名、レポート、カテゴリ集計に使われる。
_Avoid_: localized category name

**Scene Kind**:
Scene Catalogが各sceneへ付ける機械判定可能な内容種別。値は`combat`、`exploration`、`interface`、`event`、`other`で、自由なScene Slugや表示名から戦闘などの意味を推測しないために使う。`other` sceneのScene Kindは必ず`other`とする。
_Avoid_: Scene Slug naming convention, Scene Selection Role, per-frame content kind

**Scene-numbered Output Name**:
既存の画像入力selectorが選択画像に付ける、scene slugとscene内連番からなる出力ファイル名。Video Set selectorではSelected Image Output Nameに置き換える。
_Avoid_: Video Set output name, stable image identity, original filename

**Selected Image Output Name**:
Video Set selectorが選択画像に付ける、最低4桁へzero-padした全体選択順、scene slug、Frame Candidate IDのdigest部分先頭12文字からなる名前。短縮digestが同一run内で衝突する場合だけ64文字まで延長し、安定identity自体はFrame Candidate IDが担う。
_Avoid_: stable image identity, scene-local index, source filename

**Selected Image Encoding**:
Video Set selectorが選択画像へ固定で使う、元frameの解像度を維持した非可逆WebP quality 95。埋め込みmetadataは除去し、v1では利用者設定による形式やqualityの変更を許可しない。
_Avoid_: PNG default, configurable image format, resized thumbnail, source metadata copy

**Selected Image Checkpoint**:
最終選定済みFrame Candidate一件について、exact source PTSから再抽出した元解像度frameをSelected Image Encodingへ変換し、WebP bytes、寸法、SHA-256をDurable Work Unitとして確定した非公開artifact。final publicationが未確定なら同じbytesを新しいstagingへ安定順でcopyし、完成Canonical Outputが検証できる場合は既存outputを変更しない。
_Avoid_: Frame Candidate Proxy, mutable output image, partial output checkpoint, original RGB cache

**Report Source Path**:
Human Selection Reportとmachine-readable reportが元動画を追跡するために示す、Video Input Folderを基準に`/`区切りへ正規化した相対path。`..`を含めず、Video OrderとVideo Source IDを併記し、Video Identityには使わない。
_Avoid_: absolute path, basename-only path, Video Fingerprint, path-based identity

**Report Video Time**:
Human Selection Reportでは24時間で折り返さない`HH:MM:SS.mmm`をhalf-upで丸めて示し、machine-readable reportでは`source_pts`、`origin_pts`、time base、既約分数の`offset_seconds`を正本として保持するVideo Time表現。frame indexとfloat秒は公開契約に含めない。
_Avoid_: float timestamp, frame index, wrapped clock time, display string as identity

**Report Video Fingerprint**:
machine-readable reportが処理した動画内容を厳密に示す、algorithm名を伴う64桁の動画全体SHA-256。Human Selection Reportでは公開せず、短いVideo Source ID、Video Order、Report Source Pathだけを表示する。
_Avoid_: truncated digest in report.json, report.md fingerprint, file stat, path hash

**Output Folder**:
選択画像を`images/`、Human Selection Reportを`report.md`、machine-readable reportを`report.json`へ書き出す実行ごとの保存先。input folderと同一または相互の配下にできず、新規公開時は未作成または空である必要がある。既存の非空folderは、Canonical Outputのschema、artifact、layout、privacyと今回のsemantic digestがすべて一致する場合だけ終端の再開状態としてbyte変更なしで受理し、異なる完成成果物や利用者fileは削除・上書きしない。
_Avoid_: append destination, blind overwrite target, unvalidated resume folder

**Atomic Output Publication**:
存在しないOutput Folderと同じ親filesystem上の隠しstaging Folderへ全画像、machine-readable report、Human Selection Reportを生成・検証・flushした後、directory renameを1回だけ行う公開境界。既存の空Output Folderは事前検証後に取り除き、fatal errorではOutput Folderを公開しない。rename後にprocessが強制終了してfinal folderだけが残った場合は、自己完結検証とsemantic digest一致を通して完成済みpublicationとして再利用する。Selection Shortfallはwarning付き正常成果物として公開し、atomic rename非対応時は処理前に失敗する。
_Avoid_: file-by-file publication, unchecked completion marker, partially visible output, cross-filesystem staging

**Human Selection Report**:
選択画像と採用理由をgallery-firstで確認でき、末尾のappendixからshortfall、near miss、score内訳、Stage provenanceを追える`report.md`。source videoとVideo Timeは各画像に示すが、動画別の長大なtimelineを主構造にはしない。
_Avoid_: machine-readable report, raw diagnostic dump, full Video Set timeline

**Report Image Embed**:
Human Selection Reportが実際の選択WebPを相対pathでinline表示し、同じ画像へのlinkで原寸を開けるgallery要素。別thumbnailを生成せず、alt textには選択順とScene Display Nameだけを使い、Spoiler Evidenceを含めない。
_Avoid_: link-only gallery, generated thumbnail, absolute image URL, spoiler evidence in alt text

**Canonical Selection Report**:
1回のrunに対して構築・schema検証され、`report.json`へserializeされる唯一のmachine-readable report object。Human Selection Reportはこの検証済みobjectだけから決定的に生成し、cacheやmodelを再参照しない。JSON key順は契約に含めず、画像ID・path・件数・理由のprojection不一致はfatalとする。
_Avoid_: independent Markdown data source, cache-backed rendering, model-backed rendering, JSON key order contract

**Report Selection Explanation**:
選択画像ごとに、Candidate Annotation由来の画像要約とRepresentative Frame理由、reason codeとscore内訳からlocal rendererが作るselector採用理由を出所別に示す説明。machine-readable reportでは`annotation`と`selection`を分離し、単一のmodel生成reason、model confidence、内部推論を採用理由にしない。
_Avoid_: blended reason, model-selected final result, confidence score, reasoning trace

**Report Spoiler Disclosure**:
全選択画像のSpoiler RiskをHuman Selection Reportとmachine-readable reportへ常時示し、`none`以外のCandidate Annotation由来evidence summaryだけをMarkdownの閉じた`details`に置く公開形式。画面内文章やContext Cueを引用せず、selector由来Spoiler Penaltyとは別fieldで扱う。
_Avoid_: expanded spoiler text, dialogue quotation, hidden risk level, blended model evidence and selector penalty

**Report Stage Provenance**:
machine-readable reportで各Processing Stageのstatus、完全なStage Fingerprintと上流fingerprint、cache結果、再計算件数、duration、attempt、validation、token、結果に影響する正規化設定、tool・runtime・model digest、prompt・schema・policy versionを示す再現情報。Human Selection Reportは短縮fingerprintと主要件数・時間だけを要約する。path、環境変数、credential、prompt本文、raw response、stack traceは含めない。
_Avoid_: host path dump, secret-bearing config, raw prompt, raw response, stack trace

**Report Schema Version**:
machine-readable reportの`game-screen-pick/report`契約を表すSemantic Version。初版は`1.0.0`とし、breaking changeはmajor、optional fieldまたはenum値の追加はminor、構造を変えない修正はpatchにする。readerはschema identityから対応majorの履歴schemaを選び、既知の必須構造を検証しながら同じmajorの追加fieldと文字列enum値を保持し、未知majorだけを拒否する。major固有の関係とMarkdown sectionを別majorへ強制せず、producerだけが現行schemaを厳密検証する。過去schemaと既存reportは書き換えない。
_Avoid_: unversioned JSON, Markdown parser contract, automatic report migration, breaking minor change

**Report Near Miss Set**:
全未採用Blog Candidateの理由別件数を集計したうえで、各理由の代表を最低1件含め、残りを反実仮想Marginal Selection Utilityの高い順に選ぶ公開診断集合。`report.json`は`min(未採用総数, 100, max(20, 要求枚数×2))`件、Human Selection Reportはその先頭最大10件までとし、全候補詳細は処理cacheに保持する。
_Avoid_: unbounded rejection ledger, random rejection sample, full cache dump

**Scene Display Name**:
scene を人が読みやすいように表す日本語名。ブログ用の画像選択やレポート表示で使われる。
_Avoid_: filename prefix, report key

**Scene Catalog**:
Scene Catalog Representative Setから作る、一つのVideo Setを横断して共有するscene、Scene Kind、Scene Selection Roleの一覧。3から8個のsceneと分類の逃げ先である`other`で構成され、Videoごとには分割しない。Scene Slug、Scene Display Name、Scene Descriptionは、一部の代表画像だけの場所・固有人物・物語上の結果をscene全体へ断定せず、そのsceneへ後から分類される画像で再利用できる視覚・操作上の役割を表す。Scene Kind `other`のsceneは自由なslugや具体的な表示名が返っても、slugを`other`、表示名を「その他」、説明を汎用の分類逃げ先へ正規化する。Scene Kindは複数sceneで重複できるが、Scene Slugはcatalog内で一意とする。domain再試行でも非`other`のScene Slugだけが重複した場合は、入力順の数値suffixで決定的に一意化する。
_Avoid_: fixed scene list, free-form per-image labels, per-video catalog

**Scene Catalog Match**:
Candidate Frameの画素だけから、選択したScene Catalog entryのScene Display NameとScene Descriptionに含まれる具体的な場所・人物・出来事まで裏付けられること。Scene Kindや会話・戦闘という大分類だけが合う場合、音声・Context Cue・Video Set Progressで補わなければ合わない場合は不一致とする。不一致のCandidate AnnotationはScene Slugを`other`へ正規化する。公開用要約はMatchの真偽にかかわらずScene Display Nameを使わず、検証済みの有限分類だけから生成する。
_Avoid_: Scene Kind match, Context Cue relevance, image quality, selection eligibility

**Scene Description**:
画像がその scene に分類された理由を、ブログ用の画像選択に役立つように短く説明する文章。
_Avoid_: internal reasoning, model trace

**Scene Selection Role**:
scene ごとに、最終選択での扱いを表す役割。値は `ordinary`、`cinematic`、`recurring_gameplay` の3種類で、other scene、その他や不明なroleは通常配分で扱う。
_Avoid_: scene label, manual quota, content reject reason, failure mode

**Scene Hint**:
scene catalog を作るときに、ユーザーがゲームジャンルやブログ画像選択の意図を補足する短い説明。
_Avoid_: fixed scene list, selection rule

**Selection Intent**:
ブログ画像として何を重視して選ぶかを表す実行ごとの意図。scene hint は selection intent を補足する入力であり、変わると scene catalog や画像分類も変わり得る。
_Avoid_: image analysis setting, cache option

**Blog Image Type**:
Representative Frameがブログ内で主に果たす説明上の役割。値は`normal_gameplay`、`event`、`menu`、`title`、`other`で、操作可否ではなくCandidate Frame Observationの画面内容からCandidate Annotation Stageが決定的に導出する。探索や戦闘に短い台詞・HUD表示が重なったものは`normal_gameplay`、会話や演出そのものが主体なら`event`として扱い、最終的なsoft coverageは決定的なVideo Set selectorが扱う。
_Avoid_: Scene, Scene Selection Role, hard quota, final selection

**Blog Image Type Soft Coverage**:
最終選定が通常時に目指すBlog Image Typeの構成。`normal_gameplay` 70%、`event` 25%、`menu` 5%を目安とし、`other`と`title`には予約枠を設けない。候補の有用性や不足に応じて構成比の超過を許すため固定quotaではなく、`title`だけは有用な候補を最大1枚まで選べる。
_Avoid_: hard quota, per-video quota, Cinematic Soft Cap, guaranteed title image

**Blog Image Type Coverage Bonus**:
最大剰余法で丸めたBlog Image Type Soft Coverageの目標枚数へ未達の`normal_gameplay`、`event`、`menu`候補に0.10、まだtitleを選んでいないときの`title`候補に0.05を加えるsoft bonus。目標到達後はbonusを外すだけで、type超過へのpenaltyや`other`の予約枠は設けない。
_Avoid_: hard quota, overflow penalty, guaranteed title image

**Selection Coverage Facet**:
要求枚数10枚以上の条件付き最低coverageに使う、Blog Candidateの画像内根拠に基づく役割。`ordinary_combat`と`event`を持つ。`ordinary_combat`はBlog Image Typeが`normal_gameplay`で、Combat Encounter Kindが`ordinary`かつ対応する`ordinary_*`のCombat Encounter Basisがある場合だけ導出し、`major`、`uncertain`、探索、移動、障害物破壊を含めない。Spoiler Riskは物語上のネタバレ評価であり、このfacetの戦闘種別判定には使わない。`event`はBlog Image Type `event`から導出するが、Screen Text KindまたはRepresentative Frame Evidenceがtitleを示す誤分類候補は含めない。Explanation Valueや最終適格性はこのfacet自体に含めない。
_Avoid_: Blog Image Type, Scene Kind, free-form scene name, final eligibility

**Conditional Coverage Minimum**:
要求枚数が10枚以上で、Explanation Valueと既存の適格性を満たすSelection Coverage Facet候補が存在する場合だけ、`ordinary_combat`と`event`を各最低1枚選ぶ決定的なVideo Set selectorの境界。複数facetが未充足なら、終端similarity ceilingで各facetから1件ずつ選べ、必要な未代表Variant Groupの代表を含めても残り出力枠へ収まる互換組合せを保持し、別facetの実現可能な最低枠を壊す高utility候補を先に選ばない。現在passで選べない候補は後続similarity passまで枠を保持する。候補が同じrecurring gameplayの既選択Variant Groupに属し、別の未代表Groupが選択の前提になる場合は、最低枠を残せる範囲でその前提Groupを先に選ぶ。終端でもSemantic Duplicate Group、Visual Near-Duplicate、title上限、Spoiler Monotonicity Guardなどの制約に反する場合、または有効候補がなければ枠を他候補へ解放する。終端で解放した場合、または緩和ceilingで最後の最低枠を満たした場合は、選択済み画像を保持し、設定されたbase similarity ceilingから残りの通常選定を再開する。要求枚数10枚以上では、既知facetが未発見または未充足の間もSelection Shortlistを拡張し、全Candidate Momentを使い切るまで候補を探索する。残り枚数はBlog Image Type Soft CoverageとMarginal Selection Utilityで動的に配分し、Selection Shortfallを低品質候補で埋めない。
_Avoid_: fixed quota, output count guarantee, per-video minimum, invalid fallback

**Explanation Value**:
Representative FrameとそのCandidate Momentがブログ本文でplayや出来事を説明できる度合い。値は`none`、`low`、`medium`、`high`で、Candidate Annotationが意味評価として付与する。`none`はCandidate Annotation自体の失敗やmodelによる最終採否ではないが、決定的selectorが要求枚数の穴埋めに使わない適格性境界になる。
画面内容分類にかかわらず、画面内台詞文字だけで人物・event構図・動作のない会話風frame、攻撃相手をブログ画像の被写体として識別できない戦闘、または小さな相手を識別できてもplayerとの直接的な戦闘関係が見えないframeは`none`へ正規化する。
_Avoid_: Quality Score, model confidence, final selection score

**Screen Text Kind**:
Representative Frame内で意味を持つ画面内テキストの役割。値は`none`、`dialogue`、`menu`、`title`、`hud`、`other`で、生成された逐語転記は含めない。
_Avoid_: Context Cue, OCR transcript, generated quotation, Blog Image Type

**Context Cue Relevance**:
Candidate Annotationへ渡したContext Cueが、Representative FrameとCandidate Momentの説明をどれだけ補強するかを表す`unavailable`、`none`、`weak`、`strong`の評価。補強に使ったContext Cue IDを伴うが、単独でframeを適格にしない。
_Avoid_: Context Cue reliability, frame acceptance, independent candidate score

**Spoiler Risk**:
Representative FrameとCandidate Momentが物語上の重要情報を明かす可能性を表す`none`、`low`、`medium`、`high`の意味評価。物語上の進行や重要情報を明かさない探索・戦闘は`none`、軽微な進行情報は`low`、固有bossや終盤固有areaなどが意味のある進行情報を明かす場合は`medium`、Major Spoiler Signalの具体的な意味証拠があるものは`high`とし、Candidate Annotationが付与する。敵名、HP・status bar、Combat Encounter Kindだけではriskを決めず、利用者設定に応じた減点は決定的なVideo Set selectorが扱う。
_Avoid_: spoiler sensitivity, spoiler penalty, late-video hard reject

**Major Spoiler Signal**:
エンディング、最終ボスの正体・形態、主要人物の生死、裏切り・犯人・真の正体、物語の中心的な種明かしを画像、画面内テキスト、Context Cueが具体的に示すこと。Video Set内の進行位置だけではMajor Spoiler Signalにならない。
_Avoid_: late-video position, generic battle, ordinary progression detail

**Spoiler Sensitivity**:
Spoiler Riskを最終選定でどれだけ避けるかを表す実行ごとの`low`、`medium`、`high`設定。既定値は`medium`で、値を高くしても候補を除外するhard policyにはしない。
_Avoid_: Spoiler Risk, story progress, hard reject

**Spoiler Penalty**:
Spoiler SensitivityとSpoiler Riskの組み合わせから、0から1の選定utilityに適用する決定的な減点。`low`ではriskが`medium`、`high`のとき0.02、0.05、`medium`では`low`、`medium`、`high`のとき0.01、0.04、0.10、`high`では0.02、0.08、0.18とし、riskが`none`なら常に0とする。
_Avoid_: hard reject, late-video penalty, model confidence

**Spoiler Monotonicity Guard**:
同じBlog Candidate集合でSpoiler Sensitivityを上げたとき、Major Spoiler Signalを持つ選択画像数が増えないようにする最終選定の件数境界。`medium`は`low`、`high`は`medium`の選択件数を上限としてgreedy選定を再実行し、個々の候補をriskだけで常時除外するhard cutoffとは区別する。
_Avoid_: per-candidate spoiler rejection, late-video cutoff, model safety filter

**Quality Score**:
blog candidate がブログ画像としてどれだけ使いやすいかを表す評価値。scene の種類やゲームジャンルの指示ではなく、画像そのものの見やすさを表す。
_Avoid_: scene hint, user-facing mode, selection profile

**Selection Base Utility**:
Blog Candidate単体の有用性を0から1で表す決定的な値。Quality Scoreを70%、Explanation Valueを25%、Context Cue Relevanceを5%として合成し、動画内位置、Blog Image Typeの構成、視覚・時間的多様性、Spoiler Penaltyは含めない。
_Avoid_: final selection score, model confidence, diversity bonus, spoiler-adjusted utility

**Marginal Selection Utility**:
greedyなVideo Set selectorが次の1枚を選ぶたびに再計算する値。Selection Base UtilityからSpoiler PenaltyとTemporal Diversity Penaltyを引き、Blog Image Type Coverage Bonusを加える。Conditional Coverage Minimumが未充足なら対象facet候補の中でこのutilityを比較し、最低枠を満たした後または解放後は全適格候補を比較する。視覚類似度はutilityではなく適格条件として扱い、同点はSpoiler Penalty、Quality Score、選択済み画像との最大視覚類似度、Video Order、Video Time、Frame Candidate IDの順で解消する。
_Avoid_: static candidate score, Ollama output, global optimization result

**Blog Candidate**:
Candidate Annotationが完了し、Representative Frameと最終選定に必要な意味情報を持つCandidate Moment。明らかな暗転、白飛び、単色画面、遷移フレームはVideo Stageで既に除外されている。
_Avoid_: all Candidate Moments, Selection Shortlist, selected output

**Selection Shortlist**:
有効なFrame Candidateを持つCandidate Momentのうち、Neutral Image Analysisによる品質と見た目の多様性から、Candidate Annotationへ進めるものをVideo Set全体でlocalに絞った集合。複数Momentが同じRepresentative Frameを共有する場合は決定済みshortlist順の最初のMomentだけを残し、後続の一意なFrameを持つMomentの探索を続けるため、集合内のFrame Candidate IDは一意になる。
Primary Representative Frameを全Momentについて先に予約し、その後fallback候補をshortlist順で未予約IDだけに制限するため、早いMomentのfallbackが後続Primaryを奪わない。
_Avoid_: all Candidate Moments, annotated Blog Candidate, selected output

**Selection Shortfall**:
有効な未注釈Candidate Momentを決定的なshortlist順で追加し、許可された視覚類似度緩和をすべて適用しても、適格なBlog Candidateが要求枚数に満たない状態。選べた画像とshortfall理由をreportへ出してwarning付きで正常終了し、Ollama Stage Failureとは区別する。
_Avoid_: Candidate Annotation failure, silent omission, fabricated output, invalid-frame fallback

**Selection Rejection Reason**:
未採用Blog Candidateの主因を表すstable enum。`title_limit`、`semantic_duplicate`、`visual_near_duplicate`、`similarity_ceiling`、`spoiler_monotonicity_guard`、`lower_marginal_utility`を持つ。`semantic_duplicate`は同じSemantic Duplicate Groupの代表が既に選択されたことを示し、そのblocking selected IDとGroup判定根拠を伴う。Explanation Valueが`none`の候補はCounterfactual Selection Scoreを保持した`lower_marginal_utility`として説明し、model自由文や例外messageから理由を推測しない。
_Avoid_: free-text rejection, Content Reject Reason, Ollama Stage Failure

**Counterfactual Selection Score**:
未採用Blog Candidateが選定中に持ち得た最も高いMarginal Selection Utilityと、そのBase、coverage、spoiler、temporal、similarity passの内訳。採用を妨げた制約を説明するnear-miss診断であり、制約を無視して実際に選び直した結果ではない。
_Avoid_: final selected score, model confidence, regenerated explanation

**Neutral Image Analysis**:
sceneやSelection Intent、modelに依存せず、Frame Candidateそのものから得られる画質metrics、Quality Score、正規化済みHSV・輝度・edge視覚特徴。各視覚特徴成分は尺度差によって他成分を消失させないよう個別に正規化して等しく組み合わせ、画像の内容分類ではなくblog candidate判定や動画横断のcosine similarity判定の土台にする。明確な無効frameには絶対条件を使い、純白だけでなく主対象を覆う大きな連結白領域、画面全体では小さくても中央の主対象を覆う連結した白い発光、画面の大半を覆う低情報の淡い白もwhiteoutにする。構造が判別できる明るいmenuは明るさだけで除外しない。暗いgameなど入力特性にはRefinement Window Group内の分布を使う。Transition Frameには同一streamとtime baseでdurationどおりに連続するnative frameだけの前後関係を使い、0.25秒以内に画面の一部から大半へ拡大または縮小する淡い明領域もtemporal transitionにする。CLIPやHugging Face model identityをVideo Stageへ持ち込まない。
_Avoid_: scene classification, selection intent, CLIP embedding, model-dependent feature

**Content Reject Reason**:
refinement frameを有効なFrame Candidateにしなかった理由を表す安定enum。`blackout`、`whiteout`、`single_tone`、`blur`、`fade_transition`、`temporal_transition`を持ち、free textだけの除外理由にはしない。
_Avoid_: model classification, selection rejection, free-form reason

**Transition Frame**:
シーン移動や画面切り替えの途中に現れる、ブログ画像として説明価値が低い一時的な画面。
_Avoid_: event scene, cutscene

**Cinematic Scene**:
ゲームの進行操作より演出、会話、イベントの見せ場を主に写した scene。ブログ画像として少量は有用だが、入力全体の代表性を崩さないよう通常 gameplay より控えめに扱う。
_Avoid_: transition frame, hard reject, movie frame

**Cinematic Soft Cap**:
既存の画像入力selectorで、すべてのcinematic sceneの合計選択枚数を通常は少量に抑えつつ、他の有用候補が足りない場合だけ超過を許す上限。Video Set selectorではBlog Image Type Soft Coverageに置き換え、併用しない。
_Avoid_: Video Set selection rule, per-scene cinematic quota, hard reject, exact quota

**Recurring Gameplay Pattern**:
戦闘UI、探索画面、パズル盤面など、ゲーム中に頻繁に表示される通常playの画面構造。同じ構図でも状態や進行の違いがブログ上の説明価値になるため、複数のvariantを選ぶ余地がある。
_Avoid_: duplicate image, cinematic scene, static menu

**Variant Expansion**:
recurring gameplay pattern で、同じ variant group から複数の画像を選ぶこと。要求選択枚数が多いほど強まり、同じ画面構造の中にある状態差や進行差を拾うために使う。Combat Subject Group、Combat Encounter Groupまたは他のSemantic Duplicate Groupに属する2枚目は、Variant Expansionより強い上限によって選ばない。
_Avoid_: duplicate flooding, one-representative-only selection, manual expansion mode

**Semantic Duplicate Group**:
Blog Image Type、Scene Slug、Variant Groupの分類境界をまたいでも、同じブログ上の役割を重ねて示す候補のまとまり。同じGroupからは要求枚数不足時も最大1枚だけを選び、最初に選ばれた最高Marginal Selection Utilityの候補を代表とする。判定は`combat_subject_appearance`、`combat_encounter_sequence`、`title_semantics`、`visual_role_similarity`のprivacy-safeなSemantic Duplicate Basisを持ち、Neutral Image Analysisだけを全体へ一律適用しない。
_Avoid_: global similarity threshold, Variant Group, duplicate filename, model confidence

**Combat Subject Evidence**:
一枚のCandidate Frameの攻撃相手本体だけから独立観測する、body plan、scale、surface、最大2色、最大4特徴、`distinctive`・`generic`・`unclear`の有限enum。敵名、Scene Slug、画面内文字、HP・status UI、背景、player、Context Cue、前後frame、別requestの結果を含めない。body plan・scale・surface・色・特徴がすべてそろう`distinctive`だけが動画横断の同一対象Profileに使える。`generic`の具体的なfieldは同一遭遇内で異なるCandidate Momentに2回以上現れた場合だけ別対象の裏付けに使い、動画横断の同一性を成立させない。不完全なfieldのまま`distinctive`とされた応答はCombat Subject Evidenceとして受理せず、Candidate Annotationのdomain validation retry対象にする。
_Avoid_: boss name, encounter identity, free-form description, multi-image comparison

**Combat Encounter Subject Profile**:
一つのCombat Encounter Group内にある異なるCandidate MomentのCombat Subject Evidenceを決定的に集約した主要戦闘対象の遭遇内Profile。Candidate Momentごとに一つの`distinctive`観測だけを数え、一意な最頻のbody plan・scale・surfaceと、複数Momentで2回以上反復した色・特徴を保持する。単一画像の観測値を変更せず、一時的effect・部分表示・色変化による孤立した矛盾と、複数画像で裏付けられた別対象を区別する。
_Avoid_: multi-image model response, boss name identity, averaged embedding, corrected Candidate Annotation

**Combat Subject Group**:
Semantic Duplicate Groupのうち、Video Source、時刻、Scene Slug、名称の正誤をまたいで同じ主要戦闘対象を示す候補のまとまり。元のCombat Encounter境界を保持したCombat Encounter Subject Profile間の裏付け済み共通特徴とNeutral視覚根拠で同一性を完全結合判定し、孤立した単一画像の矛盾では分断しない。各Profile対が互換なら、3件以上の全Profileに共通する単一の色・特徴がなくても同じGroupになる。同じGroupは要求枚数不足時も代表1枚を上限とする。
_Avoid_: Combat Encounter Group, enemy name identity, single-frame complete linkage, generic boss category, global visual cluster

**Combat Encounter Group**:
Semantic Duplicate Groupのうち、同一Video Source内で時系列に連続する`major`戦闘候補を同じ遭遇として扱う補助的なまとまり。非`major`場面と決定的なScene runを境界とし、原則として遭遇全体を代表1枚へまとめる。別対象へ分割するには、それぞれの対象を明瞭に示す異なるCandidate Momentの画像が2枚以上必要で、集約後の各Profileも互いに非互換でなければならない。単一画像の外見矛盾や、互換な集約Profileでは分割しない。
_Avoid_: boss name truth, all major combat in one video, Combat Encounter Kind, Variant Group

**Semantic Duplicate Basis**:
Semantic Duplicate Groupを構成した決定的で公開可能な根拠enum。`combat_subject_appearance`は互換なCombat Encounter Subject Profileと、それぞれを支持する候補間の0.80以上のNeutral視覚類似度、`combat_encounter_sequence`は主要戦闘の時系列run、`title_semantics`はBlog Image Type・Screen Text Kind・Representative Frame Evidenceのいずれかが示すtitle、`visual_role_similarity`は同一source内30秒以内、同じ画像内content kindとCombat Encounter Kind、0.93以上のNeutral視覚類似度を示す。複数の根拠が重なるcomponentでは、元Groupがcomponent全memberを実際に含む場合だけ、その公開contractで表せる最上位basisへ統合する。全memberを説明する元Groupがない場合は、優先度順に元Groupを処理し、先に使われたmemberを除いた残余が2件以上なら同じbasisの残余Groupとして保持する。これによりbasisを無関係なmemberへ推移的に拡張せず、未使用member間の低優先重複制約も失わない。`combat_subject_appearance`は元の遭遇境界を保持した全Profileで一致するbody plan・scale・surfaceと、全Profileに共通するcolor・traitだけをprivacy-safe evidenceとして公開する。完全結合が成立していれば共通color・traitが空でも中核tokenだけでGroupを公開し、Group memberに孤立した異なる単一画像Evidenceや`unclear`が含まれてもProfileを分断しない。`recurring_gameplay`で`visual_role_similarity`を使う場合は、独立評価された正規化済み画像summaryも一致させ、異なる技・敵・結果の追加説明価値を維持する。
_Avoid_: free-form rejection explanation, raw model response, global threshold

**Visual Near-Duplicate**:
Video Set selectorで使う正規化済み視覚特徴のcosine similarityが0.995を超えるRepresentative Frameの組。要求枚数が不足しても同時には選択しない。
_Avoid_: recurring gameplay variant, same scene, temporal neighbor

**Variant Group**:
同じ scene の中で、見た目や構図が近くブログ上の役割が重複する画像のまとまり。最終選択では原則として各 variant group から代表画像を1枚だけ選ぶが、recurring gameplay pattern では variant expansion の対象になる。
_Avoid_: scene, duplicate file

**Ollama Stage Failure**:
Scene Catalog、Candidate Annotationの主推論、Combat Encounter Verification、Combat Visibility Verification、またはPublication Boundary Verificationが、それぞれ同じsemantic入力による初回と1回の再試行後もtransport、schema、domain validationを完了できなかった状態。`other`への分類とは区別し、fallbackや失敗Candidateの除外で処理を継続せず、最終選定とoutput公開を中止する。
_Avoid_: other scene, silent exclusion, catalog fallback, partial output

**Resumable Run**:
中断された画像選択を、再利用可能なCompleted Stageから後で続ける実行。Video Setや設定が変わった場合は、一致するStage Fingerprintの成果物だけを再利用する。
_Avoid_: fresh run, output overwrite
