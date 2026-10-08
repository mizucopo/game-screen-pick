# Rust移行の比較契約

[Issue #338](https://github.com/mizucopo/game-screen-pick/issues/338)で固定する、Python版の観測可能な契約とRust版の受入基準。
移行全体の判断は[ADR 0010](adr/0010-migrate-video-cli-with-contract-gates.md)、用語は[CONTEXT.md](../CONTEXT.md)、通常利用は[技術リファレンス](technical-reference.md)に従う。
この文書の「現行」はPython版、「移行後」は親[Issue #337](https://github.com/mizucopo/game-screen-pick/issues/337)で承認された変更を指す。後続実装の完了を宣言する文書ではない。

## 比較基準

| 基準 | commit | version | 用途 |
| --- | --- | --- | --- |
| Issue作成時のmain | `2799e02b1c3c5623de275aa7b02148d9b7ea6d69` | `1.19.3` | 当初の調査根拠 |
| 2026-10-08着手時のmain | `e5e19c01d65cb48e9c5e1d6ccc171b4e2082228c` | `1.19.4` | fixture生成・Python並行比較の固定基準 |

両commitのdiffは20ファイル。`src/`、既存の動画・CLI・画像評価テスト、`config.example.toml`、`CONTEXT.md`、ADR 0004〜0009に差分はない。
変更は依存lock、開発依存、Copier／品質gate、リリース分類・自動採番と関連文書・テスト。
versionだけを上書きして旧環境を再現せず、比較実行は着手commitの`uv.lock`とpackage metadataを使用する。
根拠: [固定基準間のdiff](https://github.com/mizucopo/game-screen-pick/compare/2799e02b1c3c5623de275aa7b02148d9b7ea6d69...e5e19c01d65cb48e9c5e1d6ccc171b4e2082228c)、[pyproject.toml](../pyproject.toml)、[CONTRIBUTING.md](../CONTRIBUTING.md)。

## 移植する実行経路

現行entrypointは`game-screen-pick = src.main:cli_main`。`cli_main → run → execute → run_video_application → VideoSelector.run`が両方式の共通経路。

| 現行CLIから到達する責務 | 移植対象と根拠 |
| --- | --- |
| CLI・設定・値object | [main.py](../src/main.py)、[VideoRunConfigLoader](../src/utils/video_run_config_loader.py)、[video_run_config](../src/models/video_run_config.py)、[video_selection_request](../src/models/video_selection_request.py)、[video_selection](../src/models/video_selection.py)、[semantic_video](../src/models/semantic_video.py)、[vllm_config](../src/models/vllm_config.py)、[vllm_runtime_config](../src/models/vllm_runtime_config.py) |
| orchestration・sampling・機械評価・diversity・report | [run_video.py](../src/application/run_video.py)、[video_selector.py](../src/services/video_selector.py)内の関数。現行の閾値・重み・tie-breakを保持する |
| FFmpeg・cache・file安全性・contact sheet | [video_frame_extractor.py](../src/services/video_frame_extractor.py)、[video_phase_cache.py](../src/services/video_phase_cache.py)、[video_selection_files.py](../src/utils/video_selection_files.py)、[contact_sheet.py](../src/utils/contact_sheet.py) |
| sampled_frames画像評価 | [ollama_frame_assessor.py](../src/services/ollama_frame_assessor.py)、[FrameAssessor](../src/protocols/frame_assessor.py)。評価契約を移植し、接続実装はvLLMへ変更する |
| semantic_video動画理解・両画像評価・runtime | [semantic_video_planner.py](../src/services/semantic_video_planner.py)、[vllm_client.py](../src/services/vllm_client.py)、[vllm_runtime_session.py](../src/services/vllm_runtime_session.py)、[http_transport.py](../src/utils/http_transport.py) |
| Game Context生成・log | [game_context_generator.py](../src/services/game_context_generator.py)、[elapsed_log_formatter.py](../src/utils/elapsed_log_formatter.py)。生成の意味を保持し、検索・provider境界を変更する |

旧静止画経路`application/run.py → GameScreenPicker → ImageQualityAnalyzer`、`analyzers/`のCLIP・torch・transformers、scene catalog、`DynamicSceneSelector`、`WholeInputProfiler`、`StaticRejectClassifier`、旧`ReportWriter`／`ConfigResolver`／neutral-analysis cacheは現行CLIから呼ばれない。
特に動画の機械的rejectは`video_selector.measure_candidate`であり、旧`content_filter_thresholds.py`の閾値を代用しない。
`single_video_selector.py`は旧import互換用のre-exportで、別pipelineではない。
根拠: [ADR 0004](adr/0004-select-images-from-a-single-video.md)、[test_import.py](../tests/test_import.py)、[test_video_selector.py](../tests/services/test_video_selector.py)の`test_legacy_single_video_selector_import_path_is_available`。

## CLI・設定・探索・終了code

現行構文は`game-screen-pick [-c FILE] -n COUNT (--game-title TITLE | --game-context TEXT) INPUT_VIDEO_DIR OUTPUT_DIR`。`--help`も提供する。

| 公開入力 | 現行の型・検証・既定 |
| --- | --- |
| `-c / --config` | 存在する非directoryの設定file path、既定`config/config.toml`は実行時cwd基準。現行Click path検証は設定fileへのsymlinkを一律拒否する契約ではない |
| `-n / --num` | 必須integer、`1..999`。boolを設定から枚数へ転用しない |
| `--game-title` | 任意の表記を受け、前後空白を除いた非空文字列を生成に使う |
| `--game-context` | 前後空白を除いた直接文章。titleと常にXOR。直接文章には生成用4見出し／2,400文字制限を強制しない |
| `INPUT_VIDEO_DIR` | directory必須、動画file単体・空の入力集合を拒否 |
| `OUTPUT_DIR` | 新規時は空。所有が確認できる既存出力のみ再使用できる |

処理順はClick option型・必須検証 → TOML解決 → 入力探索 → title/context XOR → title生成用provider/model必須検証 → 実効設定log → application。
applicationは安価なrequest/path検証・cache/output lock・所有権確認 → context確定 → probe →候補処理へ進む。未管理出力を外部推論より先に拒否する。
現行では`VideoSelector`の構築時にも`ffmpeg`／`ffprobe`の存在を確認するため、completed runにもこれらのcommandは必要。
根拠: [main.py](../src/main.py)の`execute`／`run`、[video_selector.py](../src/services/video_selector.py)の`_prepare_paths`／`_prepare_run`、[VideoFrameExtractor.__init__](../src/services/video_frame_extractor.py)。

TOMLは`[run]`だけを受け、未知section/keyと型違いを拒否する。省略されたkeyだけ組み込み既定を補う。数値keyのboolは拒否。
CLIには枚数・title/context・入出力だけがあり、下表の設定用optionは存在しない。旧静止画向けCLI optionを復活させない。

| `[run]` key | 現行既定・制約 |
| --- | --- |
| `selection_method` | string、`sampled_frames`（既定）または`semantic_video` |
| `game_context_provider`, `game_context_model` | string、省略可。title指定時だけ両方非空必須。providerは`ollama / openai / gemini / xai` |
| `ollama_api_key`, `openai_api_key`, `gemini_api_key`, `xai_api_key` | string。非空TOML値 > 対応する`OLLAMA_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY / XAI_API_KEY`。trim後空値は環境へfallback。直接context時は生成key不要。cache hitに備えてkey不足の検証は生成時まで延期 |
| `primary_model`, `secondary_model` | string、既定`qwen3.8:27b`／`muse-glimmer:30b`。sampled_framesだけで使用し、liveで存在・vision能力・digestを確認 |
| `ollama_host` | string、TOML > `OLLAMA_HOST` > `127.0.0.1:11434`。scheme／portを正規化。semantic_videoはOllama生成または明示unload時のみ使用 |
| `ollama_timeout` | number、既定900秒、正の有限値 |
| `allow_cpu`, `debug` | boolean、既定false。sampled_framesは既定GPU必須、semantic_videoはGPU配置を検証しない |
| `ffmpeg_workers` | integer、既定2、`1..4` |
| `sample_interval_seconds` | number、省略時自動、有限かつ`>=0.25`秒。semantic_videoでは指定を拒否 |
| `vllm_base_url` | string、既定`http://127.0.0.1:8000/v1`。HTTP(S)、hostname・有効port必須、userinfo/query/fragmentを拒否。末尾slash除去後`/v1`を補う |
| `vllm_model`, `vllm_cache_revision` | string、semantic_videoのmodelは非空必須。revision既定`"1"`、trim後非空必須 |
| `vllm_api_key` | string、trim後非空TOML > trim後非空`VLLM_API_KEY` > 認証headerなし |
| `vllm_timeout` | number、既定900秒、正の有限値、runtime準備後のAPI header/body全受信にdeadline |
| `semantic_chunk_seconds`, `semantic_overlap_seconds` | number、既定30／2秒。`1<=chunk<=120`、`0<=overlap<chunk`、`chunk-overlap>=1`、有限 |
| `ollama_unload_before_vllm` | boolean、既定false。trueの時だけOllama model解放。runtime副作用設定はsemantic_videoだけ許可 |
| `vllm_start_command`, `vllm_stop_command` | 空でないNULなしstringの配列。空配列は未設定。start/stopは両方指定、shellを暗黙起動しない。`{model}`／明示SSH用`{model_shell}`の置換契約を保持 |
| `vllm_command_timeout`, `vllm_startup_timeout`, `vllm_shutdown_timeout` | number、既定60／900／60秒、正の有限値。未使用の設定だけでは操作を開始しない |

根拠: [VideoRunConfigLoader](../src/utils/video_run_config_loader.py)、[main.resolve_video_run_config](../src/main.py)、[モデル設定](../src/models/vllm_config.py)、[runtime設定](../src/models/vllm_runtime_config.py)、[semantic設定](../src/models/semantic_video.py)、[config.example.toml](../config.example.toml)。

Input Videoはdirectory直下で、symlinkを除く通常fileだけをfilenameのPython文字列昇順で列挙する。大文字小文字の違うfilenameは同一視しない。拡張子判定だけcase-insensitive。
対象拡張子は`.avi .flv .m2ts .m4v .mkv .mov .mp4 .mpeg .mpg .mts .ts .webm .wmv`。
非再帰、callerの任意順を受け付けず、内部requestは同じ直下親にある重複なしordered tuple。case-fold sortや自然順sortへの変更は同等ではない。

終了codeは成功／help `0`、Clickのusage・option・TOML・入力探索エラー `2`、applicationの失敗（probe、推論、所有権、lock、artifact不一致、cleanup失敗など）`1`、applicationのCtrl+C `130`。
option解析中の中断と`SIGKILL`はapplicationの再開メッセージ契約に含めない。stderrにはClickエラー、通常logはstdout。
根拠: [main.py](../src/main.py)、[run_video_application](../src/application/run_video.py)、[test_main.py](../tests/test_main.py)。

## 候補・評価・選択の決定性

- probeはattached pictureを除く最初のvideo streamを選び、stream durationを優先し、無ければcontainer durationから遅延startを差し引く。実last packet時刻を終端制約に使う。抽出は同じstream indexを`-map`し、時刻を小数6桁でFFmpegへ渡す。
- sampled_framesは各動画全編を覆う等間隔時刻。自動の最長間隔は10秒、目標数は`max(36*n,120)`、先頭／末尾余白と0.25秒の最小間隔で制限し、`np.linspace`相当を小数6桁へPythonの丸めで確定する。指定intervalは最大間隔であり、候補数固定上限はない。動画追加時に既存動画の自動時刻を変えない。
- semantic_videoは全編のoverlap chunkを1fps・幅512以下・16MiB以下のH.264動画として送る。固定応答eventから`[-1,-0.5,0,0.5,1]`秒のoffsetを原動画時刻へ戻し、event・端点・excluded intervalで制限、6桁round・重複除去・昇順。eventなし／失敗時に等間隔方式へfallbackしない。
- 候補JPEGは幅960以下、Selected Imageはfull resolution。動画の機械評価はOpenCVのBGR→gray、mean、population std、`Laplacian(CV_64F)`のpopulation variance、256-bin Shannon entropy。reject条件は`brightness<7 && contrast<4`、`brightness>248 && contrast<4`、`contrast<2 && sharpness<1`。境界の`<`を`<=`へ変えない。
- qualityは現行のexposure 0.10／contrast 0.25／log-sharpness 0.35／entropy 0.30。dHashはPillowの`L`変換→LANCZOS 9×8→右pixel `>` 左pixelの64bit row-major。Hamming distance、signed/unsigned、leading zeroを保持する。
- 動画ごとの一次候補最大`12*n`、二次最大`3*n`から評価し、survivor不足や代表動画欠落時は未評価候補を追補する。候補抽出・機械評価は未完了jobをworkerの2倍以下に抑え、完了順から入力順へ復元する。画像評価は一次12／二次6件のbatchで逐次checkpointする。
- 一次順はquality降順・時刻昇順・frame ID昇順（semanticではimportanceも考慮）。二次／最終のgreedy utilityは現行係数・scene/source/time/visual分散を保持する。utility tupleが完全同点ならPythonの`max`と同じ最初の候補を選ぶ。frame IDによる追加tie-breakを導入しない。
- aggregateは一次0.4＋二次0.6。最終rankはaggregate降順、video index昇順、時刻昇順のstable sort。title 1枚、map `max(1,ceil(n*.10))`、menu `max(2,ceil(n*.15))`のsoft capで、不足時は超過可能。最終選定は現在のpool内で既選択全frameからdHash距離5以上の候補を優先するが、該当候補が無ければ元のpoolへfallbackする。距離5未満かつ時間距離60秒未満の場合だけutilityから35を減点し、近い重複だけでも非遷移候補が足りれば要求枚数を満たす。一次または二次で遷移と判定した候補は除外し、非遷移候補が要求枚数未満なら失敗する。
- sceneは評価文字列をtrim・80文字へ、reasonはtrim・300文字へ正規化。scene集計はcasefold→PythonのUnicode `isalnum()`がtrueの文字だけ（日本語を含む）→48文字、空は`その他`。ASCII限定filterへ変えない。batch表示ID `A01..`をcontact sheet／prompt／schemaへ用い、未知・重複・欠落IDを拒否してstable IDへ戻してから保存する。blog_scoreは有限number `0..100`、bool不可、transitionはboolean。

根拠: [video_frame_extractor.py](../src/services/video_frame_extractor.py)、[semantic_video_planner.py](../src/services/semantic_video_planner.py)、[video_selector.py](../src/services/video_selector.py)の`make_timestamps`／`measure_candidate`／`image_difference_hash`／`select_primary_candidates`／`select_diverse_candidates`／`select_final_frames`、[ollama_frame_assessor.py](../src/services/ollama_frame_assessor.py)、[vllm_client.py](../src/services/vllm_client.py)。

## Identity・cache・所有権

Input Video Identityは`{identity_version:1, relative_path:<直下filename>, size:<byte>}`のJSON digest。絶対path・mtime・入力全体SHA-256を使わない。
stable frame IDは`f`＋identity key先頭16 hexをintegerへ変換した20桁decimal＋1始まりsample indexの最低5桁decimal。sample indexが5桁を超えても切らない。入力集合内のvideo indexを含めない。
同名・同sizeの内容置換を検出しない。正常resumeは候補manifestのdigest＋通常file／size照合だけでJPEGを全件再decode／hashしないため、同sizeのJPEG置換もbulk検査の検出対象外。明示cache削除で再生成する。
根拠: [video_phase_cache.py](../src/services/video_phase_cache.py)、[ADR 0008](adr/0008-cache-video-selection-phases-by-input-video.md)。

cache rootは`INPUT_VIDEO_DIR/cache-game-screen-pick/`、説明は`CACHE_INFO.txt`、動画は`videos/<identity>/`、runは`runs/<run-key>/`、生成contextは`game-context/`。旧Output Folder内`.game-screen-pick/`は再利用しない。
phase keyは`{cache_schema_version:1,phase,phase_version,conditions}`を`json.dumps(ensure_ascii=False,sort_keys=True,separators=(",",":"))`→UTF-8→SHA-256で算出する。
数値表現・Unicode key順・丸め・`null`・配列順も入力の一部。Rustの既定JSON serializationへ置き換えて互換と宣言しない。payload digestは同じcanonical JSON、`mechanical_state_digest`は保存JSON file本文のSHA-256なのでindent／改行まで影響する。

| 現行version／phase | key・検証の主な依存 |
| --- | --- |
| cache schema 1、video identity 1 | 共通envelopeのschema/phase/version/key/data型。相違・破損・symlinkはmiss |
| video-probe 2 | identity key。identity＋metadata全体の`probe_payload_digest`と有限metadataを検証 |
| semantic_video／chunk 1、plan 3 | identity・metadata・vLLM metadata・chunk geometry・Game Context・prompt/schema・動画encoding・offset。chunkはplan request key＋index/start/end、結果digest＋liveと同じschema。plan keyは実応答evidenceと時刻を含み、意味変更も後続を失効 |
| candidate-extraction 2 | identity・probe version・metadata・ordered時刻・最大幅960・semantic plan key。ordered ID/時刻/size/生成時JPEG SHAをmanifest payload digestで保護 |
| mechanical-analysis 4 | candidate key。source manifest digestとusable/rejected全ID・quality・16桁hex dHashをpayload digestで保護。全frameを過不足なくcover |
| primary-assessment 3 | identity・candidate/mechanical keyとmechanical file digest・prompt・model/digest・endpoint/GPUまたはvLLM revision・context・枚数・batch 12・ordered candidate JPEG digest・model options |
| secondary-context 2 | candidate key・前後offset0.35秒・metadata。各before/after JPEG digest照合、要求された破損frameだけ再抽出 |
| secondary-assessment 2 | 一次と同種の条件、batch 6、context key/JPEG digest、一次assessment key。評価順を保つ |
| global candidate 1、final selection 1、artifacts 2 | manifestのphase versionsに記録。入力集合・条件変更時にglobal選定・成果物を再生成 |
| run manifest 1、algorithm `multi-video-selection-v7`、prompt `blog-image-selection-v6` | run identity、各phase version、ordered入力相対identity・metadata/sample位置、最終context、model metadata、倍率・batch・optionsをmanifest digestで固定 |
| Game Context checkpoint schema 2 | title/provider/model、Ollamaだけ正規化host。request＋result全体のpayload digest、生成用4見出し・2,400文字、provider／実modelを検証 |
| report schema 1 | JPEG集合のsize/SHAとreportの構造から所有権再確立可能。completionにはreportを含む全成果物digest |

assessment cacheは現評価単位の「完了batch prefix」だけを再利用する。途中穴、batch regroup、最終batch以外の端数、digest違い、異常score／scene/reasonは全phase miss。
同名modelも一次／二次のdigestを別に持ち、未評価batch前のlive検証で変化すれば該当phaseを再実行。
vLLM digestは`provider/base_url/model/cache_revision`の設定fingerprintであり、immutable weight hashやGPU配置証明ではない。operatorが重み・量子化・processor・runtime条件変更時にrevisionを変更する。
timeout・start/stop command・API keyは推論条件／cache keyへ含めない。fully cached runは推論HTTP・runtime commandを呼ばない。
根拠: [video_selector.py](../src/services/video_selector.py)の`_build_manifest`／`_assessment_cache_key`／`_load_assessment_state_or_miss`、[semantic_video_planner.py](../src/services/semantic_video_planner.py)、[VllmClient.model_metadata](../src/services/vllm_client.py)、[ADR 0009](adr/0009-discover-candidates-with-vllm-video-understanding.md)。

生成contextの4見出しはASCII colonを含む`ジャンル:`、`基本的なゲーム進行と主なプレイ要素:`、`代表的な画面や場面:`、`画像選定で重視する視覚的要素:`がsubstringとして全て存在すること。現行検証は前後の`strip()`だけを行い、改行（CRLFを含む）・見出し順・colonを正規化しない。全角colonはASCII見出しの代替にならない。長さはstrip後のPython Unicode文字数で2,400以下。`ambiguous / insufficient / conflict`など`ok`以外は停止し、contextをcheckpointしない。
根拠: [normalize_generated_context](../src/services/game_context_generator.py)、[game_context_expectations.json](../tests/fixtures/rust_migration/game_context_expectations.json)、[test_game_context_contract.py](../tests/migration/test_game_context_contract.py)。

Output Folderはcache root自身／配下を拒否。新規は空、再使用は正規化した同じ絶対Output Folderの正常registration、またはmanifestと全artifactのsize/SHAが一致するcompletion、またはintegrity付きreportで所有を確立する。
completion／report経由はmanaged-looking artifact集合の完全一致が必要。未管理file・余計なselected JPEG・staging symlinkを自動削除しない。
registration単体は未完了runの再開にも使うため、完成artifact digestの保証ではない。この違いを保持する。

cache component directoryはsymlink不可、lockは`O_NOFOLLOW`＋通常file＋nonblocking `fcntl.flock`、Output Folder directoryにもlockを取る。
候補cacheは正常manifestがある場合、`lstat`によるregular file・記録sizeの一致だけで再利用する。欠損・leaf symlink・非regular file・size不一致は再抽出し、再抽出後にJPEG型・幅960以下・画像の有効性を確認する。同sizeの非JPEG・幅超過・画像破損はこのbulk検査では検出せず、再抽出を保証しない。要求されたcontext frameは毎回画像の有効性・JPEG型・幅960以下と記録SHA-256を検査し、symlink・空画像・decompression bomb・digest不一致などで検査に失敗すれば再抽出する。
予測不能なexclusive temporary fileを同一filesystemへ作り、umaskを反映、flush/fsync後renameする。別origin redirectに認証を転送せず、API keyや応答error bodyをlog/report/cacheへ漏らさない。
根拠: [video_selection_files.py](../src/utils/video_selection_files.py)、[http_transport.py](../src/utils/http_transport.py)、[video_selector.py](../src/services/video_selector.py)の所有権関数、`_matches_candidate_file`／`_save_candidate_manifest`／`_is_valid_cached_image`／`_extract_context_frames`。

既存のowned completed outputはprobe・model検証・推論・staging作成が失敗しても保持する。置換成果物はOutput Folder内のstagingで全部完成してから、各fileをatomic renameし、余った旧managed fileを除き、最後にcompletionを保存する。
これは**各fileのatomic置換**であり、集合全体のtransactionではない。publication途中の強制終了では旧新混在があり得る。completionを先に失効させるので混在集合をcompletedと判定しない。
所有確認後だけabandoned実staging directoryを回収する。新規Output Folderの未完了成果物と確定済みbatch/chunkは再開用に残る。
runtimeはlive requestまでlazy、明示managed endpointが既に応答すれば所有を奪わず失敗。attempted startには中断・部分失敗もstopを試み、cleanup失敗は元の推論errorを隠さない。強制終了時のprocess回収はoperatorのservice supervisionに委ねる。
根拠: [VideoSelector._publish_selected_artifacts](../src/services/video_selector.py)、[vllm_runtime_session.py](../src/services/vllm_runtime_session.py)、[ADR 0008](adr/0008-cache-video-selection-phases-by-input-video.md)、[ADR 0009](adr/0009-discover-candidates-with-vllm-video-understanding.md)。

## 成果物・report・logの比較

file集合は`selected-<rank>.jpg`（幅`max(2,len(str(n)))`）、`selected-contact-sheet.jpg`、`report.json`の`n+2`個。
full-resolution抽出後のdHashと評価候補の距離が10を超えれば失敗。contact sheetはrank・動画相対名・時刻を表示する。

| report schema 1の領域 | 必須の意味 |
| --- | --- |
| top level | `report_schema_version`, `manifest_digest`, `videos`, `game_context`, `output_count`, `sample_count`, `models`, `selected`, `artifact_integrity` |
| 生成context時だけ | `game_context_generation:{provider,model}`。Game Titleやraw検索結果はreport／run manifestへ保存しない（context生成checkpointのrequestにはtitleを保存） |
| semantic_video時だけ | `selection_method:"semantic_video"`, `video_understanding`（model/endpoint/revision/options）、各videoの`semantic_analysis`、各selectedの`semantic_provenance`。sampled_framesはこれらを省略 |
| `videos[]` | 1始まり`video_index`, 絶対`path`, `duration_seconds`, `sample_count`。入力探索順と一致 |
| `models.{primary,secondary}` | `name`, `resolved_name`, `digest`。現行semanticでは同一vLLM modelを両段階に使う |
| `selected[]` | 1始まり`rank`, basename `output_path`, stable `frame_id`, 1始まり`video_index`, 絶対`video`, `video_name`, 小数6桁`timestamp_seconds`, 小数2桁`aggregate_score`, `candidate_output_dhash_distance`, `primary`, `secondary` |
| primary／secondary | `frame_id`, `blog_score`, `is_transition`, `scene`, `reason`。応答順によらず候補IDへ対応 |
| `artifact_integrity[]` | report自身を除く全JPEGのbasename `path`, byte `size`, lowercase hex `sha256`。completionはreport自身も含む |

logの文字列全体・timestamp・前行からの経過秒はgolden一致を要求しない。ただしversion、実効非秘密設定、選定方式／model、context確定か再利用か、入力probe/cache状態、全候補数、一次/二次の初期予定・追補上限・最終対象数、完了batch進捗、追補による母数変更理由、再試行／失敗、旧output保持／置換、完了検証／中断再開の意味を比較する。
log進捗をCompleted RunやHuman Reviewの合格と解釈しない。動的文字列は一物理行へescapeする。
根拠: [VideoSelector._write_selected_artifacts](../src/services/video_selector.py)、[contact_sheet.py](../src/utils/contact_sheet.py)、[ElapsedLogFormatter](../src/utils/elapsed_log_formatter.py)、[pipeline log tests](../tests/services/test_video_pipeline.py)。

## 同等性gateとfixture

保存fixtureは[tests/fixtures/rust_migration](../tests/fixtures/rust_migration/)、検証は[tests/migration](../tests/migration/)に置く。
合成動画とpixel recipeからの境界画像、固定AI入力媒体・応答、期待report、正常／partial／corrupt／digestのみ破損したcacheを使い、実録画・API key・検索結果を入れない。
テストはネット・実GPU不要。動画fixtureの検証には外部FFmpeg／ffprobeが必要で、不在skipを配布targetの合格と扱わない。
Pythonコードをtest中に再実装したoracleだけで判定せず、保存された期待値と実行結果を比較する。後続Rust版は同じfixtureを読む。

| 比較領域 | 固定する判定 |
| --- | --- |
| 採否・選択 | usable/rejected集合、stable ID、6桁丸め時刻、rank、scene／transition／reason、source coverage、greedy同点時の先着、cold/warm/resumeの最終結果は厳密一致。dHashの64bitと距離も厳密 |
| 同じdecoded pixelに対する計算 | 有限quality・utility・aggregateと画像metricは`abs(a-b)<=1e-6 OR abs(a-b)<=1e-8*max(abs(a),abs(b))`。NaN/infは禁止。2桁へ丸めたreport aggregateは厳密一致。数値許容内でもthreshold採否や順位が変われば不合格 |
| JPEG encoder差 | byte一致は要求しない。decoded RGBの寸法と向きは厳密、channel MAE `<=1.0`（0..255）、最大絶対差`<=16`、PSNR `>=40dB`（完全一致は∞）。さらに採否・dHash・選択結果が一致すること。別decoder/FFmpeg差にも同じgateを適用 |
| resize・thumbnail | 同じ入力寸法・縦横比・center placement・padding/cell構成を保持。thumbnail pixel領域に上のRGB gateを適用。OpenCV gray/Laplacianのedge borderとPillow LANCZOS/dHashを境界画像で確認 |
| contact sheet文字 | rank／Frame Display ID／動画名／時刻とbefore/selected/after順、cell位置は厳密。font glyph raster差だけを既知label領域でmaskし、thumbnailまでmaskしない。AI入力sheetも両版を並べHuman Review |
| report正規化 | 実行root絶対pathをplaceholder化。JPEG byte size/SHA、そこから派生するmanifest／phase digestはcross-encoder比較時だけ別比較とする。各実装自身のreport/completion integrity検証は必須。任意fieldやsemantic provenanceを丸ごと落とさない |
| cache比較 | digest/key/version計算はcanonical JSONの個別vectorで厳密比較。`1.0 / 0.0 / 1e-6`のPython JSON数値表記も維持する。既存cacheのread結果、miss範囲、未処理batch/chunk数、completed shortcut、再開結果を比較。AI接続変更の意図したmissはADRのmatrixに従う |

許容値は移植前の初期gateであり、現在のPython fixtureで測定されたRust差を意味しない。差がgateを超えれば原因を記録し、画像実装を修正する。
画像処理方式を替えるために後から許容値を広げる場合は、threshold前後・AI入力・最終品質の影響とHuman Reviewを示す別の設計判断が必要。期待値の無説明再生成で差を隠さない。

固定goldenの入口は[image_expectations.json](../tests/fixtures/rust_migration/image_expectations.json)（16種類のPNGとpixel recipe、production writerで先頭ゼロを保持するdHash）、[wire_expectations.json](../tests/fixtures/rust_migration/wire_expectations.json)（UTF-8 canonical JSON、Identity、stable frame ID、phase key）、[game_context_expectations.json](../tests/fixtures/rust_migration/game_context_expectations.json)（trim／CRLF／生成失敗、Unicode 2,400文字受理／2,401文字拒否）、[selection_expectations.json](../tests/fixtures/rust_migration/selection_expectations.json)（casefold／Unicode isalnumによるscene集計と最終選択）である。
動画は[pipeline/synthetic-game.mkv](../tests/fixtures/rust_migration/pipeline/synthetic-game.mkv)と[生成recipe](../tests/fixtures/rust_migration/pipeline/generate_video.py)、固定応答は`pipeline/responses/`、期待値は`pipeline/expected/`、保存cacheは`pipeline/stored-cache/`、再開条件は[cache-replay.json](../tests/fixtures/rust_migration/pipeline/cache-replay.json)に置く。`pipeline/inference-media/`は実際のAI入力request・sheet全体・動画全frame／PTSを固定し、誤った媒体でも同じ応答が返る比較抜けを防ぐ。`pipeline/reference-images/`には最終contact sheet全体も保存し、label・画像順・paddingを含めた同じRGB gateで検証する。

主要fixture/testの対応は次のとおり。既存テストは移行fixtureだけで置き換えず保持する。

| 契約 | 現行テスト → Rust側の受入試験 |
| --- | --- |
| CLI／TOML／秘密値／探索／exit | [test_main.py](../tests/test_main.py)、[test_video_run_config_loader.py](../tests/utils/test_video_run_config_loader.py)、[test_vllm_runtime_config.py](../tests/models/test_vllm_runtime_config.py) → 同じvalid/invalid入力、境界枚数、未知key/section、型、優先順位、error発生前の副作用なし |
| probe／全編／VFR／短尺／遅延stream／map | [test_video_frame_extractor.py](../tests/services/test_video_frame_extractor.py)、[test_video_selector.py](../tests/services/test_video_selector.py)のtimestamp tests → 同じstream/metadata/sample positions、終端を越えない |
| 画像threshold・score・dHash | [test_video_selector.py](../tests/services/test_video_selector.py)の`test_measure_candidate_rejects_black_and_scores_visible_frame`、[test_image_contract.py](../tests/migration/test_image_contract.py) → 黒/白/単色、閾値前後、RGB/gray、Laplacian border、LANCZOS・dHashの先頭ゼロを含む16桁wire・quality |
| 多動画／diversity／backfill／tie | [test_video_selector.py](../tests/services/test_video_selector.py)、[test_video_pipeline.py](../tests/services/test_video_pipeline.py)のsource/backfill tests、[test_pipeline_contract.py](../tests/migration/test_pipeline_contract.py) および[test_selection_contract.py](../tests/migration/test_selection_contract.py) → 一次/二次採否、最終ID/time/rank/source/scene、Unicode scene集計一致 |
| sampled cold／warm／中断再開 | [test_video_pipeline.py](../tests/services/test_video_pipeline.py)、保存pipeline fixture → cold推論はprimary／secondary各1回、warmでprobe/抽出/再計算/推論なし、保存batch prefixからresume、coldと同じ成果物意味 |
| semantic cold／warm／中断再開 | [test_semantic_video_planner.py](../tests/services/test_semantic_video_planner.py)、[test_semantic_video_pipeline.py](../tests/services/test_semantic_video_pipeline.py)、[test_semantic_video_end_to_end.py](../tests/services/test_semantic_video_end_to_end.py)、保存pipeline fixture → cold推論はvideo／primary／secondary各1回、chunk/event/exclusion/plan、部分checkpoint、意味変更時後続miss、等間隔fallbackなし |
| cache破損・移動・入力追加・version | [test_video_pipeline.py](../tests/services/test_video_pipeline.py)のmove/added_video/phase_version/corrupt/same_size tests → 正常/partial/corrupt replay、前段維持、変更phase以降だけmiss、同size置換の検出限界 |
| cache serialization・identity・frame ID | [test_cache_wire_contract.py](../tests/migration/test_cache_wire_contract.py)、保存wire vector → key順・非ASCII UTF-8・float/null・SHA-256・sample IDの桁を厳密比較。Rust serializer差があれば再利用せずversion境界を明示 |
| Run Manifest全項目 | [test_pipeline_contract.py](../tests/migration/test_pipeline_contract.py)、[test_run_manifest_contract.py](../tests/migration/test_run_manifest_contract.py)、保存pipeline golden → 全nested fieldの値・key・型・digestを厳密比較。倍率・batch・context offset・options・model metadata・run identityを誤変更し、report/completionのdigestを整合させても拒否 |
| output所有／staging／既存出力保護 | 同pipelineのownership/publication/preserves_completed_output tests、[test_semantic_video_pipeline.py](../tests/services/test_semantic_video_pipeline.py)のfailure test、保存output fixtureおよび[test_output_media_contract.py](../tests/migration/test_output_media_contract.py) → 最終sheetのblank／thumbnail順／誤label拒否、unmanaged拒否、tamper検出、staging失敗で旧成果物保持、publication中断でcompleted誤判定なし |
| symlink／lock／bounded jobs／atomic file | [test_video_selection_files.py](../tests/utils/test_video_selection_files.py)、同pipelineのsymlink/concurrent/bounded/cancel tests、[test_contact_sheet.py](../tests/utils/test_contact_sheet.py) → no-follow、排他、worker窓、予測不能tmp・同filesystem rename |
| AI schema／context／HTTP | [test_ollama_frame_assessor.py](../tests/services/test_ollama_frame_assessor.py)、[test_vllm_client.py](../tests/services/test_vllm_client.py)、[test_game_context_generator.py](../tests/services/test_game_context_generator.py)、[test_game_context_contract.py](../tests/migration/test_game_context_contract.py)、[test_http_transport.py](../tests/utils/test_http_transport.py) → 固定応答のID/型/範囲、生成失敗／checkpoint、ASCII見出し・CRLF保持、trim後Unicode 2,400／2,401文字境界、redirect・deadline・key非漏洩 |
| runtime所有／lazy／cleanup | [test_vllm_runtime_session.py](../tests/services/test_vllm_runtime_session.py)、[test_semantic_video_runtime.py](../tests/services/test_semantic_video_runtime.py) → warm無操作、既存server保護、partial start/interrupt/stop失敗で元error保持 |
| package・品質・リリース | [test_import.py](../tests/test_import.py)、[test_quality_gate.py](../tests/test_quality_gate.py)、[test_release_workflow.py](../tests/test_release_workflow.py) → 比較期間はPython import/packageと既存全品質gate保持、最終binaryはPython/uv/toolchainなしで起動 |

### live modelとHuman Review

固定応答gateとlive gateは別に記録する。OllamaとvLLM／Brave生成の応答一致は要求しない。
#343では同じ実動画を両方式・各配布targetで比較し、直接contextとBrave生成contextを分け、warm/resume、通信失敗・中断・stop失敗を確認する。
実録画は許可されたlocal場所に保持し、fixtureへcommitしない。記録には入力identity、両commit、FFmpeg版、model/revision/推論レベル、設定、cold/warm状態、実行回数を付ける。

少なくとも各方式3回のcold live runを行い、各runのSelected Contact Sheetと必要なfull-resolution/前後frameを人が確認する。
合格条件は全runで要求枚数と追跡可能なsource/timeが揃い、暗転・白飛び・loading・fade途中や評価候補と異なる出力が0枚、明らかな近似重複が0組、有効候補と枠があれば全Input Videoをcoverすること。
通常進行と有用な特別画面・時刻の分散がPython基準より劣化せず、soft cap超過時は候補不足の根拠が説明できること。
検索contextは4項目・簡潔さ・出典から検証可能なゲーム同定を満たし、曖昧／不足／矛盾を推測で補わないこと。
reviewer、日時、runごとの合否・具体的な問題画像IDと理由を保存し、1runでも不合格または人の確認が無ければ切替gateは未通過。
モデルscoreだけでHuman Reviewを代用しない。Braveの課金／無料枠・保存条件と採用model／推論レベルは#342の確認事項で、未確定値を既定として配布しない。

## 意図した互換性変更と後続判断

親#337により、移行後は両方式の画像評価・動画理解とGame Context生成modelをローカルvLLMへ統一し、context自動生成はBrave Search APIで検索する。
Ollama本体/API/Web Search/unload、OpenAI/Gemini/xAI/OpenRouter連携は移植しない。modelと推論レベルはTOMLの明示必須、暗黙defaultを持たない。
旧`primary_model`／`secondary_model`やprovider/key設定を黙って別意味へ転用しない。旧keyの拒否と新設定への案内を行う。以前の`gpt-6-luna`／`low`をvLLMへ流用しない。
新key名、採用modelごとの推論レベル対応、Brave無料枠の超過停止・保存条件は#342で確定する。
この変更によるAI cache miss、2.0.0へのメジャー切替、配布target・並行検証・rollbackは[ADR 0010](adr/0010-migrate-video-cli-with-contract-gates.md)に従う。
