# Issue #169: 動画入力runtimeとOllama model契約の一次調査

- 調査日: 2026-07-14
- 対象: Issue #169「動画入力CLI・config・運用ドキュメント契約を確定する」
- 調査方法: 公式ドキュメント、公式model library、公式package metadata、公式release notes、`ssh winpc`による読み取り専用inventoryと限定的な24画像能力probe
- 対象外: production実装、modelの追加download、service設定変更、全modelの品質benchmark、CLI/config precedenceの最終決定

## 結論

### runtimeの推奨baseline

1. Ollama serverのproject最低versionは **0.31.2** とする根拠がある。Qwen3-VL自体の公式下限は0.12.7だが、0.31.2はこのprojectが使う「thinking無効 + structured output」の修正をrelease noteで明示し、Issue #165の能力検証にも使われたversionである。[Qwen3-VL model page](https://ollama.com/library/qwen3-vl:8b-instruct) [Ollama v0.31.2 release](https://github.com/ollama/ollama/releases/tag/v0.31.2)
2. system `ffmpeg` / `ffprobe`のproject最低versionは、実動画と合成fixtureを検証済みの **6.1.1** とする。両binaryが同じbuildであることに加え、必要なdemuxer、decoder、filter、JSON probeをpreflightする。versionだけでcodec能力を推測しない。[FFmpeg documentation](https://ffmpeg.org/ffmpeg.html) [ffprobe documentation](https://ffmpeg.org/ffprobe.html) [Ubuntu Noble ffmpeg package](https://launchpad.net/ubuntu/noble/%2Bpackage/ffmpeg)
3. STT runtimeの最低versionは **faster-whisper 1.2.1 / CTranslate2 4.8.1** とする。Issue #166で検証したmodel名と実行optionは組み込み既定値にするが、model名、device、compute type、beam、VAD、chunk、overlapはTOMLで変更可能にする。snapshot revisionは設定へ書かず、実行時に解決した値をStage Fingerprintへ含める。faster-whisper 1.2.1はSilero VAD v6を採用し、CTranslate2 4.8.1はWhisper alignmentでframeがないwindowによりprocessが終了し得る不具合を修正している。[faster-whisper v1.2.1 release](https://github.com/SYSTRAN/faster-whisper/releases/tag/v1.2.1) [CTranslate2 v4.8.1 release](https://github.com/OpenNMT/CTranslate2/releases/tag/v4.8.1)
4. 対象GPUはGeForce RTX 5090で、NVIDIA公式仕様は32 GB GDDR7、compute capability 12.0である。OllamaはRTX 5090を対応表に明記している。[RTX 5090 official specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda/gpus) [Ollama hardware support](https://docs.ollama.com/gpu)

### modelの推奨

初期の既定model名は **`qwen3-vl:8b-instruct`** とするのが最も根拠が強い。調査時にこのtagが解決したQ4_K_M artifactは、公式libraryでText/Image、256K最大context、6.1 GB artifactとして提供され、Issue #165の3画像 + 3 Context Cue + JSON SchemaのCandidate Annotation probeと、Issue #169の24画像Scene Catalog probeの両方に成功している。digestはTOMLへpinせず、実行時に解決してfreezeする。[Qwen3-VL tags](https://ollama.com/library/qwen3-vl/tags) [Issue #165 prototype](../../prototypes/issue_165_ollama_stages/README.md) [Issue #169 probe](../../prototypes/issue_169_runtime_contract/README.md)

最大24画像のScene Catalogを32K contextで3回検証し、3回ともJSON Schemaとlocal domain validationに成功した。cold runは14.261秒（model load 7.409秒）、warm runは3.795秒と3.742秒、prompt evaluationは12,514 tokenだった。Ollamaのmodel sizeと`size_vram`はいずれも10,210,393,456 bytesで100% GPU load、`nvidia-smi`のglobal memory peakは14,629 MiBだった。warm 2回は完全一致したがcold runのscene slug表現には軽微な差があり、temperature 0をbit-for-bit安定性の保証にはしない。[Issue #169 probe](../../prototypes/issue_169_runtime_contract/README.md)

registryの「6.1 GB」はdownloaded model artifactの容量であり、必要VRAMの保証値ではない。Ollamaはcontextを増やすほどmemory使用量が増えると説明し、24–48 GiB VRAMでは現在32K contextを既定にする。実際のVRAMはcanonical request実行中の`/api/ps`にある`size_vram`、`context_length`と`ollama ps`のprocessor splitで確認する。[Ollama context length](https://docs.ollama.com/context-length) [List running models API](https://docs.ollama.com/api/ps) [Ollama FAQ](https://docs.ollama.com/faq)

## 既存contractから必要になる能力

[Issue #165の確定contract](../../prototypes/issue_165_ollama_stages/README.md)では、Ollama operationは次の2種類に限定されている。

| operation | vision入力 | text入力 | structured output | 規模 |
|---|---|---|---|---|
| Scene Catalog | Representative Frame最大24枚 | Selection Intent / Scene Hint | Scene Catalog schema | Video Setごとに1回 |
| Candidate Annotation | Frame Candidate 1〜3枚 | Scene Catalog、Context Cue、進行位置、Selection Intent | Candidate Annotation v1 | Selection ShortlistのCandidate Momentごとに1回 |

どちらも画像分類、画像説明、複数画像間の比較、JSON Schema出力が必要である。Ollama公式vision APIはmessage内の`images`配列を受け取り、vision modelが画像の説明・分類・質問応答を行えるとしている。[Ollama Vision](https://docs.ollama.com/capabilities/vision)

Ollama公式structured outputs documentationは、vision modelにも同じJSON Schema `format`を使えること、低いtemperatureが再現性に有効であることを示している。[Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)

音声はOllamaへ渡さない。Issue #166で確定した専用STT経路がtimestamp付きContext Cueへ変換し、Ollamaにはtextだけを渡す。[Issue #166 research](issue-166-audio-subtitle-context.md)

## Ollama model候補

### 公式tagの比較

次は2026-07-14にOllama公式libraryで確認したlocal model tagである。容量はregistryが示すartifact容量であり、VRAM測定値ではない。

| tag | parameter / quantization | artifact | 最大context | 位置付け |
|---|---|---:|---:|---|
| `qwen3-vl:4b-instruct-q4_K_M` | 4B / Q4_K_M | 3.3 GB | 256K | 低resource代替。schema品質は未検証 |
| `qwen3-vl:8b-instruct-q4_K_M` | 8.77B / Q4_K_M | 6.1 GB | 256K | 初期既定候補。Candidate Annotationと24画像Scene Catalogを同一digestで実測済み |
| `qwen3-vl:8b-instruct-q8_0` | 8.77B / Q8_0 | 9.8 GB | 256K | 同parameterの品質比較候補。未実測 |
| `qwen3-vl:8b-instruct-bf16` | 8.77B / BF16 | 18 GB | 256K | quantization影響の比較候補。未実測 |
| `qwen3-vl:30b-a3b-instruct-q4_K_M` | 30B級MoE / Q4_K_M | 20 GB | 256K | 高capacity challenger。未実測 |
| `qwen3-vl:32b-instruct-q4_K_M` | 32B / Q4_K_M | 21 GB | 256K | 高capacity dense challenger。未実測 |
| `gemma4:12b-it-q4_K_M` | 12B / Q4_K_M | 7.6 GB | 256K | 別family challenger。未実測 |
| `gemma4:26b-a4b-it-q4_K_M` | 26B級MoE / Q4_K_M | 18 GB | 256K | 別family高capacity challenger。未実測 |

Qwen3-VLの各tagはText/Image入力を明記し、instruct、thinking、Q4_K_M、Q8_0、BF16を別tagとして公開している。[Qwen3-VL tags](https://ollama.com/library/qwen3-vl/tags)

Gemma 4の各tagもText/Image入力、context、quantization別容量を公開している。[Gemma 4 tags](https://ollama.com/library/gemma4/tags)

### Qwen3-VLが現在最有力である根拠

- 公式model pageは、visual recognition、複数言語OCR、空間理解、長いmultimodal contextをQwen3-VLの能力として挙げている。ゲーム画面のscene、画面内text kind、Representative Frame選択に必要な能力と方向が一致する。[Qwen3-VL model page](https://ollama.com/library/qwen3-vl:8b-instruct)
- `instruct` tagは対象機の`/api/show`で`completion`、`vision`、`tools`を返し、`thinking` capabilityを返さなかった。Issue #165のcontractは`think=false`とJSON Schemaを使うため、thinking modelではなくinstruct tagを明示する方が境界が小さい。`/api/show`がcapabilities、parameter size、quantization、model metadataを返すことは公式API contractである。[Show model details API](https://docs.ollama.com/api-reference/show-model-details)
- 対象機の`qwen3-vl:8b-instruct` digest `0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319`で、Candidate Annotation schemaが3回同じdomain結果を返した。[Issue #165 prototype](../../prototypes/issue_165_ollama_stages/README.md)
- 用途が曖昧な`latest`やparameter数だけのaliasは既定にせず、`instruct`を明示したtagを設定する。実行時には`/api/tags`から完全digestとquantizationを取得し、Stage Fingerprintへ含める。公式APIもnameとは別にdigest、size、parameter size、quantizationを返す。[List models API](https://docs.ollama.com/api/tags)

### 代替候補の扱い

`qwen3-vl:8b-instruct-q8_0`は同じ8B級でquantization差だけを比較しやすい。32 GB GPUに対してartifactは9.8 GBだが、画像token、KV cache、runnerを含むVRAMはcanonical probeで測る必要がある。[Qwen3-VL tags](https://ollama.com/library/qwen3-vl/tags) [Ollama context length](https://docs.ollama.com/context-length)

`qwen3-vl:30b-a3b-instruct-q4_K_M`と`qwen3-vl:32b-instruct-q4_K_M`はartifactだけで20–21 GBを占める。32 GB GPUでは残り11–12 GBしかなく、24画像、32K context、runtime overheadを含めた100% GPU loadを公式表だけから保証できない。採用には`ollama ps`で100% GPUまたは許容したoffload率とlatencyを実測する。[Qwen3-VL tags](https://ollama.com/library/qwen3-vl/tags) [Ollama FAQ](https://docs.ollama.com/faq)

同じ30B/32BのQ8_0 artifactは34–36 GBで、GPUの公称32 GBよりartifact自体が大きい。少なくとも単一RTX 5090へ全modelを載せる候補にはできない。[Qwen3-VL tags](https://ollama.com/library/qwen3-vl/tags) [RTX 5090 official specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)

Gemma 4は公式にmultimodal understanding向けとされ、対象機にも`gemma4:latest`がある。しかしIssue #165の最終schemaを使った画像評価を実測していない。`latest`はmutable aliasでもあるため、既定にはせず、比較する場合は明示tagとdigestでcanonical probeを行う。[Gemma 4 model library](https://ollama.com/library/gemma4) [List models API](https://docs.ollama.com/api/tags)

modelが失敗したとき別modelへ自動fallbackすると、Candidate Annotationの意味とcache keyが暗黙に変わる。候補はdocument上の明示的な代替に留め、1 run内では設定したtag/digestを固定する。

## VRAMとcontextの契約化

### 公式情報から言えること

- RTX 5090は32 GB GDDR7である。[RTX 5090 official specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
- Ollamaは24–48 GiB VRAMで32K contextを現在の既定とし、contextを増やすとmemory必要量が増えると明記している。[Ollama context length](https://docs.ollama.com/context-length)
- parallel requestはcontext memoryを増やし、必要RAMは`OLLAMA_NUM_PARALLEL * OLLAMA_CONTEXT_LENGTH`に比例する。単一modelでも並列度を上げれば同じVRAM見積もりは使えない。[Ollama FAQ](https://docs.ollama.com/faq)
- `/api/ps`は実際にload中のmodelについて`size_vram`と`context_length`を返す。[List running models API](https://docs.ollama.com/api/ps)

### projectで採るべき目安

「必要VRAM 8 GB」のような固定値をregistryのartifact容量から作らない。代わりに、target profileを次の観測可能な受け入れ条件で定義する。

1. GPUは32 GB classのRTX 5090をreference targetとする。
2. `OLLAMA_NUM_PARALLEL=1`相当でScene CatalogとCandidate Annotationをserialに測る。
3. projectが明示する`num_ctx`でcanonical 24-image Scene Catalog requestと3-image Candidate Annotation requestを実行する。
4. request中の`/api/ps`から`size_vram`、`context_length`、processor splitを保存する。
5. 100% GPUを推奨条件にし、CPU offloadを許す場合は別profileとしてlatency上限を明示する。
6. prompt/eval token、load/prompt/eval duration、最大GPU memory、schema/domain validation結果をmodel確認日と一緒に残す。

32 GB targetでQwen3-VL 8B Q4_K_Mのartifactは6.1 GBであり、30B/32B Q4_K_Mの20–21 GBより大きな余白を持つ。この比較は候補の優先順位には使えるが、正確なVRAM保証には使わない。

## version下限と互換性

### Ollama

Qwen3-VL公式pageのvendor floorは0.12.7である。[Qwen3-VL model page](https://ollama.com/library/qwen3-vl:8b-instruct)

一方、Ollama 0.31.2 releaseは「thinkingを無効にしたときのstructured output」を修正したと明記している。Issue #165 contractは`/api/chat`、JSON Schema、`stream=false`、`think=false`を同時に使うため、project floorは0.31.2が妥当である。[Ollama v0.31.2 release](https://github.com/ollama/ollama/releases/tag/v0.31.2)

Ollama APIは厳密にはversionedではないが、backward compatibleを期待すると説明されている。このためsemver比較だけでなく、`/api/version`、`/api/show`、canonical structured-output probeをpreflightに含める。[Ollama API introduction](https://docs.ollama.com/api/introduction) [Get version API](https://docs.ollama.com/api-reference/get-version)

### FFmpeg / ffprobe

targetのUbuntu 24.04 packageは`ffmpeg`と`ffprobe`を同じbinary packageで配布する。[Ubuntu Noble ffmpeg package](https://launchpad.net/ubuntu/noble/%2Bpackage/ffmpeg)

project floorは、Issue #166でAV1 + AAC実動画、subtitle packet PTS、audio decode/resampleを検証した6.1.1とする。distro suffixを許し、targetの`6.1.1-3ubuntu5`は合格する。FFmpeg 6.1 branchにはより新しいpatch releaseがあるため、運用ではOSのsecurity updateを適用しつつ、最低値とcapability testを分ける。[FFmpeg download](https://ffmpeg.org/download.html)

最低versionだけでは不十分である。`ffprobe`はJSON writer、`-show_streams`、`-show_packets`、stream selectorを提供し、`ffmpeg`は`-xerror`で処理errorをfatalにできる。preflightはこれらのoptionと対象codecを実行して確認する。[ffprobe documentation](https://ffmpeg.org/ffprobe.html) [ffmpeg documentation](https://ffmpeg.org/ffmpeg.html)

### faster-whisper / CTranslate2

faster-whisper 1.2.1のpackage metadataはPython 3.9以上、`ctranslate2>=4.0,<5`、PyAV 11以上を宣言する。[faster-whisper 1.2.1 metadata](https://pypi.org/project/faster-whisper/1.2.1/)

このprojectはCTranslate2の広いrangeをそのまま採用せず、4.8.1を最低versionにする。4.8.1 releaseはWhisper `align()`がframeのないwindowを受け取った場合のprocess-killing division by zeroを修正しており、Issue #166のword timestamp contractに直接関係する。[CTranslate2 v4.8.1 release](https://github.com/OpenNMT/CTranslate2/releases/tag/v4.8.1)

faster-whisperの公式READMEは、現行GPU executionにCUDA 12向けcuBLASとCUDA 12向けcuDNN 9を要求する。CUDA 11 + cuDNN 8ならCTranslate2 3.24.0、CUDA 12 + cuDNN 8なら4.4.0へdowngradeする案も記載するが、どちらもこのprojectのCTranslate2 4.8.1 contractとは一致しない。[faster-whisper requirements](https://github.com/SYSTRAN/faster-whisper#requirements)

CTranslate2 4.8.1のinstallation pageはGPU wheelにCUDA 12.x、speech recognitionのconvolution層にcuDNNを要求する。[CTranslate2 installation](https://opennmt.net/CTranslate2/installation.html)

従ってIssue #166で実測済みの次を組み込み既定profileとし、解決された実値を一組としてfingerprintする。

| component | v1 contract |
|---|---|
| Python | project contract `>=3.13`。targetにはuv管理の3.13.12あり |
| faster-whisper | 最低`1.2.1` |
| CTranslate2 | 最低`4.8.1` |
| model | 既定`dropbox-dash/faster-whisper-large-v3-turbo`。TOMLにはmodel名だけを置き、実行時に解決したrevisionをfingerprintする |
| compute | 既定CUDA / float16、TOMLで変更可能 |
| user-space CUDA libraries | `nvidia-cublas-cu12 12.9.2.10`、`nvidia-cudnn-cu12 9.24.0.43`（実測lock値） |

model・解決されたrevision・device・compute type・beam・VAD・chunk・overlapまたはdependencyを変更した場合は、STT Stage Fingerprintを変える。word timestamp、Video Time mapping、overlap所有規則、reliability gateは固定contractとし、設定変更で外さない。runtimeをupgradeする場合は最低versionを満たすだけでcache互換とせず、Issue #166のfixtureを再実行する。

## TOMLによるmodel更新policy

### 更新確認で使える公式interface

Ollamaの`GET /api/tags`が返すdigestはlocal storeに現在installされたtagのmanifest digestである。一方、Ollama 0.31.2のdocumented API / CLIには、remote tagの現行digestだけを返すendpointや`pull --dry-run`はない。`POST /api/pull`は公式documentation上もmodelのdownload operationであり、read-onlyな更新確認として扱わない。[Ollama List models API](https://docs.ollama.com/api/tags) [Ollama Pull API](https://docs.ollama.com/api/pull)

0.31.2の公式sourceでは、既存modelに`pull`してもremote manifestを取得し、content-addressed blobがlocalにあれば再利用、なければdownloadし、manifestを書き換え、不要になった旧layerをpruneする。従って既存tagへの`pull`は「差分を再利用する同期 / upgrade」であり、実際に変更がなくてもmutating operationである。[Ollama 0.31.2 `PullModel`](https://github.com/ollama/ollama/blob/v0.31.2/server/images.go#L962-L1073) [Ollama 0.31.2 manifest fetch](https://github.com/ollama/ollama/blob/v0.31.2/server/images.go#L1221-L1243) [Ollama 0.31.2 blob download/cache hit](https://github.com/ollama/ollama/blob/v0.31.2/server/download.go#L467-L508)

2026-07-14時点の公式registryは`HEAD https://registry.ollama.ai/v2/library/qwen3-vl/manifests/8b-instruct`に対し、bodyなしで`ollama-content-digest: 0533...6319`を返した。また、同URLの893-byte manifestを`GET`してSHA-256を計算するとlocal digestと一致した。これによりblobをdownloadせずread-only比較は技術的には可能である。しかし、このregistry URLとcustom headerはOllama APIの公開contractに文書化されておらず、Ollama client自身もmanifestの`GET`と認証処理を内部実装する。OCI Distribution Specificationの`HEAD` contractが要求するheaderも`Docker-Content-Digest`であり、観測したcustom headerとは異なる。そのためv1実装はこのundocumented surfaceへ依存しない。[Ollama official registry manifest](https://registry.ollama.ai/v2/library/qwen3-vl/manifests/8b-instruct) [OCI Distribution Specification](https://github.com/opencontainers/distribution-spec/blob/main/spec.md#checking-if-content-exists-in-the-registry)

Hugging Faceは公式`huggingface_hub.HfApi.model_info(repo_id, revision=...)`が`ModelInfo.sha`としてそのrevisionのcommit SHAを返すため、model artifactをdownloadせずremote identityを取得できる。branchやtagは変動selector、40文字のcommit SHAはimmutable identityとして分ける。`RepositoryNotFoundError`は誤ったrepo IDとprivate repoの認証不足を区別できない場合があり、存在しないrevisionは`RevisionNotFoundError`になる。`HF_HUB_OFFLINE=1`時のHTTP requestは`OfflineModeIsEnabled`になる。[HfApi `model_info`](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.model_info) [Hugging Face HTTP/offline errors](https://huggingface.co/docs/huggingface_hub/en/package_reference/utilities#http-errors)

2026-07-14のmetadata-only確認では、旧repo ID `mobiuslabsgmbh/faster-whisper-large-v3-turbo`はcanonical `dropbox-dash/faster-whisper-large-v3-turbo`へredirectされ、`main`のSHAはIssue #166で検証したsnapshot `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`と同じだった。既定configにはcanonical repo IDを使い、requested repo IDと`ModelInfo.id`、解決済みfull SHAをprovenanceへ保存する。[Configured model metadata](https://huggingface.co/api/models/mobiuslabsgmbh/faster-whisper-large-v3-turbo?expand=sha) [Verified revision metadata](https://huggingface.co/api/models/mobiuslabsgmbh/faster-whisper-large-v3-turbo/revision/0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf?expand=sha)

Hugging Face cacheは`refs/<branch-or-tag>`に既知のcommit SHA、`snapshots/<commit>`にそのrevisionのfile treeを保持し、新しいrevisionをdownloadしても古いsnapshotを自動削除しない。`snapshot_download(revision=<resolved SHA>)`は既存blobを再利用し、必要なfileだけを取得できる。ただしprojectがlockする`huggingface_hub 0.36.2`の`local_files_only=True`は、snapshot directoryがあることを確認できても全fileの完全性まで確認できない。実行前のCTranslate2 model loadまで成功して初めて「利用可能なlocal model」とする。[Hugging Face cache layout](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache) [Hugging Face `snapshot_download`](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download#huggingface_hub.snapshot_download) [`snapshot_download` 0.36.2 source](https://github.com/huggingface/huggingface_hub/blob/v0.36.2/src/huggingface_hub/_snapshot_download.py)

### `auto_upgrade`の正確な意味

TOMLは全model共通の`[models] auto_upgrade = true | false`を持ち、既定値は`true`とする。model digestやsnapshot revisionはTOMLへ書かず、実行時に解決してStage Fingerprint、cache manifest、`report.json`へ保存する。

| `auto_upgrade` | Ollama tag store | Hugging Face snapshot store |
|---|---|---|
| `false` | exact configured tagが`/api/tags`にあればregistryへ問い合わせず、現在のfull local digestを実行identityにする。missingなら`/api/pull`でbootstrapする | configured repoのlocal `refs/main`が指すsnapshotを実行identityにする。missingならremote `main`を解決してbootstrapする |
| `true`（既定） | stage開始前にdistinctなconfigured tagごとに`POST /api/pull`を完了し、`/api/tags`からpost-pull full digestを再取得する | `model_info(..., revision="main")`でremote SHAを一度解決し、そのimmutable SHAを`snapshot_download(revision=<SHA>)`で取得する |

`auto_upgrade=true`でoffline、timeout、registry / Hub障害、token不在、gated / scope不足、repo不在が起きた場合は、完全で実際にload可能なlocal modelがあれば`update_status=unavailable`のwarningを記録してそのidentityで継続する。利用可能なlocal modelがなければfatalにする。partial downloadは実行identityにせず、`auto_upgrade=false`でもlocal modelがなくbootstrapに失敗した場合はfatalにする。Hugging Faceの`GatedRepoError`は`RepositoryNotFoundError`のsubclassなので先にcatchし、`model_info`にはfinite timeoutを明示する。[Hugging Face 0.36.2 error mapping](https://github.com/huggingface/huggingface_hub/blob/v0.36.2/src/huggingface_hub/utils/_http.py#L349-L476)

Hugging Faceの変動する`main`は更新先のselectorにだけ使い、このrunの実行identityには解決後のfull commit SHAを使う。新snapshotを取得しても旧snapshotは削除しない。Ollamaで同じtagがScene CatalogとCandidate Annotationに設定されている場合は、1回だけpullして両operationへ同じ解決済みdigestをfreezeする。

### cache fingerprintの解決順序

1. TOMLのstore kind、configured model名、`auto_upgrade`、認証に秘密値でないendpoint identityをvalidateする。
2. local storeを先に解決し、Ollamaならfull manifest digest、Hugging Faceならcached commit SHAを`local_identity_before_update`として記録する。
3. missingならinstallする。既存なら`auto_upgrade=false`でno network、`true`でupgradeを試みる。upgrade不能でも手順2のlocal artifactが完全でload可能ならwarning付きで継続する。
4. 更新処理後、またはbest-effort継続時のlocal identityを再解決し、Ollama vision / context / structured-output probeまたはCTranslate2 model loadで利用可能性を検証する。
5. このrunが実際に使う`resolved_execution_identity`をfreezeする。stage開始後にmodelを更新・再解決しない。
6. Stage Fingerprintはconfigured repo / tag、resolved full digest / commit SHA、runtime version、operation別model optionsから作る。update時刻や`auto_upgrade`自体はrun diagnosticsに保存するが、解決された実行modelが同じならsemantic Stage Fingerprintを変えない。

`configured model`、`local identity before update`、`resolved execution identity`を一つの`model_version`欄に混ぜない。この分離により、pullがno-opならcache hitを保ち、upgradeで実際にmodelが変わった場合だけdownstream stageを再計算できる。

## RTX 5090 / WSL2 / Windows Ollama

### 公式要件

- RTX 5090はcompute capability 12.0である。[NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda/gpus)
- OllamaはNVIDIA GPUにcompute capability 5.0以上とdriver 531以上を要求し、RTX 5090を明示的に掲載する。[Ollama hardware support](https://docs.ollama.com/gpu)
- Windows版Ollamaの現在のsystem requirementはWindows 10 22H2以降とNVIDIA driver 551.61以降で、APIは通常`http://localhost:11434`に提供される。[Ollama for Windows](https://docs.ollama.com/windows)
- NVIDIAはWSL2でWindows側のNVIDIA driverだけを導入し、WSL内へLinux display driverを入れないよう指示する。WSL kernelは`wsl --update`で更新する。[CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)
- WSL2の標準NATではLinuxからWindows serverへWindows host IPで接続する。Windows 11 22H2以降のmirrored networkingでは`127.0.0.1`で接続できる。[Microsoft WSL networking](https://learn.microsoft.com/en-us/windows/wsl/networking)
- Ollama serverは既定で127.0.0.1:11434へbindし、`OLLAMA_HOST`で変更できる。remote IPへbindする場合はLAN exposureとfirewallを別途考慮する。[Ollama FAQ](https://docs.ollama.com/faq)

### runtime配置の選択肢

対象機には2つのOllama実行面がある。

1. Windows native Ollama 0.31.2
   - WSL2の現在のgatewayからAPIへ到達可能。
   - `qwen3-vl:8b-instruct`と検証済みdigestがinstall済み。
   - Issue #165の能力検証に使われた。
2. WSL2内のLinux Ollama client 0.30.2
   - systemd serviceはinactive / disabled。
   - service model storeにmanifestがない。
   - 0.31.2のproject floorを満たさない。

現状を最小変更で運用するならWindows native Ollamaを明示的に選ぶのが合理的である。WSL側serviceを使う設計へ変更する場合は、server upgrade、service enable/start、model install、model digest確認が別のsetup作業になる。実行中にWindows/WSLを自動探索して黙って切り替えない。

Windows host IPはWSL restartで変わり得るため、観測した`172.20.32.1`を既定値へ固定しない。configで明示hostを受け取るか、documentedなWSL networking modeに基づく解決を行い、最終的には`/api/version`で到達性を検証する。[Microsoft WSL networking](https://learn.microsoft.com/en-us/windows/wsl/networking) [Get version API](https://docs.ollama.com/api-reference/get-version)

現在のWindows user環境は`OLLAMA_HOST=0.0.0.0`で、port 11434はall-interface listenerになっている。これによりNAT modeのWSLからgateway経由で到達できる一方、MicrosoftとOllamaのdocumentationが注意するnetwork exposureも生じる。reference runtimeとして採用する場合、READMEは「WSLから届くこと」だけでなくWindows Firewall / network profileの確認を案内し、可能ならmirrored networking + localhostまたは明示的に制限したbindを選択肢にする。[Microsoft WSL networking](https://learn.microsoft.com/en-us/windows/wsl/networking) [Ollama FAQ](https://docs.ollama.com/faq)

## 対象機の読み取り専用snapshot

2026-07-14に`ssh winpc`から確認した。環境変更、service起動、model pull、inferenceは行っていない。

| 項目 | 観測値 |
|---|---|
| Windows | build `10.0.26200.8655` |
| WSL | `2.7.10.0` |
| Ubuntu | `24.04.4 LTS (Noble Numbat)` |
| kernel | `6.18.33.2-microsoft-standard-WSL2` |
| CPU | Intel Core i7-13700KF、24 logical CPU |
| GPU | NVIDIA GeForce RTX 5090、compute capability 12.0 |
| NVIDIA driver | Windows KMD `610.62`、NVIDIA-SMI `610.43.02` |
| GPU memory | 32,607 MiB total、2,738 MiB used at observation |
| WSL memory | 32,792,232 kB total、8 GiB swap。`.wslconfig`なし |
| FFmpeg | `/usr/bin/ffmpeg` `6.1.1-3ubuntu5` |
| ffprobe | `/usr/bin/ffprobe` `6.1.1-3ubuntu5`、同一library versions |
| system Python | 3.12.3 |
| uv-managed Python | 3.13.12 installed |
| uv | 0.11.18 |
| WSL Ollama | client 0.30.2、service inactive / disabled、modelなし |
| Windows Ollama API | gatewayから到達、server 0.31.2 |
| Windows Ollama bind | user `OLLAMA_HOST=0.0.0.0`、listener `::`:11434 |
| installed vision model | `qwen3-vl:8b-instruct`、8.8B、Q4_K_M、6,140,415,975 bytes、256K model context |
| installed model digest | `0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319` |
| loaded Ollama models | `/api/ps`は空。調査時にmodelをloadしなかった |

`nvidia-smi`はnon-interactive SSHのPATHでは見つからず、`/usr/lib/wsl/lib/nvidia-smi`なら実行できた。`uv`もnon-interactive PATHにはなく、`/home/mizu/.local/bin/uv`なら実行できた。login shellでは両方が見つかる。production preflightは`nvidia-smi`のPATHだけをGPU可否の正本にせず、CTranslate2のCUDA初期化とOllama APIのprocessor情報を使う。

WSL VMの実memoryは約31.3 GiBであり、Windows hostの64 GBをそのまま利用可能RAMとして表示しない。Microsoftは`.wslconfig`未指定時のmemoryをWindows totalの50%、swapを25%と文書化しているが、運用診断は`/proc/meminfo`の実測値を保存する。[Advanced settings configuration in WSL](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)

## preflightで分離すべきfailure

### 外部tool

1. `ffmpeg_not_found`
2. `ffprobe_not_found`
3. `unsupported_ffmpeg_version`
4. `ffmpeg_ffprobe_version_mismatch`
5. `missing_required_demuxer_or_decoder`
6. `stt_runtime_version_mismatch`
7. `stt_model_revision_unavailable`
8. `cuda_initialization_failed`

### Ollama

1. `ollama_host_unreachable`: TCP / HTTPへ到達できない
2. `unsupported_ollama_version`: `/api/version`が0.31.2未満
3. `ollama_model_missing`: configured exact tagが`/api/tags`にない
4. `ollama_model_resolution_failed`: configured tagの完全digestをlocal storeから解決できない
5. `ollama_vision_capability_missing`: `/api/show`に`vision`がない
6. `ollama_context_insufficient`: canonical requestに必要な明示`num_ctx`を確保できない
7. `ollama_gpu_offload_unacceptable`: canonical probeが設定したprocessor / latency profileを満たさない
8. `ollama_structured_output_probe_failed`: JSON Schemaまたはlocal domain validationに失敗

binaryの存在、server到達性、model存在、vision capability、GPU load、structured-output成立性を一つの`ollama_unavailable`へ畳み込まない。各failureには観測version、hostのscrubbed表示、model tag/digest、remediationを対応させる。

## model採用probeと今後の比較条件

既定8B modelについて次の限定的なinference probeを対象機で完了した。

1. `qwen3-vl:8b-instruct` Q4_K_Mで最大24枚のScene Catalog schemaが3回成立した。
2. Scene Catalogは32K context、Candidate AnnotationはIssue #165の3画像caseとContext Cueなしcaseで成立した。
3. request中の`/api/ps`で`size_vram`と`context_length`を採取し、100% GPU loadを確認した。

8B Q8_0または30B A3B Q4_K_Mを代替として比較する場合は、同一fixtureで意味品質の改善がlatency / VRAM増加に見合うか確認する。これらは未導入であり、今回pullや推論を行っていない。

30B/32BやGemma 4を「RTX 5090だから動くはず」という理由だけで推奨modelにしない。

## Issue #169へ渡す推奨判断

- Ollama project floor: `>=0.31.2`。
- reference runtime: Windows native OllamaへWSL2から接続。WSL local Ollamaとの自動切替なし。
- initial model default: `qwen3-vl:8b-instruct`。TOMLへdigestをpinせず、実行時に解決した完全digestとquantizationをStage Fingerprintへ保存。
- Scene CatalogとCandidate Annotationは初期状態では同一modelを使えるが、設定とfingerprintはoperationごとに保持する。
- model変更は明示config変更とし、runtime fallbackにしない。
- reference hardware: RTX 5090 32 GB、serial request。VRAM必要量はartifact容量でなくcanonical probeの`size_vram`で公開する。
- FFmpeg / ffprobe floor: upstream `6.1.1`、同build、必要codec / option capability probe必須。
- STT: faster-whisper 1.2.1 / CTranslate2 4.8.1を最低versionにする。検証済みmodel名 / CUDA float16を既定値とし、model名・device・compute type・beam・VAD・chunk・overlapはTOMLで変更可能にする。解決済みrevisionは設定ではなくprovenanceへ保存する。
- preflightはbinary、server、model、capability、GPU、schemaを別々に検査し、別reason codeとremediationを出す。
- model recommendationには用途、tag、quantization、artifact容量、実測VRAM、代替、確認日を必須項目にする。

## 公式一次資料

### Ollama

- [Qwen3-VL model library](https://ollama.com/library/qwen3-vl)
- [Qwen3-VL tags](https://ollama.com/library/qwen3-vl/tags)
- [Gemma 4 tags](https://ollama.com/library/gemma4/tags)
- [Vision](https://docs.ollama.com/capabilities/vision)
- [Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Context length](https://docs.ollama.com/context-length)
- [Show model details API](https://docs.ollama.com/api-reference/show-model-details)
- [List models API](https://docs.ollama.com/api/tags)
- [List running models API](https://docs.ollama.com/api/ps)
- [Get version API](https://docs.ollama.com/api-reference/get-version)
- [Ollama FAQ](https://docs.ollama.com/faq)
- [Ollama v0.31.2 release](https://github.com/ollama/ollama/releases/tag/v0.31.2)

### GPU / Windows / WSL

- [RTX 5090 official specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
- [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda/gpus)
- [Ollama hardware support](https://docs.ollama.com/gpu)
- [Ollama for Windows](https://docs.ollama.com/windows)
- [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [Microsoft WSL networking](https://learn.microsoft.com/en-us/windows/wsl/networking)
- [Advanced settings configuration in WSL](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)

### FFmpeg / STT

- [FFmpeg download](https://ffmpeg.org/download.html)
- [ffmpeg documentation](https://ffmpeg.org/ffmpeg.html)
- [ffprobe documentation](https://ffmpeg.org/ffprobe.html)
- [Ubuntu Noble ffmpeg package](https://launchpad.net/ubuntu/noble/%2Bpackage/ffmpeg)
- [faster-whisper v1.2.1 release](https://github.com/SYSTRAN/faster-whisper/releases/tag/v1.2.1)
- [faster-whisper package](https://pypi.org/project/faster-whisper/1.2.1/)
- [CTranslate2 v4.8.1 release](https://github.com/OpenNMT/CTranslate2/releases/tag/v4.8.1)
- [CTranslate2 4.8.1 installation](https://opennmt.net/CTranslate2/installation.html)
