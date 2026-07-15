# Issue #166: 音声・内蔵字幕文脈の取得方式と失敗境界

- 調査日: 2026-07-14
- 対象: Issue #166「音声・内蔵字幕文脈の取得方式と失敗境界を検証する」
- 調査方法: 公式ドキュメント、公式ソースコード、公式モデルカードと、対象機での最小能力検証
- 対象外: 外部字幕、画面内文字の OCR、本番コードの実装、全 STT 候補の網羅的 benchmark

## 結論

1. 音声・内蔵字幕の列挙と抽出には、WSL2 の PATH 上にある system `ffprobe` / `ffmpeg` を使う。起動時に両方の実行と必要 codec を検査し、能力不足は Video Set の処理前に失敗させる。
2. 内蔵字幕と音声は、明示 stream index、設定言語、default disposition、候補の一意性の順に選ぶ。同順位が複数なら最小 index を推測せず `ambiguous_*_stream` で preflight を失敗させる。
3. 一意に選べる non-forced text subtitle は STT より優先する。text subtitle がない場合と forced-only の場合に限り音声を STT へ送り、bitmap subtitle は「字幕なし」へ畳み込まない。
4. 音声は mono 16 kHz signed 16-bit PCM に復号し、10分、5秒 overlap の chunk にする。chunk と cue の時刻は PCM sample index から Video Time へ写像し、overlap 中央で区切った所有区間に cue midpoint が入る側だけを採用する。
5. STT は `faster-whisper 1.2.1` / CTranslate2 4.8.1、`large-v3-turbo`、CUDA float16 を採用する。現行 Ollama 公式 API には音声入力契約がなく、Gemma 4 CLI も実測で遅く誤判定したため、Ollama には timestamp 付きテキストだけを渡す。
6. STT の公開 cue は word timestamp を1.5秒以下の gap でまとめ直す。`avg_logprob >= -0.8` かつ3文字以上を暫定の利用可能条件とし、落とした raw cue は診断には残すが画像評価へ渡さない。
7. 「track 不在」「発話なし」「低信頼」は別々の正常な非 fatal 結果にする。「時刻ずれ」「unsupported selected track」「抽出失敗」「解析失敗」は別の fatal reason code とし、成功分だけを公開したり別経路へ暗黙 fallback したりしない。

以降では、一次資料から直接確認できる事実と、このプロジェクトの採用判断を分けて記す。

## 一次資料から確認できる事実

### 1. FFmpeg / ffprobe の配布と検出

- FFmpeg 公式サイト自身が配布するのはソースコードであり、Windows・Linux の実行ファイルはリンク先のビルドまたは各 OS のパッケージを使う構成である。[FFmpeg Download](https://ffmpeg.org/download.html)
- `ffprobe` はコンテナとストリームの情報を機械可読形式で出力できる。入力を開けない、または認識できない場合は正の終了コードを返す。[ffprobe Documentation](https://ffmpeg.org/ffprobe.html)
- `-show_streams`、`-show_format`、`-show_error`、`-show_packets`、`-show_frames`、JSON writer が公式に用意されている。`-select_streams` では音声 `a`、字幕 `s`、metadata `m:key[:value]`、disposition `disp:...` による選択ができる。[ffprobe Documentation](https://ffmpeg.org/ffprobe.html)
- FFmpeg は既定では一部の復号エラーを許容し得る。`-xerror` はエラー時に停止して終了する指定である。[ffmpeg Documentation](https://ffmpeg.org/ffmpeg.html)

したがって、アプリケーションは PATH 上の存在だけでなく、次の二つを別々に実行して終了コードを検査できる。

```console
ffmpeg -version
ffprobe -version
```

バージョン文字列は診断情報として保存する。`ffmpeg` が見つかって `ffprobe` が見つからない場合も能力不足である。

### 2. 音声・字幕ストリームの metadata と disposition

最低限、各ストリームについて次を保存できる。

- `index`
- `codec_type` / `codec_name`
- `time_base` / `start_pts` / `start_time` / `duration_ts` / `duration`
- `tags.language` / `tags.title`
- `disposition.default` / `disposition.forced`
- 利用できる場合は `hearing_impaired` などその他の disposition

FFmpeg の `-map` は入力ストリームを明示的に選択でき、末尾の `?` を使った optional mapping、metadata language を使った mapping も文書化されている。出力ストリームの disposition は `-disposition` で設定できる。[ffmpeg Documentation](https://ffmpeg.org/ffmpeg.html)

Matroska では `FlagDefault` はプレイヤーの自動選択候補であることを示す。`FlagForced` は、通常字幕を無効にしていても、選択言語に対して表示すべき字幕を示す。`LanguageBCP47` が存在する場合、従来の `Language` より優先される。[Matroska Element Specification](https://www.matroska.org/technical/elements.html)

このため、`default` は「最良の全文字幕」、`forced` は「全文文字起こし」と同義ではない。forced track は異言語発話や画面上の翻訳だけを含む可能性がある。

### 3. 内蔵字幕の種類と抽出

FFmpeg の字幕自動選択は、字幕 encoder が扱える text subtitle と image subtitle を区別する。手動で選んだ字幕と encoder の組み合わせに互換性がない場合、処理全体が中止され得る。[ffmpeg Documentation](https://ffmpeg.org/ffmpeg.html)

FFmpeg の字幕データ構造は、字幕矩形に plain text / ASS text を保持でき、字幕イベントには PTS と display start / end time がある。[AVSubtitleRect](https://ffmpeg.org/doxygen/trunk/structAVSubtitleRect.html) [AVSubtitle](https://ffmpeg.org/doxygen/trunk/structAVSubtitle.html)

従って、内蔵字幕には少なくとも次の三状態がある。

1. text subtitle: 文字列と時刻を抽出可能
2. bitmap subtitle: OCR なしには文字列文脈を取得できない
3. codec / data corruption 等により抽出失敗

2 と 3 を「字幕なし」に畳み込むことはできない。

### 4. timestamp と音声チャンク

FFmpeg の進捗・filter 情報に現れる `pts` は整数で、`tb` はその time base である。時刻は `pts * time_base` として解釈される。[ffmpeg Documentation](https://ffmpeg.org/ffmpeg.html)

`-copyts` は入力 timestamp を処理せず保持する指定だが、出力 muxer や vsync などの処理で timestamp が変わり得ることも明記されている。`-start_at_zero` は `-copyts` と併用して開始時刻をずらす。[ffmpeg Documentation](https://ffmpeg.org/ffmpeg.html)

音声 filter については次が確認できる。[FFmpeg Filters Documentation](https://ffmpeg.org/ffmpeg-filters.html)

- `atrim` は time または sample number で範囲を選べるが、timestamp 自体は変更しない。
- `asetpts=N/SR/TB` はサンプル数から timestamp を生成できる。
- `aresample` の async 処理は timestamp に合わせて音声を伸縮したり、サンプルを注入・削除したりできる。
- `ashowinfo` は PTS、秒換算 PTS、sample rate、sample count を観測できる。

FFmpeg の segment muxer は `segment_time` による切断が正確でない場合があると明記されている。`reset_timestamps=1` も全ての muxer / codec で正しく動く保証はない。[FFmpeg Formats Documentation](https://ffmpeg.org/ffmpeg-formats.html)

したがって、segment muxer のファイル境界を正確な Video Time とみなす根拠はない。また、字幕を SRT / WebVTT に変換した後の表示用 timestamp だけから、元の packet PTS が完全に保存されたとも断定できない。

### 5. Ollama API と Gemma 4 の音声能力

Ollama の公式 OpenAPI 定義と API ドキュメントでは、生成 API は text prompt と任意の base64 `images`、chat API は message content と `images`、embedding API は text input を受け取る。音声 field、音声 upload、transcription endpoint は定義されていない。[Ollama OpenAPI](https://docs.ollama.com/openapi.yaml) [Generate API](https://docs.ollama.com/api/generate) [Chat API](https://docs.ollama.com/api/chat)

Google の Gemma 4 公式モデルカードでは、次の modality が示されている。[Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4)

| Gemma 4 variant | 入力 | 出力 |
|---|---|---|
| E2B / E4B / 12B | text / image / audio | text |
| 26B A4B / 31B | text / image | text |

Gemma 4 の公式 audio guide は、音声対応モデルの入力条件を最長 30 秒、mono 16 kHz、float32 `[-1, 1]` とし、音声を毎秒 25 token に符号化すると説明している。掲載例は Hugging Face Transformers の any-to-any pipeline である。[Gemma Audio Guide](https://ai.google.dev/gemma/docs/capabilities/audio?hl=en)

モデルそのものの音声能力と、Ollama が公開 API でその入力経路を提供していることは別である。2026-07-14 時点の一次資料から、Ollama 経由の音声入力を安定した契約として採用する根拠は得られなかった。

### 6. RTX 5090 / WSL2 の制約

- NVIDIA の公式 GPU table では GeForce RTX 5090 の compute capability は 12.0 である。[CUDA GPUs](https://developer.nvidia.com/cuda/gpus)
- CUDA 12.8 release notes は、Blackwell の SM 120 を compiler library が初めてサポートしたことを示す。[CUDA Toolkit 12.8 Release Notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
- Ollama は RTX 5090 を NVIDIA 対応 GPU として明記し、NVIDIA driver 550.40.07 以降と compute capability 5.0 以降を要件としている。[Ollama GPU Support](https://docs.ollama.com/gpu)
- NVIDIA の WSL guide は Windows 側の NVIDIA driver を使い、WSL 内に Linux display driver を導入しないよう指示している。CUDA toolkit が必要な場合も WSL-Ubuntu 向けまたは `cuda-toolkit-12-x` を使い、`cuda` / `cuda-12-x` / `cuda-drivers` meta-package を避けるよう注意している。[CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- WSL2 には完全な Unified Memory、同時 CPU/GPU access、pinned system memory、NVML query の一部に制約がある。[CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- `.wslconfig` 未設定時の WSL2 VM memory は Windows memory の 50% が既定である。64 GB machine なら既定値は概ね 32 GB になる。[Advanced settings configuration in WSL](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)

RTX 5090 が公式対応一覧にあることは、任意の Python wheel / CUDA runtime / cuDNN / model combination がそのまま動くことを保証しない。各 STT runtime の pinned version で実機検証が必要である。

## 専用 STT 候補の比較

### 比較表

| 候補 | segment timestamp | word timestamp | VAD | CUDA / 配布 | Issue #166 での位置付け |
|---|---|---|---|---|---|
| OpenAI Whisper | あり | cross-attention と DTW による optional 出力 | 専用 VAD なし。`no_speech_prob` 等の heuristic | PyTorch。system FFmpeg が必要 | 参照実装・正確性比較 |
| faster-whisper | あり | optional、word ごとの probability あり | Silero VAD filter あり | CTranslate2。現行 GPU stack は CUDA 12 + cuDNN 9 を要求 | 採用 |
| whisper.cpp | あり | experimental | Silero VAD あり | C/C++ binary。CMake で CUDA build | Python runtime 非依存の代替 backend |
| Kotoba-Whisper v2.0 | あり | model card では未確認 | model card に統合 VAD の記載なし | Transformers / PyTorch、BF16 | 日本語向け challenger |

### OpenAI Whisper

OpenAI Whisper の公式 repository は system `ffmpeg` を要求し、README 上の想定 Python は 3.8–3.11 である。日本語を明示して実行でき、長い音声は 30 秒窓をずらしながら処理する。[OpenAI Whisper README](https://github.com/openai/whisper)

固定 revision の公式 source では、`word_timestamps` は cross-attention pattern と dynamic time warping から word-level timestamp を抽出する option である。結果には text、segments、language が含まれる。[transcribe.py](https://github.com/openai/whisper/blob/04f449b8a437f1bbd3dba5c9f826aca972e7709a/whisper/transcribe.py#L38-L125)

各 segment は start、end、text、temperature、`avg_logprob`、`compression_ratio`、`no_speech_prob` を持つ。[transcribe.py segment fields](https://github.com/openai/whisper/blob/04f449b8a437f1bbd3dba5c9f826aca972e7709a/whisper/transcribe.py#L246-L261) `no_speech_prob` と `avg_logprob` の組み合わせによる silence 判定は heuristic であり、独立した VAD ではない。[transcribe.py silence heuristic](https://github.com/openai/whisper/blob/04f449b8a437f1bbd3dba5c9f826aca972e7709a/whisper/transcribe.py#L298-L310)

この repository の declared Python range と本プロジェクトの Python 3.13 以上にはずれがあるため、同一 environment への直接統合は検証が必要である。また、`avg_logprob` や `no_speech_prob` を校正済みの「信頼度」と呼ぶ根拠はない。

### faster-whisper / CTranslate2

faster-whisper は Python 3.9 以上を要求し、音声 decode には FFmpeg library を同梱する PyAV を使う。現行 README は GPU execution に CUDA 12 の cuBLAS と cuDNN 9 を要求する。[faster-whisper README](https://github.com/SYSTRAN/faster-whisper/blob/ed9a06cd89a93e47838f564998a6c09b655d7f43/README.md#L57-L70)

segment timestamp、word timestamp、Silero VAD filter が公式に用意されている。既定の VAD は 2 秒より長い無音だけを除去し、batched transcription では VAD が既定で有効になる。[faster-whisper README: timestamps and VAD](https://github.com/SYSTRAN/faster-whisper/blob/ed9a06cd89a93e47838f564998a6c09b655d7f43/README.md#L125-L218)

固定 revision の source では、word に start / end / probability、segment に start / end / text / avg_logprob / compression_ratio / no_speech_prob / words、transcription info に detected language probability、duration、duration after VAD がある。[faster-whisper transcription types](https://github.com/SYSTRAN/faster-whisper/blob/ed9a06cd89a93e47838f564998a6c09b655d7f43/faster_whisper/transcribe.py#L31-L108)

VAD 後の duration と speech chunk mapping が source で保持されている。[faster-whisper VAD path](https://github.com/SYSTRAN/faster-whisper/blob/ed9a06cd89a93e47838f564998a6c09b655d7f43/faster_whisper/transcribe.py#L875-L905) CTranslate2 の current wheel は Linux / Windows と CUDA 12.x を対象にする一方、hardware / installation docs の要件表現は faster-whisper README と完全には一致しないため、version を固定して組み合わせ全体を検証すべきである。[CTranslate2 Installation](https://opennmt.net/CTranslate2/installation.html) [CTranslate2 Hardware Support](https://opennmt.net/CTranslate2/hardware_support.html)

### whisper.cpp

whisper.cpp は CMake option `-DGGML_CUDA=1` で CUDA build を提供する。[whisper.cpp README: CUDA](https://github.com/ggml-org/whisper.cpp/blob/080bbbe85230f624f0b52127f1ae1218247989f9/README.md#L316-L331) 公式例にある個別 architecture 値は RTX 5090 の compute capability 12.0 と一致しないため、その値をコピーせず、CUDA 12.8 以上で SM 120 を対象にした build を実機確認する必要がある。

通常の timestamp 出力に加え word-level timestamp を出せるが、公式 README は word-level を experimental と明記する。[whisper.cpp README: timestamps](https://github.com/ggml-org/whisper.cpp/blob/080bbbe85230f624f0b52127f1ae1218247989f9/README.md#L535-L600) Silero VAD model と threshold / speech duration / padding 等の option もある。[whisper.cpp README: VAD](https://github.com/ggml-org/whisper.cpp/blob/080bbbe85230f624f0b52127f1ae1218247989f9/README.md#L780-L856)

### Kotoba-Whisper v2.0

Kotoba Technologies の公式 model card は、Kotoba-Whisper v2.0 を日本語向けに蒸留した 756M parameter の Whisper model として公開している。Transformers 4.39 以降を使う BF16 example、long-form sequential / chunked inference、`return_timestamps=True` の segment timestamp example がある。[Kotoba-Whisper v2.0 Model Card](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0)

同 model card に掲載された自己評価では、Whisper large-v3 と比べて ReazonSpeech held-out の CER は低い一方、Common Voice 8 と JSUT の CER は高い。日本語向けであることだけから、このゲーム動画で常に高精度とは判断できない。

## 確定済みのプロジェクト判断

- audio / subtitle track が存在しないことは正常な不在であり、非 fatal とする。
- audio track は存在するが発話がないこと、および ASR 結果が低信頼であることは非 fatal とする。
- 選択対象の track が存在するのに抽出または解析に失敗した場合は fatal とする。別 track や別 modality へ暗黙 fallback しない。
- 上記の結果を同じ空配列へ畳み込まず、reason code で区別する。

## 採用設計

ここからは一次資料と実機検証を踏まえて Issue #166 で採用する設計であり、上記の公式仕様そのものではない。

### 1. 処理境界

```text
環境 preflight
  -> ffprobe で全 stream を列挙
  -> 一意に選べる non-forced text subtitle があれば抽出
  -> non-forced text subtitle がなければ audio を PCM 化して VAD + STT
       forced-only text subtitle があれば補助 cue として併せて抽出
  -> word timestamp を共通の Context Cue に正規化
  -> scene / candidate の該当時刻周辺だけ Ollama に text として渡す
```

- Ollama を音声 decoder / STT の責務から外す。
- non-forced text subtitle は音声内容の正確性と時刻の再現性が STT より高いため、その track が選べた Video Source では STT を実行しない。
- forced-only subtitle は全文字幕とはみなさず、選択 audio の STT と併用する。双方の provenance を保持し、同時刻の重複 text は後段へ二重加点しない。
- track 不在、発話なし、低信頼は context の正常な非 fatal 結果として扱う。一方、存在する track の抽出・解析失敗は、空の context として visual pipeline へ渡さない。

### 2. ストリーム選択 policy

#### 字幕

1. 明示 stream index があれば、その subtitle stream だけを選び、type と codec を検証する。
2. それ以外は text subtitle を non-forced と forced-only に分け、non-forced を先に選ぶ。
3. 設定言語と container の language tag を primary language で正規化して一致候補を絞る。`ja` と legacy `jpn` は同じ言語として扱う。
4. 一致候補がなければ、language が `und` / 未設定の候補を残す。
5. unique `default=1`、unique remaining candidate の順で一つへ絞る。
6. 同順位が複数なら index 順で推測せず `ambiguous_subtitle_stream` を返す。

指定 index が存在しない、subtitle type でない、選択結果が bitmap subtitle だった場合は、暗黙 fallback せず preflight の fatal error とする。forced-only track は全文性を推測せず補助 source と明示する。

#### 音声

1. 明示 stream index があれば、その audio stream だけを選ぶ。
2. それ以外は設定言語との一致、unique `default=1`、unique remaining candidate の順で一つへ絞る。設定言語と一致する track がなくても、`und` / 未設定が一つだけなら選択して、その設定言語を STT に渡す。
3. 同順位が複数なら index 順で推測せず `ambiguous_audio_stream` を返す。

commentary、descriptive audio、複数言語 mix を metadata だけで完全には判別できないため、選択理由と全候補を診断情報に残す。設定言語がない場合だけ backend の言語検出を使い、chunk ごとの検出結果と確率を保存する。

### 3. 字幕 timestamp

- packet / decoded subtitle event から取得できる場合は、元 stream の `PTS` と `time_base` を保存する。
- 本プロジェクトの Video Time には、既存の原点規則に従って rational arithmetic で写像する。
- SRT / WebVTT 変換後の値しか得られない backend では、millisecond quantization と変換時の offset を provenance に記録する。
- `-copyts` の使用だけを「時刻保持の検証」としない。fixture ごとに input PTS、抽出 cue、Video Time の対応を比較する。

### 4. 音声 decode と chunk timestamp

- 選択 audio を mono 16 kHz signed 16-bit PCM に一度連続 decode する。
- chunk は10分、隣接 chunk との overlap は5秒とし、PCM sample index で切る。segment muxer の切断点を使わない。
- 各 chunk に `source_sample_start`、`sample_count`、`sample_rate` と、元 stream の開始 PTS / time base から計算した `video_time_origin` を持たせる。
- STT が返す秒数は、`chunk_origin + local_sample_index / sample_rate` に写像する。整数 sample index に量子化し、内部で binary float 秒を正本にしない。
- STT timestamp は model による推定値であり、source packet PTS と同じ精度区分にしない。`timestamp_basis=asr_sample_grid_estimate` のように provenance を分ける。
- `aresample=async` を使って sample を注入・削除した場合は補正量を記録する。無記録で timeline を変更しない。
- overlap の中央を隣接 chunk の所有境界とし、global Video Time 上の cue midpoint が自身の半開所有区間に入る chunk の結果だけを採用する。
- backend segment をそのまま Context Cue にしない。word 間 gap が1.5秒を超えた箇所で分割し、各 word group の最初と最後の timestamp を cue の範囲にする。
- STT cue は whitespace と記号を除く文字数が3以上、かつ source segment の `avg_logprob >= -0.8` の場合だけ利用可能とする。条件未満の cue は diagnostics に残すが後段へ渡さない。

### 5. STT backend

- 既定 backend は `faster-whisper 1.2.1` / CTranslate2 4.8.1、model alias `large-v3-turbo`、CUDA float16、beam size 5、Silero VAD、word timestamp 有効、`condition_on_previous_text=False` とする。
- 検証した model artifact は `mobiuslabsgmbh/faster-whisper-large-v3-turbo` の snapshot `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` である。実装では revision を固定し、diagnostics と Stage Fingerprint に含める。
- 対象機で動作した CUDA runtime library は `nvidia-cublas-cu12 12.9.2.10` と `nvidia-cudnn-cu12 9.24.0.43` である。wheel 内 library directory を loader が解決できることも preflight の model load で検査する。
- OpenAI Whisper、Kotoba-Whisper v2.0、whisper.cpp は将来の明示 backend 候補として比較表に残すが、実行中の自動 fallback には使わない。
- Gemma 4 audio は Ollama API 契約の欠落、30秒入力制限、実測結果のため採用しない。

### 6. 統一 cue contract

字幕・STT を共通化する最小単位を次のようにする。

| field | 意味 |
|---|---|
| `source_kind` | `embedded_subtitle` / `speech_to_text` |
| `stream_index` | 元 container stream。STT でも audio stream index を保持 |
| `start` / `end` | rational Video Time |
| `timestamp_basis` | `source_pts` / `container_text_ms` / `asr_sample_grid_estimate` |
| `text` | 正規化し過ぎない元文字列 |
| `language` | metadata または推定言語。由来も保持 |
| `reliability` | `usable` / `low`。校正済み確率とは呼ばない |
| `diagnostics` | backend 固有の avg log probability、no-speech probability、word probability 等 |
| `provenance` | codec、backend / model / revision、device、chunk range、disposition |

Ollama に渡す表示形式と、内部の正確な Video Time / provenance を分離する。

Context Cue の抽出は Video Stage の再利用可能成果物にする。Stage Fingerprint には stream 選択、設定言語、FFmpeg extractor version、STT backend / model revision / compute type、chunk / overlap、VAD、word-group、reliability policy version を含める。PCM chunk は一時データとして処理後に破棄し、Context Cue と診断 manifest だけを cache する。

## 失敗・結果 contract

### 結果 category

結果は実際に選択・試行した `source_kind` ごとに記録する。forced subtitle と STT の一方が成功しても他方が fatal なら Video Stage 全体を失敗させ、成功分だけの Context Cue manifest は publish しない。

| `status` | `reason_code` 例 | 意味 | 動画全体の扱い |
|---|---|---|---|
| `available` | `context_extracted` | usable cue を取得 | 成功 |
| `available` | `context_extracted_with_rejections` | usable cue と policy 未満の raw cue が混在 | usable cue だけを公開し、除外数を診断へ保存 |
| `absent` | `no_audio_stream` | audio stream が存在しない | 非 fatal。空結果 |
| `absent` | `no_subtitle_stream` | subtitle stream が存在しない | 非 fatal。audio 経路を選択可能 |
| `no_speech` | `vad_no_speech` | 音声はあるが VAD で発話なし | 非 fatal。正常な空結果 |
| `no_speech` | `asr_no_speech` | VAD passage はあるが ASR heuristic で発話なし | 非 fatal。正常な空結果、診断を保存 |
| `low_reliability` | `asr_below_policy_threshold` | raw text はあるが usable cue が0件 | 非 fatal。raw text は診断へ隔離し、画像評価へ渡さない |
| `failed` | `ambiguous_audio_stream` / `ambiguous_subtitle_stream` | 自動選択の最上位候補が複数 | fatal。stream 指定を要求 |
| `failed` | `timestamp_drift` | cue text はあるが時刻ずれが許容値超過 | fatal。時刻付き文脈として公開しない |
| `failed` | `chunk_failed` | 一部 chunk の抽出・解析が失敗 | fatal。成功分だけを partial publish しない |
| `unsupported` | `unsupported_bitmap_subtitle` | 選択した内蔵字幕はあるが OCR 対象 | fatal。audio へ暗黙 fallback しない |
| `failed` | `audio_extraction_failed` | 選択 stream の decode / resample が失敗 | fatal。別 track / modality へ fallback しない |
| `failed` | `subtitle_extraction_failed` | 選択 text subtitle の decode / 変換が失敗 | fatal。audio へ fallback しない |
| `failed` | `stt_analysis_failed` | model load 後の inference が失敗 | fatal。空 context として扱わない |

`no_speech` と `low_reliability` は異なる。text が生成されたが品質指標が低い場合に、その結果を「発話なし」へ変換しない。また、bitmap subtitle を `no_subtitle_stream` としない。

### fatal にする境界

feature が有効なのに次の環境能力が満たせない場合は、動画処理を始める前に一度だけ fatal error にする。

- `ffmpeg` または `ffprobe` がない、実行できない、必要な decode 能力がない
- configured STT backend / model revision を読み込めない
- `device=cuda` を明示したのに CUDA device / 必要 library を初期化できない
- model artifact の破損、license / authentication 等により configured backend を利用できない

preflight 後も、存在する track に対する次の失敗は当該実行を fatal error にし、別 track / modality へ fallback しない。

- audio / subtitle の decode、変換、resample 失敗
- STT inference、VAD mapping、timestamp mapping の失敗
- 許容値を超える timestamp drift
- 選択された track の codec が対象外、または明示 stream 指定が不正

timestamp は次を全て満たす必要がある。

- subtitle は packet PTS と time base から rational Video Time へ写像できる。
- STT word は有限で `0 <= start < end <= Video Duration` に収まり、global 時刻を 16 kHz sample index へ丸めて戻した差が1 sample 以下である。
- decoded chunk の観測 origin と、宣言済み resample 補正を含む期待 origin の差が1 output sample 以下である。

これらを満たさない値を clip や推測で救済せず `timestamp_drift` とする。ASR が推定する発話境界そのものは source PTS と同精度とはみなさず、`timestamp_basis` で区別する。

一方、特定動画に audio / subtitle track がないこと、audio はあっても発話がないこと、ASR が低信頼であることは失敗ではない。

### 診断情報

失敗と非 fatal の空・低信頼 result には最低限次を保存する。

- status / reason code / stage
- input video identity
- stream index、codec、language、全 disposition
- source PTS / time base、chunk sample range、期待 offset と観測 offset
- FFmpeg / ffprobe version と終了コード
- STT backend、model id / revision、runtime version、device、compute type
- VAD / ASR threshold と backend が返した raw diagnostic values
- 成功 cue count、失敗 chunk count
- error class と scrubbed message

raw `avg_logprob`、`no_speech_prob`、word probability は backend 固有値として保存し、異なる model 間で同じ意味の「confidence score」として比較しない。

## 実機検証

### 環境

| 項目 | 検証値 |
|---|---|
| OS | WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`、x86_64 |
| GPU | NVIDIA GeForce RTX 5090、driver 610.62、32,607 MiB |
| FFmpeg | `ffmpeg` / `ffprobe` 6.1.1-3ubuntu5 |
| Python | 3.13.12 |
| STT runtime | faster-whisper 1.2.1、CTranslate2 4.8.1、CUDA device count 1、float16 |

非対話 SSH の PATH には `/home/mizu/.local/bin` と `/usr/lib/wsl/lib` が入っておらず、`uv` と `nvidia-smi` は絶対 path なら実行できた。アプリケーションの preflight は `nvidia-smi` の PATH だけに依存せず、CTranslate2 自身の CUDA 初期化で利用可否を判定する。

### 実動画の構成

提供された12本の MP4 は合計 182,426.481秒（50時間40分26.481秒）だった。全て AV1 video と AAC 48 kHz stereo audio を持ち、audio language は `und`、内蔵 subtitle stream は0件だった。このため STT は実動画、subtitle は合成 container fixture で検証した。

### faster-whisper

- 30秒の実動画音声を6地点から mono 16 kHz PCM として抽出した。Python 3.13 環境では model load が1.666秒、各 sample の推論は0.067〜0.508秒だった。
- 発話の多い600秒 sample は4.641秒で処理され、VAD 後は210.592秒、61 segment だった。監視で観測した GPU memory 使用量の最大は5,196 MiBだった。
- 画面表示と照合すると、604.5秒付近の「知り合いの爺さん」は `g さん` になったが、それ以外の文はほぼ一致した。611.5秒、618.5秒、625.5秒付近も表示中の台詞と segment / word timestamp が対応し、10,015秒付近の「この宝箱、何が入ってるんでしょう？」はほぼそのまま得られた。
- 発話なし sample は VAD 後0秒、cue 0件になった。一方、VAD 後0.96秒の雑音 sample は `ん`（0.38秒、`avg_logprob=-0.9844`）を生成した。良好 sample の `avg_logprob` は -0.5099 以上だったため、-0.8 と3文字の gate がこの誤検出を除外した。
- 600秒 sample では、一つの backend segment が82秒の無音 gap をまたいだ。word timestamp を調べると先頭1文字と残りの語の間に82秒の gap があり、segment 範囲を Context Cue に使えないことを確認した。1.5秒 gap での word group 再構成なら分離できる。

### chunk overlap と時刻

実動画の600〜620秒と615〜635秒を5秒 overlap で処理した。前 chunk では境界上の台詞が「その人は怪我をして探しに行けなく」で切れ、後 chunk では「もったいねぇ その人は怪我をして探しに行けなくて困ってるんだ」まで復元された。overlap 中央の617.5秒を所有境界にすると、両 cue の midpoint は後 chunk 側になり、完全な方だけを一意に残せた。

10分 PCM は約19.2 MBであり、5秒 overlap の追加量は約160 kBである。10分単位は対象機で4.641秒の推論時間に収まり、失敗時の再実行粒度としても十分小さい。

### 内蔵字幕 fixture

日本語 non-forced/default と英語 forced の2本の SubRip stream を持つ Matroska fixture を作成した。`ffprobe` は stream index、`jpn` / `eng`、default / forced disposition を区別でき、subtitle packet PTS は2.500秒、5.000秒、7.000秒だった。

同じ stream を FFmpeg で SRT に変換すると2.521秒、5.021秒、7.021秒となった。AAC stream の開始 PTS が -21 ms だったため container 全体の offset が加算された結果であり、変換後 SRT だけを Video Time の正本にできないことを確認した。元 subtitle packet PTS と video の原点を使えば期待時刻を保持できる。

### Ollama / Gemma 4

Windows 側の Ollama 0.31.2 と `gemma4:latest` は CLI 上で audio capability を表示し、30秒 WAV を受理した。しかし600秒地点と同じ日本語会話 sample に75.010秒かかり、結果は `No discernible Japanese speech was detected` だった。同じ sample を faster-whisper は0.484秒でほぼ正しく文字起こしした。

CLI の非公開経路が音声を受け取れることは、公開 API の安定した audio contract、timestamp、STT 品質を満たさない。Ollama/Gemma 4 音声を不採用とする実測根拠になった。

### 実装時の受け入れ fixture

bitmap subtitle、non-zero start PTS、timestamp discontinuity、破損 packet、複数同順位 stream は本番実装の統合 fixture として Issue #170 の受け入れ戦略へ渡す。Issue #166 ではそれぞれの選択・失敗 contract まで確定しており、backend 選定の未決事項にはしない。

## 一次資料一覧

### FFmpeg / subtitle container

- [FFmpeg Download](https://ffmpeg.org/download.html)
- [ffprobe Documentation](https://ffmpeg.org/ffprobe.html)
- [ffmpeg Documentation](https://ffmpeg.org/ffmpeg.html)
- [FFmpeg Filters Documentation](https://ffmpeg.org/ffmpeg-filters.html)
- [FFmpeg Formats Documentation](https://ffmpeg.org/ffmpeg-formats.html)
- [FFmpeg AVSubtitleRect](https://ffmpeg.org/doxygen/trunk/structAVSubtitleRect.html)
- [FFmpeg AVSubtitle](https://ffmpeg.org/doxygen/trunk/structAVSubtitle.html)
- [Matroska Element Specification](https://www.matroska.org/technical/elements.html)

### Ollama / Gemma / GPU / WSL

- [Ollama OpenAPI](https://docs.ollama.com/openapi.yaml)
- [Ollama Generate API](https://docs.ollama.com/api/generate)
- [Ollama Chat API](https://docs.ollama.com/api/chat)
- [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma Audio Guide](https://ai.google.dev/gemma/docs/capabilities/audio?hl=en)
- [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda/gpus)
- [Ollama GPU Support](https://docs.ollama.com/gpu)
- [CUDA Toolkit 12.8 Release Notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
- [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Advanced settings configuration in WSL](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)

### STT

- [OpenAI Whisper](https://github.com/openai/whisper)
- [OpenAI Whisper transcribe.py](https://github.com/openai/whisper/blob/04f449b8a437f1bbd3dba5c9f826aca972e7709a/whisper/transcribe.py)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [faster-whisper transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/ed9a06cd89a93e47838f564998a6c09b655d7f43/faster_whisper/transcribe.py)
- [CTranslate2 Installation](https://opennmt.net/CTranslate2/installation.html)
- [CTranslate2 Hardware Support](https://opennmt.net/CTranslate2/hardware_support.html)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [Kotoba-Whisper v2.0 Model Card](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0)
