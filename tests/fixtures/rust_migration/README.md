# Rust 移行用の固定 fixture

この directory は Python 現行実装の比較基準です。秘密情報、実ゲームの録画、
実モデル応答を含みません。通常のテストは期待値を生成せず、保存済みの JSON、
画像、動画、cache を読みます。契約全体は
[rust-migration-contract.md](../../../docs/rust-migration-contract.md) を参照してください。

## 数値・wire・Game Context

- `images/` と `image_expectations.json`: 15 個の公開合成 pixel recipe と計算済み
  brightness / contrast / Laplacian variance / entropy / quality / 64-bit dHash。
  暗転、白飛び、低コントラストの境界、gray step / stripe、RGB edge pattern、
  dHash の向き・bit 順を固定します。採否・dHash は厳密一致、raw float は
  `abs <= 1e-6` または `rel <= 1e-8` です。
- `wire_expectations.json`: cache の canonical UTF-8 JSON / float 表現、digest、
  phase key、stable frame ID の固定 vector。旧 cache の再利用には wire の
  完全一致が必要です。
- `game_context_expectations.json`: Game Context の固定値と 4 つの固定 HTTP 応答。
  見出しの欠落・曖昧さ・全角 colon を拒否し、trim 後の CRLF を保ちます。

## 両パイプライン

`pipeline/synthetic-game.mkv` は 160 × 96、4 fps、6 秒の full-range YUV444 lossless
FFV1 動画です。neutral chroma、16 × 16 pixel に整列した luma block の決定的な
pattern を 1 秒ずつ切り替え、2–3 秒を黒にしています。
探索・戦闘・会話などの名前は固定 AI fixture の架空の分類で、モデルの認識性能を
評価する素材ではありません。`pipeline/generate_video.py` は pixel recipe と
FFmpeg command を公開しています。Matroska container / encoder の version 差で
再生成時の file bytes や size が変わり得るため、回帰テストは保存済み動画を使います。

初期 RGB pattern の非整列 block は Mac FFmpeg 9.0.2 と Ubuntu FFmpeg 6.1.1 の
JPEG 変換で最大 2 pixel 値の差を生み、同じ decoded pixel 用の数値 gate を適用
できませんでした。素材を YUV・DCT 境界へ整列させ、上記の両 tool で候補画像の
decoded RGB・quality・dHash が完全一致することを確認して、明示的に基準を
更新しました。数値・画像の許容値、採否・時刻・順位の厳密比較は変更していません。
RGB/gray の色変換・閾値計算は 15 個の固定 PNG で別途検証します。

`pipeline/responses/` は Ollama / vLLM の HTTP response JSON を固定します。
HTTP 境界以外は production の FFmpeg、ffprobe、画像計算、選定、cache、出力処理を
使用します。Ollama の `/api/ps` 応答も fixture で、CPU 許可設定です。
GPU、モデル download、外部 network は不要です。

`pipeline/inference-media/` は実際に AI へ送った primary / secondary JPEG、
動画 MP4、decode 済み reference PNG、媒体 bytes を marker に置換した HTTP request
を保存します。通常の `FixtureHttp` は prompt・schema・表示 ID・media options を
厳密照合し、sheet 全体（label・padding・候補の順・before/selected/after の順も含む）
を既宣言の RGB gate と dHash で比較します。今回の Pillow default font は bundled
なので、label 領域は mask しません。動画は codec・寸法・fps・duration・全 frame の
PTS と、順序付きの全 decoded frame を照合します。6 frame の MP4 は PTS
0–5 秒・duration 6 秒、prompt の原動画区間は 0–5.8 秒です。blank、候補の並び替え、
label 欠落、before/after 入れ替え、動画の逆順・PTS 変更・誤区間の negative test と、
JPEG 再 encode だけなら合格する positive control を用意しています。
各 media reference の JSON に記録元 revision を残します。通常テストは生成せず、
媒体だけの明示更新には次の command を使います。

```bash
PYTHONPATH=. uv run python tests/fixtures/rust_migration/pipeline/record_baseline.py --reviewed-update --inference-media-only
```

この command は既存 report / cache golden と `baseline-provenance.json` を更新しません。
媒体変更を review したあとに provenance の inventory を明示更新してください。

動画 metadata は native の `pix_fmt` と明示 `color_range`（未表示なら
`unspecified`）を保存します。FFmpeg 9 の clip は `yuvj420p` / `pc`、FFmpeg 6 は
`yuv420p` / range 未表示になり、decoded RGB は channel MAE 最大約 0.617・最大差 1
でした。意味比較では 8-bit YUV420 layout の別名 `yuvj420p` を `yuv420p` と扱い、
range 表記単独で同一 pixel の意味を決めません。全 decoded frame の既宣言 RGB
gate と dHash は必須、codec・寸法・fps・PTS・duration は厳密一致のままです。
正しい full→limited 変換の positive control は合格し、pixel を変換せず SPS の
range flag だけ変えた negative control は decoded pixel の相違で不合格になります。

| 方式 | 抽出・採否 | 固定した最終選択 |
| --- | --- | --- |
| `sampled_frames` | 0.5–5.5 秒の 6 候補。暗転 2.5 秒を機械評価で除外。一次 AI 評価で 3.5 秒を transition とする | rank 1 = 1.5 秒 / 戦闘、rank 2 = 4.5 秒 / 会話 |
| `semantic_video` | 1 chunk の固定 events / excluded intervals から 1.5、4.5、5.5 秒を抽出。2.5 秒は excluded interval 内 | rank 1 = 1.5 秒 / 戦闘、rank 2 = 4.5 秒 / 会話 |

`pipeline/expected/` は frame ID、時刻、順位、scene、transition、採否、report の
全 field / nested type、phase version / key / payload shape、機械評価、両 AI 評価、
出力 JPEG の寸法を保存した golden です。`reference-images/` は最終 JPEG を
decode した固定 RGB PNG です。`baseline-provenance.json` に Python / dependency /
FFmpeg / 元 revision と各 fixture の SHA-256 を記録しています。

`pipeline/stored-cache/` は実際に現行 Python が保存した旧 cache の bytes です。
run manifest、probe、candidate JPEG / manifest、mechanical envelope、secondary
context JPEG / envelope、primary / secondary assessment、semantic chunk が含まれます。
一時的な絶対 path、output registration / completion、再生成可能な contact sheet は
保存していません。これらの cache は output directory を変えても再利用できます。

`pipeline/cache-replay.json` の recipe を旧 cache のコピーへ適用します。

- `normal`: 固定 cache をそのまま再利用し、AI 通信・候補抽出・機械評価を行わない。
- `partial`: secondary assessment だけ削除し、固定済み primary / context から再開。
- `corrupt`: mechanical cache を wrong-key / broken-digest envelope に置換し、
  機械評価だけ再生成。候補画像・AI 評価は正常 cache から再利用。
- `corrupt-digest`: 正しい key と完全な payload を保ち、digest だけ変更する。
  digest 検証だけで拒否して機械評価を再生成し、他の正常 cache は再利用する。

別の中断テストは cold run の primary batch 保存後、secondary の HTTP 境界で
`KeyboardInterrupt` を送出します。再開後は secondary と最終出力だけ実行します。
完了済み画像を利用者が変更した場合、再実行を拒否して全出力 bytes を保護します。

## 比較時の正規化と除外項目

選択 ID / 時刻 / 順位 / scene / transition、機械的採否、dHash、run / identity /
probe / candidate / semantic key、phase version は厳密一致です。
機械評価の `quality_score` だけ raw float の許容誤差を適用し、report の
`aggregate_score` は現行の小数 2 桁丸めを含め厳密一致します。

report の入力絶対 path は basename へ置換します。`manifest_digest` と出力 JPEG の
byte size / SHA-256 は report 比較用の marker へ置換します。ただし実際の report と
completion は、各実装が生成した artifact の size / SHA-256 と完全一致することを
別途検証します。全 phase payload digest、assessment payload digest、manifest digest
も保存した中身から再計算して照合します。欠落 field を正規化で消しません。

`assessment_envelopes[*].cache_key` は候補 JPEG bytes / mechanical payload bytes /
context JPEG bytes に依存します。cold / interrupted run の golden 比較時だけ
`<jpeg-content-dependent>` に置換します。元 key は golden と旧 cache に残しており、
保存済み cache replay では厳密一致します。他の key は置換しません。

最終画像は同寸法、decoded RGB の MAE ≤ 1.0、最大差 ≤ 16、PSNR ≥ 40 dB とし、
固定 reference の dHash も厳密一致させます。JPEG file 自体の byte 完全一致は
cross-encoder の条件にしません。contact sheet は format / 寸法と自身の完全性を
検証します。font や rasterizer が異なる移植版は契約文書に従い label 位置・内容を
別途比較し、この fixture の寸法だけで合格にしないでください。

## 実行と期待値の更新

FFmpeg と ffprobe が必要です。macOS では `brew install ffmpeg`、Ubuntu の CI では
`sudo apt-get update && sudo apt-get install -y ffmpeg` で用意します。
この新規 pipeline suite は prerequisite が無い場合に失敗します。skip marker は
ありません。既存の optional FFmpeg E2E test の skip 条件は変更していません。

repository root で実行します。

```bash
uv sync --locked
uv run pytest tests/migration -q -rA --junitxml=/tmp/rust-migration.xml
uv run task check
```

CI の証跡には `tests/migration` の JUnit / log を保存し、全ケースが pass、skipped = 0
であることを確認してください。FFmpeg が無い run は互換性 gate の証跡に使えません。

期待値更新は通常テストに含めません。挙動変更を承認したときだけ、Python baseline、
旧 cache、reference images、provenance の差分を独立に review します。

```bash
# 動画の再生成は通常不要。必要なら新しい動画 bytes / identity も review する。
python tests/fixtures/rust_migration/pipeline/generate_video.py /tmp/synthetic-game.mkv

# 固定動画・固定 HTTP 応答から、明示的に Python の基準を再記録する。
PYTHONPATH=. uv run python tests/fixtures/rust_migration/pipeline/record_baseline.py --reviewed-update
```

Rust 実装との比較にはこの保存済み golden と旧 cache を入力し、Rust 側で期待値を
自動生成し直さないでください。実モデル・実 GPU・実録画での運用検証は、この
固定応答による意味的互換性確認とは別の段階で行います。
