# Rust 移行用の固定 fixture

この directory は Python 現行実装の比較基準です。秘密情報、実ゲームの録画、
実モデル応答を含みません。通常のテストは期待値を生成せず、保存済みの JSON、
画像、動画、cache を読みます。契約全体は
[rust-migration-contract.md](../../../docs/rust-migration-contract.md) を参照してください。

## 数値・wire・Game Context

- `images/` と `image_expectations.json`: 16 個の公開合成 pixel recipe と計算済み
  brightness / contrast / Laplacian variance / entropy / quality / 64-bit dHash。
  暗転、白飛び、低コントラストの境界、gray step / stripe、RGB edge pattern、
  dHash の向き・bit 順を固定します。9 × 8 gray pixel rows の
  `0a55555555555555` は、実際の候補 JSON writer が先頭ゼロを含む16桁の
  lowercase hex を維持することも検証します。採否・dHash は厳密一致、raw float は
  `abs <= 1e-6` または `rel <= 1e-8` です。
- `wire_expectations.json`: cache の canonical UTF-8 JSON / float 表現、digest、
  phase key、stable frame ID の固定 vector。旧 cache の再利用には wire の
  完全一致が必要です。
- `game_context_expectations.json`: Game Context の固定値と 6 つの固定 HTTP 応答。
  見出しの欠落・曖昧さ・全角 colon を拒否し、trim 後の CRLF を保ちます。
  日本語と非 BMP emoji を含む trim 後の2,400 Unicode codepoints
  （UTF-8 8,347 bytes）を受理し、2,401（8,351 bytes）を拒否します。
  長さはUTF-8 bytesやUTF-16 code unitsでは数えません。
- `selection_expectations.json`: 既存の最終選択2 vectorを保持し、7候補から
  output_count 2 × production multiplier 3 = 6件を二次評価へ送る別の2 vectorを保存。
  実際のcallerの件数計算・diversity選定・時刻sortを通した候補 ID / 順序を照合します。
  `Straße` と `STRASSE`、`探索２` と `探索-２` は同じ集計 bucketになります。
  どちらも同 bucket の高得点候補を抑えて別 scene の候補を選び、`lower`だけの
  変換やASCIIだけの抽出では異なる候補が選ばれるnegative controlを備えます。
  二次 pool だけ誤正規化した場合も、最終選択が正常なまま検出します。

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
RGB/gray の色変換・閾値計算は 16 個の固定 PNG で別途検証します。

`pipeline/responses/` は Ollama / vLLM の HTTP response JSON を固定します。
HTTP 境界以外は production の FFmpeg、ffprobe、画像計算、選定、cache、出力処理を
使用します。Ollama の `/api/ps` 応答も fixture で、CPU 許可設定です。
GPU、モデル download、外部 network は不要です。
Cold run の推論 sequence は `sampled_frames` が primary → secondary、
`semantic_video` が video → primary → secondary で、各 stage を厳密に 1 回ずつ
呼びます。重複した同一要求が同じ report を返しても不合格にします。
モデル metadata の HTTP call 数は固定しません。warm / resume の再利用条件も
既存の検証を維持します。

`pipeline/inference-media/` は実際に AI へ送った primary / secondary JPEG、
動画 MP4、decode 済み reference PNG、媒体 bytes を marker に置換した HTTP request
を保存します。通常の `FixtureHttp` は prompt・schema・表示 ID・media options を
厳密照合します。sheet の label は後述の実 bitmap による意味 / 位置確認を行ってから
glyph 領域を除外し、thumbnail・padding・候補の順・before/selected/after の順を
既宣言の RGB gate と dHash で比較します。動画は codec・寸法・fps・duration・全 frame の
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
出力 JPEG の寸法を保存した golden です。`reference-images/` は選定した 2 画像と
最終 `selected-contact-sheet.jpg` を decode した固定 RGB PNG です。sheet 全体の
rank / timestamp / source label、画像の並び、padding も比較します。blank、
thumbnail の並び替え、誤った rank / time label を、自己 hash / size を整合させた
実 pipeline 出力へ適用する negative test と、JPEG 再 encode の positive control を
用意しています。AI 入力と最終 sheet に同じ文字検証を適用します。
`baseline-provenance.json` に Python / dependency /
FFmpeg / 元 revision と各 fixture の SHA-256 を記録しています。

`sheet_contract.py` の label / cell geometry は6つの保存 PNG から独立に転記した
期待値です。実画像の占有 label strip を、Pillow 同梱 Aileron Regular 10px または
bitmap default の完全な文字 / 背景 template と既存 RGB 閾値で照合します。
[sheet-label-profiles.json](sheet-label-profiles.json) に各 strip の RGB pixel SHA-256、
label / geometry と監査した環境を固定保存し、そのhashに一致するtemplateだけを
使います。実行時のfontやrasterizerがactualとtemplateを同時に変えても、未承認の
描画を合格にしません。通常testでprofile hashを再生成・更新しません。
rank / Frame Display ID、動画名、時刻、context legend、x/y位置と黒い余白を検証後、
既知の32px / 42px stripだけを比較の双方で黒くします。thumbnail の先頭 pixel row
と未使用 cell は除外しません。request / report の文字や提出 font 名だけでは
合格にしません。未登録 font / rasterizer は拒否し、同じ意味・配置を保つ profile の
独立確認、positive / negative controls、Human Review を行ってから別途追加します。
正しい別 font の JPEG再encodeは合格し、blank・誤ID/rank/time/source・文字順・1pxの
位置差・context legend・label余白・thumbnail境界・未使用cellの改変は不合格です。

`run_manifest` は保存した Run Manifest 全体、`run_manifest_schema` は全 field の
nested type の固定 snapshot です。`run_identity`、全 models / provider metadata、
candidate multipliers、batch sizes、context offset、model options、GPU 条件と
自己 digest を含め、欠落・追加・型・値を厳密比較します。現 fixture の manifest は
relative input path と固定モデル条件だけで、絶対 root や JPEG bytes 由来の digest
を含まないため、manifest に対する正規化 allowlist は空です。既存 subset の keys も
保持します。manifest / report / completion の digest と artifact integrity を全て
再計算した誤設定でも、出力 report の採否・選択が同じなら合格とすることはありません。

完全 manifest snapshot だけの明示更新には次の command を使います。既存 golden の
他の keys、旧 cache、選定画像、AI 入力媒体、`baseline-provenance.json` は更新しません。
`pipeline/expected/run-manifest-provenance.json` に capture 元 revision（初回
`660a811e`）と両 manifest の JSON digest を残します。snapshot 差分を review した
あとに provenance の inventory を明示更新してください。

```bash
PYTHONPATH=. uv run python tests/fixtures/rust_migration/pipeline/record_baseline.py --reviewed-update --run-manifest-only
```

最終 sheet だけを記録する場合は、次の明示 command を使います。選定画像、
report / cache golden、AI 入力媒体、`baseline-provenance.json` は更新しません。
PNG text の `source_revision` に capture 元の revision（初回は `9713b663`）を残し、
sheet 差分を review したあとに provenance の inventory を明示更新してください。

```bash
PYTHONPATH=. uv run python tests/fixtures/rust_migration/pipeline/record_baseline.py --reviewed-update --selected-contact-sheet-only
```

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

report の `videos[].path` / `selected[].video` は、`video_index` に対応する
実入力の resolved absolute path と一致することを先に確認してから basename へ
置換します。basenameだけや同じbasenameの foreign root は許容しません。
`manifest_digest` と出力 JPEG の
byte size / SHA-256 は report 比較用の marker へ置換します。ただし実際の report と
completion は、各実装が生成した artifact の size / SHA-256 と完全一致することを
別途検証します。全 phase payload digest、assessment payload digest、manifest digest
も保存した中身から再計算して照合します。欠落 field を正規化で消しません。

candidate manifest の各 `image_sha256` とsizeは各実装自身の候補JPEGへ照合し、
固定stored-cache recipeの全候補ID・順序・時刻を確認します。JPEGとmechanical
記録を残してreceiptの欠落・並べ替え・時刻変更を行い、全digestを整合させても拒否します。
さらに両方式の全候補（sampled 6枚／semantic 3枚）を、同じ方式・video identity・
candidate keyの`stored-cache/`内JPEGへID・時刻を対応させてdecode比較します。
採用／棄却、AI未送信、最終未選定を問わず、JPEG形式・寸法・EXIF向き（未指定と1は
同じ向き）・dHashを厳密に確認し、既宣言のchannel MAE ≤ 1.0、最大差 ≤ 16、
PSNR ≥ 40 dBを適用します。方式間で同じframe IDが別時刻を指すため、IDだけで
referenceを共有しません。既存`baseline-provenance.json`でSHA固定されたJPEGを
独立referenceとして使い、提出runから基準を再生成しません。
`test_candidate_pixel_contract.py`は棄却候補を別の正常かつ依然棄却されるJPEGへ替え、
own receipt・phase link・実依存から計算する全assessment keyを整合させても拒否する
回帰を含みます。最終出力やAI応答が同じだけでは、候補抽出の同等性を合格にしません。
mechanical の `source_frames_digest` は同じ動画のcandidate payload digestへ照合します。
secondary-context は必要なreceipt SHAを自身のJPEGへ照合し、保存goldenの
`context_record_names` で必要な before / after のname集合・順序を確認します。
必要集合は変更しないstored-cache recipeから独立に取得します。過去の未使用receiptは
構造・nameの一意性・SHA形式を確認して保持でき、そのJPEGの存在やhashは要求しません。
必要名だけをsnapshotに残し、未使用JPEGが欠損した正常resumeを誤って拒否しません。
このfieldは既存Python stored-cacheのrecordから追加し、旧goldenの全項目と媒体は
維持しました。candidate通常cache hitのsize-only shortcutやcompleted warm runの
無操作をproduction側で変更するものではありません。完了artifactはexpected setに加え
exact record countを検証してvalidな重複も拒否します。
`test_durable_integrity_contract.py` は誤receipt・欠落・誤link・誤raw path・重複を
自己digest/size/hashが整合する状態で拒否します。別encoderのpositiveはcandidate/contextを
productionのreceipt保存前に再圧縮してcold pipeline全体を実行し、全assessment keyを
実際の依存bytesから照合してからgolden比較します。completionを除いて再開しても
推論・probe・candidate/context再抽出なしで同じkeyを再利用することを確認します。
必要contextだけを再抽出する部分再開、completed shortcutでcontextを読まない条件、誤mechanical
linkで機械評価だけmissし旧AI keyを保つ条件も確認します。
completionの同size・正常JPEGのCOM byte改変は
metadata不変のまま拒否し、正規registration経路でもSHA検査が必要なことを証明します。

`assessment_envelopes[*].cache_key` は候補 JPEG bytes / mechanical payload bytes /
context JPEG bytes に依存します。cold / interrupted run の golden 比較時だけ
`<jpeg-content-dependent>` に置換します。元 key は golden と旧 cache に残しており、
保存済み cache replay では厳密一致します。他の key は置換しません。

最終画像と最終 contact sheet は同寸法、decoded RGB の channel MAE ≤ 1.0、
最大差 ≤ 16、PSNR ≥ 40 dB とし、
固定 reference の dHash も厳密一致させます。JPEG file 自体の byte 完全一致は
cross-encoder の条件にしません。sheet は上記の独立した文字 / 位置検証に通った
label stripだけを除外し、それ以外の全pixelを同じgateで比較します。format / 寸法と
自身の完全性の照合も必須です。AI入力sheetのHuman Reviewも維持します。

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
