# 設定と抽出

## 設定

[設定例](../config.example.toml) が現在の全設定です。全 table で未知 key を拒否し、型・必須項目・値域を確認します。
`selection.method` は `sampled_frames`／`semantic_video`、`ai.backend` は `strata`／`vllm`。
`base_url` は HTTP(S) の接続先を指定し、URL 内の認証・query・fragment は拒否します。
`model`、`inference_level`、`timeout_seconds`、`cache_revision` は明示必須です。
推論レベルは `none`／`low`／`medium`／`high` を受け付け、timeout は1〜3600秒とします。
これは要求設定の検証です。backend/model ごとの実対応は共通 client の能力検証で決定します。
空白のみの認証値は設定なしとして扱います。config の非空値が環境変数より優先します。
認証値を診断・Debug・report・cache へ出しません。AI 通信はまだ行いません。

`validate` は設定・枚数・Game Title/Context の排他・入力探索・出力 path を確認します。
通常実行も同じ事前確認を行い、未実装の選定・生成 context・AI 接続を成功扱いしません。

## media

入力の対応拡張子は `.mp4`／`.mkv`／`.mov`／`.webm`（大小文字を区別しない）。
UTF-8 ファイル名順で、直下の通常 file だけを扱います。対象動画の symlink、特殊 file、空集合は拒否します。
映像 stream は attached picture を除いた最小 index を使います。default flag は選択条件にしません。
FFprobe の有効な frame PTS を使い、stream offset・VFR を算術的な fps 換算で推測しません。
stream metadata は1 MiB、frame metadata は64 MiBを上限とし、超過を拒否します。
表示回転は90度単位に対応し、途中の解像度変更・不正な PTS・最終 duration 不明は拒否します。
現在の cold probe は全 frame metadata を読み、抽出は decode 順序で原画像を確定します。
長い録画の probe/反復抽出コスト削減と warm 再利用は後続の cache/選定工程で扱います。
時刻は最初の有効 PTS からの動画内秒数です。
抽出は要求時刻以降の最初の frame、最終端点は最後の frame とし、範囲外・非有限値は拒否します。
FFmpeg の表示回転を適用した full-resolution PNG を抽出し、要求／実時刻を区別します。
前後 context は指定 delta から両端点へ clip し、中央の source/stream と一致させます。

FFmpeg/FFprobe は shell を経由せず argv で実行します。媒体は専用の排他的 temporary directory 内へ作り、
失敗・通常中断では一時 file を削除し、所有した子 process を停止・回収します。
`extract` の出力は新規 directory に限定し、失敗時はこの command が作った file だけを片付けます。
cache/output の排他・完成済み画像セットの staging/公開・再開は別途実装します。
SIGINT は130、SIGTERM/HUPも成功にせず停止します。強制 kill や filesystem transaction の保証は含めません。

## 検証

`sh tests/check.sh` が format・Clippy・unit/integration と入力素材の確認を行う単一の品質 gate です。
実 FFmpeg の抽出を [独立した入力 recipe](../tests/fixtures/README.md) の全画素・PTS・向きへ照合します。
素材確認の単独実行と明示生成は同 README を参照してください。
製品の選定・AI 能力・配布 target の最小 OS 条件は、それぞれ担当機能の検証後に確定します。
