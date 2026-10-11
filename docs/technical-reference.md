# 設定・抽出・選定

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
通常実行も同じ事前確認を行い、未接続の生成 context・AI 接続・安全な公開を成功扱いしません。

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
PTS 差分、duration、前後 context の計算時刻は FFprobe の microsecond 精度へ丸め、小数の加減算誤差で端点を飛ばさないようにします。
抽出は要求時刻以降の最初の frame、最終端点は最後の frame とし、範囲外・非有限値は拒否します。
FFmpeg の表示回転を適用した full-resolution PNG を抽出し、要求／実時刻を区別します。
前後 context は指定 delta から両端点へ clip し、中央の source/stream と一致させます。
semantic 用の内部 chunk は lossless PNG/MOV、microsecond timebase と `setts` により
元 frame の間隔と末尾の表示時間を保持します。PNG/FFV1 encode・`setts` 対応を含む
外部 FFmpeg 条件は対象環境で検証し、最小対応 version は配布検証で確定します。

FFmpeg/FFprobe は shell を経由せず argv で実行します。媒体は専用の排他的 temporary directory 内へ作り、
失敗・通常中断では一時 file を削除し、所有した子 process を停止・回収します。
`extract` の出力は新規 directory に限定し、失敗時はこの command が作った file だけを片付けます。
cache/output の排他・完成済み画像セットの staging/公開・再開は別途実装します。
SIGINT は130、SIGTERM/HUPも成功にせず停止します。強制 kill や filesystem transaction の保証は含めません。

## 品質・選定・一覧（#340）

`pipeline::run` は probe 済み動画と同期評価 callback を受け取り、専用 temporary directory の
`Artifacts` を返します。一次/二次 callback へは実際の候補 ID・画像・sheet・前後 context を渡します。
候補は最大12枚ずつ抽出して一次評価し、評価後は ID・時刻・品質指標を保持して原画像を削除します。
二次評価と最終 JPEG には必要な元 frame だけを再抽出します。評価用の原画像・sheet・前後 context は
各 callback の実行中だけ存在し、完了後に削除します。成功後は完成した JPEG・一覧・report だけを保持します。
HTTP・応答 retry は #362/#342、cache・再開・所有権・公開は #341 の責務です。
この段階の固定応答テストは外部 API を呼ばず、通常 CLI の完成を意味しません。

等間隔 sampling は両端を含み、1候補なら中央を使います。semantic event は有効区間・代表時刻・重要度・
scene・根拠・source を検証し、代表とその両側の候補を抽出します。event のない入力元も許容します。
両方式とも計画を実 frame へ解決し、重複を除いた後に全入力4096候補の上限を適用します。同じ実 PTS は一度だけ評価します。
抽出した実 PTS に対応する event が複数ある場合は、共通の優先規則で根拠を選びます。
二次評価で不足した場合は、抽出済みの未評価候補を先に試します。それでも不足すると、未抽出の最も広い時間の隙間を
入力元の巡回位置を追補間で引き継ぎ、semantic 方式では event 区間内に限ります。event の並びを重要度・source・区間・代表時刻・scene・根拠で固定します。
初期計画を含む実候補数は4096、評価 batch は12、抽出/decode/job は逐次です。
有効な異なる画像が指定枚数に達しなければ、除外・不足理由を示して停止します。

PNG/JPEG は decode 前に encoded file を64 MiB、各辺を16384、画素を3200万、縦横比を32までに制限します。
16未満の寸法、壊れた画像、symlink/特殊 file も拒否します。JPEG の向き情報を反映します。
512px以内の preview の RGB 輝度平均・標準偏差・隣接勾配を使い、近黒/近白が96%超、平均が2.5%未満/97.5%超、
標準偏差2.5%未満、平均勾配0.2%未満を除外します。遷移と有用性20点未満は評価結果から除外します。
この保守的な機械判定は実ゲーム品質の保証ではなく、#343 の実録画確認で評価します。

32×18 RGB の平均距離3.5%未満を近似重複とします。日本語等の scene を空白・英字大小だけ正規化し、
未知 source、重複/欠落/未知 ID、非有限/範囲外 score、空 scene/根拠を拒否します。
通常進行を最初に選び、通常進行が残る間は特別画面を概ね3分の1（2枚時は1枚）に抑えます。
許容候補から未採用 source、scene、source内の時間3分割を優先し、次に有用性80%＋機械品質20%＋event重要度15%と
見た目の距離で選びます。同点は有用性、source の UTF-8 順、実時刻、要求時刻、ID 順です。
不足時は代表画像の選び直しを試し、通常画面の枚数を減らさず、追加比較・探索訪問を各4096回までに制限します。
探索内では枚数、source・scene・source内の時間帯の分散の順に集合を比較します。
最大枚数を数学的に保証する探索ではありません。最終集合から順位・除外理由・source 分散の説明を再計算します。

候補 sheet は12枚ごとの JPEG。二次評価の before/center/after は候補の実 PTS を中心に抽出し、実時刻付きで表示します。
最終一覧は全選定画像を一度ずつ rank 順で、ID・source・要求/実時刻・scene と表示します。
縦横比を維持して余白を付け、日本語の Noto Sans JP（[OFL](../assets/OFL.txt)）を同梱します。
sheet は200MPの RGB canvas を上限とし、一度に1画像だけ decode/resize します。上限の canvas 自体は600MBです。
通常の枚数では小さい canvas を使います。999枚や長い入力名の一覧は拡大して確認してください。
未収録の Unicode 文字は ASCII code point 表記へ置き換え、元の backslash は二重化して表示を区別します。
元の source・scene は保持し、表示の計測・折り返しにも同じ置換を適用します。
source は255 UTF-16 unitsまで表示し、通常の最終一覧は14pxです。小型の最終一覧が長い名前の展開で
canvas 上限に収まらない場合だけ12・10・8pxの順で再計測します。
不正な rank/context 順序・異常 metadata は読めない一覧のまま成功させません。

Selected JPEG は縮小せず品質95で encode し、番号は全枚数が100以上なら3桁、それ未満なら2桁に揃えます。
report は rank・source・要求/実時刻・寸法・一次/二次評価・event根拠・
除外理由・非秘密の model 条件を記録します。遷移・低有用性等の確定除外と、最終選定に基づく重複・順位理由を分けます。
Game Context は保存可能と確認した caller だけ明示して渡します。

## 検証

`sh tests/check.sh` が format・Clippy・unit/integration と入力素材の確認を行う単一の品質 gate です。
実 FFmpeg の抽出を [独立した入力 recipe](../tests/fixtures/README.md) の全画素・PTS・向きへ照合します。
素材確認の単独実行と明示生成は同 README を参照してください。
固定応答の選定と一覧の pixel/source/PTS/向き検証は Rust integration test で行います。
実 model の品質・AI 能力・配布 target の最小 OS 条件は、それぞれ担当機能の検証後に確定します。
