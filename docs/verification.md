# 成果物・再開・品質の検証

固定応答と合成動画の検証は、Rust CLI の接続・出典対応・障害回復の証拠である。
実モデルの理解能力、実録画の選定品質、Brave の適用契約や live 接続、本人の画像確認を証明しない。
実モデル・Human Review の証拠は未実施のまま残す。Issue の close やコード完了から実検証完了を推定しない。

## 自動検証

```sh
sh tests/check.sh
```

FFmpeg、FFprobe、jq が必要。CI の同じ quality gate が Rust の固定応答と合成動画 E2E を実行する。
独立した入力 recipe と承認 tuple を使い、通常テストで再生成しない。

| 要件 | 検証 |
| --- | --- |
| 両方式 × 単一／複数動画 × 直接／生成 context | `tests/workflow.rs` の `cli_evidence_matrix_checks_pixels_labels_events_and_zero_call_warm`。8通りの通常 CLI cold／warm。生成は固定2検索の loopback mock。 |
| 原寸、source、実 PTS、向き、rank、理由、event | `tests/support/evidence.rs` の独立 recipe と画素照合。画像要求の sheet・label・前後順と semantic chunk の全 frame を照合してから応答する。成果物画像は JPEG 誤差を許容し、report から再構成した sheet と照合する。renderer 自体のラベル・配置・日本語は `tests/contact_sheet.rs` が別に検証する。 |
| 複数映像 track と回転 | `cli_rotated_multitrack_artifacts_keep_full_resolution_display_orientation`。最小 index の映像が 96×160 の表示向きで画像・動画要求・成果物に対応する。入力の全 PTS は `tests/fixture_inputs.rs`、start offset／VFR は `tests/media.rs` が検証する。 |
| cold と途中再開の同一性 | `cli_resume_reuses_committed_media_and_context_and_matches_clean_cold`。8通りで最初の batch／chunk 成功後に HTTP 401 を返し、完了済み媒体評価と生成 context を再利用。再開後の画像・sheet・report は同じ接続先・入力・条件の clean cold と byte 一致。これは回復可能な HTTP 失敗の再開であり、SIGINT 実験とは区別する。 |
| 全 hit の外部操作0、入力追加／移動／枚数変更、破損 cache | CLI warm は tools のない PATH と `--cache-only` で確認。`tests/cached_run.rs`、`tests/run_workflow_reuse.rs` は必要 checkpoint のみの再計算、出力が正常なときの未使用 checkpoint 非走査、cache 全消失後の正常出力再利用を検証する。 |
| 誤画像／時刻／label、不完全出力 | `evidence_rejects_wrong_pixels_times_labels_and_incomplete_artifacts` は前後画素交換、要求 metadata の誤時刻、report のsource／時刻／rank／scene改変、出力画像交換・欠損を拒否する。製品の所有権拒否は `tests/output.rs`、`tests/cached_run.rs` が別に確認する。 |
| 不足、FFmpeg／HTTP／schema 失敗 | `cli_shortage_http_and_schema_failures_preserve_previous_complete_output`、`cli_ffmpeg_failure_preserves_previous_complete_output_without_inference`。不足理由、有限attempt、以前の正常出力の byte 保持、秘密非出力を確認。 |
| disk／公開／cleanup／中断 | `src/output.rs` の staging・rename・journal recovery・部分 cleanup の負例、`tests/runtime.rs` の所有session・停止失敗、`tests/http_cancellation.rs`／`tests/workflow.rs` の実 SIGINT と終了130を使用する。強制 kill や任意filesystemの保証へ拡張しない。 |
| 台本の時間分散 guard | ソース checkout の `tests/fixtures/README.md` の固定 tuple を `fixture_contract.jq` が検査。candidate・eventの同時時刻縮小、source交換、context変更、ID取り違えの負例を拒否する。 |

合成素材だけを手元へ保存するには、空の検証用 directory を指定する。
archive の検証には `BUILD-INFO.json` の `commit` と同じソース checkout を使う。

```sh
GSP_QA_DIR=/absolute/path/to/empty-synthetic-qa sh tests/check.sh
```

`cli-*` は8通りの JPEG・sheet・report・cold/warm 計測、`resume-*` は再開の計測と外部呼出数を保存する。
公開用にはこの合成証拠のみを使用する。利用者の config、録画、cache、実モデル request／response をコピーしない。
既存の `cached-*` 証拠も合成素材だけである。GUIで sheet を開いても実録画の Human Review 完了とはしない。

## 計測の意味

成功した CLI は stderr の `metrics: {JSON}` に非秘密の計測を出す。report には計測を混ぜず、cold／warm／再開の成果物同一性を維持する。
`elapsed_seconds` と旧 `cache.elapsed_seconds` は `Session::prepare` 以降の累積時間。
`stage_seconds` の範囲は次の通りで、工程合計は elapsed の完全な内訳ではない。

| field | 範囲 |
| --- | --- |
| `extraction` | FFmpeg の候補・前後・chunk・最終原寸 frame 抽出。 |
| `machine_quality` | 候補画像のdecodeと機械的品質評価。 |
| `inference_callbacks` | assessment／semantic／context callback の wall time。HTTP retry、検索、runtime準備やcallback内の再検証を含み得るため、純粋なモデル推論時間ではない。 |
| `cache_verification` | 入力metadata／identity／hash（identity index書込を含む）、checkpoint／domain、assessment-key媒体hash、preflight／staged／公開済み／復旧時の出力所有権確認。通常checkpoint書込・公開copyは除く。 |
| `media_probe` | 実FFprobe呼出。cache lookupを除く。 |

callback内で検証する場合は計測区間が重なることがある。通常warmでは extraction・machine_quality・inference_callbacks・media_probe は0。
bytes、input hash、probe、assessment／semantic／context callback数も記録する。mock server側の AI／検索呼出数と設定なしruntimeの0件を合成QA計測に併記する。

`peak_memory` は macOS／Linux の `getrusage(RUSAGE_SELF)` による当該process生存期間の最大RSSをbytesへ正規化した値。
開始前の処理・同processのthreadを含み、FFmpeg／FFprobe子processとremote backendは含まない。
取得不能は `unavailable`、非対応OSは `unsupported`。利用者が実backendを測定するときはserver側の計測を別に残す。
abortしたCLIは完成した `RunOutcome` がなくmetrics行を出さない。失敗までの外部呼出数はmock記録、実測ではbackend／検索側の非秘密集計を使用する。
環境・build・素材・条件で時間は変わる。Rust化だけの速度向上率を推測したり、時間閾値を品質の合格条件にしたりしない。

## 未実施の実モデルと画像評価

必要な入力は、利用許可のある実録画のローカルpath、明示されたendpoint・backend/server version・model/revision・vision設定・上限、既存の利用可能な認証、
Brave生成を試す場合の適用契約（無課金停止、推論用途、生成文保存）と既存の利用可能な認証、画像を確認するreviewerである。
model自動インストール、GPU server構成、秘密設定、認証変更や外部有料サービスへの切替はこの検証で行わない。

各対応backendの画像／動画／JSON Schema／tools・上限を実測し、設定だけを切り替えた対応経路でE2Eを行う。
非対応semantic経路は理由付き停止を確認し、別方式やtext-onlyへfallbackしない。
両方式をそれぞれ複数回実行し、単一／複数動画と直接／生成contextの実施可否を記録する。
同じ条件のcold／warm／batchまたはchunk後の再開について、出典・順位・内容と外部呼出数を比較する。

記録は公開repositoryやartifactの外のprivateな場所に保存し、次の項目を埋める。現在はすべて未実施である。

| 項目 | 記録する内容 |
| --- | --- |
| 実行条件 | 日時、CLI head、OS/arch、FFmpeg/FFprobe version、backend/server version、model/revision、vision/上限、方式、count、context種別。secretとendpoint認証部分を除く。 |
| private入力 | ローカル素材の識別子、動画数・長さ・寸法・向き。公開pathや録画uploadを要求しない。 |
| 能力 | image/video/schema/toolsと上限ごとの実測結果・不対応理由。mock結果で代用しない。 |
| 計測 | cold/warm/再開のstage time、CLI peak RSS、必要ならbackend側peak、AI/search/runtime件数と計測方法。 |
| reviewerと合否 | 実際に画像を確認した人・日時、各runのpass/fail。自動検証担当やこの文書の作成者をHuman reviewerとして記録しない。 |
| 問題画像 | privateなrank/source/要求・実時刻と具体的理由。暗転、白飛び、loading/fade、近似重複、出典や向き誤り、偏りを記録する。 |
| 分散と有用性 | 有効候補と枠がある場合の入力元・時刻・scene分散、通常進行と有用な特別場面。満たせない理由と解決結果。 |

実能力と具体的な画像品質・障害回復・性能の証拠が揃い、利用を妨げる未解決事項がなくなってから製品の実受入を判断する。
archive の build・checksum・clean 環境検証と、実公開の承認境界は [配布・公開](release.md) を参照する。
