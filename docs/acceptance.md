# Rust CLI の受入基準

ゲーム録画からブログ用の画像を指定枚数選び、画像・一覧・根拠を一緒に確認できることを合格条件とする。
この仕様と [入力素材](../tests/fixtures/README.md) を各担当 Issue の Rust テストへ接続する。
検証素材の確認だけでは CLI や実 model の品質が合格したことにはならない。

## 入力と CLI（#339）

```text
game-screen-pick --config config.toml --count 2 \
  --game-context "探索・戦闘・会話の見やすい場面を選ぶ" INPUT_DIRECTORY OUTPUT_DIRECTORY
```

- `--config`、`--count`（1〜999）、入力・出力 directory は必須。
  `--game-title` と非空の `--game-context` はどちらか一つを指定する。`--help`／`--version` は処理しない。
- TOML は `[selection] method = "sampled_frames" | "semantic_video"` と `[ai]` を基本とする。
  `[ai]` は `backend`（`strata`／`vllm`）、`base_url`、`model`、`inference_level`、
  `timeout_seconds`、`cache_revision` を明示必須とする。認証は非空 `api_key` 優先、
  なければ `GAME_SCREEN_PICK_API_KEY` を使う。model／推論レベルに暗黙 default を置かない。
  その他の検索・media・runtime 設定は #362／#342 で必要なものだけ定義する。
- 未知 key・型・範囲・排他違反・非対応の推論レベル／media を説明して拒否する。
  設定 file の不在や誤りで暗黙に別の設定へ切り替えない。
- 入力直下の通常 file のみを、拡張子を大文字小文字無視で `.mp4`／`.mkv`／`.mov`／`.webm` と判定し、
  UTF-8 ファイル名の順で探索する。再帰探索・symlink 追跡はしない。対象動画の symlink、不正 path、空集合は拒否する。
- 時刻は、attached picture を除いた映像 stream の最初の有効 PTS を 0 秒とした動画内秒数。
  report には要求時刻と実際に抽出した時刻を区別する。start offset、端点、VFR は frame の有効 PTS で照合する。
  抽出の内容・向き・寸法を既知の入力 recipe と直接比較し、選定や cache の試験で代用しない。
- 終了 code は `0`＝全成果物の公開・検証完了（または help/version）、`2`＝引数・設定・入力の誤り、
  `1`＝処理失敗・候補不足・非対応経路、`130`＝通常の SIGINT 中断。
  中断以外の signal は成功扱いしない。未実装経路は `1` で理由を示す。
- stderr の進捗は方式、工程、完了／予定数、追補理由、cache 再利用、失敗と復旧方法が分かること。
  動的文字列を安全に表示し、認証値・HTTP header/body の秘密情報を表示しない。

## 成果物（#340／#343）

| 成果物 | 合格条件 |
| --- | --- |
| `selected-01.jpg` … 指定枚数 | 元動画の正しい内容・実時刻・表示向き・full resolution。評価用縮小画像を出力しない。暗転・白飛び・単色・loading/fade途中・明らかな近似重複を避ける。不足時は枚数だけ埋めず失敗する。 |
| `selected-contact-sheet.jpg` | 全画像が順位順に一度ずつ表示される。rank・入力元・実時刻が report と一致し、画像・日本語 label が読める。縦横比・向き・前後 context の対応を実画像で検証する。 |
| `report.json` | 方式、指定／実枚数、入力元、要求／実時刻、rank、画像 path、scene・選定理由、使用 backend/model・非秘密の実効条件を追跡できる。semantic 方式は event の区間・説明・根拠を含む。保存可能と確認した Game Context のみ記録する。 |

有効候補と枠があれば、入力元・時刻・scene・見た目を分散し、通常進行画面を中心に有用な特別場面も含める。
偏りや除外・不足の理由を report またはエラーで説明する。固定応答時の順序と同点規則は Rust で決定し回帰テストにする。
同じ入力・条件の cold／warm／中断再開で、Rust の同じ選定結果を得る。
内部 ID、score の式、JPEG bytes、font、report 内部 schema はこの仕様で固定しない。

## 再開と出力保護（#341）

新しい namespace/schema の Rust cache を使う。入力同一性と変更検出範囲は #341 で決定・明記する。
backend/model、推論レベル、prompt/schema、media 処理、revision の意味が変われば依存結果を無効化し、
認証値を key や保存内容に含めない。動画追加・directory 移動・枚数変更でも再利用可能な処理を保つ。

warm は推論・検索・runtime 操作が 0 件。batch/chunk 完了後の中断は完了済み結果を再利用し、欠損だけ処理する。
payload 破損・誤参照・画像欠損を正常な hit にしない。warm 検証の時間・decode/hash 回数を測定する。

未所有 file・未知 report・symlink・特殊 file・path 逸脱・競合実行を拒否し、既存 file を保持して空の別出力先を案内する。
Rust report と出力集合の整合性で所有権を検証でき、cache 消失だけで正常出力を失わない。
入力・設定・path・出力所有権と cache/output の排他を、FFmpeg/FFprobe・Brave 検索・推論接続・runtime 操作より先に確認する。
事前確認で拒否した run は外部呼出を 0 件とし、検索枠や server 起動に影響させない。
公開前の失敗は以前の正常成果物を保持する。staging／公開途中の失敗も不完全・混在した出力を成功扱いせず復旧する。
lock、同一 filesystem 内の atomic 書込、cleanup の保証を対象 OS で検証する。未実測の強制 kill／filesystem 横断 transaction は約束しない。

## AI と品質確認（#362 → #342 → #362／#343）

#362 が小さな共通 OpenAI 互換 client・設定・mock 契約を作り、#342 が context 生成、二段階画像評価、動画理解を接続する。
Brave は固定検索に絞り、無料枠・停止条件・保存条件を確認する。生レスポンスを永続保存せず、
生成 context の保存・再利用も契約確認前に有効化しない。直接 context は検索しない。有料 API への自動 fallback はしない。

固定応答は ID／前後画像／動画の順序・時刻を照合した mock にだけ返す。
media を取り違えても同じ応答を返す stub では能力や抽出を合格扱いしない。
JSON/schema、未知・欠落・重複 ID、時刻逸脱、拒否／打切り、不正値を検証し、有限 retry を尽くしたら失敗する。
timeout、同一 origin redirect、認証非出力、未設定 runtime 操作 0 件、自己が開始した session の cleanup を確認する。

mock 成功と実能力は別に記録する。Strata／vLLM ごとに server version、model、vision 設定、
image/video/JSON Schema／必要な tools、解像度・枚数・token/context 上限を実測する。
非対応 `semantic_video` は理由付きで停止し、text-only や一枚の静止画へ黙って置換しない。
時刻付き画像列を使う場合も順序・時間・上限・成果物品質を検証する。

#343 は両方式×単一／複数動画×直接／生成 context の E2E と、各方式の複数回の実録画・実 model の Human Review を行う。
reviewer、実行条件、問題画像と理由、合否を記録し、秘密情報や実録画を公開 artifact に入れない。
抽出・機械評価・推論・cache 確認時間、peak memory、cold／warm／再開の外部呼出数を測定する。速度向上率を推測しない。

## 担当と配布（#344）

| 順序 | 担当 |
| --- | --- |
| #338 | この受入仕様、入力の事実・固定応答・障害ケース、簡潔な更新手順 |
| #339 | Cargo／CLI／Rust config／探索／FFprobe・FFmpeg 抽出、Rust 開発設定・quality gate |
| #340 | 品質・多様性・候補不足・contact sheet |
| #341 | Rust cache・必要部分の再計算・所有権・排他・安全な公開 |
| #362 → #342 | 共通 client・設定・mock → Brave／context／画像／動画／任意 runtime 接続 → backend 別実能力確認 |
| #343 | 成果物 E2E、負例、実 model 品質、障害回復・性能 |
| #344 | `aarch64-apple-darwin`／`x86_64-unknown-linux-gnu` artifact の clean 環境検証と短い利用文書 |

配布 binary は Python／uv／venv／Rust toolchain／OpenCV の利用者 setup を要求しない。
FFmpeg／FFprobe と推論 server は外部依存。最小 macOS／glibc、dynamic library、FFmpeg 条件は #344 で実測し、
未検証 target を対応済みとしない。作業 PR は `integration/rust-migration` 向け。
最終 main PR と公開は全検証後の別工程で、採番は [既存 release 規約](../CONTRIBUTING.md) に従う。
