# 動画入力

> [!IMPORTANT]
> Effective Configuration、動画探索・identity、Input Lock、Completed Stage cache、動画単位のFrame Candidate・Context Cueを作るVideo Stageは内部実装済みですが、installed public CLIはIssue #190までscreenshot入力のままです。現在のconsole commandではまだ動画入力を実行できません。

## 実行単位

動画入力版はsubcommandを持たない一つのcommandとして公開します。

```bash
game-screen-pick [OPTIONS] <VIDEO_INPUT_FOLDER> <OUTPUT_FOLDER>
```

`VIDEO_INPUT_FOLDER`から1本以上の動画を発見し、一つのVideo Setとしてまとめて選定します。`OUTPUT_FOLDER`には選択画像、`report.json`、`report.md`をatomicに公開します。

## CLI

| Option | 意味 | CLI未指定時 |
|---|---|---|
| `-n, --image-count INTEGER` | 要求する画像枚数。10以上 | Effective Configurationで解決。組み込み既定値100 |
| `-r, --recursive` | 子directoryも探索する | TOML、組み込み既定値`false`へ委譲 |
| `--no-recursive` | CLIから再帰探索を明示的に無効化する | TOML、組み込み既定値へ委譲 |
| `--config PATH` | 読み込むTOMLを明示する | configを読まない |
| `--scene-hint TEXT` | ゲームやブログ選定意図の補足 | hintなし |
| `--spoiler-sensitivity low\|medium\|high` | Spoiler Riskへ適用するsoft penaltyの強さ | 組み込み既定値`medium` |
| `--similarity-threshold FLOAT` | 通常選定を開始する類似度上限 | 組み込み既定値`0.72` |
| `--ollama-host URL` | 接続するOllama server | TOML、`OLLAMA_HOST`、localhostへ委譲 |
| `--reset-cache` | このVideo Input Folderの処理cache全体を安全なpreflight後に消す | `false` |
| `--debug` | stack traceを含む安全化済み診断を表示する | `false` |

`--help`と`--version`も通常のCLI情報として提供します。path、実行意図、運用操作だけをCLIに置き、モデル・抽出・STT・並列度はTOMLで設定します。

現行screenshot版の`--num`、`--similarity`、`--ollama-scene-hint`はaliasとして残しません。動画入力版への切り替えは後方互換性を持たない移行です。

## 動画の発見と順序

- 対応拡張子は`.mp4`、`.mov`、`.mkv`、`.webm`です。
- 既定ではroot直下だけを探索し、recursive時だけ子directoryを探索します。
- directory symlinkは辿りません。対応拡張子を持つfile symlinkは許可し、link先の内容を処理します。
- Video Orderは入力rootからの正規化済み相対pathの自然順です。mtimeやfilesystem列挙順は使いません。
- Video Identityはfile全体のSHA-256で決まります。renameやmtime変更では変わらず、内容変更時だけ変わります。cache missではstat-content-statの順に全体を読み、動画1本のSHA-256が確定するたびにpathを含まないVideo Identity cacheへatomic保存します。後続動画の失敗やprocess再起動で、確定済み動画を再hashしません。
- Video Set FingerprintはVideo OrderどおりのVideo Fingerprint列から決まり、input rootや設定値を含みません。
- 対応動画0本、壊れた動画、同一内容の重複動画は、cacheやoutputを作る前に実行全体を失敗させます。
- 発見後にpath、size、`mtime_ns`が変化した場合はsnapshot不一致としてrunを中止します。Video Identity cacheはengine version、入力rootと相対pathから作るprivacy-safeなlogical source key、size、`mtime_ns`が一致する場合だけwhole-file SHA-256を再利用します。device、inode、ctimeは判定に使いません。Input Lock取得後、media probe、各Video Stage、Vision batch、publisherのstaging開始時・final rename直前は同じmetadataを検査し、同じ大容量fileをStageごとに再hashしません。同じsize・mtimeへ意図的に内容を書き換えた場合は検知できないため、入力管理者がmtimeを正しく更新することを契約とします。通常CLIの`--reset-cache`ではidentityも再計算されます。

入力と出力は同一pathにも相互の親子にもできません。新しい`OUTPUT_FOLDER`は存在しないか空である必要があります。atomic rename後に完了記録だけが失われた場合に限り、既存の非空folderをCanonical Outputとして完全検証し、今回のsemantic digestと一致すれば一byteも変更せず再利用します。不正なfolderや異なる意味結果は削除・上書きせず拒否します。

## 例

```bash
game-screen-pick \
  --config ./video-selection.toml \
  --image-count 100 \
  '/mnt/g/Captures/14_冒険家エリオットの千年物語/movie' \
  ./output/elliot-blog-images
```

設定値と優先順位は[設定](configuration.md)、処理順と安全な再開は[Pipelineと安全な再開](pipeline-resume.md)、cache、進捗、エラー、WSL2運用は[運用](operations.md)、成果物は[report](report.md)を参照してください。
