# 動画入力の成果物

> [!IMPORTANT]
> この文書のWebP encoder、Canonical Selection Report producer、Markdown renderer、検証、atomic publisherは内部実装済みです。現在のpublic commandはまだscreenshot入力版であり、Video Set applicationへの接続とCLI切り替えまでは生成されません。

成功した一つのrunは、Output Folderへ次の成果物をatomicに公開します。

```text
output/
├── images/
│   ├── 0001_opening_38f1a9c2e642.webp
│   └── 0002_exploration_6a4d812e6241.webp
├── report.json
└── report.md
```

- `images/`: 元frame解像度、lossy WebP quality 95、metadata除去済みの選択画像
- `report.json`: `game-screen-pick/report@1.0.0`の唯一のmachine-readable正本
- `report.md`: 検証済みJSON objectだけから決定的に作るgallery-firstの人間向けreport

画像名は全体選択順、Scene Slug、Frame Candidate IDの短縮digestから作ります。安定identityはfilenameではなく、完全なFrame Candidate IDです。

Video Source IDは通常、whole-file SHA-256の先頭12文字を使います。同じVideo Set内でそのprefixが衝突した場合は、衝突したsourceだけを64文字の完全digestへ拡張し、一意性を保ちます。

producerが検証する厳密なschema実体は[`report-1.0.0.schema.json`](../src/video_selection/schemas/report-1.0.0.schema.json)です。readerは同じmajorの将来minor field／enumを保持できるcompatibility gateを別に持ち、producerの厳密検証とは分離しています。report schema versionはpackage versionから独立します。

## provenanceとmodel更新

`report.json`は各Processing Stageについて、完全なStage Fingerprint、検証済みCompleted
Stage manifestの上流fingerprint、現在runで観測したcache hit/miss・再計算件数・試行回数・
実行時間、正規化済み設定、tool/runtime version、実行時に解決した完全なmodel digestまたは
commit SHAを保持します。coldとwarmのreportは同じartifact identityでも、各runで実際に
再計算または再利用された結果を別々に示します。`report.md`は短縮fingerprintと主要診断だけを
表示します。

`run.started_at`はApplicationへ入った時点、`run.completed_at`は全Processing Stageと選定に
加えて選択画像のstagingが完了し、atomic publisherがCanonical reportを最終化する時点をUTCで
記録します。公開済みreportを後書きせず、最終directory renameまで同じ検証済みstagingを使う
ための最新のatomic-safeな完了境界です。

設定ファイルにはmodel hashを書きません。modelが更新されるとResolved Model Identityが変わり、そのmodelに依存するStageだけが新しいfingerprintで再計算されます。実際に同じidentityへ解決された場合は既存cacheを再利用します。

## 公開しない情報

絶対path、環境変数、credential、prompt本文、raw model response、stack trace、字幕・STT本文、生成した画面内textの逐語引用はreportへ含めません。Context CueはID、source、正確な時間範囲、reliability、選定への関連度だけを公開します。逐語転載の検査はmodel由来の公開自由文を対象とし、独立生成と区別できない1〜2文字の一般語は引用判定から除外します。

Context Cueの時刻がsource streamの整数PTS gridへlosslessに対応する場合は、`source_pts`、`origin_pts`、`time_base`を保持します。containerが与えるCue時刻とPTS gridが非整列の場合は、`timestamp_basis`を維持しつつ、reduced rationalの`offset_seconds`へlosslessにfallbackします。

Spoiler Riskは常時表示しますが、`none`以外の短いevidence summaryはMarkdownで閉じた`details`にします。選定に使ったSpoiler Penaltyとは別fieldです。

Selection Shortfallはwarning付き正常成果物として選択済みsubsetを公開します。model更新が利用不能でも完全でload可能なlocal artifactを再検査できた場合は、`model_update_unavailable`と対象roleをwarningとして公開します。それ以外のfatal error、schema不正、renderer失敗、成果物不一致ではOutput Folderを公開しません。

publisherはartifact生成前に同じparent内のdirectory renameをprobeし、元動画のpath・device・inode・size・mtime・ctime snapshotを開始時と最終rename直前に再検証します。hidden sibling staging内のfileとdirectoryをflushし、schema・画像hash／寸法／path・JSON serialization・Markdown再render・privacy・layoutを検証してから一回だけfinal renameを行い、そのparent directoryもflushします。renameの前後を含む失敗時はstagingとfinal Output Folderを除去します。

完全なfield、命名、時刻、near miss、schema evolution、atomic publication契約は[ADR 0005](adr/0005-publish-video-selection-artifacts-atomically.md)を参照してください。
