# 動画入力の成果物

> [!IMPORTANT]
> この文書は将来のVideo Set selectorの確定済み契約です。現在のscreenshot入力実装ではまだ生成されません。

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

## provenanceとmodel更新

`report.json`は各Processing Stageについて、完全なStage Fingerprint、上流fingerprint、cache結果、実行時間、正規化済み設定、tool/runtime version、実行時に解決した完全なmodel digestまたはcommit SHAを保持します。`report.md`は短縮fingerprintと主要診断だけを表示します。

設定ファイルにはmodel hashを書きません。modelが更新されるとResolved Model Identityが変わり、そのmodelに依存するStageだけが新しいfingerprintで再計算されます。実際に同じidentityへ解決された場合は既存cacheを再利用します。

## 公開しない情報

絶対path、環境変数、credential、prompt本文、raw model response、stack trace、字幕・STT本文、生成した画面内textの逐語引用はreportへ含めません。Context CueはID、source、正確な時間範囲、reliability、選定への関連度だけを公開します。

Spoiler Riskは常時表示しますが、`none`以外の短いevidence summaryはMarkdownで閉じた`details`にします。選定に使ったSpoiler Penaltyとは別fieldです。

Selection Shortfallはwarning付き正常成果物として選択済みsubsetを公開します。model更新が利用不能でも完全でload可能なlocal artifactを再検査できた場合は、`model_update_unavailable`と対象roleをwarningとして公開します。それ以外のfatal error、schema不正、renderer失敗、成果物不一致ではOutput Folderを公開しません。

完全なfield、命名、時刻、near miss、schema evolution、atomic publication契約は[ADR 0005](adr/0005-publish-video-selection-artifacts-atomically.md)を参照してください。
