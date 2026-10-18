# Game Screen Pick

1本以上のゲーム録画から、ブログ用の画像を指定枚数選び、一覧と選定根拠を得る。
合格条件は [受入基準](docs/acceptance.md) に置く。

## 責務と安全条件

CLI/config/media、選定、cache/output、AI 接続の責務を分ける。
Game Context 生成・画像評価・動画理解は [共通 OpenAI 互換 client](docs/ai-client.md) を使い、
Strata／vLLM、model・推論レベルを設定で明示する。非対応 media を成功扱いしない。
[Brave 検索](docs/brave-search.md) と [任意 runtime 管理](docs/technical-reference.md) は用途側に置く。

利用者の入力・設定・出力・cache は保持する。所有権を証明できない出力は保持して別の空出力先を案内する。
staging と完了検証で既存成果物を保護し、強制 kill や filesystem の保証は実測の範囲に限る。
配布対象・共通テンプレートと製品 hook／guard の境界・採番・承認条件は [配布・採番・公開](docs/release.md) に従う。

## 用語

| 用語 | 意味 |
| --- | --- |
| Input Video Directory | 入力集合を指定する directory。直下の対応する通常動画 file を安定したファイル名順で探索し、再帰探索・symlink 追跡はしない。 |
| Input Video / Input Videos | 入力録画1本／その集合。ほぼ先頭から末尾までを対象とし、元動画と実時刻を出力まで追跡する。 |
| Input Video Identity | 内容SHA-256をmetadata（Unixのdevice/inode・mtime/ctime、Windowsのvolume/File ID・LastWriteTime/ChangeTime、サイズ）で再利用する条件。親directory移動を保ち、通常の変更は再hashする。metadata偽装復元は保証範囲外。 |
| Game Title | Brave 検索から Game Context を作るためのゲーム表記。ファイル名から推測しない。 |
| Game Context | ゲームの進行・代表画面・選定で重視する点を説明する文章。直接指定、または Brave 検索と明示 model による生成。固定カテゴリ/quotaや命令として扱わない。検索由来の保存・再利用は契約確認後だけ有効にする。 |
| Sample Position | 各動画内の候補抽出時刻。等間隔方式は全編を覆い、動画理解方式は全編の重要場面周辺を使う。 |
| Semantic Event | 動画理解で得た重要区間・代表時刻・説明・重要度。画像候補の根拠であり、最終採用を保証しない。 |
| Frame Candidate | 元動画から抽出した画像候補。全 decode frame を一括で未完了 job にしない。 |
| Frame Display ID | 評価 batch 内の短い表示 ID。contact sheet・prompt・応答検証で実画像に対応させ、未知・重複・欠落を拒否する。cache の内部 ID 形式は固定しない。 |
| Primary / Secondary Candidate | 一次画像評価の対象／一次評価後に品質・scene・見た目・時刻を分散した二次評価の対象。 |
| Transition Context | 二次評価の直前・対象・直後の画像。中央の採否に使い、3枚を別々の最終候補としない。 |
| Scene | 同種の画面をまとめる短い場面名。日本語等を扱い、実行前の固定 catalog は置かない。 |
| Normal Progress / Special Screen | 探索・戦闘・会話等の通常進行／title・map・menu・result等の有用だが偏りを招く特別画面。ジャンルに存在しない場面を要求しない。 |
| Selected Image | 品質・多様性・前後 context を評価して選んだ full-resolution 画像。元動画・実時刻・rank・理由を追える。 |
| Selected Contact Sheet | 全 Selected Image を rank・入力元・実時刻付きでまとめた人間確認用の `selected-contact-sheet.jpg`。 |
| Output Folder | Selected Image、一覧、report の公開先。Rust reportの宣言集合と内容digestで所有権を検証し、未所有・編集済みfileは保持する。 |
| Phase / Assessment Cache | Rust の再生成可能な工程／完了 batch の処理状態。入力・条件の変更で依存する部分だけ無効化し、認証値を含めない。 |
| Completed Run | 指定枚数の画像・一覧・report と完了記録が整合する状態。候補不足や不完全公開を含めない。 |
| Human Review | 実 model の出力一覧と個別画像を人が確認し、条件・reviewer・問題画像・合否を記録する品質確認。mock 成功や model score で代用しない。 |
