# Game Screen Pick

1本以上のゲーム録画から、ブログ用の画像を指定枚数選び、一覧と選定根拠を得る。
設計の判断は [ADR 0010](docs/adr/0010-rust-product-boundaries.md)、合格条件と担当は
[受入基準](docs/acceptance.md) に置く。

| 用語 | 意味 |
| --- | --- |
| Input Video Directory | 入力集合を指定する directory。直下の対応する通常動画 file を安定したファイル名順で探索し、再帰探索・symlink 追跡はしない。 |
| Input Video / Input Videos | 入力録画1本／その集合。ほぼ先頭から末尾までを対象とし、元動画と実時刻を出力まで追跡する。 |
| Input Video Identity | Rust cache が入力を区別する条件。directory 移動と再利用、内容変更の検出範囲を #341 で決める。 |
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
| Output Folder | Selected Image、一覧、report の公開先。未所有の既存 file は保持する。Rust の所有権と完了記録は #341 が検証する。 |
| Phase / Assessment Cache | Rust の再生成可能な工程／完了 batch の処理状態。入力・条件の変更で依存する部分だけ無効化し、認証値を含めない。 |
| Completed Run | 指定枚数の画像・一覧・report と完了記録が整合する状態。候補不足や不完全公開を含めない。 |
| Human Review | 実 model の出力一覧と個別画像を人が確認し、条件・reviewer・問題画像・合否を記録する品質確認。mock 成功や model score で代用しない。 |
