# Rust CLI の成果物と責務境界

[#337](https://github.com/mizucopo/game-screen-pick/issues/337) の製品は、ゲーム録画からブログ用画像・一覧・根拠を得る Rust CLI とする。
合格条件は [成果物の受入基準](../acceptance.md) に置き、score や内部形式ではなく内容・品質・使いやすさ・再開効率・安全性を検証する。

CLI/config/media、選定、cache/output、AI 接続の必要最小限の責務を分ける。
共通 OpenAI 互換 client を先に作り、Strata／vLLM を設定で切り替える。
Brave 検索と任意 runtime 管理は用途側に置く。model・推論レベルは明示し、media 非対応を成功扱いしない。
画像と動画の実能力、固定応答、実 model の成果物品質を別々に確認する。

現在の実装・テストに必要な入力の事実と文書だけを採用する。
利用者の入力・設定・出力・cache は削除しない。

Rust は新しい config と cache で開始する。所有権を証明できない出力は保持して別の空出力先を案内する。
staging と完了検証で既存成果物を保護し、強制 kill や filesystem に関する保証は実測の範囲に限る。

実装順は #338 → #339 → #340 → #341 → #362 → #342 → #343 → #344。
#362 の接続・mock 基盤の後に #342 を接続し、backend 別能力を両 Issue で仕上げる。
PR は `integration/rust-migration` へ向け、全検証・独立レビュー・配布確認後に main 向け最終 PR を別途作る。
配布検証 target は `aarch64-apple-darwin` と `x86_64-unknown-linux-gnu`。最小 OS/glibc・実依存は #344 で確認する。
途中 PR は公開を希望せず、最終のメジャー切替は [CONTRIBUTING.md](../../CONTRIBUTING.md) の既存 controller に任せる。手動採番しない。
