# 言語ツールと製品設定の境界

共通の品質設定を変更するときは、下流の生成済みファイルを手でコピーせず、
repo-template 側を変更してから、専用ブランチで通常の `copier update` を実行する。
生成値と製品固有の値が同じ箇所で変わり競合した場合は、テストして解決する。
最初から上書きする `copier recopy` は通常の更新に使わない。

## Rust / Cargo

- Rust edition、`rust-toolchain.toml` の共通コンポーネント、
  fmt／Clippy（警告をエラー扱い）／test の品質検査は上流が管理する。
- ツールチェーンは Copier の `rust_version` で選ぶ。Cargo の `rust-version`（MSRV）、
  リリース済みバージョン、workspace、crate 名、依存・features、`Cargo.lock`、
  build script と配布時の追加処理は製品側で管理する。
- `rust_cli_release_profile=thin-lto-strip` は `Cargo.toml` に
  `[profile.release]` の thin LTO と strip を生成する任意設定。既定の `default` は
  profile を追加しない。Tauri には適用しない。
- 既存の手書き profile がこの2項目と等価な standalone CLI は、移行ブランチで
  重複する節を除いてから opt-in し、最終差分で設定値とビルド結果が変わらないことを確認する。
  workspace で cargo profile が別の root にある場合、または独自 profile が必要な場合は
  opt-in せず、プロジェクト設定を保持する。

Copier の version 回答は初回生成値であり、製品の現在のリリースバージョン
ではない。更新時は版、lockfile、追加設定が保持されたことと、既存の
品質コマンドが引き続き通ることを確認する。
