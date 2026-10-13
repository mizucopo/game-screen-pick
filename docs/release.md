# 配布・採番・公開

## 配布対象と検証

Rust binary の配布対象は次の二つです。これ以外の target は対応済みとしません。

| target | build の下限・検査 |
| --- | --- |
| `aarch64-apple-darwin` | `MACOSX_DEPLOYMENT_TARGET=14.0` で build し、Mach-O の deployment target と dynamic library を検査する。 |
| `x86_64-unknown-linux-gnu` | Ubuntu 22.04 で build し、ELF が要求する GLIBC symbol version の最大値を2.35以下に制限する。 |

build の下限宣言とリンク検査は、その最小 OS 上での実行証拠と区別します。
FFmpeg／FFprobe と推論 endpoint は外部依存です。font は binary に内蔵し、本体の実行に Rust toolchain や Python 環境は要求しません。
FFmpeg の PNG／FFV1、表示回転、`setts` と実 PTS の保持は製品抽出・動画 chunk の試験で確認します。
検証した OS／FFmpeg version と dynamic library は配布物の `BUILD-INFO.json`／`DEPENDENCIES.txt` と PR の検証結果に記録します。
未実測の FFmpeg 最小 version や、宣言値以外の OS の互換性を推測しません。

archive 名は `game-screen-pick-<Cargo version>-<target>.tar.gz`、checksum は同名の `.sha256` です。
展開すると同じ名前の directory に `bin/game-screen-pick`、`config.example.toml`、README、利用文書、license と build・依存情報が入ります。
依存 package の license 一覧、font の OFL と本体の MIT license を同梱します。設定・録画・生成画像・cache は含めません。
利用者の取得・checksum・配置・設定・更新手順は [README](../README.md) を参照してください。

native の対象環境で、公開せず配布物を作れます。
Rust toolchain の `rust-docs`／`rust-src`（license収録用）、FFmpeg／FFprobe、jq、checksum tool、GNU tar が必要です。
rustup を使う場合は `rustup component add rust-docs rust-src` で収録用資料を用意します。macOS は GNU tar を `gtar` として用意します。

```sh
sh tests/check.sh
sh scripts/package-release.sh aarch64-apple-darwin dist
# Linux x86_64 GNU では次を使う:
# sh scripts/package-release.sh x86_64-unknown-linux-gnu dist
```

build は `cargo build --release --locked` を使います。`Cargo.toml` と `Cargo.lock` を一致させ、実行 file・dynamic library・license・archive を確認してから checksum を作ります。
各 target の clean 環境で展開した binary の help/version、readonly validate、実 FFmpeg 抽出と合成 mock E2E を確認します。
成功した固定応答 E2E を、実 backend/model の能力、Brave の適用契約と live 接続、実録画の Human Review の証拠にはしません。
これらの未実施項目と実施条件は [検証状況](verification.md) に残します。

## 公開前の承認境界

現在の作業 PR は `integration/rust-migration` 向けの通常 PR です。途中 PR は公開を希望せず `release:*` ラベルを付けません。
integration branch で全品質 gate、独立レビュー、配布準備を揃え、最終の `main` 向け PR を別途作ります。
設定・CLI・cache・運用の変更を含むメジャー切替として、最終 PR の希望 version は **2.0.0**、分類は `release:major` とします。
手動で version を変更せず、最終 PR の差分・分類と公開対象を確認した後に既存 controller が採番します。

実公開の前に、対象 version、GitHub Releases の二つの archive と checksum、対象 head、検証結果・未実施項目を本人へ提示します。
承認は、最終 PR の squash merge、controller による `main` の採番 commit と tag の atomic push、build・asset upload・Release 公開・Latest 更新を含む一連の操作に対して得ます。
途中 PR の実装依頼や Issue の close だけから、この承認を推定しません。認証・権限設定、Homebrew 等の下流公開は個別の承認対象です。
Issue の close とコード完了から、実録画・実 backend・契約・Human Review の完了を推定しません。

## 設定と公開の流れ

分類と公開の選択は [CONTRIBUTING.md](../CONTRIBUTING.md#リリース分類) に従います。
正本は `Cargo.toml` の `package.version`、lockfile は `Cargo.lock` の `game-screen-pick` package です。
`.github/release.json` は SemVer と GitHub Release を宣言しています。Docker image の公開設定はありません。
共有の `.github/scripts/release.py` は採番インフラであり、製品の実行・開発品質 gate の依存とは区別します。
`project_version` は Copier の初期値で、採番の正本ではありません。

公開 workflow は `main` push または `main` を指定した手動実行で動きます。別 branch／tag は checkout・認証設定・採番の前に停止します。
標準 `GITHUB_TOKEN` に `contents:write` と `pull-requests:read` を許可し、`main` と tag への通常 push が許可された repository 設定を使います。
保護設定や認証で拒否された場合は停止し、workflow で保護を回避しません。

1. PR の品質 CI と配布検証を完了し、本人の承認後に最終 PR を squash merge する。
2. controller が最新 `main` に一つの commit として取り込まれた最新のマージ済み PR と、その分類だけを検証する。ラベルなしなら採番・公開を成功扱いでスキップする。
3. controller が SemVer を計算し、Git tag と draft を含む Release の未使用を確認する。認証・照会の失敗を未使用と扱わない。
4. version source と lockfile の version 項目だけを更新し、元 SHA・PR・version・run ID の採番記録を付けた commit と tag を atomic push する。
5. 同じ Actions run で採番 commit の SHA を checkout し、品質 gate と両 target の配布物を作る。draft Release へ不足 asset を upload し、再 download した全 archive・checksum・source SHA を検証してから公開する。
6. 安定版の最新の完成済み Release だけを GitHub Latest にする。prerelease は Latest を変更しない。

workflow は `release-main`、`queue:max`、`cancel-in-progress:false` で直列化します。
待機中に実行定義・script・設定が変わった場合は新しい run に任せます。履歴・PR base・マージイベントの照会失敗は採番前に停止します。
merge commit と複数 main commit の rebase merge は拒否します。一つの commit として取り込む方法は履歴上区別できないため、GitHub では squash を選びます。
複数の過去 PR の分類を集計せず、最新 PR の分類で現在の `main` 全体を公開します。
採番と tag の競合時は最新 `main` の対象 PR を選び直し、最大三回再計算します。force push はしません。
後続 job の対象は採番 SHA であり、元イベントの `GITHUB_SHA` を配布に使いません。tag の別 workflow 起動も待ちません。

SemVer の prerelease patch は末尾数値を増やし、channel／core／安定版への変更は明示 version の下限が必要です。
core を増やす分類ではその core の prerelease も指定でき、巻き戻しは拒否します。初回の設定導入以前の PR は再分類しません。

## 失敗と復旧

| 状態 | 操作 |
| --- | --- |
| 分類・設定・権限の拒否 | 示された PR ラベル・設定を確認し、必要な変更の承認後に元 run を再実行する。 |
| squash／履歴証拠の不成立 | PR base・main 履歴・API を確認する。merge／複数 commit rebase 後は新しい squash PR で修正し、その最新 PR の分類を使う。 |
| `main` 競合 | 自動再計算が三回失敗したら元 run を再実行する。 |
| push 応答不明 | remote の採番記録と tag を確認し、成功済みなら同じ採番 commit を使う。 |
| build・検証失敗 | 採番 commit と tag を残す。元 run は同じ番号・source を使う。製品修正は新 PR・新番号にする。 |
| asset の部分 upload | 元 run を再実行し、同じ採番 commit の不足分を継続する。異なる内容で上書きしない。 |
| 公開済み状態の不整合 | 上書きせず停止し、公開状態を確認する。 |
| 古い run の再実行 | 既存公開は復旧できるが、新しい完成済み Release の Latest を巻き戻さない。 |

元 run の再実行は run ID に対応する採番 commit を再利用し、後から別 PR がマージされても再採番しません。
前回採番以後に新しいマージ済み PR がない通常の手動実行は、新しい Release を作りません。過去の公開復旧には元 run を使います。
