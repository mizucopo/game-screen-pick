# 配布・採番・公開

## 配布対象と検証

Rust binary の配布対象は次の四つです。各 target の native build と実行を検証し、クロス build だけで対応済みとはしません。

| target | build の下限・検査 |
| --- | --- |
| `aarch64-apple-darwin` | `macos-15` ARM64 runner で `MACOSX_DEPLOYMENT_TARGET=14.0` を指定して build し、Mach-O の deployment target と dynamic library を検査する。 |
| `x86_64-unknown-linux-gnu` | Ubuntu 22.04 で build し、ELF が要求する GLIBC symbol version の最大値を2.35以下に制限する。 |
| `x86_64-pc-windows-msvc` | `windows-latest` x64 runner。Windows 10 x64 を下限として宣言し、PE x64・system DLL のみへの依存・static CRT を検査する。 |
| `aarch64-pc-windows-msvc` | `windows-11-arm` ARM64 runner。Windows 11 ARM64 を下限として宣言し、PE ARM64・system DLL のみへの依存・static CRT を検査する。 |

build の下限宣言とリンク検査は、その最小 OS 上での実行証拠と区別します。
macOS CI の実行環境は 15 であり、14.0 上の実行を検証したことにはしません。
Windows x64 の hosted runner は Server 系であり、Windows 10 の実機検証ではありません。ARM64 は ARM64 runner 上で実行し、x64 emulator 上の成功と区別します。対象 OS と filesystem の記録は CI 結果・配布物で確認します。
`macos-14` runner は [GitHub の廃止予定](https://github.com/actions/runner-images/issues/13518) に従って使いません。
FFmpeg／FFprobe と推論 endpoint は外部依存です。font は binary に内蔵し、本体の実行に開発用 toolchain は要求しません。
FFmpeg の PNG／FFV1、表示回転、`setts` と実 PTS の保持は製品抽出・動画 chunk の試験で確認します。
検証した OS／FFmpeg version と dynamic library は配布物の `BUILD-INFO.json`／`DEPENDENCIES.txt` と PR の検証結果に記録します。
未実測の FFmpeg 最小 version や、宣言値以外の OS の互換性を推測しません。

archive 名と配置は共通テンプレートに従います。`game-screen-pick-<version>-windows-x86_64.zip` / `windows-arm64.zip`、`game-screen-pick-<version>-macos-arm64.tar.gz` / `linux-x86_64.tar.gz` と、同名の `.sha256` を作ります。
展開先の直下に `game-screen-pick.exe`（Windows）または `game-screen-pick`、`config.example.toml`、README、利用文書、license と build・依存情報が入ります。
既存 `2.0.0` の macOS / Linux asset は旧 target suffix と `bin/` 配置のまま保持します。既存 Release の asset は改名・置換しません。Windows を含む新形式の実公開は別途承認後です。
依存 package の license 一覧、font の OFL と本体の MIT license を同梱します。設定・録画・生成画像・cache は含めません。
利用者の取得・checksum・配置・設定・更新手順は [README](../README.md) を参照してください。

native の対象環境で、公開せず配布物を作れます。
Rust toolchain は `rust-toolchain.toml` の **1.99.0** に固定します。Rust を使う品質・配布 CI job は checkout したこの file を rustup で有効にし、元 run の再実行も同じ採番 commit の pin を使います。
配布用の開発環境には Python 3.11 以降（共有 builder）、`rust-docs`／`rust-src`（license収録用）、FFmpeg／FFprobe、jq、checksum tool が必要です。Windows は Git Bash と Visual Studio の MSVC build tools、`dumpbin` も使います。これらは配布 binary の runtime 要件ではありません。
rustup を使う場合は `rustup component add rust-docs rust-src` で収録用資料を用意します。macOS は GNU tar を `gtar` として用意します。

```sh
sh tests/check.sh
sh scripts/package-release.sh aarch64-apple-darwin dist
# Linux x86_64 GNU では次を使う:
# sh scripts/package-release.sh x86_64-unknown-linux-gnu dist
# Windows の native Git Bash では次のいずれかを使う:
# PYTHON=python sh scripts/package-release.sh x86_64-pc-windows-msvc dist
# PYTHON=python sh scripts/package-release.sh aarch64-pc-windows-msvc dist
```

build・Cargo executable の選択・archive・checksum・native 起動 smoke は変更していない共有 `.github/scripts/release.py` を使います。`.github/release.json` の `publication.rust.targets` と Copier の `rust_release_targets` が四対象を選びます。
`stage_command` の `scripts/stage-release.sh` は製品固有の依存検査・license・`BUILD-INFO.json`・文書だけを追加します。Windows の配布 build は `-C target-feature=+crt-static` を指定し、MSVC の別途 runtime 導入を要求しない system DLL allowlist を検査します。
macOS / Linux の配布 CI では `scripts/install-ci-ffmpeg.sh` に記録した公式 FFmpeg **6.1.6** の source と SHA-256 から、CI の一時 prefix に FFmpeg／FFprobe を build します。
Windows は `scripts/install-ci-windows-tools.ps1` の固定 BtbN FFmpeg **9.0.2 系** archive と SHA-256 を使い、FFmpeg／FFprobe 自身も対象 CPU の PE であることを確認してから実行します。
macOS の Homebrew FFmpeg 更新によって archive 内の依存記録が変わることを防ぎます。これは CI tool の準備で、利用者の FFmpeg を自動 install・更新する処理や archive への同梱はありません。
各 target の clean 環境で展開した binary の help/version、readonly validate、実 FFmpeg 抽出と合成 mock E2E を確認します。
成功した固定応答 E2E を、実 backend/model の能力、Brave の適用契約と live 接続、実録画の Human Review の証拠にはしません。
これらの未実施項目と実施条件は [検証状況](verification.md) に残します。

## 公開前の承認境界

変更は非 main branch で実装し、`main` 向け PR で全品質 gate、独立レビュー、配布検証を揃えます。
公開を希望しない PR は `release:*` ラベルを付けません。文書整理や CI 検証のために実公開は行いません。
公開する場合は PR の実差分から分類を選び、公開対象を確認してから既存 controller に採番を任せます。手動で product version は変更しません。

実公開の前に、対象 version、GitHub Releases の四つの archive と checksum、対象 head、検証結果・未実施項目を本人へ提示します。
承認は、PR の squash merge、controller による `main` の採番 commit と tag の atomic push、build・asset upload・Release 公開・Latest 更新を含む一連の操作に対して得ます。
PR の実装依頼や Issue の close だけから、この承認を推定しません。認証・権限設定、Homebrew 等の下流公開は個別の承認対象です。
Issue の close とコード完了から、実録画・実 backend・契約・Human Review の完了を推定しません。

## 設定と公開の流れ

分類と公開の選択は [CONTRIBUTING.md](../CONTRIBUTING.md#リリース分類) に従います。
正本は `Cargo.toml` の `package.version`、lockfile は `Cargo.lock` の `game-screen-pick` package です。
`.github/release.json` は SemVer と GitHub Release を宣言しています。Docker image の公開設定はありません。
共有の `.github/scripts/release.py` は採番・配布インフラであり、製品の runtime 依存とは区別します。製品の品質 gate は配布契約も検証します。
`project_version` は Copier の初期値で、採番の正本ではありません。

公開 workflow は `main` push または `main` を指定した手動実行で動きます。別 branch／tag は checkout・認証設定・採番の前に停止します。
標準 `GITHUB_TOKEN` に `contents:write` と `pull-requests:read` を許可し、`main` と tag への通常 push が許可された repository 設定を使います。
保護設定や認証で拒否された場合は停止し、workflow で保護を回避しません。

1. PR の品質 CI と配布検証を完了し、本人の承認後に最終 PR を squash merge する。
2. controller が最新 `main` に一つの commit として取り込まれた最新のマージ済み PR と、その分類だけを検証する。ラベルなしなら採番・公開を成功扱いでスキップする。
3. controller が SemVer を計算し、Git tag と draft を含む Release の未使用を確認する。認証・照会の失敗を未使用と扱わない。
4. version source と lockfile の version 項目だけを更新し、元 SHA・PR・version・run ID の採番記録を付けた commit と tag を atomic push する。
5. 同じ Actions run で採番 commit の SHA から共通 matrix を読み、四つの native runner で品質 gate と配布物を作る。共通 uploader が draft Release へ不足 asset を upload し、製品 guard が再 download した全 archive・checksum・source SHA を検証してから共通 controller が Downloads と公開を完了する。製品 guard の exact SHA `target_commitish` に合わせ、draft 作成では既存の `--target <採番 SHA>` 指定を保持する。
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
| asset の部分 upload | 元 run を再実行し、同じ採番 commit の不足分を継続する。draft に残った期待名の空の `starter` は削除後に再 upload する。異なる内容で上書きしない。 |
| 公開済み状態の不整合 | 上書きせず停止し、公開状態を確認する。 |
| 古い run の再実行 | 既存公開は復旧できるが、新しい完成済み Release の Latest を巻き戻さない。 |

元 run の再実行は run ID に対応する採番 commit を再利用し、後から別 PR がマージされても再採番しません。
前回採番以後に新しいマージ済み PR がない通常の手動実行は、新しい Release を作りません。過去の公開復旧には元 run を使います。

GitHub の [upload API](https://docs.github.com/en/rest/releases/assets#upload-a-release-asset) は upstream 502 の後に空の `starter` を残すことがあります。
復旧では既存の uploaded asset の bytes をすべて照合した後、draft・source・期待名・asset ID・`starter`・size 0 を再確認したものだけを削除し、不足分として再 upload します。
公開済み Release、期待外の名前、非空の `starter`、uploaded asset の不一致は削除せず停止します。

Rust version と FFmpeg source の固定は、同じ source を別の tool version で再 build する変化を抑えます。ただし hosted runner image、macOS SDK・system library、C compiler、共通 archiver の時刻情報、FFmpeg の build 環境まで固定した完全な再現 build は保証しません。部分 upload の復旧では、成功済み native builder の retained artifact を使い、失敗した公開 job の再実行を優先します。
これらが更新されると binary や `DEPENDENCIES.txt`、license 資料、archive bytes が変わる場合があります。元 run の再実行で既存 asset と一致しない場合も上書きせず停止し、元の検証済み成果物と build 環境を確認します。製品・配布の修正は新 PR と新番号で扱います。

## Copier 更新時に保持する配布要件

共有 `.github/scripts/release.py` は最新 template の生成物をそのまま使います。
`prepare`・`plan`・`rust-matrix`・`rust-build`・`distribution`・`latest` により、採番 commit、対象選択、build・archive・checksum、upload、Downloads、公開・Latest を共通化します。PR / ローカルの `package-release.sh` も同じ共通 builder を呼び、別の packaging 実装を持ちません。
最低 OS・依存 library・license・BUILD-INFO の追加は `stage_command` の製品 hook に置きます。最終 archive の FFmpeg / mock E2E、全 target の provenance・再取得・既存 bytes 照合は製品側で維持します。
共有 `create` は `target_commitish` を指定しないため、採番 SHA の明示を必要とする既存の作成処理を保持します。
Release は `--target "$RELEASE_SHA"` を指定して draft として作ります。製品 guard が既存 bytes を検証し、共通 uploader で不足分を補った後に全 asset を再取得・検証してから、共通 `distribution` を使って公開します。空 starter の削除は製品 guard が同一性と削除完了まで確認し、uploaded asset を上書きしません。

Copier の既存回答 `rust_version: stable` は保持し、実際の build toolchain は上記の `rust-toolchain.toml` の固定値を維持します。
clean な非 main branch で正式 `repo-template/main` に対する `copier update --trust --defaults --vcs-ref HEAD` を使います。`rust_release_targets` は上記四 target とし、`publication.rust.targets` と native CI の対象を一致させます。独自 controller を追加せず、Copier conflict は共通 build と製品 hook / guard の境界を保って解決します。
品質 workflow の FFmpeg／FFprobe、`sh tests/check.sh`、四つの native 配布 builder、Cargo と Copier の release inputs、
この文書の配布・承認・復旧手順は repository 固有の差分です。`.codex/project.md` の project 指示も保持します。
CodeQL の新しい回答 `use_gh_actions_codeql` は template の既定値 `false` を採用し、workflow や GitHub の解析設定は追加・変更しません。
