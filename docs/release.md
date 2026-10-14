# 配布・採番・公開

## 配布対象と検証

Rust binary の配布対象は次の二つです。これ以外の target は対応済みとしません。

| target | build の下限・検査 |
| --- | --- |
| `aarch64-apple-darwin` | `macos-15` ARM64 runner で `MACOSX_DEPLOYMENT_TARGET=14.0` を指定して build し、Mach-O の deployment target と dynamic library を検査する。 |
| `x86_64-unknown-linux-gnu` | Ubuntu 22.04 で build し、ELF が要求する GLIBC symbol version の最大値を2.35以下に制限する。 |

build の下限宣言とリンク検査は、その最小 OS 上での実行証拠と区別します。
macOS CI の実行環境は 15 であり、14.0 上の実行を検証したことにはしません。
`macos-14` runner は [GitHub の廃止予定](https://github.com/actions/runner-images/issues/13518) に従って使いません。
FFmpeg／FFprobe と推論 endpoint は外部依存です。font は binary に内蔵し、本体の実行に Rust toolchain や Python 環境は要求しません。
FFmpeg の PNG／FFV1、表示回転、`setts` と実 PTS の保持は製品抽出・動画 chunk の試験で確認します。
検証した OS／FFmpeg version と dynamic library は配布物の `BUILD-INFO.json`／`DEPENDENCIES.txt` と PR の検証結果に記録します。
未実測の FFmpeg 最小 version や、宣言値以外の OS の互換性を推測しません。

archive 名は `game-screen-pick-<Cargo version>-<target>.tar.gz`、checksum は同名の `.sha256` です。
展開すると同じ名前の directory に `bin/game-screen-pick`、`config.example.toml`、README、利用文書、license と build・依存情報が入ります。
依存 package の license 一覧、font の OFL と本体の MIT license を同梱します。設定・録画・生成画像・cache は含めません。
利用者の取得・checksum・配置・設定・更新手順は [README](../README.md) を参照してください。

native の対象環境で、公開せず配布物を作れます。
Rust toolchain は `rust-toolchain.toml` の **1.99.0** に固定します。Rust を使う品質・配布 CI job は checkout したこの file を rustup で有効にし、元 run の再実行も同じ採番 commit の pin を使います。
`rust-docs`／`rust-src`（license収録用）、FFmpeg／FFprobe、jq、checksum tool、GNU tar が必要です。
rustup を使う場合は `rustup component add rust-docs rust-src` で収録用資料を用意します。macOS は GNU tar を `gtar` として用意します。

```sh
sh tests/check.sh
sh scripts/package-release.sh aarch64-apple-darwin dist
# Linux x86_64 GNU では次を使う:
# sh scripts/package-release.sh x86_64-unknown-linux-gnu dist
```

build は `cargo build --release --locked` を使います。`Cargo.toml` と `Cargo.lock` を一致させ、実行 file・dynamic library・license・archive を確認してから checksum を作ります。
両 target の配布 CI では `scripts/install-ci-ffmpeg.sh` に記録した公式 FFmpeg **6.1.6** の source と SHA-256 から、CI の一時 prefix に FFmpeg／FFprobe を build します。
macOS の Homebrew FFmpeg 更新によって archive 内の依存記録が変わることを防ぎます。これは CI tool の準備で、利用者の FFmpeg を自動 install・更新する処理や archive への同梱はありません。
各 target の clean 環境で展開した binary の help/version、readonly validate、実 FFmpeg 抽出と合成 mock E2E を確認します。
成功した固定応答 E2E を、実 backend/model の能力、Brave の適用契約と live 接続、実録画の Human Review の証拠にはしません。
これらの未実施項目と実施条件は [検証状況](verification.md) に残します。

## 公開前の承認境界

変更は非 main branch で実装し、`main` 向け PR で全品質 gate、独立レビュー、配布検証を揃えます。
公開を希望しない PR は `release:*` ラベルを付けません。文書整理や CI 検証のために実公開は行いません。
公開する場合は PR の実差分から分類を選び、公開対象を確認してから既存 controller に採番を任せます。手動で product version は変更しません。

実公開の前に、対象 version、GitHub Releases の二つの archive と checksum、対象 head、検証結果・未実施項目を本人へ提示します。
承認は、PR の squash merge、controller による `main` の採番 commit と tag の atomic push、build・asset upload・Release 公開・Latest 更新を含む一連の操作に対して得ます。
PR の実装依頼や Issue の close だけから、この承認を推定しません。認証・権限設定、Homebrew 等の下流公開は個別の承認対象です。
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

## release 基盤の Python 依存

共有の `.github/scripts/release.py` は、現役の採番・公開 controller として保持します。
PR の分類、番号の未使用確認、採番 commit と tag の一致、同じ run の再実行、公開状態と Latest の保護を担うため、代替なしには削除しません。
Rust 本体の旧実装ではなく、Copier 管理の release 基盤です。

| 用途 | Python／uv の扱い |
| --- | --- |
| 利用者の binary 実行・最終配布物 | Python／uv／venv は不要。runtime 用 Docker image でも Python 不在を検査する。 |
| ローカル build/test、`tests/check.sh`、Rust 品質・配布検証 CI | Rust／shell／jq を使い、Python／uv は要求しない。 |
| PR の release 分類チェック、採番・公開制御 CI | `python3 -I` で controller を実行する。標準 library の `tomllib` を使うため Python 3.11 以上が必要。uv／venv／pip package は不要。 |

PR の分類チェックは公開ラベルがない場合にも起動するため、PR の全 CI が Python 不要という意味ではありません。
通常 build/test と利用手順を Rust 専用とする範囲には、共有 release controller の実行を含めません。

controller の呼び出し元は次の五箇所です。

| workflow / job / step | command | 用途 |
| --- | --- | --- |
| `release-classification.yml` / `release-classification` / Validate optional classification and its reason | `classification` | PR の公開分類と理由を確認する。 |
| `release.yml` / `prepare` / Number latest main and reserve its tag | `prepare` | 最新 main の採番、commit と tag の予約・再利用を行う。 |
| `release.yml` / `release` / Verify numbering commit and read publication data | `plan` | 採番記録と公開対象を確認する。 |
| `release.yml` / `promote-latest` / Verify numbering commit and read publication data | `plan` | Latest 判定前に同じ採番 source を再確認する。 |
| `release.yml` / `promote-latest` / Check newest completed release | `latest` | 完成した最新の安定版を判定し、後続 step で Latest に昇格する。 |

[残存整理 #398](https://github.com/mizucopo/game-screen-pick/issues/398) での棚卸しでは、追跡対象の `.py` はこの controller 一件だけです。
旧 Python アプリ・テスト、`.pyi`、`pyproject.toml`、`uv.lock`、`.python-version`、requirements／pytest 設定はありません。
`.copier-answers.yml` の `use_python: false` は製品・開発環境の選択であり、release controller の無効化指定ではありません。
README・受入条件の Python 不要という契約、`scripts/Dockerfile.artifact-runtime` の不在検査、旧 cache を拒否する Rust テストの Python 文字列は保持します。
いずれも Python の実行依存や削除待ちの旧資産ではありません。

release 分類・公開 CI も Python 不要にする場合は、先に `mizucopo/repo-template` で共通 controller の代替言語・実行環境と互換性を決めます。
採番・同一 commit/tag・再実行・公開/Latest の保護を上流テストで確認し、controller と全呼び出し元を一緒に Copier 更新で反映してから、Python の例外を外します。
consumer 単独で controller を書き換えたり、品質・artifact smoke・Python 不在検査を削除して代替にはしません。
[repo-template #204](https://github.com/mizucopo/repo-template/issues/204) の Docker 公開直後の確認再試行は別の上流作業として進め、この整理では移植・撤去しません。

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

Rust version と FFmpeg source の固定は、同じ source を別の tool version で再 build する変化を抑えます。ただし hosted runner image、macOS SDK・system library、C compiler、GNU tar／gzip、FFmpeg の build 環境まで固定した完全な再現 build は保証しません。
これらが更新されると binary や `DEPENDENCIES.txt`、license 資料、archive bytes が変わる場合があります。元 run の再実行で既存 asset と一致しない場合も上書きせず停止し、元の検証済み成果物と build 環境を確認します。製品・配布の修正は新 PR と新番号で扱います。

## Copier 更新時に保持する配布要件

共有 `.github/scripts/release.py` は最新 template の生成物をそのまま使います。
`prepare`・`plan`・`latest` による採番と公開分類を採用し、汎用の `create`・`distribution` はこの配布 workflow から呼びません。
汎用 Rust 配布の Linux 単体 archive では、上記の二つの target、checksum、同梱資料、実行検証を満たせないためです。
共有 `create` は `target_commitish` を指定しないため、採番 SHA の明示を必要とする既存の作成処理を保持します。
Release は `--target "$RELEASE_SHA"` を指定して draft として作り、既存の配布 script で採番 SHA・分類・全 asset の bytes を確認してから公開します。
共通の配布案内処理による draft 公開も、この検証経路の代わりには使いません。

Copier の既存回答 `rust_version: stable` は保持し、実際の build toolchain は上記の `rust-toolchain.toml` の固定値を維持します。
品質 workflow の FFmpeg／FFprobe、`sh tests/check.sh`、二つの native 配布 builder、Cargo と Copier の release inputs、
この文書の配布・承認・復旧手順は repository 固有の差分です。`.codex/project.md` の project 指示も保持します。
CodeQL の新しい回答 `use_gh_actions_codeql` は template の既定値 `false` を採用し、workflow や GitHub の解析設定は追加・変更しません。
