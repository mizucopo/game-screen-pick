# Rust の検証素材

秘密情報・実録画・外部 AI への通信を含まない最小入力 corpus。
[受入基準](../../docs/acceptance.md) の抽出・選定・再開・成果物テストに採用する。
この directory に正解の製品出力や保存 cache は置かない。

## 入力の事実

- `videos/01-blocks.mkv`: FFV1、160×96、YUV444 full range、4 fps、6秒、24 frame。
  frame `i` の PTS は `i/4` 秒、duration は 0.25 秒、最初の PTS は 0。
  回転なし。各秒内の4 frame は同じ模様で、2≤t<3 は黒。
- `videos/02-mirrored.mkv`: 同じ条件で各 frame を左右反転した別 source。
  同一 PTS でも黒区間以外の内容が違うため、入力取り違え・向きを確認できる。
- `videos/03-multitrack-rotated.mov`: lossless PNG／RGB24 full range の2映像 track。
  両方とも160×96・4 fps・6秒・24 frame。stream 0 は blocks、非 default、90°反時計回りの display metadata を持つ。
  stream 1 は mirrored、default、回転なし。どちらも attached picture ではない。
  選ぶ stream は最小 index の0で、表示は96×160。表示座標 `(x,y)` は元画像の `(159-y,x)` に対応する。
  neutral chroma から RGB 各 channel は luma と同値。全 PTS・raw pixels・表示寸法と回転後 pixels を照合する。
- `video-tiles.tsv`: `second, tile_y, 10個のluma` の36行。各 tile は16×16、U/V は全画素128。
  原点は左上、x は右向き、y は下向き。二つ目の動画は tile_x を `9-tile_x` とする。
  この入力 recipe は製品の選定・cache から独立している。
- `image-facts.tsv` と `images/`: 左右二分の近黒・近白・低コントラスト画像（64×32）、
  RGB channel と向きを判別する33×19画像。RGB grid の各値は
  `R=(37x+17y)%256, G=(13x+71y)%256, B=(97x+29y)%256`。
  採否閾値や品質の式は #340 が成果物品質から決める。

`tests/check-fixtures.sh` は Rust の確認処理で FFprobe の寸法・codec・pixel format・color range・時間条件と
FFmpeg decode の全画素を入力の事実へ照合する。画像 facts と媒体 directory の file 集合、
固定応答・scenario の必須一覧と、ID・source・候補時刻・chunk/event/transition の整合性も確認する。
Rust 標準 library のみで動き、Python や製品の選定処理を呼ばない。
通常の確認は入力や期待値を書き換えない。FFmpeg/FFprobe 不在は失敗とする。

```sh
sh tests/check-fixtures.sh
```

必要な開発 tool は rustc/rustfmt、FFmpeg/FFprobe、JSON 確認用 jq。明示生成には `-display_rotation` 対応の FFmpeg 6+ が必要。
#339 がこの facts を実際の Rust 抽出 API のテストに接続し、start offset・端点・VFR・attached picture・
異常 probe・日本語/空白 path を追加する。`extraction_cases` の複数 track／display rotation も製品経路で確認する。
入力媒体自体の確認は製品抽出テストの代わりにはならない。最小 FFmpeg 対応条件の製品実測は #344 が行う。

## 固定応答と実 model

`responses.json` は用途側の論理応答と、表示 ID → source／時刻／前後 context の小さな台本。
二段階画像評価、同点、暗転除外、semantic event、直接/生成 Game Context と不正応答の負例を使える。
探索・戦闘・会話は合成模様に付けた架空の scene 名で、実ゲームの認識能力の証拠ではない。
semantic の台本は source 別に定義し、各代表時刻を同じ source の候補・前後 context に対応させる。
単一動画の試験では `B01` を候補・応答から除く。各 stage は実際に要求した ID のみを応答に含める。

#362／#342 は Rust の request/schema に合わせてこの台本を mock HTTP の契約へ接続する。
元動画・実 PTS・画像内容、sheet の表示 ID、前後順序、動画 chunk の frame 順・時間を検証してから固定応答を返す。
ここでの ID/score は mock の入力であり、製品内部 ID、score 式、最終順位を規定しない。
選定順位・同点規則は #340 の採用仕様へ固定し、#343 で同じ Rust の cold／warm／再開結果を照合する。

実 server/model の確認は #362／#343 が別途行う。backend、server/model version、vision 設定、
image/video/JSON 能力と上限、実録画の問題画像・reviewer・合否を記録し、mock 成功と区別する。

## 再開・障害ケース

`scenarios.json` は両方式×単一/複数入力×直接/生成 mock context×cold/warm/途中再開の最小 matrix と期待する成果物。
cache の bytes/schema を先取りせず、#341 の Rust run 自身が作った checkpoint に破損・欠損・中断を注入する。
生成 title／生成条件／解決済み context／接続先変更による無効化、編集済み出力の保持、warm の decode/hash/I/O 上限、未所有出力、
競合、symlink、HTTP・disk・公開・runtime 失敗も同じ記録から試験できる。
これは後続 Issue のテスト入力で、#338 では製品 E2E を実行済みとは扱わない。

## 素材と期待値の更新

1. 受入仕様のどの条件を検証する変更かを PR に書く。新 source の内容・寸法・向き・PTS と recipe、
   必要な応答/障害条件を入力の事実から追加する。製品出力を正解として取り込まない。
2. facts を編集したら、明示 command で別の新規 directory にだけ候補を生成する。既存 directory は拒否する。

   ```sh
   rustc --edition=2024 --deny warnings tests/fixture_inputs.rs -o /tmp/generate-gsp-fixtures
   /tmp/generate-gsp-fixtures /tmp/gsp-fixtures-candidate
   GSP_FIXTURES=/tmp/gsp-fixtures-candidate sh tests/check-fixtures.sh
   ```

3. 新旧入力の内容・PTS・向きと影響する抽出/選定テストを確認する。
   変更理由、facts の根拠、検証 command/結果、reviewer を PR に記録して必要な素材だけ取り込む。
   追加した recipe/source と台本/case は `fixture_inputs.rs`／`fixture_contract.jq` の必須一覧・検証と #339 以降の Rust テストへ接続する。
   通常テストには生成 command を入れず、失敗を消すために期待値を再生成しない。

初回採用（#338）: 4 PNG と一つ目の動画の内容を入力として再利用し、独立した decode/probe で上記 facts を確認した。
2026-10-09 の確認環境は Mac mini、FFmpeg 9.0.2。保存動画の実 luma は19/93/168/242であり、
以前の生成 script の32/96/160/224ではなかった。新 recipe は実入力の tile 値を明記し、
rawvideo の入力・出力とも full range を指定する。容器/JPEG の bytes の同一性は合格条件にしない。
