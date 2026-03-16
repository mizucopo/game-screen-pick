# game-screen-pick
ゲームスクリーンショットから `play` / `event` をバランスよく選択するAIツールです。

## インストール

```bash
uv sync
```

## 使用方法

### 実行方法

```bash
uv run game-screen-pick [オプション] <入力フォルダ> <出力フォルダ>
```

### オプション

- `-n <数値>`, `--num <数値>`: 選択枚数
- `-s <数値>`, `--similarity <数値>`: 類似度しきい値（0.0〜1.0、デフォルト: 0.72）
- `-r`, `--recursive`: サブフォルダも検索
- `--profile <auto|active|static>`: 品質スコア重みのプロファイル
- `--config <パス>`: TOML設定ファイル
- `--scene-mix <文字列>`: 画面種別比率。例: `play=0.7,event=0.3`
- `--report-json <パス>`: JSONレポートの出力先
- `--rename`: `play0001.ext` / `event0001.ext` 形式でリネームして出力
- `--seed <数値>`: 乱数シード
- `--batch-size <数値>`: CLIP推論のバッチサイズ
- `--result-max-workers <数値>`: 結果構築の並列ワーカー数
- `--max-dim <数値>`: 画像リサイズ時の長辺の最大ピクセル数
- `--max-memory-mb <数値>`: チャンク処理時のメモリ予算MB

### 使用例

```bash
# スクリーンショットから15枚選択して出力フォルダにコピー
uv run game-screen-pick -n 15 ./screenshots ./output

# play:event = 70:30 で選択
uv run game-screen-pick --scene-mix play=0.7,event=0.3 ./screenshots ./output

# scene ごとに play0001.ext / event0001.ext 形式へリネームして出力
uv run game-screen-pick --rename ./screenshots ./output

# 設定ファイルとJSONレポートを使用
uv run game-screen-pick --config ./picker.toml --report-json ./report.json ./screenshots ./output
```

## 処理の流れ

現在の実装は次の流れです。

1. 入力画像をすべて解析し、CLIP特徴・結合特徴・画質メトリクスを作る
2. 解析結果をもとに `content filter` を実施し、暗転・白飛び・単色・遷移フレームを除外する
3. 残った画像について、画像全体の中でどれだけ近傍類似画像が多いかを `density_score` として計算する
4. `density_score` が高い側を `play`、低い側を `event` として既定で 70/30 に割り当てる
5. `play` / `event` ごとに外れ値を除外し、低スコア帯から高スコア帯まで均等に候補順を組む
6. その順序を保ったまま、全カテゴリ横断で類似画像を除外しながら最終出力を決める
7. 選ばれた画像を単一の出力フォルダへコピーする

### `play` / `event` の意味

- `play`: 入力画像全体から見て、類似画像が多い群。頻出する通常プレイ画面寄りのクラスター
- `event`: 入力画像全体から見て、類似画像が少ない群。希少な演出・変化の大きい画面

### スコアの扱い

- `density_score` は高いほど `play` 寄り、低いほど `event` 寄りです
- ただし、高スコアだけを優先して出力するわけではありません
- 外れ値を除いたうえで、各カテゴリの低スコア帯から高スコア帯までまんべんなく選びます

### 類似度フィルタリング

- 類似度判定は `play` / `event` の分類後に実行します
- 既に選ばれた画像と似すぎる候補は、カテゴリをまたいでも除外します
- `--similarity` はこの除外判定の基準値で、高いほど緩く、低いほど厳しくなります

## 設定ファイル

```toml
[selection]
profile = "auto"

[scene_mix]
play = 0.7
event = 0.3

[thresholds]
similarity = 0.72
```

## 性能チューニング

- `--max-dim`: 小さいほど高速ですが、精度が下がる可能性があります
- `--max-memory-mb`: 大きいほどチャンクサイズが増え、GPU利用率が上がりやすくなります
- `--batch-size`: 大きいほど高速ですが、VRAM消費量が増えます
- `--result-max-workers`: CPU並列度を調整します
