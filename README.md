# game-screen-pick
ゲームスクリーンショットから通常画面・イベント画面・その他画面を
バランスよく選択するAIツール

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

- `-n <数値>`, `--num <数値>`: 選択枚数 (デフォルト: 10、多様性を満たせない場合はこれより少なくなることがあります)
- `-s <数値>`, `--similarity <数値>`: 類似度しきい値
- `-r`, `--recursive`: サブフォルダも検索
- `--profile <auto|active|static>`: 選定プロファイル
- `--config <パス>`: TOML設定ファイル
- `--scene-mix <文字列>`: 画面種別比率。例: `gameplay=0.5,event=0.4,other=0.1`
- `--report-json <パス>`: JSONレポートの出力先
- `--seed <数値>`: 乱数シード（再現可能な結果を得るために指定）
- `--batch-size <数値>`: CLIP推論のバッチサイズ（デフォルト: 32、大きいほど高速だがメモリ消費増加）
- `--result-max-workers <数値>`: 結果構築の並列ワーカー数（デフォルト: 自動設定、0指定時はシングルスレッド）
- `--max-dim <数値>`: 画像リサイズ時の長辺の最大ピクセル数（デフォルト: 720、小さいほど高速だが精度低下）
- `--max-memory-mb <数値>`: チャンク処理時のメモリ予算MB（デフォルト: 512、大きいほどチャンクが大きくなる）

### 使用例

```bash
# スクリーンショットから15枚選択して出力フォルダにコピー
uv run game-screen-pick -n 15 ./screenshots ./output

# 通常画面:イベント画面:その他画面 = 50:40:10 で選択
uv run game-screen-pick --scene-mix gameplay=0.5,event=0.4,other=0.1 ./screenshots ./output

# 設定ファイルとJSONレポートを使用
uv run game-screen-pick --config ./picker.toml --report-json ./report.json ./screenshots ./output
```

### 出力フォルダ

出力フォルダが存在しない場合は自動的に作成されます。
親ディレクトリも含めて必要な階層が自動作成されます。

## 選択アルゴリズム

### 画面種別ミックス

候補画像を `gameplay` / `event` / `other` に分類し、
既定で以下の比率を目標に選択します。

- `gameplay`: 50%
- `event`: 40%
- `other`: 10%

`other` にはメニュー、タイトル、ゲームオーバーなどの非プレイ系画面が含まれます。
比率は `--scene-mix` または設定ファイルで後から調整できます。
ただし、似た画像ばかりになる場合は scene 比率より多様性を優先するため、
最終的な `scene_mix_actual` は目標値からずれることがあります。

### スコアリング

各画像は以下の2段階で評価されます。

- **画面種別スコア**: CLIP のゼロショット判定と OpenCV ヒューリスティクスから `gameplay` / `event` / `other` を推定
- **画質スコア**: `blur_score`, `contrast`, `color_richness`, `visual_balance`, `edge_density`, `action_intensity`, `ui_density`, `dramatic_score`

最終的な選択順は、画面種別スコアを主軸にしつつ、画質スコアを補助として使用します。

### 活動量ミックス

各画面種別バケットの中では、活動量バケットを使って
似たテンポの画面ばかりに偏らないようにしています。

### 類似度フィルタリング

最終的に選ばれる画像全体に対して類似度を計算し、
既に採用済みの画像と似すぎる候補を除外します。
scene bucket ごとに不足が出た場合は他 scene から再配分しますが、
しきい値緩和後も十分に多様な候補がない場合は、
指定枚数より少ない件数で結果を返します。

## 設定ファイル

TOML で以下のように指定できます。

```toml
[selection]
profile = "auto"

[scene_mix]
gameplay = 0.5
event = 0.4
other = 0.1

[thresholds]
similarity = 0.72
```

## 性能チューニング

大規模な画像セットを扱う場合、以下のオプションでメモリ使用量と処理速度を調整できます：

- `--max-dim`: 画像リサイズ時の長辺の最大ピクセル数（デフォルト: 720）
  - 小さいほど高速ですが、精度が低下する可能性があります
- `--max-memory-mb`: チャンク処理時のメモリ予算（デフォルト: 512MB）
  - 大きいほどチャンクサイズが大きくなり、GPU利用率が向上します
- `--batch-size`: CLIP推論のバッチサイズ（デフォルト: 32）
  - 大きいほど高速ですが、VRAM消費量が増加します
- `--result-max-workers`: 結果構築の並列ワーカー数（デフォルト: 自動設定）
  - CPUコア数に応じて自動調整されます。0でシングルスレッドになります
