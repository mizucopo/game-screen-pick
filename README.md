# game-screen-pick
ゲームスクリーンショットから品質・多様性を考慮して最適な画像を自動選択するAIツール

## インストール

```bash
uv sync
```

## 使用方法

### 実行方法

```bash
uv run game-screen-pick <入力フォルダ> [オプション]
```

### オプション

- `-c <フォルダ>`, `--copy-to <フォルダ>`: 選択した画像をコピーする出力フォルダ
- `-n <数値>`, `--num <数値>`: 選択枚数 (デフォルト: 10)
- `-s <数値>`, `--similarity <数値>`: 類似度しきい値 (デフォルト: 0.72)
- `-r`, `--recursive`: サブフォルダも検索
- `--seed <数値>`: 乱数シード（再現可能な結果を得るために指定）
- `--cache-file <パス>`: キャッシュデータベースのパス
- `--no-cache`: キャッシュを無効化する
- `--batch-size <数値>`: CLIP推論のバッチサイズ（デフォルト: 32）
- `--result-max-workers <数値>`: 結果構築の並列ワーカー数

### 使用例

```bash
# スクリーンショットから15枚選択
uv run game-screen-pick ./screenshots -n 15

# 選択した画像をコピー
uv run game-screen-pick ./screenshots -c ./output -n 20

# サブフォルダを含めて検索
uv run game-screen-pick ./screenshots -r -n 10
```

## 選択アルゴリズム

### スコアリング

ブログ用途向けの単一重み設定で画像を評価します：

- `blur_score`: 15% - ぼけのなさ
- `contrast`: 14% - コントラスト
- `color_richness`: 16% - 色の豊富さ
- `visual_balance`: 17% - 視覚的バランス
- `edge_density`: 11% - エッジ密度
- `action_intensity`: 12% - アクションの激しさ
- `ui_density`: 10% - UI要素の密度
- `dramatic_score`: 5% - ドラマチックさ

### 活動量ミックス

「激しい画面だけに偏らない」ことを保証するため、活動量バケット選択を導入しています：

- **活動量計算**: `action_intensity` (55%) + `edge_density` (25%) + `dramatic_score` (20%)
- **バケット分割**: q30/q70で `low/mid/high` の3バケットに分位点自動分割
- **選択比率**: 30% (low) / 40% (mid) / 30% (high) を目標に配分
- **保証**: `num>=3` かつ全バケット非空なら各バケット最低1枚を先取り

これにより、静的な風景画像から激しいアクションシーンまで幅広く選択されます。

### 類似度フィルタリング

選択された画像同士が似すぎないよう、CLIP特徴量を用いて類似度を計算し、
類似度がしきい値を超える画像を除外します。

## キャッシュ機能

解析結果は自動的にキャッシュされ、2回目以降の実行で再利用されます。

- **保存場所**:
  - `--copy-to` 指定時: `{出力フォルダ}/cache.sqlite3`
  - それ以外: `~/.cache/game-screen-pick/cache.sqlite3`
  - `--cache-file` でカスタムパスを指定可能
- **キャッシュ対象**: CLIP特徴量、HSV特徴量、画質メトリクス
- **キャッシュキー**: ファイルパス、ファイルサイズ、更新時刻、モデル名、ターゲットテキスト、解像度

### キャッシュの無効化

一時的にキャッシュを無効化する場合：

```bash
uv run game-screen-pick ./screenshots --no-cache
```

キャッシュを削除する場合：

```bash
# 出力フォルダを指定した場合
rm ./output/cache.sqlite3

# デフォルトの場所
rm ~/.cache/game-screen-pick/cache.sqlite3
```

### キャッシュの無効化条件

以下の場合、キャッシュは無効になり再解析が行われます：

- ファイルが変更された（ファイルサイズまたは更新時刻が異なる）
- アルゴリズムが更新された（内部バージョン変更）
