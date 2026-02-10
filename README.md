# game-screen-pick
ゲームスクリーンショットから品質・ジャンル・多様性を考慮して最適な画像を自動選択するAIツール

## インストール

```bash
uv sync
```

## 使用方法

### CLIコマンド (推奨)

```bash
uv run game-screen-pick <入力フォルダ> [オプション]
```

### 直接実行

```bash
uv run python src/main.py <入力フォルダ> [オプション]
```

### オプション

- `-c <フォルダ>`, `--copy-to <フォルダ>`: 選択した画像をコピーする出力フォルダ
- `-n <数値>`, `--num <数値>`: 選択枚数 (デフォルト: 10)
- `-g <ジャンル>`, `--genre <ジャンル>`: ゲームジャンル (2d_rpg, 3d_rpg, fps, tps, 2d_action, 2d_shooting, 3d_action, puzzle, racing, strategy, adventure, mixed)
- `-s <数値>`, `--similarity <数値>`: 類似度しきい値 (デフォルト: 0.82)
- `-r`, `--recursive`: サブフォルダも検索

### 使用例

```bash
# FPSゲームのスクリーンショットから15枚選択
uv run game-screen-pick ./screenshots -g fps -n 15

# 2D RPGゲームのスクリーンショットを選択してコピー
uv run game-screen-pick ./screenshots -c ./output -g 2d_rpg -n 20

# サブフォルダを含めて検索
uv run game-screen-pick ./screenshots -r -n 10
```
