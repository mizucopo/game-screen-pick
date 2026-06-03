# game-screen-pick
ゲームスクリーンショットから、ブログで使いやすい画像をOllamaの画像分類で選択するAIツールです。

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
- `--ollama-model <文字列>`: Ollamaの画像分類モデル名。未指定の場合はエラー
- `--ollama-host <URL>`: OllamaホストURL。`OLLAMA_HOST` より優先
- `--ollama-timeout <秒>`: Ollama APIタイムアウト秒数（デフォルト: 60）
- `--ollama-max-workers <数値>`: Ollama分類の並列ワーカー数（デフォルト: 1）
- `--no-ollama-cache`: Ollama分類キャッシュを使わない
- `--scene-hint <文字列>`: scene catalog 作成時に渡す補助情報
- `--report-json <パス>`: JSONレポートの出力先
- `--rename`: `battle0001.ext` のように scene slug 別の連番でリネームして出力
- `--batch-size <数値>`: CLIP推論のバッチサイズ
- `--result-max-workers <数値>`: 結果構築の並列ワーカー数
- `--max-dim <数値>`: 画像リサイズ時の長辺の最大ピクセル数
- `--max-memory-gb <数値>`: チャンク処理時のメモリ予算GB

### 使用例

```bash
# スクリーンショットから15枚選択して出力フォルダにコピー
uv run game-screen-pick --ollama-model gemma4 -n 15 ./screenshots ./output

# scene ごとに battle0001.ext / conversation0001.ext 形式へリネームして出力
uv run game-screen-pick --ollama-model gemma4 --rename ./screenshots ./output

# 設定ファイルとJSONレポートを使用
uv run game-screen-pick --config ./picker.toml --report-json ./report.json ./screenshots ./output

# アドベンチャーゲーム向けの補助ヒント
uv run game-screen-pick --ollama-model gemma4 \
  --scene-hint "アドベンチャーゲーム。会話差分が多く、表情や背景の違いを重視したい" \
  ./screenshots ./output

# パズルゲーム向けの補助ヒント
uv run game-screen-pick --ollama-model gemma4 \
  --scene-hint "パズルゲーム。盤面の状態が似やすいので、進行や結果が分かる画像を優先したい" \
  ./screenshots ./output

# RPG向けの補助ヒント
uv run game-screen-pick --ollama-model gemma4 \
  --scene-hint "RPG。戦闘、探索、会話、メニューが混在している" \
  ./screenshots ./output
```

## 処理の流れ

現在の実装は次の流れです。

1. 入力画像をすべて解析し、CLIP特徴・結合特徴・画質メトリクスを作る
2. 解析結果をもとに `content filter` を実施し、暗転・白飛び・単色・遷移フレームを明示的な reject reason 付きで除外する
3. 残った blog candidate から代表画像を最大24枚選び、Ollamaでその実行用の scene catalog を作る
4. scene catalog は3〜8個の scene で構成され、必ず `other` を含む
5. 各 blog candidate を scene catalog のいずれかへ分類し、分類失敗した画像は最終選択対象から外す
6. 同じ scene 内で見た目や構図が近い画像を variant group にまとめ、原則として各 group から代表画像を1枚だけ選ぶ
7. scene ごとの自動均等配分、画質、分類信頼度、類似度除外を組み合わせて最終出力を決める
8. 選定結果を copy / console / JSON report 共通の出力recordへ変換する
9. `OutputPlanner` が `--rename`、scene slug別連番、同名衝突回避、report用 `output_path` をcopyなしで計画する
10. 計画済みの出力先へ画像をコピーし、同じrecordから表示とJSONレポートを生成する

### scene の意味

- `scene_slug`: `battle`、`conversation`、`menu` など、ファイル名やJSONキーに使う英語slug
- `scene_display_name`: `戦闘`、`会話`、`メニュー` など、人が読む日本語名
- `scene_description`: ブログ画像選択に役立つ短い説明文
- `variant_group`: 同じscene内でブログ上の役割が重複する差分画像のまとまり

### 類似度フィルタリング

- 類似度判定は Ollama の scene 分類後に実行します
- 既に選ばれた画像と似すぎる候補は、scene をまたいでも除外します
- `--similarity` はこの除外判定の基準値で、高いほど緩く、低いほど厳しくなります
- アドベンチャーゲームやパズルゲームのように差分が少ない画像が多い場合は、variant group によって同じような画像ばかり選ばれることを避けます

### Ollama分類キャッシュ

- 分類結果は入力画像のあるフォルダ配下の `.game-screen-pick/cache/ollama-scenes.json` に保存されます
- キャッシュキーには画像パス、更新時刻、サイズ、モデル名、scene catalog が含まれます
- キャッシュを使わない場合は `--no-ollama-cache` を指定します

## 設定ファイル

```toml
[selection]
profile = "auto"

[ollama]
model = "gemma4"
host = "http://localhost:11434"
timeout = 60
max_workers = 1

[thresholds]
similarity = 0.72
```

Ollama host の優先順位は `--ollama-host`、`OLLAMA_HOST`、`[ollama].host`、`http://localhost:11434` です。

## 性能チューニング

- `--max-dim`: 小さいほど高速ですが、精度が下がる可能性があります
- `--max-memory-gb`: 大きいほどチャンクサイズが増え、GPU利用率が上がりやすくなります
- `--batch-size`: 大きいほど高速ですが、VRAM消費量が増えます
- `--result-max-workers`: CPU並列度を調整します
