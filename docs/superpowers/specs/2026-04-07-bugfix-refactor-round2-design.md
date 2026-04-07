# バグ修正・リファクタリング 第2弾 デザイン

日付: 2026-04-07
ブランチ: feature/refactor_laro
アプローチ: 全27項目を1PRでまとめる

---

## グループA: バグ修正

### A-1: `id()` → `path` による辞書キー変更

**問題:** `id(image)`（メモリアドレス）を辞書キーに使用。オブジェクトのコピー・再構築でlookupが静かに失敗する。

**対象ファイル:**
- `src/models/content_filter_result.py` — フィールド `adaptive_scores_by_image_id: dict[int, ...]` → `adaptive_scores_by_path: dict[str, ...]`
- `src/services/content_filter.py` — `id(image)` → `image.path`
- `src/services/whole_input_profiler.py` — `id(image)` → `image.path`
- `src/utils/file_utils.py` — `id(res)` → `res.path`、戻り値 `dict[str, str]`
- `src/utils/report_writer.py` — `id(candidate)` → `candidate.path`
- 関連テストファイル

### A-2: 重複メソッド削除

**問題:** `MetricCalculator.calculate_raw_norm_metrics` と `calculate_all_metrics` が同一実装。

**対象ファイル:**
- `src/analyzers/metric_calculator.py` — `calculate_raw_norm_metrics` を削除
- `tests/analyzers/test_metric_calculator.py` — 呼び出しを `calculate_all_metrics` に更新

### A-3: 無駄な計算の遅延評価

**問題:** `ContentFilter._classify_static_rejection_reason` で `system_ui_signal` と `veiled_transition_score` が早期リターン前に計算される。`calculate_bright_washout_score` も二重計算。

**対象ファイル:**
- `src/services/content_filter.py` — `system_ui_signal` / `veiled_transition_score` の計算を `_is_fade_transition` 呼び出し直前に移動。`bright_washout_score` を `_classify_static_rejection_reason` 内で一度だけ計算し、`_is_fade_transition` に引数で渡す
- `src/utils/transition_metrics.py` — `calculate_veiled_transition_score` のシグネチャを変更し、呼び出し元から `bright_washout_score` を受け取るようにする（内部での再計算を排除）

### A-4: HSV特徴量の二重計算解消

**問題:** `extract_combined_features` と `extract_content_features` がそれぞれ独立して `extract_hsv_features` を呼び出し、同じ画像に対してresize・HSV変換・ヒストグラム計算が2回実行される。

**対象ファイル:**
- `src/analyzers/feature_extractor.py` — `extract_combined_features` / `extract_content_features` がHSV結果を受け取るようシグネチャ変更
- `src/analyzers/batch_pipeline.py` — `_process_single_result` でHSV特徴量を一度だけ計算し、両メソッドに渡す

### A-5: `select_diverse_indices` の冗長スキャン修正

**問題:** 各閾値ステップでインデックス0から再スキャンし、`rejected_by_similarity_set` のチェックがない。

**対象ファイル:**
- `src/utils/vector_utils.py` — 内側ループに `if idx in rejected_by_similarity_set: continue` を追加

### A-6: `_execute` パラメータリネーム

**問題:** `input` パラメータがPython組み込みをシャドウ（PEP 8違反）。

**対象ファイル:**
- `src/main.py` — `input` → `input_path`、`output` → `output_path` にリネーム（Click引数名・内部参照も含む）

---

## グループB: 堅牢性向上

### B-1: `CLIPModelManager` にスレッドロック追加

**問題:** `_ensure_model_loaded` の check-then-act がアトミックでなく、スレッドセーフではない。

**対象ファイル:**
- `src/analyzers/clip_model_manager.py` — `threading.Lock` を導入し、ダブルチェックロッキングパターンを実装

### B-2: ユーティリティモジュールのテスト追加

**問題:** `ConfigLoader`、`FileUtils`、`ReportWriter`、`ResultFormatter` に専用テストがない。

**追加ファイル:**
- `tests/utils/test_config_loader.py`
- `tests/utils/test_file_utils.py`
- `tests/utils/test_report_writer.py`
- `tests/utils/test_result_formatter.py`

### B-3: 例外キャッチの絞り込み

**問題:** `_compute_chunk_boundaries` で `except Exception` が `PermissionError` 等を黙殺。

**対象ファイル:**
- `src/analyzers/batch_pipeline.py` — `ExceptionHandler.get_expected_image_errors()` に変更

### B-4: `SelectionProfile` にウェイト合計検証追加

**問題:** `quality_weights` の合計値が検証されていない。

**対象ファイル:**
- `src/models/selection_profile.py` — `__post_init__` で合計が 0.99〜1.01 の範囲内か検証

### B-5: バリデーションメソッドのエッジケーステスト追加

**問題:** CLI バリデータのエッジケースが未テスト。

**対象ファイル:**
- `tests/test_main.py` — parametrize テストケースを追加

### B-6: テストヘルパーの重複解消

**問題:** `_feature` / `_near_duplicate` が3ファイルに重複定義。

**対象ファイル:**
- `tests/conftest.py` — ヘルパーを集約
- `tests/services/test_scene_scorer.py` — conftest からインポート
- `tests/services/test_scene_mix_selector.py` — 同上
- `tests/services/test_game_screen_picker.py` — 同上

### B-7: 未使用importの修正

**問題:** `clip_model_manager.py` で `Optional` import が不要、`Optional[str]` と `str | None` が混在。

**対象ファイル:**
- `src/analyzers/clip_model_manager.py` — `Optional` import 削除、`Optional[str]` → `str | None`

### B-8: `NormalizedMetrics` に範囲検証追加

**問題:** ドキュメントで「0.0〜1.0の範囲」と明記されているが検証がない。

**対象ファイル:**
- `src/models/normalized_metrics.py` — `__post_init__` で全フィールドの範囲を検証

### B-9: `conftest.py` ドキュメント修正

**問題:** `create_scored_candidate` のdocstringに `outlier_rejected` パラメータが欠落。

**対象ファイル:**
- `tests/conftest.py` — docstring に `outlier_rejected` を追加

---

## グループC: 設計改善

### C-1: `cv2.setNumThreads(1)` のスコープ化

**問題:** `BatchPipeline.__init__` でプロセス全体のOpenCVスレッド数を変更。理由のコメントがなく元に戻せない。

**対象ファイル:**
- `src/analyzers/batch_pipeline.py` — `__init__` から削除
- `src/main.py` — エントリーポイントで設定、コメントで理由を記載

### C-2: `ContentFilter` の責務分離

**問題:** 382行・12メソッド・3責務（静的検出、時系列検出、オーケストレーション）が混在。

**対象ファイル:**
- `src/services/content_filter.py` — 静的検出メソッド群を `StaticRejectClassifier` に抽出、`ContentFilter` はコンポジションで利用
- `src/services/__init__.py` に変更はない（空ファイルを維持）
- 対象メソッド: `_classify_static_rejection_reason`, `_detect_blackout`, `_detect_whiteout`, `_detect_single_tone`, `_detect_bright_transition`, `_detect_dark_transition`, `_is_fade_transition`
- `C-3`（`_is_fade_transition` の分割）はこの作業に含める

### C-3: `_is_fade_transition` の分割（C-2に内包）

65行・8パラメータのメソッドを、4つの独立条件チェックを個別メソッドに分割。

### C-4: `SceneScorer` のゼロベクトル対応

**問題:** ゼロノルム特徴ベクトルの画像が density=0.0 になる。セマンティックに不正確。

**対象ファイル:**
- `src/services/scene_scorer.py` — ゼロノルム検出時に density=`-inf` を付与

### C-5: `SceneMixSelector._build_band_queues` をPythonスライスに変更

**問題:** PythonオブジェクトのNumPy配列化は意味がなく `None` フィルタが必要。

**対象ファイル:**
- `src/services/scene_mix_selector.py` — `np.array_split` → プレーンPythonのリストスライス（天井除算）

### C-6: CLIP推論パスの精度統一

**問題:** 単一画像推論（fp32）とバッチ推論（fp16）で精度が異なる。

**対象ファイル:**
- `src/analyzers/clip_model_manager.py` — `get_normalized_image_features` に `torch.autocast` を追加

### C-7: 定数クラスをモジュール定数に変更

**問題:** インスタンス化されないクラスで定数を保持する擬似ネームスペース。

**対象ファイル:**
- `src/constants/content_filter_thresholds.py` — クラス定数 → モジュールレベル定数
- `src/constants/transition_thresholds.py` — 同上
- 全呼び出し側 — `Thresholds.X` → `X` に変更

### C-8: `LayoutAnalyzer` の `np.std` → `cv2.meanStdDev` に統一

**対象ファイル:**
- `src/analyzers/layout_analyzer.py`

### C-9: `AnalyzerConfig.__post_init__` のボイラープレート削減

**問題:** 7箇所のほぼ同一バリデーションパターン。

**対象ファイル:**
- `src/models/analyzer_config.py` — `_validate_positive(name, value)` ヘルパーを抽出

### C-10: `ConfigLoader` の未知セクション警告追加

**問題:** タイプ等の設定ミスが黙って無視される。

**対象ファイル:**
- `src/utils/config_loader.py` — 未知キーを `logging.warning` で出力

### C-11: `LayoutAnalyzer` のマジックナンバーを名前付き定数に

**対象ファイル:**
- `src/analyzers/layout_analyzer.py` — クラスレベル定数として抽出

### C-12: `_score_candidates` 戻り値の整理

**問題:** `_profile_scores` が常に破棄される。

**対象ファイル:**
- `src/services/game_screen_picker.py` — 戻り値タプルから `_profile_scores` を削除、呼び出し側も更新

---

## 除外事項

- 既存のパブリックAPI（CLI引数、TOML設定キー）の互換性は維持する
- 機能追加やアルゴリズムの変更は行わない
- パフォーマンス計測・ベンチマークは本スコープ外

## 前提条件

- Python 3.13+、pytest、mypy、ruff
- `uv run task test` で ruff + mypy + pytest が全て通ること
