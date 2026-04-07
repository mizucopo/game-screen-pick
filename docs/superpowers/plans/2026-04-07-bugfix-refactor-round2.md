# バグ修正・リファクタリング 第2弾 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** コードレビューで指摘された27項目のバグ修正・リファクタリングを1PRで実装する

**Architecture:** グループA（バグ修正）→ グループB（堅牢性向上）→ グループC（設計改善）の順で実装。依存関係のある変更（定数リファクタ→ContentFilter分離）は順序を守る。

**Tech Stack:** Python 3.13+, pytest, mypy, ruff, uv

---

## File Structure

### 新規ファイル
- `src/services/static_reject_classifier.py` — ContentFilterから抽出した静的検出ロジック
- `tests/utils/test_config_loader.py`
- `tests/utils/test_file_utils.py`
- `tests/utils/test_report_writer.py`
- `tests/utils/test_result_formatter.py`

### 変更ファイル（影響度順）
- `src/services/content_filter.py` — C-2/C-3: StaticRejectClassifier抽出、A-3: 遅延評価
- `src/models/content_filter_result.py` — A-1: id()→path
- `src/services/whole_input_profiler.py` — A-1: id()→path
- `src/utils/transition_metrics.py` — A-3: bright_washout_score引数追加
- `src/utils/file_utils.py` — A-1: id()→path、戻り値型変更
- `src/utils/report_writer.py` — A-1: id()→path、引数型変更
- `src/analyzers/feature_extractor.py` — A-4: HSV引数追加
- `src/analyzers/batch_pipeline.py` — A-4: HSV事前計算、B-3: 例外絞り込み、C-1: cv2.setNumThreads移動
- `src/services/game_screen_picker.py` — C-12: 戻り値整理、A-1: id()→path
- `src/analyzers/clip_model_manager.py` — B-1: ロック追加、B-7: Optional修正、C-6: autocast追加
- `src/constants/content_filter_thresholds.py` — C-7: クラス→モジュール定数
- `src/constants/transition_thresholds.py` — C-7: クラス→モジュール定数
- `src/services/scene_scorer.py` — C-4: ゼロベクトル対応
- `src/services/scene_mix_selector.py` — C-5: Pythonスライス化
- `src/utils/vector_utils.py` — A-5: rejectedチェック追加
- `src/utils/config_loader.py` — C-10: 未知キー警告
- `src/analyzers/layout_analyzer.py` — C-8: cv2.meanStdDev化、C-11: 名前付き定数
- `src/analyzers/metric_calculator.py` — A-2: 重複メソッド削除
- `src/models/selection_profile.py` — B-4: ウェイト検証
- `src/models/normalized_metrics.py` — B-8: 範囲検証
- `src/models/analyzer_config.py` — C-9: バリデーションヘルパー
- `src/main.py` — A-6: パラメータリネーム、C-1: cv2.setNumThreads移動
- `tests/conftest.py` — B-6: ヘルパー集約、B-9: docstring修正
- `tests/analyzers/test_metric_calculator.py` — A-2: メソッド名更新
- `tests/services/test_scene_scorer.py` — B-6: ヘルパー移動
- `tests/services/test_scene_mix_selector.py` — B-6: ヘルパー移動
- `tests/services/test_game_screen_picker.py` — B-6: ヘルパー移動
- `tests/test_main.py` — A-6: パラメータ名更新、B-5: テスト追加
- `tests/services/test_content_filter.py` — A-1: id()→path追従

---

## Task 1: 重複メソッド削除と未使用import修正（A-2, B-7）

**Files:**
- Modify: `src/analyzers/metric_calculator.py:140-153`
- Modify: `src/analyzers/clip_model_manager.py:5, 24`
- Modify: `tests/analyzers/test_metric_calculator.py`

- [ ] **Step 1: `calculate_raw_norm_metrics` を削除**

`src/analyzers/metric_calculator.py` の lines 140-153 を削除:

```python
# 削除対象:
    def calculate_raw_norm_metrics(
        self, img: np.ndarray
    ) -> tuple[RawMetrics, NormalizedMetrics]:
        raw = self.calculate_raw_metrics(img)
        norm = MetricNormalizer.normalize_all(raw)
        return raw, norm
```

- [ ] **Step 2: テストの呼び出しを更新**

`tests/analyzers/test_metric_calculator.py` の `calculate_raw_norm_metrics` → `calculate_all_metrics` に変更。

- [ ] **Step 3: `clip_model_manager.py` の Optional を修正**

`src/analyzers/clip_model_manager.py`:
- line 5: `from typing import Optional` を削除
- line 24: `device: Optional[str] = None` → `device: str | None = None`

- [ ] **Step 4: テスト実行**

Run: `uv run task test`

- [ ] **Step 5: Commit**

```bash
git add src/analyzers/metric_calculator.py src/analyzers/clip_model_manager.py tests/analyzers/test_metric_calculator.py
git commit -m "refactor: 重複メソッド削除と未使用import修正"
```

---

## Task 2: パラメータリネームとモデル検証追加（A-6, B-4, B-8）

**Files:**
- Modify: `src/main.py:310-333, 363-364, 377-381`
- Modify: `src/models/selection_profile.py:6-12`
- Modify: `src/models/normalized_metrics.py:6-28`

- [ ] **Step 1: `_execute` のパラメータをリネーム**

`src/main.py`:
- line 310-316: `@click.argument("input", ...)` → `@click.argument("input_dir", ...)`, `@click.argument("output", ...)` → `@click.argument("output_dir", ...)`
- line 332-333: `input: str, output: str` → `input_dir: str, output_dir: str`
- line 363-364: docstring内 `input:` → `input_dir:`, `output:` → `output_dir:`
- line 377: `input_path = Path(input)` → `input_path = Path(input_dir)`
- line 380: `f"指定パスはフォルダではありません: {input}"` → `f"指定パスはフォルダではありません: {input_dir}"`
- line 381: `param_hint="input"` → `param_hint="input_dir"`
- line 406: `output,` → `output_dir,`

- [ ] **Step 2: `SelectionProfile` にウェイト合計検証を追加**

`src/models/selection_profile.py` の `@dataclass(frozen=True)` クラスに `__post_init__` を追加:

```python
@dataclass(frozen=True)
class SelectionProfile:
    """選定プロファイル."""

    name: str
    quality_weights: dict[str, float]

    def __post_init__(self) -> None:
        """ウェイト合計が1.0であることを検証する."""
        total = sum(self.quality_weights.values())
        if not (0.99 <= total <= 1.01):
            msg = f"quality_weightsの合計は1.0である必要があります(許容誤差±0.01): {total}"
            raise ValueError(msg)
```

- [ ] **Step 3: `NormalizedMetrics` に範囲検証を追加**

`src/models/normalized_metrics.py` の `@dataclass(frozen=True)` クラスに `__post_init__` を追加:

```python
    def __post_init__(self) -> None:
        """全フィールドが0.0〜1.0の範囲内であることを検証する."""
        for field_name in (
            "blur_score", "contrast", "color_richness", "edge_density",
            "dramatic_score", "visual_balance", "action_intensity", "ui_density",
        ):
            value = getattr(self, field_name)
            if not (0.0 <= value <= 1.0):
                msg = f"{field_name}は0.0〜1.0の範囲である必要があります: {value}"
                raise ValueError(msg)
```

- [ ] **Step 4: テスト実行**

Run: `uv run task test`

- [ ] **Step 5: Commit**

```bash
git add src/main.py src/models/selection_profile.py src/models/normalized_metrics.py
git commit -m "refactor: パラメータリネームとモデル検証追加"
```

---

## Task 3: `select_diverse_indices` の冗長スキャン修正（A-5）

**Files:**
- Modify: `src/utils/vector_utils.py:102-106`

- [ ] **Step 1: rejected_by_similarity_set チェックを追加**

`src/utils/vector_utils.py` の line 104-105 の後 (`if idx in selected_index_set: continue` の後) に追加:

```python
                if idx in rejected_by_similarity_set:
                    continue
```

変更後の該当箇所:
```python
        for threshold in threshold_steps:
            for idx, candidate_feat in enumerate(normalized_features):
                if idx in selected_index_set:
                    continue
                if idx in rejected_by_similarity_set:
                    continue

                if len(selected_indices) >= target_count:
                    break
```

- [ ] **Step 2: テスト実行**

Run: `uv run task test`

- [ ] **Step 3: Commit**

```bash
git add src/utils/vector_utils.py
git commit -m "fix: select_diverse_indicesでrejected候補の再スキャンをスキップ"
```

---

## Task 4: 定数クラスをモジュール定数に変更（C-7）

**Files:**
- Modify: `src/constants/content_filter_thresholds.py`
- Modify: `src/constants/transition_thresholds.py`
- 全呼び出し側（content_filter.py, transition_metrics.py, test_content_filter.py 等）

**前提:** ContentFilter分離（Task 9）の前に実施する必要がある。

- [ ] **Step 1: `content_filter_thresholds.py` を変換**

クラス定義を削除し、インデントを除去してモジュールレベル定数にする。値は変更しない。

Before:
```python
class ContentFilterThresholds:
    """..."""
    WHITEOUT_NEAR_WHITE_THRESHOLD: float = 0.85
    ...
```

After:
```python
"""コンテンツフィルターの閾値定数."""

WHITEOUT_NEAR_WHITE_THRESHOLD: float = 0.85
...
```

- [ ] **Step 2: `transition_thresholds.py` を同様に変換**

- [ ] **Step 3: 全呼び出し側を更新**

機械的な置換:
- `ContentFilterThresholds.XXX` → `XXX`（import元を `from ..constants.content_filter_thresholds import XXX` に変更）
- `TransitionThresholds.XXX` → `XXX`（同上）
- `t = ContentFilterThresholds` / `t = TransitionThresholds` のalias削除

対象ファイル:
- `src/services/content_filter.py` — `t = ContentFilterThresholds` → 直接参照
- `src/utils/transition_metrics.py` — `t = TransitionThresholds` → 直接参照
- `tests/services/test_content_filter.py` — もしあれば

- [ ] **Step 4: テスト実行**

Run: `uv run task test`

- [ ] **Step 5: Commit**

```bash
git add src/constants/ src/services/content_filter.py src/utils/transition_metrics.py tests/services/test_content_filter.py
git commit -m "refactor: 定数クラスをモジュールレベル定数に変更"
```

---

## Task 5: `id()` → `path` 辞書キー変更（A-1）

**Files:**
- Modify: `src/models/content_filter_result.py:15`
- Modify: `src/services/whole_input_profiler.py:85-90, 110, 122-131, 138-148`
- Modify: `src/services/content_filter.py:45, 67, 339, 373-375`
- Modify: `src/utils/file_utils.py:82, 92, 97, 115`
- Modify: `src/utils/report_writer.py:20, 36`
- Modify: `src/services/game_screen_picker.py:228, 233`
- 関連テストファイル

- [ ] **Step 1: `ContentFilterResult` のフィールドを変更**

`src/models/content_filter_result.py`:
- `adaptive_scores_by_image_id: dict[int, AdaptiveScores]` → `adaptive_scores_by_path: dict[str, AdaptiveScores]`

- [ ] **Step 2: `WholeInputProfiler` を更新**

`src/services/whole_input_profiler.py`:
- `score_images` 戻り値型: `dict[int, AdaptiveScores]` → `dict[str, AdaptiveScores]`
- `id(image)` → `image.path` (lines 85, 86, 87)
- `_calculate_information_scores` 戻り値型: `dict[int, float]` → `dict[str, float]`
- `information_scores[id(image)]` → `information_scores[image.path]` (line 130)
- `_calculate_visibility_scores` 戻り値型: `dict[int, float]` → `dict[str, float]`
- `visibility_scores[id(image)]` → `visibility_scores[image.path]` (line 141)

- [ ] **Step 3: `ContentFilter` を更新**

`src/services/content_filter.py`:
- line 45: `adaptive_scores[id(image)]` → `adaptive_scores[image.path]`
- line 67: `adaptive_scores_by_image_id=adaptive_scores` → `adaptive_scores_by_path=adaptive_scores`
- line 339: `adaptive_scores: dict[int, AdaptiveScores]` → `adaptive_scores: dict[str, AdaptiveScores]`
- lines 373-375: `adaptive_scores[id(images[index])]` → `adaptive_scores[images[index].path]` (3箇所)

- [ ] **Step 4: `FileUtils.copy_selected_items` を更新**

`src/utils/file_utils.py`:
- line 82: 戻り値型 `dict[int, str]` → `dict[str, str]`
- line 92: docstring "candidate object id" → "candidate path"
- line 97: `copied_paths_by_candidate_id: dict[int, str]` → `copied_paths_by_candidate_id: dict[str, str]`
- line 115: `copied_paths_by_candidate_id[id(res)]` → `copied_paths_by_candidate_id[res.path]`

- [ ] **Step 5: `ReportWriter.write` を更新**

`src/utils/report_writer.py`:
- line 20: `output_paths_by_candidate_id: dict[int, str] | None` → `dict[str, str] | None`
- line 36: `output_paths_by_candidate_id.get(id(candidate))` → `output_paths_by_candidate_id.get(candidate.path)`

- [ ] **Step 6: `GameScreenPicker.select_from_analyzed` を更新**

`src/services/game_screen_picker.py`:
- line 228: `selected_ids = {id(candidate) for candidate in selected}` → `selected_paths = {candidate.path for candidate in selected}`
- line 233: `if id(candidate) not in selected_ids` → `if candidate.path not in selected_paths`

- [ ] **Step 7: テスト実行**

Run: `uv run task test`

- [ ] **Step 8: Commit**

```bash
git add src/models/content_filter_result.py src/services/whole_input_profiler.py src/services/content_filter.py src/utils/file_utils.py src/utils/report_writer.py src/services/game_screen_picker.py tests/
git commit -m "fix: id()をpathベースの辞書キーに変更して堅牢性を向上"
```

---

## Task 6: 無駄な計算の遅延評価（A-3）

**Files:**
- Modify: `src/services/content_filter.py:189-257`
- Modify: `src/utils/transition_metrics.py:61-66`

- [ ] **Step 1: `calculate_veiled_transition_score` に `bright_washout_score` 引数を追加**

`src/utils/transition_metrics.py` line 61-66 のシグネチャを変更:

```python
def calculate_veiled_transition_score(
    raw_metrics: "RawMetrics",
    adaptive_scores: "AdaptiveScores",
    heuristics: "LayoutHeuristics",
    normalized_metrics: "NormalizedMetrics",
    bright_washout_score: float | None = None,
) -> float:
```

line 77 の内部計算を条件分岐に:

```python
    _bright_washout_score = (
        bright_washout_score
        if bright_washout_score is not None
        else calculate_bright_washout_score(raw_metrics)
    )
```

以降の `bright_washout_score` 参照を `_bright_washout_score` に変更。

- [ ] **Step 2: `_classify_static_rejection_reason` を遅延評価に変更**

`src/services/content_filter.py` lines 189-257 を変更:

```python
    @staticmethod
    def _classify_static_rejection_reason(
        image: AnalyzedImage,
        profile: WholeInputProfile,
        adaptive_scores: AdaptiveScores,
    ) -> str | None:
        raw = image.raw_metrics
        p10_range = max(LUMINANCE_RANGE_P10_MIN, profile.luminance_range.p10)
        p25_range = max(LUMINANCE_RANGE_P25_MIN, profile.luminance_range.p25)

        # 全パスで必要な計算
        bright_washout_score = calculate_bright_washout_score(raw)
        (
            relative_bright_transition_score,
            relative_dark_transition_score,
            _relative_transition_score,
            _relative_transition_polarity,
        ) = calculate_relative_transition_scores(raw, profile)

        if (reason := ContentFilter._detect_blackout(raw, p10_range)) is not None:
            return reason
        if (
            reason := ContentFilter._detect_whiteout(
                raw, profile, adaptive_scores, bright_washout_score
            )
        ) is not None:
            return reason
        if (
            reason := ContentFilter._detect_single_tone(raw, profile, p10_range)
        ) is not None:
            return reason
        if (
            reason := ContentFilter._detect_bright_transition(
                raw, profile, adaptive_scores,
                bright_washout_score, relative_bright_transition_score,
            )
        ) is not None:
            return reason
        if (
            reason := ContentFilter._detect_dark_transition(
                raw, adaptive_scores, profile, relative_dark_transition_score
            )
        ) is not None:
            return reason

        # _is_fade_transition に到達した時点でのみ計算
        system_ui_signal = calculate_system_ui_signal(image.layout_heuristics)
        veiled_transition_score = calculate_veiled_transition_score(
            raw, adaptive_scores,
            image.layout_heuristics, image.normalized_metrics,
            bright_washout_score=bright_washout_score,
        )
        if ContentFilter._is_fade_transition(
            raw, profile, adaptive_scores, p25_range,
            bright_washout_score, veiled_transition_score, system_ui_signal,
        ):
            return "fade_transition"

        return None
```

- [ ] **Step 3: テスト実行**

Run: `uv run task test`

- [ ] **Step 4: Commit**

```bash
git add src/services/content_filter.py src/utils/transition_metrics.py
git commit -m "perf: 静的検出の遅延評価で不要な計算を削減"
```

---

## Task 7: HSV特徴量の二重計算解消（A-4）

**Files:**
- Modify: `src/analyzers/feature_extractor.py:53-72, 91-112`
- Modify: `src/analyzers/batch_pipeline.py:340-347`

- [ ] **Step 1: `extract_content_features` に hsv_features 引数を追加**

`src/analyzers/feature_extractor.py` line 53-72:

```python
    @staticmethod
    def extract_content_features(
        img: np.ndarray,
        raw_metrics: RawMetrics,
        hsv_features: np.ndarray | None = None,
    ) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        gray_hist = cv2.normalize(gray_hist, gray_hist).flatten()
        _hsv_features = (
            hsv_features
            if hsv_features is not None
            else FeatureExtractor.extract_hsv_features(img)
        )
        metric_features = np.array(
            [
                raw_metrics.luminance_entropy / 8.0,
                raw_metrics.luminance_range / 255.0,
                raw_metrics.near_black_ratio,
                raw_metrics.near_white_ratio,
                raw_metrics.dominant_tone_ratio,
            ],
            dtype=np.float32,
        )
        return np.concatenate([gray_hist, _hsv_features, metric_features]).astype(np.float32)
```

- [ ] **Step 2: `extract_combined_features` に hsv_features 引数を追加**

`src/analyzers/feature_extractor.py` line 91-112:

```python
    def extract_combined_features(
        self,
        img: np.ndarray,
        clip_features: np.ndarray,
        hsv_features: np.ndarray | None = None,
    ) -> np.ndarray:
        _hsv_features = (
            hsv_features
            if hsv_features is not None
            else FeatureExtractor.extract_hsv_features(img)
        )
        return np.concatenate([_hsv_features, clip_features])
```

- [ ] **Step 3: `_process_single_result` でHSVを事前計算**

`src/analyzers/batch_pipeline.py` lines 340-347:

```python
            hsv_features = self.feature_extractor.extract_hsv_features(img)
            combined_features = self.feature_extractor.extract_combined_features(
                img,
                clip_features,
                hsv_features=hsv_features,
            )
            content_features = self.feature_extractor.extract_content_features(
                img,
                raw_metrics,
                hsv_features=hsv_features,
            )
```

- [ ] **Step 4: テスト実行**

Run: `uv run task test`

- [ ] **Step 5: Commit**

```bash
git add src/analyzers/feature_extractor.py src/analyzers/batch_pipeline.py
git commit -m "perf: HSV特徴量の二重計算を解消"
```

---

## Task 8: スレッド安全性と例外処理の改善（B-1, B-3）

**Files:**
- Modify: `src/analyzers/clip_model_manager.py:5, 21-39, 55-68, 113-133`
- Modify: `src/analyzers/batch_pipeline.py:260`

- [ ] **Step 1: CLIPModelManager にスレッドロックを追加**

`src/analyzers/clip_model_manager.py`:
- import追加: `import threading`
- `__init__` に `self._lock = threading.Lock()` を追加
- `_ensure_model_loaded` をダブルチェックロッキングに変更:

```python
    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            logger.info(f"CLIPモデルをロードしています ({self.model_name})...")
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            logger.info(f"モデルをデバイスに転送しています ({self.device})...")
            self._model.to(torch.device(self.device))  # type: ignore[arg-type]
            self._model.eval()
            if self.device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
```

- [ ] **Step 2: `get_normalized_image_features` に autocast を追加（C-6）**

同じファイル line 125-133:

```python
    def get_normalized_image_features(self, pil_image: Image.Image) -> torch.Tensor:
        import contextlib

        autocast_ctx = (
            torch.autocast(device_type=self.device, dtype=torch.float16)
            if self.device in ("cuda", "mps")
            else contextlib.nullcontext()
        )
        with autocast_ctx, torch.inference_mode():
            inputs = self.processor(
                images=pil_image,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            image_features = self.model.get_image_features(**inputs)
            return F.normalize(image_features, p=2, dim=-1)[0]
```

- [ ] **Step 3: `_compute_chunk_boundaries` の例外を絞り込み**

`src/analyzers/batch_pipeline.py` line 260:

```python
            except ExceptionHandler.get_expected_image_errors():
```

- [ ] **Step 4: テスト実行**

Run: `uv run task test`

- [ ] **Step 5: Commit**

```bash
git add src/analyzers/clip_model_manager.py src/analyzers/batch_pipeline.py
git commit -m "fix: CLIPModelManagerのスレッド安全性向上と例外処理の絞り込み"
```

---

## Task 9: ContentFilter の責務分離（C-2/C-3）

**Files:**
- Create: `src/services/static_reject_classifier.py`
- Modify: `src/services/content_filter.py`
- 関連テストファイル

**前提:** Task 4（定数モジュール化）完了後

- [ ] **Step 1: `StaticRejectClassifier` を新規作成**

`src/services/static_reject_classifier.py`:

ContentFilter から以下のメソッド群を抽出:
- `_detect_blackout`
- `_detect_whiteout`
- `_detect_single_tone`
- `_detect_bright_transition`
- `_detect_dark_transition`
- `_classify_static_rejection_reason`
- `_is_fade_transition`

`_is_fade_transition` 内の4つの独立条件ブロックを個別メソッドに分割:
- `_check_obvious_fade`
- `_check_bright_washout_fade`
- `_check_veiled_fade`
- `_check_direct_fade`

`ContentFilter` は `StaticRejectClassifier` をコンポジションで保持し、`filter` メソッドから委譲する。

- [ ] **Step 2: ContentFilter を委譲に変更**

`src/services/content_filter.py`:
- `__init__` に `self._static_classifier = StaticRejectClassifier()` を追加
- `filter` 内の `_classify_static_rejection_reason` 呼び出しを `self._static_classifier.classify(...)` に変更
- 抽出したメソッド群を削除

- [ ] **Step 3: テスト実行**

Run: `uv run task test`

- [ ] **Step 4: Commit**

```bash
git add src/services/static_reject_classifier.py src/services/content_filter.py
git commit -m "refactor: ContentFilterからStaticRejectClassifierを抽出"
```

---

## Task 10: 小規模な設計改善（C-1, C-4, C-5, C-8, C-9, C-10, C-11, C-12）

**Files:**
- Modify: `src/analyzers/batch_pipeline.py:61` — C-1
- Modify: `src/main.py` — C-1
- Modify: `src/services/scene_scorer.py:53-86` — C-4
- Modify: `src/services/scene_mix_selector.py:162-195` — C-5
- Modify: `src/analyzers/layout_analyzer.py` — C-8, C-11
- Modify: `src/models/analyzer_config.py:34-70` — C-9
- Modify: `src/utils/config_loader.py` — C-10
- Modify: `src/services/game_screen_picker.py:106-156, 222-224` — C-12

- [ ] **Step 1: cv2.setNumThreads のスコープ化（C-1）**

`src/analyzers/batch_pipeline.py` line 61 の `cv2.setNumThreads(1)` を削除。

`src/main.py` の `_execute` メソッド先頭（debug設定の後）に移動:

```python
        import cv2 as _cv2
        _cv2.setNumThreads(1)  # OpenCVが独自スレッドプールを作成しないよう制御
```

- [ ] **Step 2: SceneScorer のゼロベクトル対応（C-4）**

`src/services/scene_scorer.py` の `_calculate_density_scores` 内、line 76-79 の前:

```python
            feature_norm = float(np.linalg.norm(
                normalized_features[start + i]
            ))
            if feature_norm < 1e-8:
                raw_scores[start + i] = float("-inf")
                continue
```

- [ ] **Step 3: SceneMixSelector の Pythonスライス化（C-5）**

`src/services/scene_mix_selector.py` lines 172-178 を置換:

```python
        chunk_size = -(-len(ordered) // band_count)  # ceiling division
        groups = [ordered[i:i + chunk_size] for i in range(0, len(ordered), chunk_size)]
```

`numpy` import を削除可能か確認（他の箇所で使用していれば残す）。
`None` フィルタ（line 176-178）を削除。

- [ ] **Step 4: LayoutAnalyzer の改善（C-8, C-11）**

`src/analyzers/layout_analyzer.py`:
- クラスレベル定数を追加:

```python
    BOTTOM_REGION_RATIO: float = 0.65
    STD_NORMALIZATION_DIVISOR: float = 64.0
    DIALOGUE_EDGE_MULTIPLIER: float = 4.0
    DIALOGUE_BRIGHTNESS_FACTOR: float = 0.2
    MENU_EDGE_MULTIPLIER: float = 4.5
    TITLE_EDGE_MULTIPLIER: float = 3.0
    GAME_OVER_BRIGHTNESS_MULTIPLIER: float = 1.2
    GAME_OVER_EDGE_MULTIPLIER: float = 3.0
```

- `np.std` を `cv2.meanStdDev` に変更:

```python
        _, bottom_std_dev = cv2.meanStdDev(bottom_region)
        bottom_std = float(bottom_std_dev[0][0]) / self.STD_NORMALIZATION_DIVISOR
        _, upper_std_dev = cv2.meanStdDev(upper_region)
        upper_std = float(upper_std_dev[0][0]) / self.STD_NORMALIZATION_DIVISOR
        _, global_std_dev = cv2.meanStdDev(gray)
        global_std = float(global_std_dev[0][0]) / self.STD_NORMALIZATION_DIVISOR
```

- マジックナンバーを定数参照に置換

- [ ] **Step 5: AnalyzerConfig のバリデーションヘルパー（C-9）**

`src/models/analyzer_config.py` にプライベートヘルパーを追加し、7箇所のバリデーションを統一:

```python
    @staticmethod
    def _validate_positive(name: str, value: int | float, *, min_value: int | float = 1) -> None:
        if value < min_value:
            msg = f"{name}は{min_value}以上の値を指定してください: {value}"
            raise ValueError(msg)
```

各バリデーションブロックをこのヘルパー呼び出しに置換。

- [ ] **Step 6: ConfigLoader の未知セクション警告（C-10）**

`src/utils/config_loader.py`:
- import追加: `import logging`
- `logger = logging.getLogger(__name__)`
- `load` メソッド内で、既知のセクション以外のキーを警告:

```python
        KNOWN_SELECTION_KEYS = {"profile"}
        KNOWN_SECTIONS = {"selection", "scene_mix", "thresholds"}

        for section_name in raw_data:
            if section_name not in KNOWN_SECTIONS:
                logger.warning(f"未知のセクションを無視しました: [{section_name}]")

        for key in selection:
            if key not in KNOWN_SELECTION_KEYS:
                logger.warning(f"未知のキーを無視しました: [selection] {key}")
```

- [ ] **Step 7: `_score_candidates` 戻り値整理（C-12）**

`src/services/game_screen_picker.py`:
- line 109: 戻り値型から `dict[str, float]` を削除
- line 156: `return candidates, resolved_profile, scene_distribution` に変更（`profile_scores` 削除）
- line 222-224: `candidates, resolved_profile, scene_distribution = self._score_candidates(filtered_images)`

- [ ] **Step 8: テスト実行**

Run: `uv run task test`

- [ ] **Step 9: Commit**

```bash
git add src/analyzers/batch_pipeline.py src/main.py src/services/scene_scorer.py src/services/scene_mix_selector.py src/analyzers/layout_analyzer.py src/models/analyzer_config.py src/utils/config_loader.py src/services/game_screen_picker.py
git commit -m "refactor: 小規模な設計改善を一括適用"
```

---

## Task 11: テスト改善（B-5, B-6, B-9）

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/services/test_scene_scorer.py`
- Modify: `tests/services/test_scene_mix_selector.py`
- Modify: `tests/services/test_game_screen_picker.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: conftest にヘルパーを集約（B-6）**

`tests/conftest.py` に追加:

```python
def _feature(index: int, dim: int = 576) -> np.ndarray:
    """テスト用特徴ベクトルを生成する."""
    feature = np.zeros(dim, dtype=np.float32)
    feature[index] = 1.0
    return feature


def _near_duplicate(base: np.ndarray, index: int) -> np.ndarray:
    """ベース特徴ベクトルの指定位置を微小値に変更したニアデュープリケートを生成する."""
    feature = base.copy()
    feature[index] = 0.01
    return feature
```

3つのテストファイルから `_feature` / `_near_duplicate` を削除し、conftest からインポート:

```python
from tests.conftest import _feature, _near_duplicate
```

注意: `test_content_filter.py` の `_feature` はシグネチャが異なる（dim=101、delta_index/delta パラメータあり）のでそのまま残す。

- [ ] **Step 2: conftest docstring 修正（B-9）**

`tests/conftest.py` の `create_scored_candidate` docstring の Args に `outlier_rejected` を追加。

- [ ] **Step 3: バリデーションテスト追加（B-5）**

`tests/test_main.py` の `test_cli_validates_inputs` の parametrize にケースを追加:

```python
    (
        ["-n", "0"],
        "positive integer",
    ),
    (
        ["-n", "3.5"],
        "positive integer",
    ),
    (
        ["--similarity", "0.0"],
        "0.0~1.0",
    ),
    (
        ["--similarity", "1.0"],
        "0.0~1.0",
    ),
    (
        ["--scene-mix", "play=0.7"],
        "scene_mix total",
    ),
```

- [ ] **Step 4: テスト実行**

Run: `uv run task test`

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/services/test_scene_scorer.py tests/services/test_scene_mix_selector.py tests/services/test_game_screen_picker.py tests/test_main.py
git commit -m "test: テストヘルパー集約とバリデーションテスト追加"
```

---

## Task 12: ユーティリティモジュールのテスト追加（B-2）

**Files:**
- Create: `tests/utils/test_config_loader.py`
- Create: `tests/utils/test_file_utils.py`
- Create: `tests/utils/test_report_writer.py`
- Create: `tests/utils/test_result_formatter.py`

- [ ] **Step 1: `tests/utils/` ディレクトリと `__init__.py` を作成**

- [ ] **Step 2: `test_config_loader.py` を作成**

テストケース:
- `path=None` の場合、空辞書を返す
- 正常なTOMLを読み込める
- `scene_mix` セクションから SceneMix を生成
- 未知のセクションで警告ログが出力される（Task 10 の C-10 実装後）

- [ ] **Step 3: `test_file_utils.py` を作成**

テストケース:
- `rename=False` の場合、元のファイル名でコピー
- `rename=True` の場合、scene別連番でコピー
- `rename=True, requested_num=None` の場合、ValueError
- 重複ファイル名がある場合、ユニークな名前を生成

- [ ] **Step 4: `test_report_writer.py` を作成**

テストケース:
- JSONファイルが正しく出力される
- `output_paths_by_candidate_id` が `path` ベースで動作する（Task 5 の A-1 実装後）
- `whole_input_profile=None` の場合、null として出力

- [ ] **Step 5: `test_result_formatter.py` を作成**

テストケース:
- 基本的なフォーマット出力がエラーなく完了

- [ ] **Step 6: テスト実行**

Run: `uv run task test`

- [ ] **Step 7: Commit**

```bash
git add tests/utils/
git commit -m "test: ユーティリティモジュールのテストを追加"
```

---

## 最終確認

全Task完了後:

```bash
uv run task test
```

ruff + mypy + pytest が全て通ることを確認。
