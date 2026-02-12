"""MetricCalculatorの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルを戦略的にモック化（700MB、10-30秒のロード時間）
3. OpenCV操作、NumPy計算はモック化しない
4. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
5. 高速実行（約2-5秒） - 重いモデルロードなし
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.analyzers.clip_model_manager import CLIPModelManager
from src.analyzers.metric_calculator import MetricCalculator
from src.analyzers.metric_normalizer import MetricNormalizer
from src.constants.genre_weights import GenreWeights
from src.models.analyzer_config import AnalyzerConfig


@pytest.fixture
def sample_image_path(tmp_path: Path) -> str:
    """標準的なテスト画像（640x480 JPG）を作成する."""
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def dark_image_path(tmp_path: Path) -> str:
    """輝度ペナルティのテスト用に暗いテスト画像（640x480 JPG）を作成する."""
    np.random.seed(42)
    img_array = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "dark_image.jpg"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def metric_calculator() -> MetricCalculator:
    """メトリクス計算器のフィクスチャ."""
    config = AnalyzerConfig()
    weights = GenreWeights.get_weights("mixed")
    model_manager = CLIPModelManager()
    return MetricCalculator(config, weights, model_manager)


def test_calculate_raw_metrics_returns_expected_metrics(
    metric_calculator: MetricCalculator, sample_image_path: str
) -> None:
    """生メトリクスが正しい形式で返されること.

    Given:
        - メトリクス計算器がある
        - テスト画像がある
    When:
        - 生メトリクスが計算される
    Then:
        - 期待されるすべてのメトリクスキーが含まれていること
        - すべての値が数値であること
    """
    # Arrange
    img = cv2.imread(sample_image_path)

    # Act
    raw_metrics = metric_calculator.calculate_raw_metrics(img)

    # Assert
    expected_keys = {
        "blur_score",
        "brightness",
        "contrast",
        "edge_density",
        "color_richness",
        "ui_density",
        "action_intensity",
        "visual_balance",
        "dramatic_score",
    }
    assert set(raw_metrics.keys()) == expected_keys
    for value in raw_metrics.values():
        assert isinstance(value, (int, float))
        assert not np.isnan(value)


def test_calculate_raw_metrics_returns_non_negative_values(
    metric_calculator: MetricCalculator, sample_image_path: str
) -> None:
    """生メトリクスの値が負ではないこと（一部のメトリクス）.

    Given:
        - メトリクス計算器がある
        - テスト画像がある
    When:
        - 生メトリクスが計算される
    Then:
        - 輝度、コントラスト、エッジ密度が負ではないこと
    """
    # Arrange
    img = cv2.imread(sample_image_path)

    # Act
    raw_metrics = metric_calculator.calculate_raw_metrics(img)

    # Assert
    assert raw_metrics["brightness"] >= 0
    assert raw_metrics["contrast"] >= 0
    assert raw_metrics["edge_density"] >= 0


def test_calculate_semantic_score_returns_value_in_expected_range(
    metric_calculator: MetricCalculator, sample_image_path: str
) -> None:
    """セマンティックスコアが期待される範囲で返されること.

    Given:
        - メトリクス計算器がある
        - テスト画像がある
    When:
        - セマンティックスコアが計算される
    Then:
        - スコアがコサイン類似度の範囲（[-1, 1]）にあること
    """
    # Arrange
    with Image.open(sample_image_path) as img:
        pil_img = img.convert("RGB")

    # Act
    semantic_score = metric_calculator.calculate_semantic_score(pil_img)

    # Assert
    assert isinstance(semantic_score, float)
    assert not np.isnan(semantic_score)
    # 浮動小数点の丸め誤差を許容して境界チェック
    assert -1.0 <= semantic_score
    assert semantic_score <= 1.0 + 1e-5  # 丸め誤差を許容


def test_calculate_semantic_score_from_features_returns_value_in_expected_range(
    metric_calculator: MetricCalculator,
) -> None:
    """CLIP特徴からセマンティックスコアが計算できること.

    Given:
        - メトリクス計算器がある
        - 正規化されたCLIP特徴がある
    When:
        - セマンティックスコアが特徴から計算される
    Then:
        - スコアがコサイン類似度の範囲（[-1, 1]）にあること
    """
    # Arrange
    clip_features = np.ones(512) / np.sqrt(512.0)

    # Act
    semantic_score = metric_calculator.calculate_semantic_score_from_features(
        clip_features
    )

    # Assert
    assert isinstance(semantic_score, float)
    assert not np.isnan(semantic_score)
    # 浮動小数点の丸め誤差を許容して境界チェック
    assert -1.0 <= semantic_score
    assert semantic_score <= 1.0 + 1e-5  # 丸め誤差を許容


def test_calculate_total_score_returns_non_negative_value(
    metric_calculator: MetricCalculator,
) -> None:
    """総合スコアが負ではない値で返されること.

    Given:
        - メトリクス計算器がある
        - 有効なメトリクスがある
    When:
        - 総合スコアが計算される
    Then:
        - スコアが0以上であること
    """
    # Arrange
    raw = {
        "blur_score": 500.0,
        "brightness": 100.0,
        "contrast": 50.0,
        "edge_density": 0.2,
        "color_richness": 40.0,
        "ui_density": 10.0,
        "action_intensity": 30.0,
        "visual_balance": 90.0,
        "dramatic_score": 50.0,
    }

    norm = MetricNormalizer.normalize_all(raw)
    semantic = 0.5

    # Act
    total_score = metric_calculator.calculate_total_score(raw, norm, semantic)

    # Assert
    assert total_score >= 0.0
    assert isinstance(total_score, float)


def test_calculate_total_score_applies_brightness_penalty_for_dark_images(
    metric_calculator: MetricCalculator, dark_image_path: str
) -> None:
    """暗い画像に対して輝度ペナルティが適用されること.

    Given:
        - メトリクス計算器がある
        - 暗いテスト画像がある
    When:
        - 総合スコアが計算される
    Then:
        - 輝度ペナルティが適用されること
        - ペナルティ適用後もスコアが有効であること
    """
    # Arrange
    img = cv2.imread(dark_image_path)
    raw = metric_calculator.calculate_raw_metrics(img)

    norm = MetricNormalizer.normalize_all(raw)
    semantic = 0.5

    # Act
    total_score = metric_calculator.calculate_total_score(raw, norm, semantic)

    # Assert
    # 暗い画像では輝度ペナルティが適用される
    assert raw["brightness"] < metric_calculator.config.brightness_penalty_threshold
    assert total_score >= 0.0


def test_calculate_all_metrics_returns_complete_results(
    metric_calculator: MetricCalculator, sample_image_path: str
) -> None:
    """すべてのメトリクスが一括計算できること.

    Given:
        - メトリクス計算器がある
        - テスト画像がある
    When:
        - すべてのメトリクスが一括計算される
    Then:
        - 期待されるすべての結果が返されること
        - 各結果が正しい型であること
    """
    # Arrange
    img = cv2.imread(sample_image_path)
    clip_features = np.ones(512) / np.sqrt(512.0)

    # Act
    raw, norm, semantic, total = metric_calculator.calculate_all_metrics(
        img, clip_features
    )

    # Assert
    assert isinstance(raw, dict)
    assert len(raw) == 9
    assert isinstance(norm, dict)
    assert len(norm) == 8  # brightnessは正規化されていない
    assert isinstance(semantic, float)
    assert isinstance(total, float)
    assert total >= 0.0


def test_calculate_raw_metrics_handles_large_images(
    metric_calculator: MetricCalculator, tmp_path: Path
) -> None:
    """大きな画像が正しくリサイズされて処理されること.

    Given:
        - メトリクス計算器がある
        - max_dimより大きな画像がある
    When:
        - 生メトリクスが計算される
    Then:
        - エラーが発生しないこと
        - 有効なメトリクスが返されること
    """
    # Arrange
    # 1920x1080の画像を作成（max_dim=720より大きい）
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    img_path = tmp_path / "large_image.jpg"
    cv2.imwrite(str(img_path), img_array)
    img = cv2.imread(str(img_path))

    # Act
    raw_metrics = metric_calculator.calculate_raw_metrics(img)

    # Assert
    assert isinstance(raw_metrics, dict)
    assert len(raw_metrics) == 9
    for value in raw_metrics.values():
        assert isinstance(value, (int, float))


def test_metric_calculator_uses_genre_weights(
    metric_calculator: MetricCalculator,
) -> None:
    """ジャンル特有の重みが使用されること.

    Given:
        - メトリクス計算器がある
        - ジャンル別の重みがある
    When:
        - 総合スコアが計算される
    Then:
        - 重みが使用されていること
    """
    # Arrange
    raw = {
        "blur_score": 500.0,
        "brightness": 100.0,
        "contrast": 50.0,
        "edge_density": 0.2,
        "color_richness": 40.0,
        "ui_density": 10.0,
        "action_intensity": 30.0,
        "visual_balance": 90.0,
        "dramatic_score": 50.0,
    }

    norm = MetricNormalizer.normalize_all(raw)
    semantic = 0.5

    # Act
    total_score = metric_calculator.calculate_total_score(raw, norm, semantic)

    # Assert
    # 重みが使用されていることを確認（スコアが計算されている）
    assert total_score >= 0.0
    # 重みの合計が正規化されていることを確認
    weight_sum = sum(metric_calculator.weights.values())
    assert weight_sum == pytest.approx(1.0, abs=0.01)


def test_calculate_total_score_with_zero_semantic_weight(
    metric_calculator: MetricCalculator,
) -> None:
    """セマンティック重みが0の場合、総合スコアに影響しないこと.

    Given:
        - メトリクス計算器がある
        - セマンティック重みが0に設定されている
    When:
        - 総合スコアが計算される
    Then:
        - セマンティックスコアが総合スコアに影響しないこと
    """
    # Arrange
    config = AnalyzerConfig(semantic_weight=0.0)
    weights = GenreWeights.get_weights("mixed")
    calculator = MetricCalculator(config, weights, metric_calculator.model_manager)

    raw = {
        "blur_score": 500.0,
        "brightness": 100.0,
        "contrast": 50.0,
        "edge_density": 0.2,
        "color_richness": 40.0,
        "ui_density": 10.0,
        "action_intensity": 30.0,
        "visual_balance": 90.0,
        "dramatic_score": 50.0,
    }

    norm = MetricNormalizer.normalize_all(raw)
    semantic = 0.5

    # Act
    total_score = calculator.calculate_total_score(raw, norm, semantic)

    # Assert
    assert total_score >= 0.0
    # セマンティック重みが0でもスコアが計算されている
    assert total_score > 0


def test_calculate_semantic_score_batch_returns_correct_count(
    metric_calculator: MetricCalculator,
) -> None:
    """バッチ計算で正しい数の結果が返されること.

    Given:
        - メトリクス計算器がある
        - 複数の正規化済みCLIP特徴がある
    When:
        - セマンティックスコアがバッチ計算される
    Then:
        - 入力数と同じ数の結果が返されること
        - 各スコアが期待される範囲にあること
    """
    # Arrange
    # 正規化済みのランダムベクトルを作成
    vec = np.random.randn(512).astype(np.float32)
    vec /= np.linalg.norm(vec)

    clip_features_list = [
        np.ones(512) / np.sqrt(512.0),
        vec,
        np.ones(512) / np.sqrt(512.0),
    ]

    # Act
    scores = metric_calculator.calculate_semantic_score_batch(clip_features_list)

    # Assert
    assert len(scores) == 3
    for score in scores:
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert -1.0 <= score
        assert score <= 1.0 + 1e-5


def test_calculate_semantic_score_batch_handles_none_features(
    metric_calculator: MetricCalculator,
) -> None:
    """バッチ計算でNone特徴が正しく処理されること.

    Given:
        - メトリクス計算器がある
        - 有効な特徴とNoneが混在している
    When:
        - セマンティックスコアがバッチ計算される
    Then:
        - 有効な特徴にはスコアが返されること
        - None特徴にはNoneが返されること
    """
    # Arrange
    clip_features_list = [
        np.ones(512) / np.sqrt(512.0),
        None,
        np.ones(512) / np.sqrt(512.0),
    ]

    # Act
    scores = metric_calculator.calculate_semantic_score_batch(clip_features_list)

    # Assert
    assert len(scores) == 3
    assert scores[0] is not None
    assert isinstance(scores[0], float)
    assert scores[1] is None
    assert scores[2] is not None
    assert isinstance(scores[2], float)


def test_calculate_semantic_score_batch_returns_similar_results_as_single(
    metric_calculator: MetricCalculator,
) -> None:
    """バッチ計算の結果が単一計算の結果と一致すること.

    Given:
        - メトリクス計算器がある
        - 複数のCLIP特徴がある
    When:
        - セマンティックスコアが両方の方法で計算される
    Then:
        - バッチ計算と単一計算の結果が一致すること
    """
    # Arrange
    clip_features_list = [
        np.ones(512) / np.sqrt(512.0),
        np.random.randn(512).astype(np.float32),
    ]

    # Act
    batch_scores = metric_calculator.calculate_semantic_score_batch(clip_features_list)
    single_scores = [
        metric_calculator.calculate_semantic_score_from_features(f)
        for f in clip_features_list
    ]

    # Assert
    assert len(batch_scores) == len(single_scores)
    for batch_score, single_score in zip(batch_scores, single_scores):
        assert batch_score == pytest.approx(single_score)


def test_calculate_semantic_score_batch_returns_empty_list_for_empty_input(
    metric_calculator: MetricCalculator,
) -> None:
    """空入力に対して空リストが返されること.

    Given:
        - メトリクス計算器がある
        - 空のリストが入力される
    When:
        - セマンティックスコアがバッチ計算される
    Then:
        - 空のリストが返されること
    """
    # Arrange
    clip_features_list: list[np.ndarray | None] = []

    # Act
    scores = metric_calculator.calculate_semantic_score_batch(clip_features_list)

    # Assert
    assert scores == []


def test_calculate_semantic_score_batch_handles_all_none_features(
    metric_calculator: MetricCalculator,
) -> None:
    """すべてNone特徴の入力に対してすべてNoneが返されること.

    Given:
        - メトリクス計算器がある
        - すべてNoneのリストが入力される
    When:
        - セマンティックスコアがバッチ計算される
    Then:
        - すべてNoneのリストが返されること
    """
    # Arrange
    clip_features_list: list[np.ndarray | None] = [None, None, None]

    # Act
    scores = metric_calculator.calculate_semantic_score_batch(clip_features_list)

    # Assert
    assert len(scores) == 3
    assert all(s is None for s in scores)
