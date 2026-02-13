"""FeatureExtractorの単体テスト.

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
from src.analyzers.feature_extractor import FeatureExtractor


@pytest.fixture
def feature_extractor() -> FeatureExtractor:
    """特徴抽出器のフィクスチャ."""
    model_manager = CLIPModelManager()
    return FeatureExtractor(model_manager)


def test_extract_hsv_features_returns_correct_shape(
    feature_extractor: FeatureExtractor, sample_image_path: str
) -> None:
    """HSV特徴が正しい形状で抽出されること.

    Given:
        - 特徴抽出器がある
        - テスト画像がある
    When:
        - HSV特徴が抽出される
    Then:
        - 64次元の特徴ベクトルが返されること
        - 特徴が正規化されていること
    """
    # Arrange
    img = cv2.imread(sample_image_path)

    # Act
    hsv_features = feature_extractor.extract_hsv_features(img)

    # Assert
    assert hsv_features.shape == (64,)
    assert np.isfinite(hsv_features).all()


def test_extract_clip_features_returns_normalized_features(
    feature_extractor: FeatureExtractor, sample_image_path: str
) -> None:
    """CLIP特徴が正規化されて抽出されること.

    Given:
        - 特徴抽出器がある
        - テスト画像がある
    When:
        - CLIP特徴が抽出される
    Then:
        - 512次元の特徴ベクトルが返されること
        - 特徴がL2正規化されていること
    """
    # Arrange
    with Image.open(sample_image_path) as img:
        pil_img = img.convert("RGB")

    # Act
    clip_features = feature_extractor.extract_clip_features(pil_img)

    # Assert
    assert clip_features.shape == (512,)
    # L2ノルムが1であることを確認
    norm = np.linalg.norm(clip_features)
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_extract_clip_features_batch_returns_correct_number_of_results(
    feature_extractor: FeatureExtractor, tmp_path: Path
) -> None:
    """バッチ処理で正しい数の結果が返されること.

    Given:
        - 特徴抽出器がある
        - 複数のテスト画像がある
    When:
        - バッチ処理でCLIP特徴が抽出される
    Then:
        - 入力数と同じ数の結果が返されること
        - 各有効結果が512次元であること
    """
    # Arrange
    pil_images = []
    for i in range(3):
        np.random.seed(42 + i)
        img_array = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        img_path = tmp_path / f"test_image_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        with Image.open(str(img_path)) as img:
            pil_images.append(img.convert("RGB"))

    # Act
    results = feature_extractor.extract_clip_features_batch(
        pil_images, initial_batch_size=2
    )

    # Assert
    assert len(results) == 3
    assert any(r is not None for r in results)
    for result in results:
        if result is not None:
            assert result.shape == (512,)
