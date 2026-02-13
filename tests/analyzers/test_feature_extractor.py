"""FeatureExtractorの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルを戦略的にモック化（700MB、10-30秒のロード時間）
3. OpenCV操作、NumPy計算はモック化しない
4. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
5. 高速実行（約2-5秒） - 重いモデルロードなし
"""

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
