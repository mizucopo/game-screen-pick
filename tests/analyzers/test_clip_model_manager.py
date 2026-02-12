"""CLIPModelManagerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルを戦略的にモック化（700MB、10-30秒のロード時間）
3. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
4. 高速実行（約2-5秒） - 重いモデルロードなし
"""

import pytest
from unittest.mock import MagicMock

from PIL import Image

from src.analyzers.clip_model_manager import CLIPModelManager


@pytest.mark.parametrize(
    "target_text,expected_text",
    [
        (None, "epic game scenery"),  # デフォルト値
        ("beautiful landscape", "beautiful landscape"),  # カスタム値
    ],
)
def test_initialization_sets_text_and_prepares_embeddings(
    target_text: str | None, expected_text: str
) -> None:
    """ターゲットテキストが設定され、テキスト埋め込みが事前計算されること.

    Given:
        - デフォルトまたはカスタムのターゲットテキストがある
    When:
        - CLIPModelManagerが初期化される
    Then:
        - target_textが正しく設定されること
        - テキスト埋め込みが事前計算されること
        - モデル名とデバイスが適切に設定されること
    """
    # Arrange & Act
    if target_text is None:
        manager = CLIPModelManager()
    else:
        manager = CLIPModelManager(target_text=target_text)

    # Assert
    assert manager.target_text == expected_text
    assert manager.model_name == "openai/clip-vit-base-patch32"
    assert manager.device in ("cuda", "cpu")
    assert manager.get_text_embeddings().shape == (1, 512)


def test_get_image_features_returns_correct_shape() -> None:
    """単一画像の特徴抽出が正しく動作すること.

    Given:
        - CLIPModelManagerインスタンスがある
        - モックPIL画像がある
    When:
        - 画像特徴が抽出される
    Then:
        - 正しい形状のテンソルが返されること
    """
    # Arrange
    manager = CLIPModelManager()
    mock_image = MagicMock(spec=Image.Image)

    # Act
    features = manager.get_image_features(mock_image)

    # Assert
    assert features.shape == (1, 512)
