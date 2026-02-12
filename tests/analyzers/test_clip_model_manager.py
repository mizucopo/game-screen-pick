"""CLIPModelManagerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルを戦略的にモック化（700MB、10-30秒のロード時間）
3. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
4. 高速実行（約2-5秒） - 重いモデルロードなし
"""

from unittest.mock import MagicMock

from PIL import Image

from src.analyzers.clip_model_manager import CLIPModelManager


def test_initialization_with_default_parameters() -> None:
    """デフォルトパラメータで正常に初期化されること.

    Given:
        - デフォルトパラメータがある
    When:
        - CLIPModelManagerが初期化される
    Then:
        - モデル名、ターゲットテキスト、デバイスが正しく設定されること
        - テキスト埋め込みが事前計算されること
    """
    # Arrange & Act
    manager = CLIPModelManager()

    # Assert
    assert manager.model_name == "openai/clip-vit-base-patch32"
    assert manager.target_text == "epic game scenery"
    assert manager.device in ("cuda", "cpu")
    assert manager.get_text_embeddings().shape == (1, 512)


def test_initialization_with_custom_target_text() -> None:
    """カスタムターゲットテキストで正常に初期化されること.

    Given:
        - カスタムターゲットテキストがある
    When:
        - CLIPModelManagerがカスタムテキストで初期化される
    Then:
        - target_textがカスタム値であること
        - テキスト埋め込みが計算されること
    """
    # Arrange & Act
    custom_text = "beautiful landscape"
    manager = CLIPModelManager(target_text=custom_text)

    # Assert
    assert manager.target_text == custom_text
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
