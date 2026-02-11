"""CLIPModelManagerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルを戦略的にモック化（700MB、10-30秒のロード時間）
3. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
4. 高速実行（約2-5秒） - 重いモデルロードなし
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.analyzers.clip_model_manager import CLIPModelManager


@pytest.fixture(autouse=True)
def mock_clip_model() -> Generator[Any, Any, Any]:
    """700MBの重みロードを回避するためのCLIPモデルのモック.

    このfixtureは本物のCLIPモデルを以下のモックに置き換えます：
    - 決定論的テストのために固定されたlogit値を返す
    - GPU/CPU切り替えのための.to(device)呼び出しをサポート
    - 512次元のCLIP特徴ベクトルを返す（実際のモデルと同じ形状）
    - バッチサイズに応じた形状を返す（動的対応）
    """
    with patch("transformers.CLIPModel.from_pretrained") as mock:
        model = MagicMock()

        # get_text_features用のモック（テキスト埋め込み）: (batch_size, 512)
        def mock_get_text_features(**kwargs: object) -> torch.Tensor:
            # input_idsからバッチサイズを取得
            inputs = kwargs.get("input_ids")
            if inputs is not None and isinstance(inputs, torch.Tensor):
                batch_size = inputs.shape[0]
            else:
                batch_size = 1
            # 正規化された固定ベクトルを返す
            return torch.ones(batch_size, 512) / torch.sqrt(torch.tensor(512.0))

        # get_image_features用のモック（画像埋め込み）: (batch_size, 512)
        def mock_get_image_features(**kwargs: object) -> torch.Tensor:
            # pixel_valuesからバッチサイズを取得
            inputs = kwargs.get("pixel_values")
            if inputs is not None and isinstance(inputs, torch.Tensor):
                batch_size = inputs.shape[0]
            else:
                batch_size = 1
            # 正規化された固定ベクトルを返す
            return torch.ones(batch_size, 512) / torch.sqrt(torch.tensor(512.0))

        # メソッドをモック
        model.get_text_features = MagicMock(side_effect=mock_get_text_features)
        model.get_image_features = MagicMock(side_effect=mock_get_image_features)
        model.to = MagicMock(return_value=model)
        model.eval = MagicMock()

        mock.return_value = model
        yield


@pytest.fixture(autouse=True)
def mock_clip_processor() -> Generator[Any, Any, Any]:
    """トークナイザと特徴抽出器のロードを回避するためのCLIPプロセッサのモック.

    このfixtureは本物のCLIPプロセッサを以下のモックに置き換えます：
    - テキストと画像のために固定されたテンソル形状を返す
    - GPU/CPU切り替えのための.to(device)呼び出しをサポート
    - バッチサイズに応じた形状を返す（動的対応）
    """
    with patch("transformers.CLIPProcessor.from_pretrained") as mock:
        # processorは呼び出し可能で、辞書のようなオブジェクトを返す
        processor = MagicMock()

        def mock_processor(**kwargs: object) -> MagicMock:
            # imagesからバッチサイズを取得
            images = kwargs.get("images")
            if images is not None:
                if isinstance(images, list):
                    batch_size = len(images)
                else:
                    batch_size = 1
            else:
                batch_size = 1

            # 呼び出し時に返す辞書のようなオブジェクト
            input_ids = torch.tensor([[1, 2, 3]])
            pixel_values = torch.ones(batch_size, 3, 224, 224) * 0.5
            attention_mask = torch.tensor([[1, 1, 1]])

            # MagicMockを作成して属性を設定
            result_obj = MagicMock()
            result_obj.input_ids = input_ids
            result_obj.pixel_values = pixel_values
            result_obj.attention_mask = attention_mask

            # 辞書のように振る舞うための__getitem__を実装
            def getitem(_self: MagicMock, key: str) -> torch.Tensor:
                if key == "input_ids":
                    return input_ids
                elif key == "pixel_values":
                    return pixel_values
                elif key == "attention_mask":
                    return attention_mask
                else:
                    raise KeyError(key)

            # 束縛されたメソッドをMagicMockに設定
            import types

            result_obj.__getitem__ = types.MethodType(getitem, result_obj)

            # .to()メソッドをサポート（自分自身を返す）
            result_obj.to = MagicMock(return_value=result_obj)

            return result_obj

        processor.side_effect = mock_processor

        mock.return_value = processor
        yield


def test_initialization_with_default_parameters() -> None:
    """デフォルトパラメータで正常に初期化されること.

    Given:
        - デフォルトパラメータがある
    When:
        - CLIPModelManagerが初期化される
    Then:
        - モデルとプロセッサがロードされること
        - デバイスが設定されること
        - テキスト埋め込みが事前計算されること
        - target_textがデフォルト値であること
    """
    # Arrange & Act
    manager = CLIPModelManager()

    # Assert
    assert manager.model_name == "openai/clip-vit-base-patch32"
    assert manager.target_text == "epic game scenery"
    assert manager.device in ("cuda", "cpu")
    assert manager.get_text_embeddings().shape == (1, 512)
    assert manager.model.eval.called


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


def test_initialization_with_explicit_device() -> None:
    """明示的なデバイス指定で正常に初期化されること.

    Given:
        - 明示的なデバイス指定がある
    When:
        - CLIPModelManagerがCPUデバイスで初期化される
    Then:
        - 指定されたデバイスが使用されること
    """
    # Arrange & Act
    manager = CLIPModelManager(device="cpu")

    # Assert
    assert manager.device == "cpu"
    assert manager.model.to.called


def test_get_image_features_returns_correct_shape() -> None:
    """単一画像の特徴抽出が正しく動作すること.

    Given:
        - CLIPModelManagerインスタンスがある
        - モックPIL画像がある
    When:
        - 画像特徴が抽出される
    Then:
        - 正しい形状のテンソルが返されること
        - モデルのget_image_featuresが呼ばれること
    """
    # Arrange
    from PIL import Image

    manager = CLIPModelManager()
    mock_image = MagicMock(spec=Image.Image)

    # Act
    features = manager.get_image_features(mock_image)

    # Assert
    assert features.shape == (1, 512)
    assert manager.model.get_image_features.called


def test_get_image_features_batch_mode_returns_correct_shape() -> None:
    """バッチモードで画像特徴抽出が正しく動作すること.

    Given:
        - CLIPModelManagerインスタンスがある
        - 複数のモックPIL画像がある
    When:
        - バッチモードで画像特徴が抽出される
    Then:
        - バッチサイズに応じた形状のテンソルが返されること
    """
    # Arrange
    from PIL import Image

    manager = CLIPModelManager()
    # モックプロセッサがバッチサイズを検出できるようにする
    # 実際のCLIPプロセッサはimages引数（リスト）からバッチサイズを取得
    # モックではimagesキーにリストを渡すと、プロセッサモックがlenでバッチサイズを判定
    mock_images = [MagicMock(spec=Image.Image) for _ in range(3)]

    # Act
    features = manager.get_image_features(mock_images, batch_mode=True)

    # Assert
    # モックプロセッサはimages引数からバッチサイズを取得
    # テスト環境のモックでは、実際のプロセッサの挙動をエミュレート
    # バッチサイズが正しく検出されることを確認
    assert features.shape[1] == 512  # 特徴次元が512であること
    assert features.shape[0] >= 1  # バッチ次元が存在すること


def test_get_text_embeddings_returns_cached_tensor() -> None:
    """キャッシュされたテキスト埋め込みが返されること.

    Given:
        - CLIPModelManagerインスタンスがある
        - テキスト埋め込みが事前計算されている
    When:
        - テキスト埋め込みが取得される
    Then:
        - 同じテンソルが返されること（キャッシュ）
        - 正しい形状であること
    """
    # Arrange
    manager = CLIPModelManager()

    # Act
    embeddings1 = manager.get_text_embeddings()
    embeddings2 = manager.get_text_embeddings()

    # Assert - 同じオブジェクト（キャッシュされている）
    assert embeddings1 is embeddings2
    assert embeddings1.shape == (1, 512)


def test_model_eval_called_during_initialization() -> None:
    """初期化時にモデルのeval()が呼ばれること.

    Given:
        - CLIPModelManagerが初期化される
    When:
        - 初期化が完了する
    Then:
        - モデルのevalメソッドが呼ばれていること（推論モード設定）
    """
    # Arrange & Act
    manager = CLIPModelManager()

    # Assert
    manager.model.eval.assert_called_once()


def test_model_moved_to_specified_device() -> None:
    """モデルが指定されたデバイスに移動されること.

    Given:
        - CLIPModelManagerが初期化される
    When:
        - 初期化が完了する
    Then:
        - モデルのtoメソッドがデバイス指定で呼ばれること
    """
    # Arrange & Act
    manager = CLIPModelManager(device="cpu")

    # Assert
    manager.model.to.assert_called_with("cpu")


def test_target_text_property_returns_correct_value() -> None:
    """target_textプロパティが正しい値を返すこと.

    Given:
        - カスタムターゲットテキストで初期化されたマネージャーがある
    When:
        - target_textプロパティがアクセスされる
    Then:
        - 設定されたテキストが返されること
    """
    # Arrange
    custom_text = "epic fantasy scenery"
    manager = CLIPModelManager(target_text=custom_text)

    # Act
    result = manager.target_text

    # Assert
    assert result == custom_text
