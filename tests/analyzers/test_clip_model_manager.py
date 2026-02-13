"""CLIPModelManagerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルを戦略的にモック化（700MB、10-30秒のロード時間）
3. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
4. 高速実行（約2-5秒） - 重いモデルロードなし
"""

import pytest


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
        - CLIPモデルとプロセッサのモックが利用可能
    When:
        - CLIPModelManagerが初期化される
    Then:
        - target_textが正しく設定されること
        - テキスト埋め込みが事前計算されること
    """
    # Arrange & Act
    manager = (
        CLIPModelManager()
        if target_text is None
        else CLIPModelManager(target_text=target_text)
    )

    # Assert
    assert manager.target_text == expected_text
    assert manager.get_text_embeddings() is not None
