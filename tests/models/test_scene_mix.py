"""SceneMix.calculate_allocationのテスト。"""

import pytest

from src.constants.scene_label import SceneLabel
from src.models.scene_mix import SceneMix


class TestCalculateAllocation:
    """calculate_allocationメソッドのテスト。"""

    def test_正の剰余で配分合計がtotalと一致すること(self) -> None:
        """正の剰余がある場合、配分の合計がtotalと一致されること。

        Given:
            - play=0.7, event=0.3の比率が設定される
            - total=11が設定される
        When:
            - calculate_allocationが実行される
        Then:
            - 配分の合計がtotalと一致されること
        """
        # Arrange
        scene_mix = SceneMix(play=0.7, event=0.3)
        total = 11

        # Act
        result = scene_mix.calculate_allocation(total)

        # Assert
        assert sum(result.values()) == total

    def test_負の剰余で配分合計がtotalと一致すること(self) -> None:
        """負の剰余がある場合、配分の合計がtotalと一致されること。

        Given:
            - play=0.80, event=0.21の比率が設定される
            - total=200が設定される
        When:
            - calculate_allocationが実行される
        Then:
            - 配分の合計がtotalと一致されること
        """
        # Arrange
        scene_mix = SceneMix(play=0.80, event=0.21)
        total = 200

        # Act
        result = scene_mix.calculate_allocation(total)

        # Assert
        assert sum(result.values()) == total

    def test_負の剰余で小数部が小さい方から減算されること(self) -> None:
        """負の剰余がある場合、小数部が小さい方から減算されること。

        Given:
            - play=0.80, event=0.21の比率が設定される
            - total=200が設定される
        When:
            - calculate_allocationが実行される
        Then:
            - 小数部が小さいplayから減算されること
        """
        # Arrange
        scene_mix = SceneMix(play=0.80, event=0.21)
        total = 200

        # Act
        result = scene_mix.calculate_allocation(total)

        # Assert
        assert result[SceneLabel.PLAY] == 158
        assert result[SceneLabel.EVENT] == 42
        assert sum(result.values()) == total

    def test_剰余ゼロで変更されないこと(self) -> None:
        """剰余がゼロの場合、配分が変更されないこと。

        Given:
            - play=0.7, event=0.3の比率が設定される
            - total=10が設定される
        When:
            - calculate_allocationが実行される
        Then:
            - 整数切り捨ての結果がそのまま返されること
        """
        # Arrange
        scene_mix = SceneMix(play=0.7, event=0.3)
        total = 10

        # Act
        result = scene_mix.calculate_allocation(total)

        # Assert
        assert result[SceneLabel.PLAY] == 7
        assert result[SceneLabel.EVENT] == 3
        assert sum(result.values()) == total


@pytest.mark.parametrize(
    ("play", "event", "total", "expected_play", "expected_event"),
    [
        # 剰余ゼロのケース
        (0.7, 0.3, 10, 7, 3),
        # 正の剰余のケース
        (0.7, 0.3, 11, 8, 3),
        (0.3, 0.7, 11, 3, 8),
        # 負の剰余のケース
        (0.80, 0.21, 200, 158, 42),
        (0.51, 0.50, 100, 50, 50),
    ],
)
def test_calculate_allocation_パラメータ化(
    play: float,
    event: float,
    total: int,
    expected_play: int,
    expected_event: int,
) -> None:
    """各種入力値で配分が正しく計算されること。

    Given:
        - 様々な比率と総数が設定される
    When:
        - calculate_allocationが実行される
    Then:
        - 配分の合計がtotalと一致し、期待値が返されること
    """
    # Arrange
    scene_mix = SceneMix(play=play, event=event)

    # Act
    result = scene_mix.calculate_allocation(total)

    # Assert
    assert result[SceneLabel.PLAY] == expected_play
    assert result[SceneLabel.EVENT] == expected_event
    assert sum(result.values()) == total
