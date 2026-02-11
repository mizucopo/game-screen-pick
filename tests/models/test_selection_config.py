"""SelectionConfigの単体テスト."""

from src.models.selection_config import SelectionConfig


def test_selection_config_has_default_values() -> None:
    """SelectionConfigがデフォルト値で正しく初期化されること.

    Given:
        - SelectionConfigをデフォルト値で作成
    When:
        - 各属性にアクセス
    Then:
        - すべてのデフォルト値が正しく設定されていること
    """
    # Act
    config = SelectionConfig()

    # Assert
    assert config.batch_size == 32
    assert config.threshold_relaxation_steps == [0.03, 0.06, 0.10, 0.15]
    assert config.max_threshold == 0.98


def test_selection_config_can_be_customized() -> None:
    """SelectionConfigがカスタム値で初期化できること.

    Given:
        - カスタム値を指定
    When:
        - SelectionConfigを作成
    Then:
        - 指定した値が正しく設定されていること
    """
    # Arrange
    custom_batch_size = 64
    custom_steps = [0.05, 0.10, 0.20]
    custom_max_threshold = 0.95

    # Act
    config = SelectionConfig(
        batch_size=custom_batch_size,
        threshold_relaxation_steps=custom_steps,
        max_threshold=custom_max_threshold,
    )

    # Assert
    assert config.batch_size == custom_batch_size
    assert config.threshold_relaxation_steps == custom_steps
    assert config.max_threshold == custom_max_threshold


def test_compute_threshold_steps_with_base_threshold() -> None:
    """ベースしきい値から段階的な緩和ステップが計算されること.

    Given:
        - SelectionConfigインスタンスがある
        - ベースしきい値が0.72
    When:
        - しきい値ステップを計算
    Then:
        - 5段階のしきい値が正しく計算されること
        - 最終ステップがmax_threshold以下であること
    """
    # Arrange
    config = SelectionConfig()
    base_threshold = 0.72

    # Act
    steps = config.compute_threshold_steps(base_threshold)

    # Assert
    assert len(steps) == 5
    assert steps[0] == 0.72
    assert steps[1] == 0.75  # 0.72 + 0.03
    assert steps[2] == 0.78  # 0.72 + 0.06
    assert steps[3] == 0.82  # 0.72 + 0.10
    assert steps[4] == 0.87  # 0.72 + 0.15


def test_compute_threshold_steps_respects_max_threshold() -> None:
    """しきい値の上限がmax_thresholdで制限されること.

    Given:
        - SelectionConfigインスタンスがある
        - ベースしきい値が0.95（緩和後にmax_thresholdを超過）
    When:
        - しきい値ステップを計算
    Then:
        - すべてのステップがmax_threshold以下であること
    """
    # Arrange
    config = SelectionConfig(max_threshold=0.98)
    base_threshold = 0.95

    # Act
    steps = config.compute_threshold_steps(base_threshold)

    # Assert
    assert len(steps) == 5
    assert steps[0] == 0.95
    assert steps[1] == 0.98  # 0.95 + 0.03 = 0.98 (上限)
    assert steps[2] == 0.98  # 0.95 + 0.06 = 1.01 -> 0.98 (上限)
    assert steps[3] == 0.98  # 0.95 + 0.10 = 1.05 -> 0.98 (上限)
    assert steps[4] == 0.98  # 0.95 + 0.15 = 1.10 -> 0.98 (上限)


def test_compute_threshold_steps_with_custom_relaxation_steps() -> None:
    """カスタムの緩和ステップが正しく適用されること.

    Given:
        - カスタムの緩和ステップを持つSelectionConfig
        - ベースしきい値が0.70
    When:
        - しきい値ステップを計算
    Then:
        - カスタムステップに基づいて正しく計算されること
    """
    # Arrange
    config = SelectionConfig(
        threshold_relaxation_steps=[0.05, 0.15], max_threshold=0.95
    )
    base_threshold = 0.70

    # Act
    steps = config.compute_threshold_steps(base_threshold)

    # Assert
    assert len(steps) == 3
    assert steps[0] == 0.70
    assert steps[1] == 0.75  # 0.70 + 0.05
    assert steps[2] == 0.85  # 0.70 + 0.15


def test_default_list_is_not_shared_between_instances() -> None:
    """デフォルトのリストがインスタンス間で共有されないこと.

    Given:
        - 2つのSelectionConfigインスタンスを作成
    When:
        - 片方のリストを変更
    Then:
        - もう片方のリストは変更されていないこと
    """
    # Arrange
    config1 = SelectionConfig()
    config2 = SelectionConfig()

    # Act
    config1.threshold_relaxation_steps.append(0.99)

    # Assert
    assert config1.threshold_relaxation_steps == [0.03, 0.06, 0.10, 0.15, 0.99]
    assert config2.threshold_relaxation_steps == [0.03, 0.06, 0.10, 0.15]
