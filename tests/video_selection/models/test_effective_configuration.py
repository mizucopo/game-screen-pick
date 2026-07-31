"""EffectiveConfigurationの単体テスト。"""

from pathlib import Path

import pytest

from src.video_selection.models.effective_configuration import EffectiveConfiguration


def test_image_count_must_be_positive(tmp_path: Path) -> None:
    """正でない要求画像枚数が拒否されること。

    Arrange:
        - 0件の要求画像枚数が用意される
    Act:
        - Effective Configurationが構築される
    Assert:
        - image_countのvalidation errorが返されること
    """
    # Arrange
    image_count = 0

    # Act
    with pytest.raises(ValueError) as error:
        EffectiveConfiguration(
            video_input_folder=tmp_path / "videos",
            output_folder=tmp_path / "output",
            image_count=image_count,
        )

    # Assert
    assert "image_count" in str(error.value)
