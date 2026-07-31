"""Selected Image Artifact modelのtest。"""

import pytest

from src.video_selection.models.selected_image_artifact import SelectedImageArtifact


def test_relative_webp_with_complete_identity_is_accepted() -> None:
    """完全ID、relative WebP path、hash、寸法が保持されること。

    Arrange:
        - staging済みWebPの公開診断が用意される
    Act:
        - Selected Image Artifactが構築される
    Assert:
        - relative pathとbyte数が保持されること
    """
    # Arrange
    relative_path = "images/0001_test_aaaaaaaaaaaa.webp"

    # Act
    artifact = SelectedImageArtifact(
        image_id="frm_" + "a" * 64,
        relative_path=relative_path,
        sha256="b" * 64,
        width=3840,
        height=2160,
        size_bytes=1024,
    )

    # Assert
    assert artifact.relative_path == "images/0001_test_aaaaaaaaaaaa.webp"
    assert artifact.size_bytes == 1024


def test_parent_segment_in_image_path_is_rejected() -> None:
    """images外へ出るparent segment pathが拒否されること。

    Arrange:
        - parent segmentを持つWebP pathが用意される
    Act:
        - Selected Image Artifactが構築される
    Assert:
        - 公開path契約違反として拒否されること
    """
    # Arrange
    relative_path = "images/../outside.webp"

    # Act
    with pytest.raises(ValueError) as error:
        SelectedImageArtifact(
            image_id="frm_" + "a" * 64,
            relative_path=relative_path,
            sha256="b" * 64,
            width=1,
            height=1,
            size_bytes=1,
        )

    # Assert
    assert "path" in str(error.value)
