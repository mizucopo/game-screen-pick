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
    # Arrange / Act
    artifact = SelectedImageArtifact(
        image_id="frm_" + "a" * 64,
        relative_path="images/0001_test_aaaaaaaaaaaa.webp",
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
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="path"):
        SelectedImageArtifact(
            image_id="frm_" + "a" * 64,
            relative_path="images/../outside.webp",
            sha256="b" * 64,
            width=1,
            height=1,
            size_bytes=1,
        )
