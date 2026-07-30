"""Canonical Publication Request modelのtest。"""

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from src.video_selection.models.canonical_publication_request import (
    CanonicalPublicationRequest,
)
from src.video_selection.services.select_video_set_images import (
    SpoilerSensitivity,
    select_video_set_images,
)
from tests.video_selection.fakes.canonical_publication_factory import (
    build_canonical_publication_request,
)


def test_complete_domain_graph_is_accepted(tmp_path: Path) -> None:
    """Video Set、Stage、selection、Catalog、Cueが整合すると受理されること。

    Arrange:
        - 完全なcanonical publication domain graphが用意される
    Act:
        - Canonical Publication Requestが構築される
    Assert:
        - run IDとVideo Set selectionが保持されること
    """
    # Arrange
    # Act
    request = build_canonical_publication_request(tmp_path)

    # Assert
    assert isinstance(request, CanonicalPublicationRequest)
    assert request.run_id == "run_20260716T120000Z_fixture"
    assert request.selection_result.shortfall is True


def test_requested_count_mismatch_is_rejected(tmp_path: Path) -> None:
    """Effective Configurationとselectionの要求枚数不一致が拒否されること。

    Arrange:
        - 正常requestと異なるimage_countのconfigurationが用意される
    Act:
        - Canonical Publication Requestが再構築される
    Assert:
        - runとVideo Setの整合違反として拒否されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    configuration = replace(request.configuration, image_count=3)

    # Act
    # Assert
    with pytest.raises(ValueError, match="runまたはVideo Set"):
        replace(request, configuration=configuration)


def test_context_cue_from_another_video_source_is_rejected(tmp_path: Path) -> None:
    """Video Stageと異なるVideo Sourceを指すContext Cueが拒否されること。

    Arrange:
        - 正常requestのContext Cueだけが別Video Fingerprintへ変更される
    Act:
        - Canonical Publication Requestが再構築される
    Assert:
        - Context Cue graphの不整合として拒否されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    stage = request.video_stage_results[0]
    cue = replace(stage.context.cues[0], video_fingerprint="f" * 64)
    context = replace(stage.context, cues=(cue,))
    changed_stage = replace(stage, context=context)

    # Act
    # Assert
    with pytest.raises(ValueError, match="Context Cueが不正"):
        replace(request, video_stage_results=(changed_stage,))


def test_scene_catalog_may_be_absent_when_no_candidate_is_annotated(
    tmp_path: Path,
) -> None:
    """Annotationが0件のshortfallではScene Catalogなしが受理されること。

    Arrange:
        - 有効Candidateが0件のselection resultが用意される
    Act:
        - Scene CatalogなしのCanonical Publication Requestが再構築される
    Assert:
        - 0枚のwarning付き公開入力として受理されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    selection = select_video_set_images(
        (),
        requested_count=request.configuration.image_count,
        spoiler_sensitivity=cast(
            SpoilerSensitivity,
            request.configuration.spoiler_sensitivity,
        ),
        similarity_threshold=request.configuration.similarity_threshold,
    )

    # Act
    empty = replace(request, scene_catalog=None, selection_result=selection)

    # Assert
    assert empty.scene_catalog is None
    assert empty.selection_result.annotated_candidate_count == 0


def test_scene_catalog_is_required_when_candidate_is_annotated(tmp_path: Path) -> None:
    """Annotationが1件以上あるrunではScene Catalogなしが拒否されること。

    Arrange:
        - Annotation済みCandidateを持つ正常requestが用意される
    Act:
        - Scene Catalogだけを除いたrequestが再構築される
    Assert:
        - CatalogとAnnotationの不整合として拒否されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)

    # Act
    # Assert
    with pytest.raises(ValueError, match="Scene Catalog"):
        replace(request, scene_catalog=None)
