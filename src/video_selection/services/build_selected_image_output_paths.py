"""Selected Image Output Nameの決定的な構築。"""

from ..models.video_set_selection_result import VideoSetSelectionResult


def build_selected_image_output_paths(
    selection: VideoSetSelectionResult,
) -> dict[str, str]:
    """選択順、Scene Slug、衝突対応digestからrelative pathを返す。"""
    prefixes: dict[str, int] = {}
    for item in selection.selected:
        digest = item.candidate.identifier.removeprefix("frm_")
        prefix = digest[:12]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    width = max(4, len(str(len(selection.selected))))
    paths: dict[str, str] = {}
    for item in selection.selected:
        identifier = item.candidate.identifier
        digest = identifier.removeprefix("frm_")
        digest_part = digest if prefixes[digest[:12]] > 1 else digest[:12]
        scene_slug = item.candidate.annotation.scene_slug
        paths[identifier] = (
            f"images/{item.selection_index:0{width}d}_{scene_slug}_{digest_part}.webp"
        )
    return paths
