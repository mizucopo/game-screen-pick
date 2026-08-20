"""コンタクトシート生成のテスト."""

from pathlib import Path

from PIL import Image

from src.models.video_selection import FrameCandidate
from src.utils.contact_sheet import build_contact_sheet, context_frame_path


def test_build_contact_sheet_outputs_readable_selected_image(
    tmp_path: Path,
) -> None:
    """選択画像一覧が3列の有効なJPEGになること."""
    candidates = []
    for index, color in enumerate(("red", "green", "blue", "yellow"), start=1):
        image_path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (320, 180), color).save(image_path)
        candidates.append(
            FrameCandidate(
                frame_id=f"{index:02d}",
                timestamp_seconds=index * 60.0,
                path=str(image_path),
            )
        )
    output_path = tmp_path / "selected-contact-sheet.jpg"

    build_contact_sheet(candidates, output_path)

    with Image.open(output_path) as sheet:
        assert sheet.format == "JPEG"
        assert sheet.size == (1440, 604)


def test_build_context_sheet_uses_before_and_after_frames(tmp_path: Path) -> None:
    """二次評価sheetが直前・対象・直後の三枚を含むこと."""
    candidate_path = tmp_path / "candidate.jpg"
    Image.new("RGB", (320, 180), "white").save(candidate_path)
    candidate = FrameCandidate("f00001", 10.0, str(candidate_path))
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    for position, color in (("before", "red"), ("after", "blue")):
        Image.new("RGB", (320, 180), color).save(
            context_frame_path(context_dir, candidate, position)
        )
    output_path = tmp_path / "context-sheet.jpg"

    build_contact_sheet([candidate], output_path, context_dir=context_dir)

    with Image.open(output_path) as sheet:
        assert sheet.size == (1920, 222)
