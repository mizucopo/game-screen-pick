"""コンタクトシート生成のテスト."""

import stat
from pathlib import Path

import pytest
from PIL import Image

from src.models.video_selection import FrameCandidate
from src.utils.contact_sheet import build_contact_sheet, context_frame_path


def test_build_contact_sheet_uses_short_display_ids_without_renaming_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ollama用sheetには内部IDでなくbatch内表示IDを描くこと."""
    drawn_labels: list[str] = []

    class RecordingDraw:
        """描画文字列だけを記録するfake."""

        def text(
            self,
            _position: tuple[int, int],
            value: str,
            *,
            fill: str,
        ) -> None:
            """labelを記録する."""
            assert fill == "white"
            drawn_labels.append(value)

    monkeypatch.setattr(
        "src.utils.contact_sheet.ImageDraw.Draw",
        lambda _sheet: RecordingDraw(),
    )
    candidate_path = tmp_path / "candidate.jpg"
    Image.new("RGB", (320, 180), "white").save(candidate_path)
    internal_id = "f1193384472080815349902201"
    candidate = FrameCandidate(internal_id, 10.0, str(candidate_path))

    build_contact_sheet(
        [candidate],
        tmp_path / "contact-sheet.jpg",
        display_ids=["A01"],
    )

    assert drawn_labels == ["A01  00:10"]
    assert candidate.frame_id == internal_id


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


def test_build_contact_sheet_outputs_readable_jpeg_for_999_images(
    tmp_path: Path,
) -> None:
    """999枚の選定画像をJPEGの寸法上限内の一枚へまとめること."""
    image_path = tmp_path / "candidate.jpg"
    Image.new("RGB", (16, 9), "white").save(image_path)
    candidates = [
        FrameCandidate(
            frame_id=f"{index:03d}",
            timestamp_seconds=float(index),
            path=str(image_path),
        )
        for index in range(1, 1000)
    ]
    output_path = tmp_path / "selected-contact-sheet.jpg"

    build_contact_sheet(candidates, output_path)

    with Image.open(output_path) as sheet:
        assert sheet.format == "JPEG"
        assert max(sheet.size) <= 65_000
        sheet.verify()


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


def test_build_contact_sheet_does_not_follow_fixed_temporary_symlink(
    tmp_path: Path,
) -> None:
    """予測可能な旧temporary pathのsymlinkを辿らないこと."""
    candidate_path = tmp_path / "candidate.jpg"
    Image.new("RGB", (320, 180), "white").save(candidate_path)
    candidate = FrameCandidate("f00001", 10.0, str(candidate_path))
    output_path = tmp_path / "contact-sheet.jpg"
    legacy_temporary = tmp_path / ".contact-sheet.partial.jpg"
    external = tmp_path / "external.txt"
    external.write_text("user-owned", encoding="utf-8")
    legacy_temporary.symlink_to(external)
    mode_probe = tmp_path / "mode-probe"
    mode_probe.touch()
    expected_mode = stat.S_IMODE(mode_probe.stat().st_mode)

    build_contact_sheet([candidate], output_path)

    assert external.read_text(encoding="utf-8") == "user-owned"
    assert legacy_temporary.is_symlink()
    assert stat.S_IMODE(output_path.stat().st_mode) == expected_mode
    with Image.open(output_path) as sheet:
        assert sheet.format == "JPEG"
