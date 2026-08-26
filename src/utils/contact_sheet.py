"""Ollama評価用および選定結果用のコンタクトシート生成."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from ..models.video_selection import FrameCandidate
from .video_selection_files import create_exclusive_temporary_file


def context_frame_path(
    context_dir: Path,
    candidate: FrameCandidate,
    position: str,
) -> Path:
    """候補の遷移判定用フレームパスを返す."""
    return context_dir / f"{candidate.frame_id}-{position}.jpg"


def build_contact_sheet(
    candidates: Sequence[FrameCandidate],
    output_path: Path,
    *,
    context_dir: Path | None = None,
) -> None:
    """候補一覧を一枚のJPEGへまとめる.

    `context_dir` が指定された場合は、各候補を直前・対象・直後の三枚組で
    表示する。指定しない場合は対象フレームだけを表示する。
    """
    if not candidates:
        raise ValueError("空の候補からコンタクトシートは作成できません")

    contextual = context_dir is not None
    if contextual:
        cell_width = 960
        image_height = 180
        label_height = 42
        columns = 2
    else:
        cell_width = 480
        image_height = 270
        label_height = 32
        columns = 3

    rows = math.ceil(len(candidates) / columns)
    sheet = Image.new(
        "RGB",
        (cell_width * columns, (image_height + label_height) * rows),
        "black",
    )
    draw = ImageDraw.Draw(sheet)
    for index, candidate in enumerate(candidates):
        x = index % columns * cell_width
        y = index // columns * (image_height + label_height)
        suffix = "  [before | selected | after]" if contextual else ""
        source = f"  {candidate.source_label}" if candidate.source_label else ""
        draw.text(
            (x + 10, y + 8),
            (
                f"{candidate.frame_id}  "
                f"{_format_timestamp(candidate.timestamp_seconds)}{source}{suffix}"
            ),
            fill="white",
        )
        image_y = y + label_height
        if context_dir is None:
            _paste_thumbnail(
                sheet,
                Path(candidate.path),
                box_x=x,
                box_y=image_y,
                box_width=cell_width,
                box_height=image_height,
            )
            continue

        panel_width = cell_width // 3
        paths = (
            context_frame_path(context_dir, candidate, "before"),
            Path(candidate.path),
            context_frame_path(context_dir, candidate, "after"),
        )
        for panel_index, source_path in enumerate(paths):
            _paste_thumbnail(
                sheet,
                source_path,
                box_x=x + panel_index * panel_width,
                box_y=image_y,
                box_width=panel_width,
                box_height=image_height,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_path = create_exclusive_temporary_file(
        output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=".partial.jpg",
    )
    try:
        with os.fdopen(
            temporary_fd,
            mode="w+b",
        ) as temporary:
            sheet.save(temporary, format="JPEG", quality=91)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        sheet.close()


def _paste_thumbnail(
    sheet: Image.Image,
    source_path: Path,
    *,
    box_x: int,
    box_y: int,
    box_width: int,
    box_height: int,
) -> None:
    """画像を縦横比を保ったまま指定領域の中央へ貼る."""
    with Image.open(source_path) as source:
        image = source.convert("RGB")
        image.thumbnail((box_width, box_height), Image.Resampling.LANCZOS)
        sheet.paste(
            image,
            (
                box_x + (box_width - image.width) // 2,
                box_y + (box_height - image.height) // 2,
            ),
        )


def _format_timestamp(timestamp_seconds: float) -> str:
    """秒をコンタクトシート向けの時刻へ整形する."""
    rounded = max(0, round(timestamp_seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
