"""walking skeletonのoutput artifactをatomicに公開する。"""

import json
import shutil
import tempfile
from pathlib import Path

from ..models.run_status import RunStatus
from ..models.selected_image import SelectedImage
from ..models.video_set import VideoSet
from .prepared_output import PreparedOutput


class AtomicOutputPublisher:
    """同一filesystemのstaging directoryからOutput Folderを公開する。"""

    def prepare(
        self,
        output_folder: Path,
        video_set: VideoSet,
        selected_images: tuple[SelectedImage, ...],
        requested_count: int,
        run_status: RunStatus,
    ) -> PreparedOutput:
        """画像、JSON、Markdownを公開直前までstagingする。"""
        if output_folder.is_symlink() or (
            output_folder.exists()
            and (not output_folder.is_dir() or any(output_folder.iterdir()))
        ):
            msg = f"Output Folderは存在しないか空である必要があります: {output_folder}"
            raise ValueError(msg)
        output_folder.parent.mkdir(parents=True, exist_ok=True)
        staging_folder = Path(
            tempfile.mkdtemp(
                prefix=f".{output_folder.name}.",
                suffix=".staging",
                dir=output_folder.parent,
            )
        )
        try:
            report = self._write_staged_artifacts(
                staging_folder,
                video_set,
                selected_images,
                requested_count,
                run_status,
            )
        except BaseException:
            shutil.rmtree(staging_folder, ignore_errors=True)
            raise
        return PreparedOutput(staging_folder, output_folder, report)

    @staticmethod
    def _write_staged_artifacts(
        staging_folder: Path,
        video_set: VideoSet,
        selected_images: tuple[SelectedImage, ...],
        requested_count: int,
        run_status: RunStatus,
    ) -> dict[str, object]:
        """staging directoryへ全artifactを書き出す。"""
        images_folder = staging_folder / "images"
        images_folder.mkdir()
        selected_records: list[dict[str, object]] = []
        selected_count = len(selected_images)
        warnings: list[dict[str, object]] = []
        if run_status is RunStatus.COMPLETED_WITH_WARNINGS:
            warnings.append(
                {
                    "code": "selection_shortfall",
                    "requested_count": requested_count,
                    "selected_count": selected_count,
                }
            )
        markdown_lines = [
            "# Video Selection Report",
            "",
            f"Status: {run_status.value}",
            f"Requested images: {requested_count}",
            f"Selected images: {selected_count}",
            "",
        ]
        if warnings:
            markdown_lines.extend(
                (
                    "Selection Shortfall: "
                    f"requested={requested_count}, selected={selected_count}",
                    "",
                )
            )
        markdown_lines.extend(("## Selected images", ""))
        for index, selected_image in enumerate(selected_images, start=1):
            annotation = selected_image.annotation
            candidate = annotation.candidate
            relative_path = f"images/{index:04d}_{candidate.identifier}.webp"
            (staging_folder / relative_path).write_bytes(candidate.image_bytes)
            selected_records.append(
                {
                    "id": candidate.identifier,
                    "path": relative_path,
                    "summary": annotation.summary,
                    "reason_codes": list(selected_image.reason_codes),
                }
            )
            markdown_lines.append(
                f"{index}. [{candidate.identifier}]({relative_path}) — "
                f"{annotation.summary}"
            )
        report: dict[str, object] = {
            "schema": "game-screen-pick/walking-skeleton@0",
            "status": run_status.value,
            "requested_count": requested_count,
            "selected_count": selected_count,
            "warnings": warnings,
            "video_set": {"videos": list(video_set.relative_paths)},
            "selected": selected_records,
        }
        (staging_folder / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging_folder / "report.md").write_text(
            "\n".join(markdown_lines) + "\n",
            encoding="utf-8",
        )
        return report
