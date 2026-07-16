"""walking skeletonのoutput artifactをatomicに公開する。"""

import json
import shutil
import tempfile
from pathlib import Path

from ..models.resolved_models import ResolvedModels
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
        resolved_models: ResolvedModels,
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
                resolved_models,
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
        resolved_models: ResolvedModels,
    ) -> dict[str, object]:
        """staging directoryへ全artifactを書き出す。"""
        images_folder = staging_folder / "images"
        images_folder.mkdir()
        selected_records: list[dict[str, object]] = []
        selected_count = len(selected_images)
        selection_shortfall = selected_count < requested_count
        unavailable_roles = [role.value for role in resolved_models.unavailable_roles()]
        warnings: list[dict[str, object]] = []
        if run_status is RunStatus.COMPLETED_WITH_WARNINGS:
            if selection_shortfall:
                warnings.append(
                    {
                        "code": "selection_shortfall",
                        "requested_count": requested_count,
                        "selected_count": selected_count,
                    }
                )
            if unavailable_roles:
                warnings.append(
                    {
                        "code": "model_update_unavailable",
                        "roles": unavailable_roles,
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
            if selection_shortfall:
                markdown_lines.append(
                    "Selection Shortfall: "
                    f"requested={requested_count}, selected={selected_count}"
                )
            if unavailable_roles:
                markdown_lines.append(
                    "Model Update Unavailable: roles=" + ",".join(unavailable_roles)
                )
            markdown_lines.append("")
        markdown_lines.extend(("## Models", ""))
        for model in sorted(
            resolved_models.items,
            key=lambda value: value.role.value,
        ):
            markdown_lines.append(
                f"- {model.role.value}: {model.configured_name} @ "
                f"{_abbreviate_identity(model.execution_identity.identifier)} "
                f"({model.update_status.value})"
            )
        markdown_lines.append("")
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
            "models": resolved_models.provenance(),
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


def _abbreviate_identity(identifier: str) -> str:
    """report.md用にstore prefixを保った短縮identityを返す。"""
    prefix, value = identifier.rsplit(":", maxsplit=1)
    return f"{prefix}:{value[:12]}…"
