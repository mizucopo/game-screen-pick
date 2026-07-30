"""Canonical Selection ReportとSelected Imageをatomicに公開する。"""

import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from ..models.canonical_publication_request import CanonicalPublicationRequest
from ..models.decoded_video_frame import DecodedVideoFrame
from ..models.frame_candidate import FrameCandidate
from ..models.processing_stage import ProcessingStage
from ..models.selected_image_artifact import SelectedImageArtifact
from ..models.video_stage_result import VideoStageResult
from ..protocols.run_observer import RunObserver
from ..protocols.selected_frame_media_runtime import SelectedFrameMediaRuntime
from .build_canonical_selection_report import build_canonical_selection_report
from .build_selected_image_output_paths import build_selected_image_output_paths
from .canonical_report_semantic_digest import canonical_report_semantic_digest
from .render_human_selection_report import render_human_selection_report
from .selected_image_checkpoint import SelectedImageCheckpoint
from .serialize_canonical_selection_report import (
    serialize_canonical_selection_report,
)
from .validate_canonical_selection_report import (
    load_validated_canonical_selection_report,
    validate_canonical_selection_report,
)
from .validate_output_folder import validate_output_folder
from .validate_video_set_snapshot import validate_video_set_snapshot_metadata

PublicationFaultInjector = Callable[[str, Path], None]
DirectoryRenamer = Callable[[Path, Path], None]
CompletionClock = Callable[[], datetime]


class CanonicalOutputPublisher:
    """検証済みstaging Folderを一回のdirectory renameで公開する。"""

    def __init__(
        self,
        media_runtime: SelectedFrameMediaRuntime,
        *,
        fault_injector: PublicationFaultInjector | None = None,
        directory_renamer: DirectoryRenamer | None = None,
        completion_clock: CompletionClock | None = None,
        observer: RunObserver | None = None,
    ) -> None:
        self._media_runtime = media_runtime
        self._fault_injector = fault_injector or _ignore_fault
        self._directory_renamer = directory_renamer or _rename_directory
        self._completion_clock = completion_clock
        self._observer = observer
        self._reused_completed_publication = False

    @property
    def reused_completed_publication(self) -> bool:
        """既存の検証済みCanonical outputを変更せず返したか示す。"""
        return self._reused_completed_publication

    def publish(
        self,
        request: CanonicalPublicationRequest,
    ) -> dict[str, object]:
        """WebP、Canonical JSON、Markdownを検証してatomicに公開する。"""
        self._reused_completed_publication = False
        output_folder = request.configuration.output_folder
        validate_video_set_snapshot_metadata(request.video_set)
        validate_output_folder(
            request.video_set.input_folder,
            output_folder,
            allow_completed_canonical_output=True,
        )
        output_folder.parent.mkdir(parents=True, exist_ok=True)
        if output_folder.exists():
            if any(output_folder.iterdir()):
                return self._reuse_completed_publication(request, output_folder)
            output_folder.rmdir()
        _verify_atomic_directory_rename(output_folder.parent, output_folder.name)
        staging_folder = Path(
            tempfile.mkdtemp(
                prefix=f".{output_folder.name}.",
                suffix=".staging",
                dir=output_folder.parent,
            )
        )
        try:
            report = self._prepare_and_validate(request, staging_folder)
            validate_video_set_snapshot_metadata(request.video_set)
            self._fault_injector("before-rename", staging_folder)
            if output_folder.exists():
                raise ValueError("Output Folderが公開前に再作成されました")
            self._directory_renamer(staging_folder, output_folder)
            self._fault_injector("after-rename-before-parent-flush", output_folder)
            _flush_directory(output_folder.parent)
            return report
        except BaseException:
            if not staging_folder.exists() and output_folder.exists():
                try:
                    load_validated_canonical_selection_report(output_folder)
                except ValueError:
                    _remove_failed_publication(output_folder)
            shutil.rmtree(staging_folder, ignore_errors=True)
            raise

    def _reuse_completed_publication(
        self,
        request: CanonicalPublicationRequest,
        output_folder: Path,
    ) -> dict[str, object]:
        """検証済み完成outputが今回の意味結果と同じ場合だけ再利用する。"""
        completed_report = load_validated_canonical_selection_report(output_folder)
        staging_folder = Path(
            tempfile.mkdtemp(
                prefix=f".{output_folder.name}.",
                suffix=".staging",
                dir=output_folder.parent,
            )
        )
        try:
            current_report = self._prepare_and_validate(request, staging_folder)
            validate_video_set_snapshot_metadata(request.video_set)
            validate_canonical_selection_report(
                completed_report,
                output_folder,
                request,
                allow_model_update_diagnostic_mismatch=True,
            )
            if canonical_report_semantic_digest(
                completed_report
            ) != canonical_report_semantic_digest(current_report):
                msg = (
                    "Output Folderの完成済みCanonical outputが"
                    "今回の意味結果と一致しません"
                )
                raise ValueError(msg)
            self._reused_completed_publication = True
            return completed_report
        finally:
            shutil.rmtree(staging_folder, ignore_errors=True)

    def _prepare_and_validate(
        self,
        request: CanonicalPublicationRequest,
        staging_folder: Path,
    ) -> dict[str, object]:
        """全artifactをstagingしflushとcross-validationまで完了する。"""
        images_folder = staging_folder / "images"
        images_folder.mkdir()
        paths = build_selected_image_output_paths(request.selection_result)
        stages = {item.source.fingerprint: item for item in request.video_stage_results}
        artifacts: list[SelectedImageArtifact] = []
        for selected in request.selection_result.selected:
            self._fault_injector("before-image-write", staging_folder)
            frame = selected.candidate.annotation.candidate
            stage = _stage_for_frame(frame, stages)
            relative_path = paths[selected.candidate.identifier]
            checkpoint = SelectedImageCheckpoint(
                request.configuration.processing_cache_folder,
                source_fingerprint=stage.source.fingerprint,
                validate_source=lambda: validate_video_set_snapshot_metadata(
                    request.video_set
                ),
                observer=self._observer,
            )

            def extract_selected_frame(
                frame: FrameCandidate = frame,
                stage: VideoStageResult = stage,
            ) -> DecodedVideoFrame:
                return self._extract_original_frame(frame, stage)

            artifact, content = checkpoint.resolve(
                frame,
                scan_stage_fingerprint=_scan_stage_fingerprint(stage),
                relative_path=relative_path,
                extract_frame=extract_selected_frame,
            )
            output_path = staging_folder / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(content)
            artifacts.append(artifact)
        if self._completion_clock is not None:
            request = replace(request, completed_at=self._completion_clock())
        report = build_canonical_selection_report(request, tuple(artifacts))
        self._fault_injector("before-report-json-write", staging_folder)
        (staging_folder / "report.json").write_text(
            serialize_canonical_selection_report(report),
            encoding="utf-8",
        )
        self._fault_injector("before-markdown-render", staging_folder)
        markdown = render_human_selection_report(report)
        self._fault_injector("before-report-markdown-write", staging_folder)
        (staging_folder / "report.md").write_text(markdown, encoding="utf-8")
        self._fault_injector("before-flush", staging_folder)
        _flush_staging_tree(staging_folder)
        self._fault_injector("before-validation", staging_folder)
        validate_canonical_selection_report(report, staging_folder, request)
        return report

    def _extract_original_frame(
        self,
        candidate: FrameCandidate,
        stage: VideoStageResult,
    ) -> DecodedVideoFrame:
        """選択candidateのexact PTSから元寸法RGB frameを再抽出する。"""
        stream = stage.scan.primary_stream
        if (
            candidate.stream_index is None
            or candidate.source_pts is None
            or candidate.time_base is None
            or stream.width is None
            or stream.height is None
        ):
            msg = "Selected Imageのexact frame情報が不足しています"
            raise ValueError(msg)
        decoded = self._media_runtime.extract_original_video_frame(
            stage.source.path,
            candidate.stream_index,
            candidate.source_pts,
        )
        if (
            decoded.stream_index != candidate.stream_index
            or decoded.pts != candidate.source_pts
            or decoded.time_base != candidate.time_base
            or decoded.width != stream.width
            or decoded.height != stream.height
        ):
            msg = "再抽出したSelected Imageがexact PTSまたは元寸法と一致しません"
            raise ValueError(msg)
        return decoded


def _stage_for_frame(
    frame: FrameCandidate,
    stages: dict[str, VideoStageResult],
) -> VideoStageResult:
    fingerprint = frame.video_fingerprint
    if fingerprint is None:
        raise ValueError("Selected ImageにVideo Fingerprintがありません")
    return stages[fingerprint]


def _scan_stage_fingerprint(stage: VideoStageResult) -> str:
    matches = tuple(
        item.fingerprint.value
        for item in stage.completed_stages
        if item.stage is ProcessingStage.SCAN_VIDEO
    )
    if len(matches) != 1:
        msg = "Selected Image checkpointに一つのScan Stageが必要です"
        raise ValueError(msg)
    return matches[0]


def _flush_staging_tree(staging_folder: Path) -> None:
    """staging内の全fileとdirectory entryをrename前にflushする。"""
    paths = sorted(staging_folder.rglob("*"))
    for path in paths:
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as file:
                os.fsync(file.fileno())
    directories = [
        staging_folder.parent,
        staging_folder,
        *(path for path in paths if path.is_dir()),
    ]
    for directory in reversed(directories):
        _flush_directory(directory)


def _flush_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory(source: Path, destination: Path) -> None:
    source.rename(destination)


def _verify_atomic_directory_rename(parent: Path, output_name: str) -> None:
    """同じparent内のdirectory rename能力をartifact生成前に検証する。"""
    probe_source = Path(
        tempfile.mkdtemp(
            prefix=f".{output_name}.rename-probe.",
            dir=parent,
        )
    )
    probe_destination = probe_source.with_name(probe_source.name + ".renamed")
    try:
        os.rename(probe_source, probe_destination)
    except OSError as error:
        msg = "Output Folderのfilesystemでatomic directory renameを利用できません"
        raise OSError(msg) from error
    finally:
        shutil.rmtree(probe_source, ignore_errors=True)
        shutil.rmtree(probe_destination, ignore_errors=True)


def _remove_failed_publication(output_folder: Path) -> None:
    if output_folder.is_dir() and not output_folder.is_symlink():
        shutil.rmtree(output_folder, ignore_errors=True)
    else:
        output_folder.unlink(missing_ok=True)


def _ignore_fault(_checkpoint: str, _staging_folder: Path) -> None:
    return
