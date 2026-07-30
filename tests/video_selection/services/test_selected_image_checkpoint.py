"""SelectedImageCheckpointの再利用test。"""

import hashlib
import json
from fractions import Fraction
from pathlib import Path

from src.video_selection.models.decoded_video_frame import DecodedVideoFrame
from src.video_selection.services.selected_image_checkpoint import (
    SelectedImageCheckpoint,
)
from tests.video_selection.fakes.canonical_publication_factory import (
    build_canonical_publication_request,
)
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver


def test_completed_webp_is_not_extracted_or_encoded_again(tmp_path: Path) -> None:
    """完了済みSelected WebPが同じbyteのまま再利用されること。

    Arrange:
        - exact PTSを持つ選定frameと決定的な元frame extractorが用意される
    Act:
        - 同じcheckpointが2回解決される
    Assert:
        - extractorは初回だけ呼ばれartifactとWebP byteが一致すること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    selected = request.selection_result.selected[0]
    frame = selected.candidate.annotation.candidate
    source = request.video_set.sources[0]
    calls = 0
    observer = RecordingRunObserver()

    def extract() -> DecodedVideoFrame:
        nonlocal calls
        calls += 1
        return DecodedVideoFrame(
            stream_index=0,
            pts=frame.source_pts or 0,
            duration_ts=1,
            time_base=Fraction(1, 10),
            width=2,
            height=2,
            pixel_format="rgb24",
            pixels=bytes(range(12)),
        )

    def checkpoint() -> SelectedImageCheckpoint:
        return SelectedImageCheckpoint(
            request.configuration.processing_cache_folder,
            source_fingerprint=source.fingerprint,
            validate_source=lambda: None,
            observer=observer,
        )

    # Act
    first = checkpoint().resolve(
        frame,
        scan_stage_fingerprint="d" * 64,
        relative_path="images/0001.webp",
        extract_frame=extract,
    )
    second = checkpoint().resolve(
        frame,
        scan_stage_fingerprint="d" * 64,
        relative_path="images/0001.webp",
        extract_frame=extract,
    )

    # Assert
    assert first == second
    assert calls == 1
    resolutions = tuple(
        (event.recompute_count, event.reuse_count)
        for event in observer.progress_events
        if event.work_unit_kind == "selected-image-webp"
    )
    assert resolutions == ((1, 0), (0, 1))


def test_domain_invalid_webp_is_recomputed_instead_of_reused(
    tmp_path: Path,
) -> None:
    """hash整合済みでもWebPでないcheckpointが再計算されること。

    Arrange:
        - 正常なSelected Image checkpointが用意される
        - image、artifact、manifest hashが非WebP bytesへ整合するよう改変される
    Act:
        - 同じSelected Imageが再解決される
    Assert:
        - 不正checkpointが再利用されず元frameが再抽出されること
    """
    # Arrange
    request = build_canonical_publication_request(tmp_path)
    frame = request.selection_result.selected[0].candidate.annotation.candidate
    source = request.video_set.sources[0]

    def decoded_frame() -> DecodedVideoFrame:
        return DecodedVideoFrame(
            stream_index=0,
            pts=frame.source_pts or 0,
            duration_ts=1,
            time_base=Fraction(1, 10),
            width=2,
            height=2,
            pixel_format="rgb24",
            pixels=bytes(range(12)),
        )

    checkpoint = SelectedImageCheckpoint(
        request.configuration.processing_cache_folder,
        source_fingerprint=source.fingerprint,
        validate_source=lambda: None,
    )
    checkpoint.resolve(
        frame,
        scan_stage_fingerprint="d" * 64,
        relative_path="images/0001.webp",
        extract_frame=decoded_frame,
    )
    image_path = next(request.configuration.processing_cache_folder.rglob("image.webp"))
    invalid_content = b"not-a-webp"
    image_path.write_bytes(invalid_content)
    artifact_path = image_path.parent / "artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["sha256"] = hashlib.sha256(invalid_content).hexdigest()
    artifact["size_bytes"] = len(invalid_content)
    artifact_bytes = (
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    artifact_path.write_bytes(artifact_bytes)
    manifest_path = image_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["artifacts"]:
        if record["path"] == "artifact.json":
            content = artifact_bytes
        elif record["path"] == "image.webp":
            content = invalid_content
        else:
            continue
        record["size_bytes"] = len(content)
        record["sha256"] = hashlib.sha256(content).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    calls = 0

    def extract() -> DecodedVideoFrame:
        nonlocal calls
        calls += 1
        return decoded_frame()

    # Act
    _artifact, content = checkpoint.resolve(
        frame,
        scan_stage_fingerprint="d" * 64,
        relative_path="images/0001.webp",
        extract_frame=extract,
    )

    # Assert
    assert calls == 1
    assert content != invalid_content
