"""Selected Image WebPをframe単位で確定する。"""

import hashlib
from collections.abc import Callable
from fractions import Fraction
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError, features
from PIL import __version__ as pillow_version

from ..models.checkpoint_operation import CheckpointOperation
from ..models.decoded_video_frame import DecodedVideoFrame
from ..models.frame_candidate import FrameCandidate
from ..models.selected_image_artifact import SelectedImageArtifact
from ..protocols.run_observer import RunObserver
from .checkpoint_version import checkpoint_version
from .durable_work_unit_cache import DurableWorkUnitCache
from .encode_selected_webp import SELECTED_WEBP_QUALITY, encode_selected_webp

_ENGINE_VERSION = checkpoint_version(CheckpointOperation.SELECTED_IMAGE_WEBP)
_SCHEMA = "game-screen-pick/selected-image-checkpoint@1.0.0"


class SelectedImageCheckpoint:
    """元frame抽出とWebP encodeをSelected Imageごとに再利用する。"""

    def __init__(
        self,
        cache_folder: Path,
        *,
        source_fingerprint: str,
        validate_source: Callable[[], None],
        observer: RunObserver | None = None,
    ) -> None:
        self._validate_source = validate_source
        self._cache = DurableWorkUnitCache(
            cache_folder,
            subject_fingerprint=source_fingerprint,
            operation=CheckpointOperation.SELECTED_IMAGE_WEBP,
            observer=observer,
        )

    def resolve(
        self,
        frame: FrameCandidate,
        *,
        scan_stage_fingerprint: str,
        relative_path: str,
        extract_frame: Callable[[], DecodedVideoFrame],
    ) -> tuple[SelectedImageArtifact, bytes]:
        """検証済みWebPを返し、miss時だけ元frameを再抽出する。"""
        if (
            frame.video_fingerprint is None
            or frame.stream_index is None
            or frame.source_pts is None
            or frame.time_base is None
        ):
            msg = "Selected Image checkpointにexact frame情報が必要です"
            raise ValueError(msg)
        semantic_input = {
            "image_id": frame.identifier,
            "video_fingerprint": frame.video_fingerprint,
            "stream_index": frame.stream_index,
            "source_pts": frame.source_pts,
            "time_base": _fraction_value(frame.time_base),
            "scan_stage_fingerprint": scan_stage_fingerprint,
            "encoding": {
                "contract": _ENGINE_VERSION,
                "quality": SELECTED_WEBP_QUALITY,
                "pillow_version": pillow_version,
                "libwebp_version": features.version("webp"),
            },
        }
        bundle, _reused = self._cache.resolve(
            frame.identifier,
            semantic_input,
            lambda checkpoint_root: self._produce(
                frame.identifier,
                extract_frame,
                checkpoint_root,
            ),
            validate_bundle=lambda value: _restore_artifact(
                value.artifact,
                image_id=frame.identifier,
                relative_path=relative_path,
                content=value.root.joinpath("image.webp").read_bytes(),
            ),
        )
        self._validate_source()
        image_path = bundle.root / "image.webp"
        content = image_path.read_bytes()
        artifact = _restore_artifact(
            bundle.artifact,
            image_id=frame.identifier,
            relative_path=relative_path,
            content=content,
        )
        return artifact, content

    def _produce(
        self,
        image_id: str,
        extract_frame: Callable[[], DecodedVideoFrame],
        checkpoint_root: Path,
    ) -> dict[str, object]:
        """元frameをWebPへencodeしsource検証後だけ確定候補にする。"""
        artifact = encode_selected_webp(
            image_id,
            extract_frame(),
            checkpoint_root / "image.webp",
            "images/checkpoint.webp",
        )
        self._validate_source()
        return {
            "schema": _SCHEMA,
            "image_id": artifact.image_id,
            "sha256": artifact.sha256,
            "width": artifact.width,
            "height": artifact.height,
            "size_bytes": artifact.size_bytes,
            "artifact_path": "image.webp",
        }


def _restore_artifact(
    value: dict[str, object],
    *,
    image_id: str,
    relative_path: str,
    content: bytes,
) -> SelectedImageArtifact:
    expected_digest = hashlib.sha256(content).hexdigest()
    width = _integer(value.get("width"))
    height = _integer(value.get("height"))
    if (
        value.get("schema") != _SCHEMA
        or value.get("image_id") != image_id
        or value.get("artifact_path") != "image.webp"
        or value.get("sha256") != expected_digest
        or value.get("size_bytes") != len(content)
    ):
        msg = "Selected Image checkpoint artifactが不正です"
        raise ValueError(msg)
    try:
        with Image.open(BytesIO(content)) as image:
            valid_encoding = (
                image.format == "WEBP"
                and image.size == (width, height)
                and not image.getexif()
                and not ({"exif", "icc_profile", "xmp"} & image.info.keys())
            )
    except (OSError, UnidentifiedImageError):
        valid_encoding = False
    if not valid_encoding:
        msg = "Selected Image checkpoint WebPが不正です"
        raise ValueError(msg)
    return SelectedImageArtifact(
        image_id=image_id,
        relative_path=relative_path,
        sha256=expected_digest,
        width=width,
        height=height,
        size_bytes=len(content),
    )


def _fraction_value(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = "Selected Image checkpoint integerが不正です"
        raise ValueError(msg)
    return value
