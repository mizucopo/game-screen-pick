"""Hugging Face Hub snapshotを使うSTT model store adapter。"""

import re
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from faster_whisper import WhisperModel
from huggingface_hub import HfApi, __version__, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from ..models.model_artifact import ModelArtifact
from ..models.model_artifact_invalid_error import ModelArtifactInvalidError
from ..models.model_capability import ModelCapability
from ..models.model_requirement import ModelRequirement
from ..models.model_runtime_identity import ModelRuntimeIdentity
from ..models.model_store_kind import ModelStoreKind
from ..models.model_store_unavailable_error import ModelStoreUnavailableError
from ..models.resolved_model_identity import ResolvedModelIdentity

ModelInfoResolver = Callable[..., object]
SnapshotDownloader = Callable[..., object]
ModelValidator = Callable[[Path, ModelRequirement], None]

_REPO_ID_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?"
)


class HuggingFaceModelStore:
    """mutable mainをimmutable SHAへ解決してlocal loadを検証する。"""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        token: bool | str | None = None,
        metadata_timeout_seconds: float = 10.0,
        model_info_resolver: ModelInfoResolver | None = None,
        snapshot_downloader: SnapshotDownloader | None = None,
        model_validator: ModelValidator | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._token = token
        self._metadata_timeout_seconds = metadata_timeout_seconds
        self._model_info_resolver = model_info_resolver or HfApi().model_info
        self._snapshot_downloader = snapshot_downloader or snapshot_download
        self._model_validator = model_validator or _validate_faster_whisper_model

    @property
    def kind(self) -> ModelStoreKind:
        """Hugging Face store kindを返す。"""
        return ModelStoreKind.HUGGING_FACE

    def resolve_local(self, requirement: ModelRequirement) -> ModelArtifact | None:
        """local refs/mainが指すsnapshot候補をnetworkなしで返す。"""
        _require_hugging_face_requirement(requirement)
        try:
            downloaded = self._snapshot_downloader(
                requirement.configured_name,
                revision="main",
                cache_dir=self._cache_dir,
                token=self._token,
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            return None
        except Exception:
            return None
        path = _require_snapshot_path(downloaded)
        return _build_artifact(
            path,
            canonical_name=requirement.configured_name,
            expected_identity=None,
        )

    def synchronize(self, requirement: ModelRequirement) -> ModelArtifact:
        """remote mainをfull SHAへ解決してimmutable snapshotを取得する。"""
        _require_hugging_face_requirement(requirement)
        try:
            info = self._model_info_resolver(
                requirement.configured_name,
                revision="main",
                timeout=self._metadata_timeout_seconds,
                token=self._token,
            )
        except Exception:
            raise ModelStoreUnavailableError(
                "Hugging Face model metadataを取得できませんでした"
            ) from None
        sha = getattr(info, "sha", None)
        canonical_name = getattr(info, "id", None)
        if (
            not isinstance(sha, str)
            or not isinstance(canonical_name, str)
            or _REPO_ID_PATTERN.fullmatch(canonical_name) is None
        ):
            raise ModelArtifactInvalidError(
                "Hugging Face model metadataを検証できませんでした"
            )
        try:
            identity = ResolvedModelIdentity(ModelStoreKind.HUGGING_FACE, sha)
        except ValueError:
            raise ModelArtifactInvalidError(
                "Hugging Face commit SHAを検証できませんでした"
            ) from None
        try:
            downloaded = self._snapshot_downloader(
                requirement.configured_name,
                revision=identity.value,
                cache_dir=self._cache_dir,
                token=self._token,
                local_files_only=False,
            )
        except Exception:
            raise ModelStoreUnavailableError(
                "Hugging Face snapshotを取得できませんでした"
            ) from None
        artifact = _build_artifact(
            _require_snapshot_path(downloaded),
            canonical_name=canonical_name,
            expected_identity=identity,
        )
        return artifact

    def validate(
        self,
        artifact: ModelArtifact,
        requirement: ModelRequirement,
    ) -> None:
        """snapshotをlocal-onlyでSTT backendへloadできることを検証する。"""
        _require_hugging_face_requirement(requirement)
        location = artifact.location
        if (
            artifact.identity.store_kind is not self.kind
            or location is None
            or not location.is_dir()
        ):
            raise ModelArtifactInvalidError(
                "Hugging Face snapshotが完全なdirectoryではありません"
            )
        try:
            self._model_validator(location, requirement)
        except Exception:
            raise ModelArtifactInvalidError(
                "Hugging Face snapshotをSTT backendへloadできませんでした"
            ) from None

    def publish_validated(self, artifact: ModelArtifact) -> None:
        """load検証済みsnapshotだけを次回local mainとして公開する。"""
        if artifact.identity.store_kind is not self.kind:
            raise ModelArtifactInvalidError("Hugging Face artifact kindが不正です")
        _update_main_ref(artifact)


def _validate_faster_whisper_model(
    location: Path,
    requirement: ModelRequirement,
) -> None:
    model = WhisperModel(
        str(location),
        device=requirement.device or "auto",
        compute_type=requirement.compute_type or "default",
        local_files_only=True,
    )
    del model


def _build_artifact(
    path: Path,
    *,
    canonical_name: str,
    expected_identity: ResolvedModelIdentity | None,
) -> ModelArtifact:
    try:
        identity = ResolvedModelIdentity(ModelStoreKind.HUGGING_FACE, path.name)
    except ValueError:
        raise ModelArtifactInvalidError(
            "Hugging Face snapshot pathからcommit SHAを解決できませんでした"
        ) from None
    if expected_identity is not None and identity != expected_identity:
        raise ModelArtifactInvalidError(
            "Hugging Face snapshotと解決済みcommit SHAが一致しません"
        )
    return ModelArtifact(
        identity=identity,
        canonical_name=canonical_name,
        runtime_identity=ModelRuntimeIdentity(
            ModelStoreKind.HUGGING_FACE,
            __version__,
        ),
        location=path,
    )


def _require_snapshot_path(value: object) -> Path:
    if not isinstance(value, str | Path):
        raise ModelArtifactInvalidError(
            "Hugging Face snapshot locationを検証できませんでした"
        )
    return Path(value)


def _update_main_ref(artifact: ModelArtifact) -> None:
    """immutable snapshotを次回local-only解決用のmain refへatomicに記録する。"""
    location = artifact.location
    if location is None or location.parent.name != "snapshots":
        raise ModelArtifactInvalidError(
            "Hugging Face snapshot layoutを検証できませんでした"
        )
    refs_folder = location.parent.parent / "refs"
    temporary_ref = refs_folder / f".main.{uuid4().hex}.tmp"
    try:
        refs_folder.mkdir(parents=True, exist_ok=True)
        temporary_ref.write_text(artifact.identity.value, encoding="utf-8")
        temporary_ref.replace(refs_folder / "main")
    except OSError:
        raise ModelStoreUnavailableError(
            "Hugging Face local refを更新できませんでした"
        ) from None
    finally:
        with suppress(OSError):
            temporary_ref.unlink(missing_ok=True)


def _require_hugging_face_requirement(requirement: ModelRequirement) -> None:
    if (
        requirement.store_kind is not ModelStoreKind.HUGGING_FACE
        or requirement.capability is not ModelCapability.SPEECH_TO_TEXT
    ):
        raise ValueError(
            "HuggingFaceModelStoreにはspeech-to-text requirementが必要です"
        )
