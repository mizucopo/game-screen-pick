import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from src.video_selection.model_runtime.hugging_face_model_store import (
    HuggingFaceModelStore,
)
from src.video_selection.models.model_artifact_invalid_error import (
    ModelArtifactInvalidError,
)
from src.video_selection.models.model_capability import ModelCapability
from src.video_selection.models.model_requirement import ModelRequirement
from src.video_selection.models.model_role import ModelRole
from src.video_selection.models.model_store_kind import ModelStoreKind
from src.video_selection.models.model_store_unavailable_error import (
    ModelStoreUnavailableError,
)


def test_local_ref_and_remote_main_are_resolved_to_immutable_snapshots(
    tmp_path: Path,
) -> None:
    """local refとremote mainが完全commit SHAのsnapshotへ解決されること。

    Arrange:
        - local main snapshotとremote main metadata・immutable snapshotが用意される
        - load capabilityを記録するvalidatorが用意される
    Act:
        - local解決、remote同期、load検証が順に実行される
    Assert:
        - remote selectorがfull SHAへ解決されてdownloadされること
        - requested名とcanonical名が分離されること
    """
    # Arrange
    old_sha = "a" * 40
    new_sha = "b" * 40
    old_snapshot = tmp_path / "snapshots" / old_sha
    new_snapshot = tmp_path / "snapshots" / new_sha
    old_snapshot.mkdir(parents=True)
    new_snapshot.mkdir()
    download_calls: list[tuple[str, dict[str, object]]] = []
    validation_calls: list[tuple[Path, ModelRole]] = []

    def download(repo_id: str, **options: object) -> str:
        download_calls.append((repo_id, options))
        if options["local_files_only"] is True:
            return str(old_snapshot)
        return str(new_snapshot)

    def validate(path: Path, requirement: ModelRequirement) -> None:
        validation_calls.append((path, requirement.role))

    store = HuggingFaceModelStore(
        cache_dir=tmp_path / "hub-cache",
        token="token-secret",
        metadata_timeout_seconds=7.0,
        model_info_resolver=lambda *_args, **_kwargs: SimpleNamespace(
            sha=new_sha,
            id="canonical-org/faster-whisper",
        ),
        snapshot_downloader=download,
        model_validator=validate,
    )
    requirement = _requirement("legacy-org/faster-whisper")

    # Act
    local = store.resolve_local(requirement)
    synchronized = store.synchronize(requirement)
    store.validate(synchronized, requirement)

    # Assert
    assert local is not None
    assert local.identity.identifier == "hf:" + old_sha
    assert synchronized.identity.identifier == "hf:" + new_sha
    assert synchronized.canonical_name == "canonical-org/faster-whisper"
    assert synchronized.runtime_identity.startswith("huggingface-hub:")
    assert download_calls[0][1]["revision"] == "main"
    assert download_calls[0][1]["local_files_only"] is True
    assert download_calls[1][1]["revision"] == new_sha
    assert download_calls[1][1]["local_files_only"] is False
    assert validation_calls == [(new_snapshot, ModelRole.SPEECH_TO_TEXT)]


def test_missing_local_ref_is_reported_without_network_fallback() -> None:
    """local refs/main不在がadapter内のnetwork fallbackなしで返されること。

    Arrange:
        - local-only snapshot lookupがnot foundを返すstoreが用意される
    Act:
        - local modelが解決される
    Assert:
        - Noneが返されremote metadataは呼ばれないこと
    """
    # Arrange
    metadata_called = False

    def resolve_metadata(*_args: object, **_kwargs: object) -> object:
        nonlocal metadata_called
        metadata_called = True
        return object()

    def missing_snapshot(*_args: object, **_kwargs: object) -> object:
        raise LocalEntryNotFoundError("not found")

    store = HuggingFaceModelStore(
        model_info_resolver=resolve_metadata,
        snapshot_downloader=missing_snapshot,
        model_validator=lambda _path, _requirement: None,
    )

    # Act
    resolved = store.resolve_local(_requirement("org/model"))

    # Assert
    assert resolved is None
    assert metadata_called is False


def test_snapshot_mismatch_and_external_detail_are_sanitized(tmp_path: Path) -> None:
    """snapshot不一致とexternal failure detailが安全なerrorへ変換されること。

    Arrange:
        - metadata SHAと異なるsnapshot pathが用意される
        - tokenと絶対pathを含むmetadata failureが用意される
    Act:
        - 各storeでremote同期が試行される
    Assert:
        - mismatchがpartialとして拒否され秘密がmessageへ出ないこと
    """
    # Arrange
    expected_sha = "a" * 40
    wrong_snapshot = tmp_path / "snapshots" / ("b" * 40)
    wrong_snapshot.mkdir(parents=True)
    mismatch = HuggingFaceModelStore(
        model_info_resolver=lambda *_args, **_kwargs: SimpleNamespace(
            sha=expected_sha,
            id="org/model",
        ),
        snapshot_downloader=lambda *_args, **_kwargs: str(wrong_snapshot),
        model_validator=lambda _path, _requirement: None,
    )

    def fail_metadata(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("token-secret /private/model-store")

    unavailable = HuggingFaceModelStore(
        model_info_resolver=fail_metadata,
        snapshot_downloader=lambda *_args, **_kwargs: str(wrong_snapshot),
        model_validator=lambda _path, _requirement: None,
    )

    # Act / Assert
    with pytest.raises(ModelArtifactInvalidError, match="一致しません"):
        mismatch.synchronize(_requirement("org/model"))
    with pytest.raises(ModelStoreUnavailableError) as captured:
        unavailable.synchronize(_requirement("org/model"))
    assert "token-secret" not in str(captured.value)
    assert "/private/model-store" not in str(captured.value)
    formatted = "".join(
        traceback.format_exception(captured.type, captured.value, captured.tb)
    )
    assert "token-secret" not in formatted
    assert "/private/model-store" not in formatted


def test_load_failure_does_not_expose_snapshot_path(tmp_path: Path) -> None:
    """STT backend load failureでsnapshot絶対pathが公開されないこと。

    Arrange:
        - 完全SHAのsnapshotと秘密を含むload failureが用意される
    Act:
        - artifact capabilityが検証される
    Assert:
        - pathと秘密を含まないartifact invalid errorになること
    """
    # Arrange
    sha = "a" * 40
    snapshot = tmp_path / "token-secret" / "snapshots" / sha
    snapshot.mkdir(parents=True)

    def fail_load(_path: Path, _requirement: ModelRequirement) -> None:
        raise RuntimeError(f"cannot load {snapshot}")

    store = HuggingFaceModelStore(
        snapshot_downloader=lambda *_args, **_kwargs: str(snapshot),
        model_validator=fail_load,
    )
    artifact = store.resolve_local(_requirement("org/model"))
    assert artifact is not None

    # Act / Assert
    with pytest.raises(ModelArtifactInvalidError) as captured:
        store.validate(artifact, _requirement("org/model"))
    assert str(tmp_path) not in str(captured.value)
    assert "token-secret" not in str(captured.value)
    formatted = "".join(
        traceback.format_exception(captured.type, captured.value, captured.tb)
    )
    assert str(tmp_path) not in formatted
    assert "token-secret" not in formatted


def _requirement(configured_name: str) -> ModelRequirement:
    return ModelRequirement(
        role=ModelRole.SPEECH_TO_TEXT,
        store_kind=ModelStoreKind.HUGGING_FACE,
        configured_name=configured_name,
        capability=ModelCapability.SPEECH_TO_TEXT,
        device="cuda",
        compute_type="float16",
    )
