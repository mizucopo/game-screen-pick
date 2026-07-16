from collections.abc import Mapping

from src.video_selection.models.model_artifact import ModelArtifact
from src.video_selection.models.model_artifact_invalid_error import (
    ModelArtifactInvalidError,
)
from src.video_selection.models.model_requirement import ModelRequirement
from src.video_selection.models.model_store_kind import ModelStoreKind


class FakeModelStore:
    """model lifecycle contractへ決定的なstore応答を返すfake。"""

    def __init__(
        self,
        kind: ModelStoreKind,
        *,
        local_artifacts: Mapping[str, ModelArtifact | None],
        synchronized_artifacts: Mapping[str, ModelArtifact] | None = None,
        synchronization_errors: Mapping[str, Exception] | None = None,
        local_artifacts_after_synchronization_error: Mapping[
            str,
            ModelArtifact | None,
        ]
        | None = None,
        invalid_identifiers: frozenset[str] = frozenset(),
        publication_errors: Mapping[str, Exception] | None = None,
    ) -> None:
        self._kind = kind
        self._local_artifacts = dict(local_artifacts)
        self._synchronized_artifacts = dict(synchronized_artifacts or {})
        self._synchronization_errors = dict(synchronization_errors or {})
        self._local_artifacts_after_synchronization_error = dict(
            local_artifacts_after_synchronization_error or {}
        )
        self._invalid_identifiers = invalid_identifiers
        self._publication_errors = dict(publication_errors or {})
        self.local_resolution_calls: list[str] = []
        self.synchronization_calls: list[str] = []
        self.validation_calls: list[tuple[str, str]] = []
        self.identity_confirmation_calls: list[str] = []
        self.publication_calls: list[str] = []

    @property
    def kind(self) -> ModelStoreKind:
        """store kindを返す。"""
        return self._kind

    def resolve_local(self, requirement: ModelRequirement) -> ModelArtifact | None:
        """設定名に対応するlocal artifact候補を返す。"""
        self.local_resolution_calls.append(requirement.configured_name)
        return self._local_artifacts.get(requirement.configured_name)

    def synchronize(self, requirement: ModelRequirement) -> ModelArtifact:
        """設定名に対応する同期結果または注入errorを返す。"""
        name = requirement.configured_name
        self.synchronization_calls.append(name)
        error = self._synchronization_errors.get(name)
        if error is not None:
            if name in self._local_artifacts_after_synchronization_error:
                self._local_artifacts[name] = (
                    self._local_artifacts_after_synchronization_error[name]
                )
            raise error
        return self._synchronized_artifacts[name]

    def validate(
        self,
        artifact: ModelArtifact,
        requirement: ModelRequirement,
    ) -> None:
        """artifactとroleの組を記録し、指定identityだけ拒否する。"""
        identifier = artifact.identity.identifier
        self.validation_calls.append((identifier, requirement.role.value))
        if identifier in self._invalid_identifiers:
            raise ModelArtifactInvalidError("fake artifact detail")

    def confirm_current_identity(
        self,
        artifact: ModelArtifact,
        requirement: ModelRequirement,
    ) -> None:
        """mutable selectorの最終identity確認を記録する。"""
        self.identity_confirmation_calls.append(
            f"{requirement.configured_name}:{artifact.identity.identifier}"
        )

    def publish_validated(self, artifact: ModelArtifact) -> None:
        """検証済みartifactのstore-local selector公開を記録する。"""
        identifier = artifact.identity.identifier
        self.publication_calls.append(identifier)
        error = self._publication_errors.get(identifier)
        if error is not None:
            raise error
