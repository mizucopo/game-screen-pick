"""documented Ollama APIを使うmodel store adapter。"""

import json
import re
from collections.abc import Callable, Mapping
from typing import cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ..models.model_artifact import ModelArtifact
from ..models.model_artifact_invalid_error import ModelArtifactInvalidError
from ..models.model_capability import ModelCapability
from ..models.model_requirement import ModelRequirement
from ..models.model_runtime_identity import ModelRuntimeIdentity
from ..models.model_store_http_error import ModelStoreHttpError
from ..models.model_store_kind import ModelStoreKind
from ..models.model_store_unavailable_error import ModelStoreUnavailableError
from ..models.resolved_model_identity import ResolvedModelIdentity
from ..utils.http_retry_delay import http_retry_delay
from .canonicalize_ollama_model_selector import (
    canonicalize_ollama_model_selector,
)

JsonRequester = Callable[
    [str, str, Mapping[str, object] | None, float],
    object,
]

_SEMANTIC_VERSION_PATTERN = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?"
)
_MINIMUM_OLLAMA_VERSION = (0, 31, 2)
_STRUCTURED_OUTPUT_CAPABILITY_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"ready": {"type": "boolean", "const": True}},
    "required": ["ready"],
}


class OllamaModelStore:
    """Ollama tagのlocal digest、pull、vision capabilityを扱う。"""

    def __init__(
        self,
        host: str,
        *,
        timeout_seconds: float,
        requester: JsonRequester | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._requester = requester or _request_json

    @property
    def kind(self) -> ModelStoreKind:
        """Ollama store kindを返す。"""
        return ModelStoreKind.OLLAMA

    def resolve_local(self, requirement: ModelRequirement) -> ModelArtifact | None:
        """exact configured tagのlocal manifest digestを解決する。"""
        _require_ollama_requirement(requirement)
        runtime_identity = self._runtime_identity()
        return self._resolve_local_artifact(requirement, runtime_identity)

    def resolve_current_identity(
        self,
        configured_name: str,
    ) -> ResolvedModelIdentity | None:
        """local configured tagが現在指す完全identityだけを解決する。"""
        if not configured_name.strip():
            raise ValueError("Ollama model名が必要です")
        resolved = self._resolve_current_model(configured_name)
        return None if resolved is None else resolved[0]

    def synchronize(self, requirement: ModelRequirement) -> ModelArtifact:
        """configured tagをpullしpost-pull identityを解決する。"""
        _require_ollama_requirement(requirement)
        runtime_identity = self._runtime_identity()
        response = _require_mapping(
            self._request(
                "POST",
                "/api/pull",
                {"model": requirement.configured_name, "stream": False},
            )
        )
        if response.get("status") != "success":
            raise ModelArtifactInvalidError("Ollama pullの完了を確認できませんでした")
        artifact = self._resolve_local_artifact(requirement, runtime_identity)
        if artifact is None:
            raise ModelArtifactInvalidError(
                "Ollama pull後のmodel identityを確認できませんでした"
            )
        return artifact

    def validate(
        self,
        artifact: ModelArtifact,
        requirement: ModelRequirement,
    ) -> None:
        """modelがvisionと要求context lengthを提供することを検証する。"""
        _require_ollama_requirement(requirement)
        if artifact.identity.store_kind is not self.kind:
            raise ModelArtifactInvalidError("Ollama artifact kindが不正です")
        response = _require_mapping(
            self._request(
                "POST",
                "/api/show",
                {"model": requirement.configured_name, "verbose": True},
            )
        )
        capabilities = response.get("capabilities")
        if not isinstance(capabilities, list) or "vision" not in capabilities:
            raise ModelArtifactInvalidError(
                "Ollama modelのvision capabilityを確認できませんでした"
            )
        model_info = response.get("model_info")
        if not isinstance(model_info, dict):
            raise ModelArtifactInvalidError(
                "Ollama modelのcontext capabilityを確認できませんでした"
            )
        context_lengths = [
            value
            for key, value in model_info.items()
            if isinstance(key, str)
            and key.endswith(".context_length")
            and isinstance(value, int)
            and not isinstance(value, bool)
        ]
        minimum = requirement.minimum_context_length
        if minimum is None or not context_lengths or max(context_lengths) < minimum:
            raise ModelArtifactInvalidError(
                "Ollama modelのcontext capabilityを確認できませんでした"
            )
        self._validate_structured_output(requirement, minimum)

    def confirm_current_identity(
        self,
        artifact: ModelArtifact,
        requirement: ModelRequirement,
    ) -> None:
        """capability検証後もtagがfreeze対象digestを指すことを確認する。"""
        _require_ollama_requirement(requirement)
        if artifact.identity.store_kind is not self.kind:
            raise ModelArtifactInvalidError("Ollama artifact kindが不正です")
        current = self._resolve_local_artifact(
            requirement,
            artifact.runtime_identity,
        )
        if current is None or current.identity != artifact.identity:
            raise ModelArtifactInvalidError(
                "Ollama tagがcapability検証中に変更されました"
            )

    def publish_validated(self, artifact: ModelArtifact) -> None:
        """Ollamaはpull時にtagを公開済みのためkindだけを検証する。"""
        if artifact.identity.store_kind is not self.kind:
            raise ModelArtifactInvalidError("Ollama artifact kindが不正です")

    def _validate_structured_output(
        self,
        requirement: ModelRequirement,
        minimum_context_length: int,
    ) -> None:
        """固定schemaの最小応答でstructured output capabilityを検証する。"""
        response = _require_mapping(
            self._request(
                "POST",
                "/api/chat",
                {
                    "model": requirement.configured_name,
                    "stream": False,
                    "think": False,
                    "format": _STRUCTURED_OUTPUT_CAPABILITY_SCHEMA,
                    "options": {
                        "temperature": 0,
                        "num_ctx": minimum_context_length,
                    },
                    "messages": [
                        {
                            "role": "user",
                            "content": ("Return a JSON object with ready set to true."),
                        }
                    ],
                },
            )
        )
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        try:
            parsed = json.loads(content) if isinstance(content, str) else None
        except json.JSONDecodeError:
            parsed = None
        if parsed != {"ready": True}:
            raise ModelArtifactInvalidError(
                "Ollama modelのstructured output capabilityを確認できませんでした"
            )

    def _runtime_identity(self) -> ModelRuntimeIdentity:
        """supported Ollama server versionをruntime identityへ変換する。"""
        response = _require_mapping(self._request("GET", "/api/version", None))
        version = response.get("version")
        if not isinstance(version, str) or _semantic_version(version) < (
            _MINIMUM_OLLAMA_VERSION
        ):
            raise ModelArtifactInvalidError("Ollama server 0.31.2以上が必要です")
        try:
            return ModelRuntimeIdentity(ModelStoreKind.OLLAMA, version)
        except ValueError:
            raise ModelArtifactInvalidError(
                "Ollama server versionを検証できませんでした"
            ) from None

    def _resolve_local_artifact(
        self,
        requirement: ModelRequirement,
        runtime_identity: ModelRuntimeIdentity,
    ) -> ModelArtifact | None:
        resolved = self._resolve_current_model(requirement.configured_name)
        if resolved is None:
            return None
        identity, canonical_name = resolved
        return ModelArtifact(
            identity=identity,
            canonical_name=canonical_name,
            runtime_identity=runtime_identity,
            location=None,
        )

    def _resolve_current_model(
        self,
        configured_name: str,
    ) -> tuple[ResolvedModelIdentity, str] | None:
        """local一覧からidentityとcanonical nameを一度に解決する。"""
        response = _require_mapping(self._request("GET", "/api/tags", None))
        models = response.get("models")
        if not isinstance(models, list):
            raise ModelArtifactInvalidError(
                "Ollama local model一覧を検証できませんでした"
            )
        for raw_model in models:
            if not isinstance(raw_model, dict):
                raise ModelArtifactInvalidError(
                    "Ollama local model一覧を検証できませんでした"
                )
            model = cast(dict[object, object], raw_model)
            names = tuple(
                value
                for key in ("name", "model")
                if isinstance((value := model.get(key)), str)
            )
            if not any(_names_match(configured_name, name) for name in names):
                continue
            digest = model.get("digest")
            canonical_name = model.get("name")
            if not isinstance(digest, str) or not isinstance(canonical_name, str):
                raise ModelArtifactInvalidError(
                    "Ollama local model identityを検証できませんでした"
                )
            if re.fullmatch(r"[0-9a-f]{64}", digest) is not None:
                digest = f"sha256:{digest}"
            try:
                identity = ResolvedModelIdentity(ModelStoreKind.OLLAMA, digest)
            except ValueError:
                raise ModelArtifactInvalidError(
                    "Ollama local model identityを検証できませんでした"
                ) from None
            return identity, canonical_name
        return None

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None,
    ) -> object:
        try:
            return self._requester(
                method,
                f"{self._host}{path}",
                payload,
                self._timeout_seconds,
            )
        except HTTPError as error:
            retry_after = (
                error.headers.get("Retry-After") if error.headers is not None else None
            )
            raise ModelStoreHttpError(
                error.code,
                retry_after_seconds=http_retry_delay(error.code, retry_after),
            ) from None
        except (ModelArtifactInvalidError, ModelStoreUnavailableError):
            raise
        except Exception:
            raise ModelStoreUnavailableError(
                "Ollama APIを利用できませんでした"
            ) from None


def _request_json(
    method: str,
    url: str,
    payload: Mapping[str, object] | None,
    timeout: float,
) -> object:
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _require_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ModelArtifactInvalidError("Ollama API responseが不正です")
    return cast(dict[str, object], value)


def _require_ollama_requirement(requirement: ModelRequirement) -> None:
    if (
        requirement.store_kind is not ModelStoreKind.OLLAMA
        or requirement.capability is not ModelCapability.VISION_STRUCTURED_OUTPUT
    ):
        raise ValueError("OllamaModelStoreにはvision requirementが必要です")


def _semantic_version(version: str) -> tuple[int, int, int]:
    match = _SEMANTIC_VERSION_PATTERN.match(version)
    if match is None:
        return (0, 0, 0)
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch") or 0),
    )


def _names_match(configured_name: str, local_name: str) -> bool:
    return canonicalize_ollama_model_selector(
        configured_name
    ) == canonicalize_ollama_model_selector(local_name)
