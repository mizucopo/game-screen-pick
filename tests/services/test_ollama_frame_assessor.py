"""単一動画向けOllama API境界のテスト."""

from typing import Any

from src.services.ollama_frame_assessor import OllamaFrameAssessor


class ModelListingAssessor(OllamaFrameAssessor):
    """Ollama model一覧と詳細応答を固定するfake."""

    def __init__(self) -> None:
        """request履歴を初期化する."""
        super().__init__("localhost", timeout_seconds=1.0, require_gpu=False)
        self.requested_payloads: list[dict[str, Any] | None] = []

    def _request_json(
        self,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """tagsとshowのfixtureを返す."""
        assert timeout_seconds > 0
        self.requested_payloads.append(payload)
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": "llava:latest",
                        "digest": "llava-digest",
                    }
                ]
            }
        assert url.endswith("/api/show")
        return {"capabilities": ["vision"], "details": {}}


def test_normalize_host_adds_default_port_to_bracketed_ipv6() -> None:
    """portなしIPv6へOllama既定portを補うこと."""
    assert OllamaFrameAssessor.normalize_host("[::1]") == "http://[::1]:11434"
    assert (
        OllamaFrameAssessor.normalize_host("http://[::1]:11435") == "http://[::1]:11435"
    )


def test_fetch_model_metadata_resolves_untagged_latest_alias() -> None:
    """`llava`をinstalled nameの`llava:latest`へ解決すること."""
    assessor = ModelListingAssessor()

    metadata = assessor.fetch_model_metadata({"llava"})

    assert metadata["llava"]["digest"] == "llava-digest"
    assert metadata["llava"]["resolved_name"] == "llava:latest"
    assert assessor.requested_payloads[-1] == {"model": "llava:latest"}
