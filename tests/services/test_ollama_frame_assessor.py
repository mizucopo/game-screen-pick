"""単一動画向けOllama API境界のテスト."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.models.video_selection import FrameCandidate
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


class ChangingDigestAssessor(OllamaFrameAssessor):
    """2回目のbatchでloaded model digestが変わるfake."""

    def __init__(self) -> None:
        """CPU許可でmodel identityだけを検証する."""
        super().__init__("localhost", timeout_seconds=1.0, require_gpu=False)
        self.ps_digests = iter(("stable-digest", "changed-digest"))
        self.ps_calls = 0

    def _request_json(
        self,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """chatとloaded model一覧を固定応答する."""
        del payload
        assert timeout_seconds > 0
        if url.endswith("/api/chat"):
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "frames": [
                                {
                                    "id": "f00001",
                                    "blog_score": 80,
                                    "transition": False,
                                    "scene": "探索",
                                    "reason": "test",
                                }
                            ]
                        }
                    )
                }
            }
        assert url.endswith("/api/ps")
        self.ps_calls += 1
        return {
            "models": [
                {
                    "name": "llava:latest",
                    "digest": next(self.ps_digests),
                    "size": 100,
                    "size_vram": 0,
                }
            ]
        }


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


def test_assess_revalidates_loaded_model_digest_for_every_batch(
    tmp_path: Path,
) -> None:
    """同じtagが別digestへ変わったbatchを受け入れないこと."""
    contact_sheet = tmp_path / "sheet.jpg"
    contact_sheet.write_bytes(b"image")
    candidate = FrameCandidate("f00001", 1.0, "unused")
    assessor = ChangingDigestAssessor()

    assessor.assess(
        model="llava:latest",
        model_digest="stable-digest",
        prompt="test",
        candidates=[candidate],
        contact_sheet=contact_sheet,
    )
    with pytest.raises(RuntimeError, match="digest"):
        assessor.assess(
            model="llava:latest",
            model_digest="stable-digest",
            prompt="test",
            candidates=[candidate],
            contact_sheet=contact_sheet,
        )

    assert assessor.ps_calls == 2
