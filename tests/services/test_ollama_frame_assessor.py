"""単一動画向けOllama API境界のテスト."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.models.video_selection import FrameAssessment, FrameCandidate
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
                                    "id": "A01",
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


class FrameResponseAssessor(OllamaFrameAssessor):
    """指定したframe評価を返すfake."""

    def __init__(self, frames: list[dict[str, object]]) -> None:
        """Ollama応答のframe配列を保持する."""
        super().__init__("localhost", timeout_seconds=1.0, require_gpu=False)
        self.frames = frames
        self.requested_chat_payloads: list[dict[str, Any]] = []

    def _request_json(
        self,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """chat応答とloaded model一覧を固定する."""
        assert timeout_seconds > 0
        if url.endswith("/api/chat"):
            assert payload is not None
            self.requested_chat_payloads.append(payload)
            return {
                "message": {
                    "content": json.dumps({"frames": self.frames}),
                }
            }
        assert url.endswith("/api/ps")
        return {
            "models": [
                {
                    "name": "llava:latest",
                    "digest": "stable-digest",
                    "size": 100,
                    "size_vram": 0,
                }
            ]
        }


def frame_response(
    frame_id: str,
    *,
    blog_score: int = 80,
    scene: str = "探索",
    reason: str = "test",
) -> dict[str, object]:
    """有効なframe評価fixtureを返す."""
    return {
        "id": frame_id,
        "blog_score": blog_score,
        "transition": False,
        "scene": scene,
        "reason": reason,
    }


def assess_frames(
    tmp_path: Path,
    frames: list[dict[str, object]],
    candidate_ids: list[str],
) -> list[FrameAssessment]:
    """指定した応答とcandidate IDで評価を実行する."""
    contact_sheet = tmp_path / "sheet.jpg"
    contact_sheet.write_bytes(b"image")
    assessor = FrameResponseAssessor(frames)
    candidates = [
        FrameCandidate(frame_id, float(index), "unused")
        for index, frame_id in enumerate(candidate_ids)
    ]
    return assessor.assess(
        model="llava:latest",
        model_digest="stable-digest",
        prompt="test",
        candidates=candidates,
        contact_sheet=contact_sheet,
    )


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


def test_assess_maps_short_display_ids_to_internal_ids_in_candidate_order(
    tmp_path: Path,
) -> None:
    """短い表示IDを長い内部IDへ戻し、入力candidate順で返すこと."""
    first = frame_response("A01", blog_score=81)
    second = frame_response("A02", blog_score=72)
    candidate_ids = [
        "f1193384472080815349902201",
        "f1193384472080815349902202",
    ]

    assessments = assess_frames(
        tmp_path,
        [second, first],
        candidate_ids,
    )

    assert [assessment.frame_id for assessment in assessments] == candidate_ids
    assert [assessment.blog_score for assessment in assessments] == [81.0, 72.0]


def test_assess_passes_structured_output_schema_for_batch(tmp_path: Path) -> None:
    """評価fieldの型・必須性・件数・表示IDをJSON Schemaで拘束すること."""
    contact_sheet = tmp_path / "sheet.jpg"
    contact_sheet.write_bytes(b"image")
    assessor = FrameResponseAssessor([frame_response("A01"), frame_response("A02")])
    candidates = [
        FrameCandidate("f00001", 1.0, "unused"),
        FrameCandidate("f00002", 2.0, "unused"),
    ]

    assessor.assess(
        model="llava:latest",
        model_digest="stable-digest",
        prompt="test",
        candidates=candidates,
        contact_sheet=contact_sheet,
    )

    assert assessor.requested_chat_payloads[0]["format"] == {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": ["A01", "A02"]},
                        "blog_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "transition": {"type": "boolean"},
                        "scene": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "id",
                        "blog_score",
                        "transition",
                        "scene",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["frames"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected_message"),
    [
        ("blog_score", True, "blog_score"),
        ("transition", "false", "transition"),
    ],
)
def test_assess_rejects_invalid_field_types(
    tmp_path: Path,
    field: str,
    invalid_value: object,
    expected_message: str,
) -> None:
    """Structured Outputsを外れた型不正値も保存前に拒否すること."""
    response = frame_response("A01")
    response[field] = invalid_value

    with pytest.raises(ValueError, match=expected_message):
        assess_frames(tmp_path, [response], ["f00001"])


def test_assess_rejects_duplicate_display_id(tmp_path: Path) -> None:
    """内容が同じでも重複する表示IDを拒否すること."""
    with pytest.raises(ValueError, match="A01"):
        assess_frames(
            tmp_path,
            [
                frame_response("A01", blog_score=80),
                frame_response("A01", blog_score=80),
            ],
            ["f00001"],
        )


def test_assess_rejects_missing_frame_id(tmp_path: Path) -> None:
    """要求したframe IDが欠落する応答を拒否すること."""
    with pytest.raises(ValueError, match="表示IDが一致しません"):
        assess_frames(
            tmp_path,
            [frame_response("A01")],
            ["f00001", "f00002"],
        )


def test_assess_rejects_unknown_frame_id(tmp_path: Path) -> None:
    """要求していないframe IDを含む応答を拒否すること."""
    with pytest.raises(ValueError, match="表示IDが一致しません"):
        assess_frames(
            tmp_path,
            [frame_response("A01"), frame_response("A99")],
            ["f00001"],
        )
