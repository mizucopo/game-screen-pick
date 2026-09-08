"""動画理解に基づく候補とchunk単位の再開を検証する."""

import base64
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from src.models.semantic_video import SemanticVideoOptions, SemanticVideoPlan
from src.models.video_selection import VideoMetadata
from src.models.vllm_config import VllmConfig
from src.services.semantic_video_planner import SemanticVideoPlanner
from src.services.vllm_client import VllmClient
from src.utils.video_selection_files import json_digest


@pytest.mark.parametrize(
    ("chunk", "overlap"),
    [
        (0, 0),
        (30, 30),
        (30, -1),
        (math.inf, 2),
        (30, math.nan),
        (True, 0),
        (0.5, 0),
        (121, 2),
        (30, 29.5),
        (10**1000, 0),
    ],
)
def test_options_reject_invalid_chunk_geometry(chunk: float, overlap: float) -> None:
    with pytest.raises(ValueError):
        SemanticVideoOptions(chunk_seconds=chunk, overlap_seconds=overlap)


def event_response(
    *, start: float = 1, end: float = 3, timestamp: float = 2
) -> dict[str, Any]:
    return {
        "events": [
            {
                "start_seconds": start,
                "end_seconds": end,
                "timestamp_seconds": timestamp,
                "summary": "仲間と合流して戦闘が始まる",
                "importance": 85,
            }
        ],
        "excluded_intervals": [],
    }


@pytest.fixture
def client(mocker: MockerFixture) -> VllmClient:
    result = VllmClient(VllmConfig(base_url="http://localhost:8000/v1", model="video"))
    mocker.patch.object(result, "fetch_model_metadata", return_value={})
    mocker.patch.object(result, "complete_json", return_value=event_response())
    return result


@pytest.fixture
def ffmpeg(mocker: MockerFixture) -> Mock:
    def encode(command: list[str], **_kwargs: Any) -> None:
        Path(command[-1]).write_bytes(b"test-mp4")

    return mocker.patch(
        "src.services.semantic_video_planner.subprocess.run", side_effect=encode
    )


def run_plan(
    tmp_path: Path,
    client: VllmClient,
    *,
    duration: float = 10,
    start: float = 5,
    last_frame: float | None = None,
    average_frame_rate: str = "30/1",
    options: SemanticVideoOptions | None = None,
    game_context: str = "仲間との旅を紹介する",
) -> SemanticVideoPlan:
    return SemanticVideoPlanner(client, options or SemanticVideoOptions()).plan(
        video=tmp_path / "input.mp4",
        metadata=VideoMetadata(
            duration_seconds=duration,
            width=1920,
            height=1080,
            codec_name="h264",
            average_frame_rate=average_frame_rate,
            video_stream_index=2,
            start_time_seconds=start,
            last_frame_timestamp_seconds=last_frame,
        ),
        identity_key="video-identity",
        cache_root=tmp_path / "cache",
        video_cache_dir=tmp_path / "cache" / "videos" / "video-identity",
        game_context=game_context,
    )


def test_video_events_define_absolute_candidate_times(
    tmp_path: Path, client: VllmClient, ffmpeg: Mock
) -> None:
    plan = run_plan(tmp_path, client)

    assert plan.timestamps == (6.0, 6.5, 7.0, 7.5, 8.0)
    assert plan.importance(7.0) == 85
    assert plan.provenance(7.0)[0]["summary"] == "仲間と合流して戦闘が始まる"
    assert plan.provenance(7.0)[0]["timestamp_seconds"] == 7.0
    ffmpeg.assert_called_once()


def test_cache_reuses_all_chunks_without_http_or_ffmpeg_and_repairs_middle_hole(
    tmp_path: Path, client: VllmClient, ffmpeg: Mock, mocker: MockerFixture
) -> None:
    first = run_plan(tmp_path, client, duration=70)
    assert [chunk["start_seconds"] for chunk in first.evidence["chunks"]] == [5, 33, 61]
    assert [chunk["end_seconds"] for chunk in first.evidence["chunks"]] == [35, 63, 75]
    assert ffmpeg.call_count == 3
    ffmpeg.reset_mock()
    complete = mocker.patch.object(
        client, "complete_json", side_effect=AssertionError("offline")
    )
    metadata = mocker.patch.object(
        client, "fetch_model_metadata", side_effect=AssertionError("offline")
    )

    restored = run_plan(tmp_path, client, duration=70)

    assert restored == first
    ffmpeg.assert_not_called()
    complete.assert_not_called()
    metadata.assert_not_called()

    files = sorted((tmp_path / "cache").rglob("chunk-*.json"))
    assert len(files) == 3
    files[1].unlink()
    complete.side_effect = None
    complete.return_value = event_response()
    metadata.side_effect = None
    run_plan(tmp_path, client, duration=70)

    assert ffmpeg.call_count == 1
    complete.assert_called_once()
    metadata.assert_called_once()
    assert "33.000000 - 63.000000" in complete.call_args.kwargs["prompt"]


def test_new_candidate_plan_version_reuses_existing_video_understanding_chunks(
    tmp_path: Path,
    client: VllmClient,
    ffmpeg: Mock,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as legacy:
        legacy.setattr(
            "src.services.semantic_video_planner.SEMANTIC_VIDEO_PLAN_VERSION", 1
        )
        legacy_plan = run_plan(tmp_path, client)
    ffmpeg.reset_mock()
    complete = mocker.patch.object(
        client, "complete_json", side_effect=AssertionError("offline")
    )
    metadata = mocker.patch.object(
        client, "fetch_model_metadata", side_effect=AssertionError("offline")
    )

    updated_plan = run_plan(tmp_path, client)

    assert updated_plan.timestamps == legacy_plan.timestamps
    assert updated_plan.evidence == legacy_plan.evidence
    assert updated_plan.cache_key != legacy_plan.cache_key
    ffmpeg.assert_not_called()
    complete.assert_not_called()
    metadata.assert_not_called()


def test_repaired_chunk_changes_plan_identity_when_meaning_changes(
    tmp_path: Path, client: VllmClient, ffmpeg: Mock, mocker: MockerFixture
) -> None:
    first = run_plan(tmp_path, client)
    next((tmp_path / "cache").rglob("chunk-*.json")).unlink()
    changed = event_response()
    changed["events"][0]["summary"] = "仲間との対話が終わる"
    mocker.patch.object(client, "complete_json", return_value=changed)

    second = run_plan(tmp_path, client)

    assert second.timestamps == first.timestamps
    assert second.cache_key != first.cache_key
    assert ffmpeg.call_count == 2


def test_exclusions_from_overlapping_chunks_remove_all_matching_candidate_times(
    tmp_path: Path, client: VllmClient, ffmpeg: Mock, mocker: MockerFixture
) -> None:
    first = event_response(start=27, end=30, timestamp=29)
    second = event_response(start=0, end=3, timestamp=1)
    second["excluded_intervals"] = [
        {"start_seconds": 0.5, "end_seconds": 1.5, "reason": "fade"}
    ]
    mocker.patch.object(client, "complete_json", side_effect=[first, second])

    plan = run_plan(tmp_path, client, duration=40)

    assert plan.timestamps == (33.0, 35.0)
    assert all(len(plan.provenance(timestamp)) == 2 for timestamp in plan.timestamps)
    assert ffmpeg.call_count == 2


def test_eventless_video_returns_no_sampled_fallback_candidates(
    tmp_path: Path, client: VllmClient, ffmpeg: Mock, mocker: MockerFixture
) -> None:
    mocker.patch.object(
        client, "complete_json", return_value={"events": [], "excluded_intervals": []}
    )

    plan = run_plan(tmp_path, client)

    assert plan.timestamps == ()
    assert plan.importance(7) == 0
    ffmpeg.assert_called_once()


def test_refinement_stays_at_or_before_last_video_frame(
    tmp_path: Path, client: VllmClient, ffmpeg: Mock, mocker: MockerFixture
) -> None:
    mocker.patch.object(
        client,
        "complete_json",
        return_value=event_response(start=8, end=8.55, timestamp=8.5),
    )

    plan = run_plan(tmp_path, client, last_frame=13.5)

    assert plan.timestamps == (13.0, 13.5)
    ffmpeg.assert_called_once()


@pytest.mark.parametrize(
    ("average_frame_rate", "expected"),
    [
        ("1/1", (13.0, 13.5, 14.0)),
        ("2/1", (13.0, 13.5, 14.0, 14.5)),
        ("0.5", (13.0,)),
    ],
)
def test_endpoint_event_candidates_respect_frame_interval_without_last_frame_probe(
    tmp_path: Path,
    client: VllmClient,
    ffmpeg: Mock,
    mocker: MockerFixture,
    average_frame_rate: str,
    expected: tuple[float, ...],
) -> None:
    mocker.patch.object(
        client,
        "complete_json",
        return_value=event_response(start=8, end=10, timestamp=9),
    )

    plan = run_plan(tmp_path, client, average_frame_rate=average_frame_rate)

    assert plan.timestamps == expected
    ffmpeg.assert_called_once()


def test_known_last_frame_takes_priority_over_frame_rate_fallback(
    tmp_path: Path, client: VllmClient, ffmpeg: Mock, mocker: MockerFixture
) -> None:
    mocker.patch.object(
        client,
        "complete_json",
        return_value=event_response(start=8, end=9.5, timestamp=9),
    )

    plan = run_plan(tmp_path, client, average_frame_rate="1/1", last_frame=14.5)

    assert plan.timestamps == (13.0, 13.5, 14.0, 14.5)
    ffmpeg.assert_called_once()


def test_frame_interval_fallback_never_moves_candidates_before_video_start(
    tmp_path: Path, client: VllmClient, ffmpeg: Mock, mocker: MockerFixture
) -> None:
    mocker.patch.object(
        client,
        "complete_json",
        return_value=event_response(start=0, end=0.5, timestamp=0),
    )

    plan = run_plan(tmp_path, client, duration=0.5, average_frame_rate="1/1")

    assert plan.timestamps == (5.0,)
    ffmpeg.assert_called_once()


@pytest.mark.parametrize("overlap", [0.0, 2.0])
def test_probe_last_frame_bounds_chunks_with_overreported_duration(
    tmp_path: Path,
    client: VllmClient,
    ffmpeg: Mock,
    mocker: MockerFixture,
    overlap: float,
) -> None:
    mocker.patch.object(
        client,
        "complete_json",
        return_value=event_response(start=0, end=1, timestamp=0.5),
    )
    plan = run_plan(
        tmp_path,
        client,
        duration=3600,
        last_frame=34.99,
        options=SemanticVideoOptions(overlap_seconds=overlap),
    )

    assert len(plan.evidence["chunks"]) == (1 if overlap == 0 else 2)
    assert all(chunk["start_seconds"] <= 34.99 for chunk in plan.evidence["chunks"])
    assert 34.99 <= plan.evidence["chunks"][-1]["end_seconds"] <= 35.040001
    assert ffmpeg.call_count == len(plan.evidence["chunks"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_seconds", -1),
        ("start_seconds", 4),
        ("end_seconds", 11),
        ("end_seconds", math.inf),
        ("timestamp_seconds", math.nan),
        ("timestamp_seconds", True),
        ("timestamp_seconds", "2"),
        ("timestamp_seconds", 4),
        ("importance", False),
        ("importance", 101),
        ("importance", 10**1000),
        ("summary", " "),
        ("summary", "x" * 301),
        ("summary", 3),
        ("unknown", "field"),
    ],
)
def test_invalid_model_events_fail_without_checkpointing(
    tmp_path: Path,
    client: VllmClient,
    ffmpeg: Mock,
    mocker: MockerFixture,
    field: str,
    value: Any,
) -> None:
    response = event_response()
    response["events"][0][field] = value
    mocker.patch.object(client, "complete_json", return_value=response)

    with pytest.raises(ValueError, match="動画理解"):
        run_plan(tmp_path, client)

    assert list((tmp_path / "cache").rglob("chunk-*.json")) == []
    ffmpeg.assert_called_once()


@pytest.mark.parametrize("corruption", ["digest", "schema", "envelope", "truncated"])
def test_corrupt_cache_is_revalidated_and_recomputed(
    tmp_path: Path, client: VllmClient, ffmpeg: Mock, corruption: str
) -> None:
    first = run_plan(tmp_path, client)
    path = next((tmp_path / "cache").rglob("chunk-*.json"))
    payload = json.loads(path.read_text())
    if corruption == "digest":
        payload["data"]["result"]["events"][0]["summary"] = "unverified replacement"
    elif corruption == "schema":
        payload["data"]["result"]["events"][0]["timestamp_seconds"] = True
        payload["data"]["result_digest"] = json_digest(payload["data"]["result"])
    elif corruption == "envelope":
        payload["phase_version"] = 0
    path.write_text("{" if corruption == "truncated" else json.dumps(payload))

    second = run_plan(tmp_path, client)

    assert second == first
    assert ffmpeg.call_count == 2


def test_interruption_preserves_completed_chunks(
    tmp_path: Path, client: VllmClient, ffmpeg: Mock, mocker: MockerFixture
) -> None:
    complete = mocker.patch.object(
        client,
        "complete_json",
        side_effect=[event_response(), RuntimeError("interrupted")],
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        run_plan(tmp_path, client, duration=70)
    assert len(list((tmp_path / "cache").rglob("chunk-*.json"))) == 1
    complete.side_effect = None
    complete.return_value = event_response()
    complete.reset_mock()
    ffmpeg.reset_mock()

    result = run_plan(tmp_path, client, duration=70)

    assert len(result.evidence["chunks"]) == 3
    assert complete.call_count == ffmpeg.call_count == 2


@pytest.mark.parametrize(
    "change", ["model", "endpoint", "revision", "context", "options", "metadata"]
)
def test_semantic_inputs_invalidate_chunk_cache(
    tmp_path: Path, client: VllmClient, ffmpeg: Mock, mocker: MockerFixture, change: str
) -> None:
    first = run_plan(tmp_path, client)
    config = client.config
    if change == "model":
        config = replace(config, model="new-video")
    elif change == "endpoint":
        config = replace(config, base_url="http://localhost:9000/v1")
    elif change == "revision":
        config = replace(config, cache_revision="2")
    next_client = VllmClient(config)
    mocker.patch.object(next_client, "fetch_model_metadata", return_value={})
    complete = mocker.patch.object(
        next_client, "complete_json", return_value=event_response()
    )

    second = run_plan(
        tmp_path,
        next_client,
        game_context="新しい解説文" if change == "context" else "仲間との旅を紹介する",
        options=SemanticVideoOptions(chunk_seconds=20) if change == "options" else None,
        duration=11 if change == "metadata" else 10,
    )

    assert second.cache_key != first.cache_key
    complete.assert_called_once()
    assert ffmpeg.call_count == 2


def test_clip_request_uses_selected_stream_start_and_bounded_video_payload(
    tmp_path: Path, client: VllmClient, ffmpeg: Mock, mocker: MockerFixture
) -> None:
    complete = mocker.patch.object(
        client, "complete_json", return_value=event_response()
    )

    run_plan(tmp_path, client)

    command = ffmpeg.call_args.args[0]
    assert command[command.index("-ss") + 1] == "5.000000"
    assert command[command.index("-t") + 1] == "10.000000"
    assert command[command.index("-map") + 1] == "0:2"
    assert "fps=1.0" in command[command.index("-vf") + 1]
    assert "512" in command[command.index("-vf") + 1]
    assert "setpts=PTS-STARTPTS" in command[command.index("-vf") + 1]
    kwargs = complete.call_args.kwargs
    url = kwargs["media"]["video_url"]["url"]
    assert url.startswith("data:video/mp4;base64,")
    assert base64.b64decode(url.partition(",")[2]) == b"test-mp4"
    assert kwargs["media_io_kwargs"] == {"video": {"fps": 1.0, "num_frames": 10}}
    assert not list((tmp_path / "cache").rglob("*.mp4"))


@pytest.mark.parametrize("size", [0, 16 * 1024 * 1024 + 1])
def test_empty_or_oversized_clip_is_rejected_before_inference(
    tmp_path: Path, client: VllmClient, ffmpeg: Mock, mocker: MockerFixture, size: int
) -> None:
    def invalid_encode(command: list[str], **_kwargs: Any) -> None:
        with Path(command[-1]).open("wb") as output:
            output.truncate(size)

    ffmpeg.side_effect = invalid_encode
    complete = mocker.patch.object(client, "complete_json")

    with pytest.raises(RuntimeError, match="clip"):
        run_plan(tmp_path, client)

    complete.assert_not_called()
    assert not list((tmp_path / "cache").rglob("*.mp4"))
