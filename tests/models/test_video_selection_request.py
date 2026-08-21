"""動画選定リクエストの互換性を検証する."""

from dataclasses import replace

from src.models.video_selection_request import VideoSelectionRequest


def test_replace_can_update_legacy_input_video() -> None:
    """旧input_video fieldを使うreplaceが単一入力を更新できること."""
    request = VideoSelectionRequest(
        input_video="old.mp4",
        output_dir="output",
        output_count=30,
        game_title=None,
        game_context="テスト用のGame Context",
        primary_model="primary",
        secondary_model="secondary",
        ollama_host=None,
        ollama_timeout=300.0,
        allow_cpu=False,
        ffmpeg_workers=4,
        sample_interval_seconds=None,
        debug=False,
    )

    updated = replace(
        request,
        input_video="new.mp4",  # type: ignore[call-arg]  # 旧fieldの実行時互換
    )

    assert updated.input_video == "new.mp4"
    assert updated.input_videos == ("new.mp4",)
