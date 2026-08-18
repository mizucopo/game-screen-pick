"""ffmpegを使った単一動画のメタデータ取得とフレーム抽出."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..models.video_selection import VideoMetadata
from ..utils.video_selection_files import is_valid_image


class VideoFrameExtractor:
    """ffprobeとffmpegを安全な引数配列で呼び出す."""

    def __init__(self) -> None:
        """必要な外部コマンドが利用できることを確認する."""
        missing = [name for name in ("ffprobe", "ffmpeg") if shutil.which(name) is None]
        if missing:
            raise RuntimeError(
                "動画処理に必要なコマンドが見つかりません: " + ", ".join(missing)
            )

    def probe(self, video: Path) -> VideoMetadata:
        """動画時間と先頭映像streamの情報を返す."""
        payload = self._run_json(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                (
                    "format=duration,start_time:"
                    "stream=index,codec_type,codec_name,width,height,"
                    "avg_frame_rate,duration,start_time:"
                    "stream_disposition=attached_pic"
                ),
                "-of",
                "json",
                str(video),
            ]
        )
        format_payload = payload.get("format")
        streams = payload.get("streams")
        if not isinstance(format_payload, dict) or not isinstance(streams, list):
            raise ValueError("ffprobeから動画メタデータを取得できませんでした")

        video_stream = next(
            (
                stream
                for stream in streams
                if (
                    isinstance(stream, dict)
                    and stream.get("codec_type") == "video"
                    and not self._is_attached_picture(stream)
                )
            ),
            None,
        )
        if video_stream is None:
            raise ValueError("映像streamが見つかりません")
        format_start = self._finite_timestamp(format_payload.get("start_time")) or 0.0
        stream_start = self._finite_timestamp(video_stream.get("start_time"))
        start_time = max(
            0.0,
            (stream_start if stream_start is not None else format_start) - format_start,
        )
        duration = self._positive_duration(video_stream.get("duration"))
        if duration is None:
            container_duration = self._positive_duration(format_payload.get("duration"))
            duration = (
                self._positive_duration(container_duration - start_time)
                if container_duration is not None
                else None
            )
        if duration is None:
            raise ValueError("動画時間を取得できませんでした")
        return VideoMetadata(
            duration_seconds=duration,
            width=self._positive_int(video_stream.get("width"), "width"),
            height=self._positive_int(video_stream.get("height"), "height"),
            codec_name=str(video_stream.get("codec_name", "unknown")),
            average_frame_rate=str(video_stream.get("avg_frame_rate", "unknown")),
            video_stream_index=self._nonnegative_int(
                video_stream.get("index"), "stream index"
            ),
            start_time_seconds=start_time,
        )

    def extract_frame(
        self,
        video: Path,
        timestamp_seconds: float,
        output_path: Path,
        *,
        max_width: int | None,
        video_stream_index: int = 0,
    ) -> None:
        """指定時刻の映像フレームをJPEGとしてatomicに出力する."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.stem}.partial.jpg")
        temporary.unlink(missing_ok=True)
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp_seconds:.6f}",
            "-i",
            str(video),
            "-map",
            f"0:{video_stream_index}",
            "-frames:v",
            "1",
            "-an",
            "-threads",
            "2",
        ]
        if max_width is not None:
            command.extend(["-vf", f"scale='min(iw,{max_width})':-2:flags=lanczos"])
        command.extend(["-q:v", "2", "-y", str(temporary)])
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            if not is_valid_image(temporary):
                raise RuntimeError(
                    f"ffmpegが有効な画像を出力しませんでした: {output_path}"
                )
            temporary.replace(output_path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _positive_int(value: object, field_name: str) -> int:
        """ffprobeの正整数フィールドを検証する."""
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"動画の{field_name}が不正です: {value!r}")
        return value

    @staticmethod
    def _nonnegative_int(value: object, field_name: str) -> int:
        """ffprobeの非負整数フィールドを検証する."""
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"動画の{field_name}が不正です: {value!r}")
        return value

    @staticmethod
    def _is_attached_picture(stream: dict[str, Any]) -> bool:
        """cover art用video streamかを返す."""
        disposition = stream.get("disposition")
        return isinstance(disposition, dict) and disposition.get("attached_pic") == 1

    @staticmethod
    def _positive_duration(value: object) -> float | None:
        """ffprobeの有限な正のdurationを返す."""
        if isinstance(value, bool) or not isinstance(value, str | int | float):
            return None
        try:
            duration = float(value)
        except ValueError:
            return None
        if not math.isfinite(duration) or duration <= 0:
            return None
        return duration

    @staticmethod
    def _finite_timestamp(value: object) -> float | None:
        """ffprobeの有限なtimestampを返す."""
        if isinstance(value, bool) or not isinstance(value, str | int | float):
            return None
        try:
            timestamp = float(value)
        except ValueError:
            return None
        return timestamp if math.isfinite(timestamp) else None

    @staticmethod
    def _run_json(command: list[str]) -> dict[str, Any]:
        """コマンドの標準出力をJSON objectとして返す."""
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        payload: Any = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise ValueError("ffprobeがJSON objectを返しませんでした")
        return payload
