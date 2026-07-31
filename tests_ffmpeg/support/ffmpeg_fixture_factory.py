"""binaryをcommitせずFFmpeg integration fixtureを生成する。"""

import subprocess
from pathlib import Path

_FIXTURE_TEXT_FOLDER = Path(__file__).resolve().parents[1] / "fixtures"


def generate_cfr_video(output_path: Path) -> Path:
    """2fps・2秒のCFR test patternを生成する。"""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=2:duration=2",
            "-map",
            "0:v:0",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def generate_odd_dimension_video(output_path: Path) -> Path:
    """奇数の幅と高さを持つsource frameを生成する。"""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=65x49:rate=1:duration=1",
            "-map",
            "0:v:0",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "yuv444p",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def generate_nonzero_start_video(output_path: Path) -> Path:
    """5秒の非ゼロ開始PTSを持つ4fps fixtureを生成する。"""
    codec = "mpeg4" if output_path.suffix.casefold() == ".mp4" else "ffv1"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=4:duration=4",
            "-output_ts_offset",
            "5",
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def generate_delayed_video_with_audio(output_path: Path) -> Path:
    """audioより2秒遅く始まる15秒のvideo streamを生成する。"""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=2:duration=15",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=17",
            "-filter_complex",
            "[0:v]setpts=PTS+2/TB[video]",
            "-map",
            "[video]",
            "-map",
            "1:a:0",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def generate_vfr_video(output_path: Path) -> Path:
    """不均一なsource PTSを持つVFR test patternを生成する。"""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=4:duration=1.25",
            "-vf",
            "select=eq(n\\,0)+eq(n\\,1)+eq(n\\,3)+eq(n\\,4)",
            "-fps_mode",
            "vfr",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def generate_scene_change_video(output_path: Path) -> Path:
    """1秒ごとに明確なscene changeを持つ3秒fixtureを生成する。"""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:size=64x48:rate=4:duration=1",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=4:duration=1",
            "-f",
            "lavfi",
            "-i",
            "color=blue:size=64x48:rate=4:duration=1",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[video]",
            "-map",
            "[video]",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def generate_av1_aac_video(output_path: Path) -> Path:
    """AV1 videoとAAC audioを持つ短いfixtureを生成する。"""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=2:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libsvtav1",
            "-preset",
            "11",
            "-crf",
            "40",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-shortest",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def generate_stream_matrix_video(output_path: Path) -> Path:
    """video、multiple audio、embedded subtitleを持つfixtureを生成する。"""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=2:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000:duration=3",
            "-i",
            str(_FIXTURE_TEXT_FOLDER / "ja-default.srt"),
            "-i",
            str(_FIXTURE_TEXT_FOLDER / "en-forced.srt"),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "2:a:0",
            "-map",
            "3:s:0",
            "-map",
            "4:s:0",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "pcm_s16le",
            "-c:s",
            "subrip",
            "-metadata:s:a:0",
            "language=jpn",
            "-metadata:s:a:1",
            "language=eng",
            "-disposition:a:0",
            "default",
            "-disposition:a:1",
            "0",
            "-metadata:s:s:0",
            "language=jpn",
            "-metadata:s:s:1",
            "language=eng",
            "-disposition:s:0",
            "default",
            "-disposition:s:1",
            "forced",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def generate_quantized_audio(output_path: Path) -> Path:
    """packet PTSがsample gridから量子化ずれするaudio fixtureを生成する。"""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-af",
            "asetpts=PTS+gte(T\\,1)*9",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def generate_corrupt_video(output_path: Path) -> Path:
    """headerを保ち途中のMPEG-TS packetを破損させたfixtureを生成する。"""
    valid_path = output_path.with_name(f".{output_path.stem}.valid.ts")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=10:duration=2",
            "-c:v",
            "mpeg2video",
            "-f",
            "mpegts",
            str(valid_path),
        ],
        check=True,
    )
    payload = bytearray(valid_path.read_bytes())
    corrupt_start = len(payload) * 43 // 100
    corrupt_length = len(payload) * 16 // 100
    payload[corrupt_start : corrupt_start + corrupt_length] = b"\0" * corrupt_length
    output_path.write_bytes(payload)
    valid_path.unlink()
    return output_path
