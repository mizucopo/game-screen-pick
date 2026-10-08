"""Generate the public, deterministic lossless migration video without game footage."""

import argparse
import random
import subprocess
from pathlib import Path

WIDTH = 160
HEIGHT = 96
FPS = 4
SECONDS = 6
BLOCK_SIZE = 16


def frame_yuv(second: int) -> bytes:
    """Align full-range luma blocks to JPEG DCT boundaries and use neutral chroma."""
    generator = random.Random(338 + second)
    levels = [
        [generator.choice((32, 96, 160, 224)) for _ in range(WIDTH // BLOCK_SIZE)]
        for _ in range(HEIGHT // BLOCK_SIZE)
    ]
    luma = bytes(
        0 if second == 2 else levels[y // BLOCK_SIZE][x // BLOCK_SIZE]
        for y in range(HEIGHT)
        for x in range(WIDTH)
    )
    return luma + bytes([128]) * (WIDTH * HEIGHT * 2)


def main() -> None:
    """Encode fixed YUV frames; fixture tests use the committed file, not a rebuild."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "yuv444p",
            "-video_size",
            f"{WIDTH}x{HEIGHT}",
            "-framerate",
            str(FPS),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "yuv444p",
            "-color_range",
            "pc",
            "-threads",
            "1",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            "-map_metadata",
            "-1",
            "-y",
            str(args.output),
        ],
        input=b"".join(frame_yuv(second) * FPS for second in range(SECONDS)),
        capture_output=True,
        check=True,
    )


if __name__ == "__main__":
    main()
