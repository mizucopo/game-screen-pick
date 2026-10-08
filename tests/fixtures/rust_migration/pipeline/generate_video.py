"""Generate the public, deterministic lossless migration video without game footage."""

import argparse
import random
import subprocess
from pathlib import Path

WIDTH = 160
HEIGHT = 96
FPS = 4
SECONDS = 6


def frame_rgb(second: int) -> bytes:
    """Return a coarse high-margin pattern, or the deliberately black third scene."""
    if second == 2:
        return bytes(WIDTH * HEIGHT * 3)
    generator = random.Random(338 + second)
    levels = [
        [generator.choice((35, 100, 170, 235)) for _ in range(9)] for _ in range(8)
    ]
    pixels = bytearray()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            level = levels[y * 8 // HEIGHT][x * 9 // WIDTH]
            pixels.extend((level, max(0, level - 15), min(255, level + 15)))
    return bytes(pixels)


def main() -> None:
    """Encode fixed RGB frames; fixture tests use the committed file, not a rebuild."""
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
            "rgb24",
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
            "bgr0",
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
        input=b"".join(frame_rgb(second) * FPS for second in range(SECONDS)),
        capture_output=True,
        check=True,
    )


if __name__ == "__main__":
    main()
