"""Explicitly record a reviewed Python baseline; never invoked by regression tests."""

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from PIL import Image

from src.services.video_selector import VideoSelector
from src.utils.video_selection_files import file_sha256
from tests.migration.support.pipeline_fixture import (
    FIXTURE_ROOT,
    FixtureHttp,
    RecordingExtractor,
    cache_root,
    pipeline_contract,
    request_for,
)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed-update", action="store_true", required=True)
    parser.add_argument("--inference-media-only", action="store_true")
    args = parser.parse_args()
    for method in ("sampled_frames", "semantic_video"):
        with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
            request = request_for(Path(directory), method)
            FixtureHttp(
                method,
                record_media_directory=FIXTURE_ROOT / "inference-media" / method,
            ).install(patch)
            VideoSelector(request, frame_extractor=RecordingExtractor()).run()
            if args.inference_media_only:
                continue
            write_json(
                FIXTURE_ROOT / "expected" / f"{method}.json", pipeline_contract(request)
            )
            destination = FIXTURE_ROOT / "stored-cache" / method
            if destination.exists():
                shutil.rmtree(destination)
            source_root = cache_root(request)
            for path in source_root.rglob("*"):
                if not path.is_file() or path.suffix not in {".json", ".jpg"}:
                    continue
                relative = path.relative_to(source_root)
                if relative.parts[0] == "runs" and path.name != "run-manifest.json":
                    continue
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            for name in ("selected-01", "selected-02"):
                target = FIXTURE_ROOT / "reference-images" / method / f"{name}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(Path(request.output_dir) / f"{name}.jpg") as image:
                    image.convert("RGB").save(target)
    if args.inference_media_only:
        return
    write_json(
        FIXTURE_ROOT / "baseline-provenance.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "source_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "opencv-python", "pillow")
            },
            "ffmpeg": subprocess.run(
                ["ffmpeg", "-version"], check=True, capture_output=True, text=True
            ).stdout.splitlines()[0],
            "input_sha256": file_sha256(FIXTURE_ROOT / "synthetic-game.mkv"),
            "files": {
                "cache-replay.json": file_sha256(FIXTURE_ROOT / "cache-replay.json"),
                **{
                    path.relative_to(FIXTURE_ROOT).as_posix(): file_sha256(path)
                    for folder in (
                        "expected",
                        "stored-cache",
                        "reference-images",
                        "responses",
                        "inference-media",
                    )
                    for path in sorted((FIXTURE_ROOT / folder).rglob("*"))
                    if path.is_file()
                },
            },
        },
    )


if __name__ == "__main__":
    main()
