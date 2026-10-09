"""Explicitly record a reviewed Python baseline; requires --reviewed-update."""

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
from PIL.PngImagePlugin import PngInfo

from src.services.video_selector import VideoSelector
from src.utils.video_selection_files import file_sha256, json_digest
from tests.migration.support.pipeline_fixture import (
    FIXTURE_ROOT,
    FixtureHttp,
    RecordingExtractor,
    cache_root,
    load_json,
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
    only = parser.add_mutually_exclusive_group()
    only.add_argument("--inference-media-only", action="store_true")
    only.add_argument("--selected-contact-sheet-only", action="store_true")
    only.add_argument("--run-manifest-only", action="store_true")
    args = parser.parse_args()
    manifest_digests: dict[str, str] = {}
    for method in ("sampled_frames", "semantic_video"):
        with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
            request = request_for(Path(directory), method)
            FixtureHttp(
                method,
                record_media_directory=(
                    None
                    if args.selected_contact_sheet_only or args.run_manifest_only
                    else FIXTURE_ROOT / "inference-media" / method
                ),
            ).install(patch)
            VideoSelector(request, frame_extractor=RecordingExtractor()).run()
            if args.run_manifest_only:
                contract = pipeline_contract(request)
                target = FIXTURE_ROOT / "expected" / f"{method}.json"
                expected = load_json(target)
                for field in ("run_manifest", "run_manifest_schema"):
                    expected[field] = contract[field]
                write_json(target, expected)
                manifest_digests[method] = json_digest(contract["run_manifest"])
                continue
            if not args.inference_media_only:
                target = (
                    FIXTURE_ROOT
                    / "reference-images"
                    / method
                    / "selected-contact-sheet.png"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                metadata = PngInfo()
                metadata.add_text(
                    "source_revision",
                    subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip(),
                )
                with Image.open(
                    Path(request.output_dir) / "selected-contact-sheet.jpg"
                ) as image:
                    image.convert("RGB").save(target, pnginfo=metadata)
            if args.inference_media_only or args.selected_contact_sheet_only:
                continue
            # The full reviewed update replaces these pixel references below.
            # Retain receipt/recipe/integrity checks without comparing old pixels.
            contract = pipeline_contract(request, compare_candidate_pixels=False)
            write_json(FIXTURE_ROOT / "expected" / f"{method}.json", contract)
            manifest_digests[method] = json_digest(contract["run_manifest"])
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
    if args.inference_media_only or args.selected_contact_sheet_only:
        return
    write_json(
        FIXTURE_ROOT / "expected" / "run-manifest-provenance.json",
        {
            "source_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "manifest_json_digests": manifest_digests,
        },
    )
    if args.run_manifest_only:
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
