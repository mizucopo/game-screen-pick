"""PROTOTYPE: 最大24画像のScene Catalog能力をreference runtimeで測る。"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
SCENE_ROLES = {"ordinary", "cinematic", "recurring_gameplay"}
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def scene_catalog_schema() -> dict[str, Any]:
    """Issue #165のScene Catalog schemaを返す。"""
    scene = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "slug": {"type": "string", "pattern": SLUG_PATTERN.pattern},
            "display_name": {"type": "string", "minLength": 1},
            "description": {"type": "string", "minLength": 1},
            "selection_role": {
                "type": "string",
                "enum": sorted(SCENE_ROLES),
            },
        },
        "required": ["slug", "display_name", "description", "selection_role"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scenes": {
                "type": "array",
                "minItems": 3,
                "maxItems": 8,
                "items": scene,
            }
        },
        "required": ["scenes"],
    }


def parse_args() -> argparse.Namespace:
    """probe引数を返す。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--ollama-host", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--num-ctx", type=int, default=32768)
    return parser.parse_args()


def request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 10,
) -> dict[str, Any]:
    """JSON APIを呼び出す。"""
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def discover_videos(video_dir: Path) -> list[Path]:
    """root直下の対応動画を自然順相当で返す。"""
    videos = sorted(
        (
            path
            for path in video_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    if not videos:
        raise RuntimeError("supported video was not found")
    return videos


def video_duration_seconds(video: Path) -> float:
    """ffprobeから動画長を返す。"""
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def sample_times(duration: float) -> tuple[float, float]:
    """動画の1/3と2/3から安全な抽出時刻を返す。"""
    if duration <= 0:
        raise RuntimeError(f"invalid video duration: {duration}")
    margin = min(1.0, duration / 10)
    end = max(margin, duration - margin)
    return (
        min(end, max(margin, duration / 3)),
        min(end, max(margin, duration * 2 / 3)),
    )


def extract_images(videos: list[Path], destination: Path) -> list[Path]:
    """各動画から2枚ずつ960px JPEGを一時抽出する。"""
    images: list[Path] = []
    for video_index, video in enumerate(videos, start=1):
        duration = video_duration_seconds(video)
        for sample_index, timestamp in enumerate(sample_times(duration), start=1):
            output = destination / f"v{video_index:02d}-{sample_index}.jpg"
            print(
                f"extract {len(images) + 1:02d}/{len(videos) * 2}: "
                f"{video.name} @ {timestamp:.3f}s",
                file=sys.stderr,
                flush=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-ss",
                    f"{timestamp:.6f}",
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=960:-2:flags=lanczos",
                    "-q:v",
                    "3",
                    "-y",
                    str(output),
                ],
                check=True,
            )
            images.append(output)
    if len(images) != 24:
        raise RuntimeError(f"expected 24 images, got {len(images)}")
    return images


def encode_images(images: list[Path]) -> list[str]:
    """画像をOllama API用base64へ変換する。"""
    return [base64.b64encode(image.read_bytes()).decode() for image in images]


def validate_catalog(value: object) -> list[dict[str, str]]:
    """Scene Catalogのlocal domain contractを検証する。"""
    if not isinstance(value, dict) or set(value) != {"scenes"}:
        raise ValueError("catalog_shape")
    scenes = value["scenes"]
    if not isinstance(scenes, list) or not 3 <= len(scenes) <= 8:
        raise ValueError("scene_count")
    expected_keys = {"slug", "display_name", "description", "selection_role"}
    normalized: list[dict[str, str]] = []
    for scene in scenes:
        if not isinstance(scene, dict) or set(scene) != expected_keys:
            raise ValueError("scene_shape")
        if not all(isinstance(scene[key], str) for key in expected_keys):
            raise ValueError("scene_value_type")
        if not SLUG_PATTERN.fullmatch(scene["slug"]):
            raise ValueError("scene_slug")
        if not scene["display_name"] or not scene["description"]:
            raise ValueError("empty_scene_text")
        if scene["selection_role"] not in SCENE_ROLES:
            raise ValueError("scene_role")
        normalized.append({key: scene[key] for key in sorted(expected_keys)})
    slugs = [scene["slug"] for scene in normalized]
    if len(slugs) != len(set(slugs)):
        raise ValueError("duplicate_scene_slug")
    other = [scene for scene in normalized if scene["slug"] == "other"]
    if len(other) != 1 or other[0]["selection_role"] != "ordinary":
        raise ValueError("other_scene_contract")
    return normalized


def gpu_memory_used_mib() -> int | None:
    """WSLのnvidia-smiからGPU使用memoryを返す。"""
    command = Path("/usr/lib/wsl/lib/nvidia-smi")
    if not command.exists():
        return None
    completed = subprocess.run(
        [
            str(command),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return int(completed.stdout.strip().splitlines()[0])


@dataclass
class RuntimePeak:
    """request中のruntime観測最大値。"""

    gpu_memory_used_mib: int | None = None
    size_bytes: int | None = None
    size_vram_bytes: int | None = None
    context_length: int | None = None

    def observe_gpu(self, value: int | None) -> None:
        """GPU memory最大値を更新する。"""
        if value is not None:
            self.gpu_memory_used_mib = max(self.gpu_memory_used_mib or 0, value)

    def observe_model(self, model: dict[str, Any]) -> None:
        """Ollama runtime情報を更新する。"""
        for key, attribute in (
            ("size", "size_bytes"),
            ("size_vram", "size_vram_bytes"),
            ("context_length", "context_length"),
        ):
            value = model.get(key)
            if isinstance(value, int):
                previous = getattr(self, attribute)
                setattr(self, attribute, max(previous or 0, value))


def monitor_runtime(host: str, stop: threading.Event, peak: RuntimePeak) -> None:
    """request中のOllamaとGPU stateをpollする。"""
    while not stop.is_set():
        peak.observe_gpu(gpu_memory_used_mib())
        try:
            payload = request_json(f"{host}/api/ps", timeout=1)
        except Exception:
            payload = {}
        models = payload.get("models", [])
        if isinstance(models, list):
            for model in models:
                if isinstance(model, dict):
                    peak.observe_model(model)
        stop.wait(0.1)


def run_probe(
    *,
    host: str,
    model: str,
    images: list[str],
    num_ctx: int,
    run_number: int,
) -> dict[str, Any]:
    """1回のScene Catalog requestを実行して集計を返す。"""
    prompt = (
        "入力された24枚は、順序付きVideo Set全体から選ばれた代表frameです。"
        "ブログ画像選定に使う共有Scene Catalogを3〜8件で作ってください。"
        "slugは小文字英数字とhyphenだけで一意にし、otherを必ず1件だけ含め、"
        "otherのselection_roleはordinaryにしてください。"
        "個々の画像を採否せず、Video Set全体を分類できる"
        "scene vocabularyだけを返してください。"
    )
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": scene_catalog_schema(),
        "options": {"temperature": 0, "num_ctx": num_ctx},
        "keep_alive": "10m",
        "messages": [{"role": "user", "content": prompt, "images": images}],
    }
    stop = threading.Event()
    peak = RuntimePeak()
    monitor = threading.Thread(
        target=monitor_runtime,
        args=(host, stop, peak),
        daemon=True,
    )
    print(f"run {run_number}: request start", file=sys.stderr, flush=True)
    started = time.monotonic()
    monitor.start()
    try:
        response = request_json(f"{host}/api/chat", payload=payload, timeout=300)
    finally:
        stop.set()
        monitor.join(timeout=2)
    wall_seconds = time.monotonic() - started
    content = response.get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("missing_response_content")
    parsed = json.loads(content)
    scenes = validate_catalog(parsed)
    full_digest = hashlib.sha256(
        json.dumps(parsed, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    semantic_shape = [
        {"slug": scene["slug"], "selection_role": scene["selection_role"]}
        for scene in scenes
    ]
    shape_digest = hashlib.sha256(
        json.dumps(semantic_shape, sort_keys=True).encode()
    ).hexdigest()
    result = {
        "run": run_number,
        "schema_and_domain_valid": True,
        "scene_count": len(scenes),
        "scenes": semantic_shape,
        "full_output_sha256": full_digest,
        "semantic_shape_sha256": shape_digest,
        "wall_seconds": round(wall_seconds, 3),
        "total_duration_seconds": round(response.get("total_duration", 0) / 1e9, 3),
        "load_duration_seconds": round(response.get("load_duration", 0) / 1e9, 3),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_seconds": round(response.get("prompt_eval_duration", 0) / 1e9, 3),
        "eval_count": response.get("eval_count"),
        "eval_seconds": round(response.get("eval_duration", 0) / 1e9, 3),
        "done_reason": response.get("done_reason"),
        "runtime_peak": {
            "gpu_memory_used_mib": peak.gpu_memory_used_mib,
            "model_size_bytes": peak.size_bytes,
            "model_size_vram_bytes": peak.size_vram_bytes,
            "context_length": peak.context_length,
            "fully_gpu_loaded": (
                peak.size_bytes is not None and peak.size_bytes == peak.size_vram_bytes
            ),
        },
    }
    print(f"run {run_number}: request complete", file=sys.stderr, flush=True)
    return result


def main() -> None:
    """一時frameを抽出して最大24画像probeを実行する。"""
    args = parse_args()
    if args.runs < 1:
        raise ValueError("runs must be positive")
    host = args.ollama_host.rstrip("/")
    version = request_json(f"{host}/api/version")
    tags = request_json(f"{host}/api/tags")
    matching = [
        item
        for item in tags.get("models", [])
        if isinstance(item, dict) and item.get("name") == args.model
    ]
    if len(matching) != 1:
        raise RuntimeError(f"configured model was not found: {args.model}")
    model_info = matching[0]
    videos = discover_videos(args.video_dir)
    with tempfile.TemporaryDirectory(prefix="game-screen-pick-issue-169-") as temp:
        image_paths = extract_images(videos, Path(temp))
        images = encode_images(image_paths)
        results = [
            run_probe(
                host=host,
                model=args.model,
                images=images,
                num_ctx=args.num_ctx,
                run_number=index,
            )
            for index in range(1, args.runs + 1)
        ]
    summary = {
        "question": "maximum_24_image_scene_catalog_capability",
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ollama_version": version.get("version"),
        "model": args.model,
        "model_digest": model_info.get("digest"),
        "parameter_size": model_info.get("details", {}).get("parameter_size"),
        "quantization": model_info.get("details", {}).get("quantization_level"),
        "configured_num_ctx": args.num_ctx,
        "video_count": len(videos),
        "image_count": 24,
        "runs": results,
        "all_runs_valid": all(run["schema_and_domain_valid"] for run in results),
        "semantic_shape_stable": len({run["semantic_shape_sha256"] for run in results})
        == 1,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
