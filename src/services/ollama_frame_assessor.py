"""Ollama vision modelによる動画フレーム評価."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from ..models.video_selection import FrameAssessment, FrameCandidate

MODEL_OPTIONS = {"temperature": 0, "seed": 271}
MINIMUM_GPU_MEMORY_RATIO = 0.5


class OllamaModelValidationError(RuntimeError):
    """model identityまたはGPU実行条件が満たされないerror."""


class OllamaFrameAssessor:
    """Ollama APIのモデル検証・画像評価・GPU確認を担当する."""

    def __init__(
        self,
        host: str,
        *,
        timeout_seconds: float,
        require_gpu: bool,
    ) -> None:
        """接続設定を保持する."""
        self.host = self.normalize_host(host)
        self.timeout_seconds = timeout_seconds
        self.require_gpu = require_gpu
        self.gpu_evidence: dict[str, dict[str, Any]] = {}

    @staticmethod
    def normalize_host(host: str) -> str:
        """Ollama hostへschemeと既定portを補う."""
        normalized = host.strip().rstrip("/")
        if not normalized:
            raise ValueError("Ollama hostが空です")
        if "://" not in normalized:
            normalized = f"http://{normalized}"
        parsed = urlsplit(normalized)
        if parsed.port is None:
            parsed = parsed._replace(netloc=f"{parsed.netloc}:11434")
        return urlunsplit(parsed)

    def fetch_model_metadata(
        self,
        requested_models: set[str],
    ) -> dict[str, dict[str, Any]]:
        """指定モデルのdigestとvision capabilityを検証して返す."""
        payload = self._request_json(f"{self.host}/api/tags", timeout_seconds=30)
        raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            raise ValueError("Ollama /api/tagsにmodelsがありません")

        available: dict[str, str] = {}
        for raw_model in raw_models:
            if not isinstance(raw_model, dict):
                continue
            name = raw_model.get("name") or raw_model.get("model")
            digest = raw_model.get("digest")
            if isinstance(name, str) and isinstance(digest, str):
                available[name] = digest
        resolved_names: dict[str, str] = {}
        for requested_model in requested_models:
            if requested_model in available:
                resolved_names[requested_model] = requested_model
                continue
            latest_name = f"{requested_model}:latest"
            untagged = ":" not in requested_model.rsplit("/", maxsplit=1)[-1]
            if untagged and latest_name in available:
                resolved_names[requested_model] = latest_name

        missing = sorted(requested_models - resolved_names.keys())
        if missing:
            raise ValueError(
                f"Ollama modelが見つかりません: {missing}; "
                f"利用可能: {sorted(available)}"
            )

        metadata: dict[str, dict[str, Any]] = {}
        for model in sorted(requested_models):
            resolved_name = resolved_names[model]
            details = self._request_json(
                f"{self.host}/api/show",
                payload={"model": resolved_name},
                timeout_seconds=min(self.timeout_seconds, 60),
            )
            capabilities = details.get("capabilities")
            if not isinstance(capabilities, list) or "vision" not in capabilities:
                raise ValueError(f"vision非対応のOllama modelです: {resolved_name}")
            metadata[model] = {
                "digest": available[resolved_name],
                "resolved_name": resolved_name,
                "capabilities": capabilities,
                "details": details.get("details"),
            }
        return metadata

    def assess(
        self,
        *,
        model: str,
        model_digest: str,
        prompt: str,
        candidates: Sequence[FrameCandidate],
        contact_sheet: Path,
    ) -> list[FrameAssessment]:
        """一枚のコンタクトシート内の全候補を評価する."""
        payload = {
            "model": model,
            "stream": False,
            "format": "json",
            "think": False,
            "keep_alive": "1h",
            "options": MODEL_OPTIONS,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [
                        base64.b64encode(contact_sheet.read_bytes()).decode("ascii")
                    ],
                }
            ],
        }
        response = self._request_json(
            f"{self.host}/api/chat",
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        self._verify_gpu_use(model, model_digest)
        message = response.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama応答にmessageがありません")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            thinking = message.get("thinking")
            content = thinking if isinstance(thinking, str) else ""
        frames = self._parse_json_object(content).get("frames")
        if not isinstance(frames, list):
            raise ValueError("Ollama応答JSONにframesがありません")

        assessments = [self._assessment_from_item(item) for item in frames]
        expected_ids = {candidate.frame_id for candidate in candidates}
        actual_ids = [assessment.frame_id for assessment in assessments]
        if set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
            raise ValueError(
                "Ollama応答のframe IDが一致しません: "
                f"expected={sorted(expected_ids)}, actual={actual_ids}"
            )
        return assessments

    def _verify_gpu_use(self, model: str, model_digest: str) -> None:
        """各batchでloaded modelのdigestとGPU配置を確認する."""
        payload = self._request_json(f"{self.host}/api/ps", timeout_seconds=30)
        raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            raise OllamaModelValidationError(
                "Ollama /api/psからmodel利用状況を取得できません"
            )

        matching_name = [
            raw_model
            for raw_model in raw_models
            if isinstance(raw_model, dict)
            and (raw_model.get("name") or raw_model.get("model")) == model
        ]
        loaded_model = next(
            (
                raw_model
                for raw_model in matching_name
                if raw_model.get("digest") == model_digest
            ),
            None,
        )
        if loaded_model is None:
            actual_digests = sorted(
                str(raw_model.get("digest")) for raw_model in matching_name
            )
            raise OllamaModelValidationError(
                f"Ollama model digestが一致しません: {model}; "
                f"expected={model_digest}, actual={actual_digests}"
            )
        if not self.require_gpu:
            return

        size = loaded_model.get("size")
        size_vram = loaded_model.get("size_vram")
        if (
            not isinstance(size, int | float)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(size_vram, int | float)
            or isinstance(size_vram, bool)
            or size_vram / size < MINIMUM_GPU_MEMORY_RATIO
        ):
            raise OllamaModelValidationError(
                f"Ollama modelのGPU利用を確認できません: {model}"
            )
        evidence = {
            "name": model,
            "digest": model_digest,
            "size": size,
            "size_vram": size_vram,
            "gpu_memory_ratio": round(size_vram / size, 4),
        }
        self.gpu_evidence[model] = evidence

    def _request_json(
        self,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Ollama APIへJSON requestを送りJSON objectを返す."""
        data = None
        headers: dict[str, str] = {}
        method = "GET"
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
            method = "POST"
        request = Request(url, data=data, headers=headers, method=method)
        with urlopen(request, timeout=timeout_seconds) as response:
            result: Any = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError(f"Ollama APIがJSON objectを返しませんでした: {url}")
        return result

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        """応答本文の先頭または最初の`{`からJSON objectを読む."""
        stripped = content.strip()
        decoder = json.JSONDecoder()
        starts = [0]
        first_object = stripped.find("{")
        if first_object > 0:
            starts.append(first_object)
        for start in starts:
            try:
                payload, _end = decoder.raw_decode(stripped[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise ValueError("Ollama応答にJSON objectがありません")

    @staticmethod
    def _assessment_from_item(item: object) -> FrameAssessment:
        """Ollama応答の一項目を検証して値オブジェクトへ変換する."""
        if not isinstance(item, dict):
            raise ValueError("frame評価がJSON objectではありません")
        frame_id = item.get("id")
        score = item.get("blog_score")
        transition = item.get("transition")
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("frame評価にidがありません")
        if (
            not isinstance(score, int | float)
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0 <= float(score) <= 100
        ):
            raise ValueError(f"blog_scoreが0から100の範囲外です: {score!r}")
        if not isinstance(transition, bool):
            raise ValueError("transitionがbooleanではありません")
        scene = item.get("scene", "その他")
        reason = item.get("reason", "")
        if not isinstance(scene, str) or not scene.strip():
            scene = "その他"
        if not isinstance(reason, str):
            reason = ""
        return FrameAssessment(
            frame_id=frame_id,
            blog_score=float(score),
            is_transition=transition,
            scene=scene.strip()[:80],
            reason=reason.strip()[:300],
        )
