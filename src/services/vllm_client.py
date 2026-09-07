"""vLLM OpenAI互換APIによる動画理解と画像評価."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..models.video_selection import FrameAssessment, FrameCandidate
from ..models.vllm_config import VllmConfig
from ..utils.video_selection_files import json_digest
from .ollama_frame_assessor import (
    assessment_response_schema,
    batch_display_frame_ids,
)


class VllmClient:
    """動画と画像を同じserved modelへ送り、構造化応答を返す."""

    def __init__(self, config: VllmConfig) -> None:
        """接続設定とGPUの未検証状態を保持する."""
        self.config = config
        self.host = config.base_url
        self.gpu_evidence: dict[str, dict[str, Any]] = {}

    def model_metadata(self) -> dict[str, Any]:
        """通信せずcache identityを返す。weight digestやGPU配置は検証しない."""
        identity = {
            "provider": "vllm",
            "base_url": self.host,
            "model": self.config.model,
            "cache_revision": self.config.cache_revision,
        }
        return {
            **identity,
            "digest": json_digest(identity),
            "resolved_name": self.config.model,
            "identity_source": "configured_cache_revision",
            "capabilities": ["vision", "video"],
            "capabilities_source": "required_by_client",
        }

    def fetch_model_metadata(
        self,
        requested_models: set[str],
    ) -> dict[str, dict[str, Any]]:
        """設定したserved modelの存在を確認し、cache identityを返す."""
        if requested_models != {self.config.model}:
            raise ValueError("vLLM評価には設定した単一modelの指定が必要です")
        result = self._request_json("/models")
        models = result.get("data")
        if not isinstance(models, list) or not any(
            isinstance(model, dict) and model.get("id") == self.config.model
            for model in models
        ):
            raise ValueError("設定したvLLM modelが/modelsにありません")
        return {self.config.model: self.model_metadata()}

    def assess(
        self,
        *,
        model: str,
        model_digest: str,
        prompt: str,
        candidates: Sequence[FrameCandidate],
        contact_sheet: Path,
    ) -> list[FrameAssessment]:
        """contact sheetの表示IDを厳密に検証し、安定した候補IDへ戻す."""
        if (
            model != self.config.model
            or model_digest != self.model_metadata()["digest"]
        ):
            raise ValueError("vLLM評価のmodelまたはcache identityが設定と一致しません")
        display_ids = batch_display_frame_ids(len(candidates))
        encoded = base64.b64encode(contact_sheet.read_bytes()).decode("ascii")
        response = self.complete_json(
            prompt=prompt,
            media={
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            },
            schema=assessment_response_schema(display_ids),
            schema_name="frame_assessment",
        )
        frames = response.get("frames")
        if set(response) != {"frames"} or not isinstance(frames, list):
            raise ValueError("vLLM評価応答にはframes配列が必要です")
        assessments: dict[str, FrameAssessment] = {}
        for item in frames:
            assessment = self._assessment_from_item(item)
            if assessment.frame_id in assessments:
                raise ValueError("vLLM評価応答の表示IDが重複しています")
            assessments[assessment.frame_id] = assessment
        if set(assessments) != set(display_ids):
            raise ValueError("vLLM評価応答の表示IDが候補と一致しません")
        return [
            replace(assessments[display_id], frame_id=candidate.frame_id)
            for display_id, candidate in zip(display_ids, candidates, strict=True)
        ]

    @staticmethod
    def _assessment_from_item(item: object) -> FrameAssessment:
        """JSON schemaの制約を応答受信後にも検証する."""
        if not isinstance(item, dict) or set(item) != {
            "id",
            "blog_score",
            "transition",
            "scene",
            "reason",
        }:
            raise ValueError("vLLM frame評価のfieldが不正です")
        frame_id = item["id"]
        score = item["blog_score"]
        transition = item["transition"]
        scene = item["scene"]
        reason = item["reason"]
        if (
            not isinstance(frame_id, str)
            or not frame_id
            or not isinstance(score, int | float)
            or isinstance(score, bool)
            or not math.isfinite(score)
            or not 0 <= score <= 100
            or not isinstance(transition, bool)
            or not isinstance(scene, str)
            or not isinstance(reason, str)
        ):
            raise ValueError("vLLM frame評価のfield型またはscore範囲が不正です")
        return FrameAssessment(
            frame_id=frame_id,
            blog_score=float(score),
            is_transition=transition,
            scene=scene.strip()[:80] or "その他",
            reason=reason.strip()[:300],
        )

    def complete_json(
        self,
        *,
        prompt: str,
        media: dict[str, Any],
        schema: dict[str, Any],
        schema_name: str,
        media_io_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """単一mediaの理解結果をJSON objectで返す."""
        payload: dict[str, Any] = {
            "model": self.config.model,
            "stream": False,
            "temperature": 0,
            "seed": 271,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, media],
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if media_io_kwargs is not None:
            payload["media_io_kwargs"] = media_io_kwargs
        result = self._request_json("/chat/completions", payload=payload)
        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("vLLM応答には単一のchoiceが必要です")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
            raise ValueError("vLLM応答が正常終了していません")
        message = choice.get("message")
        if (
            not isinstance(message, dict)
            or message.get("role") != "assistant"
            or message.get("refusal")
            or message.get("tool_calls")
        ):
            raise ValueError("vLLM応答messageが不正または拒否されました")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("vLLM応答本文が空です")
        try:
            parsed: Any = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("vLLM応答本文が有効なJSONではありません") from None
        if not isinstance(parsed, dict):
            raise ValueError("vLLM応答本文にはJSON objectが必要です")
        return parsed

    def _request_json(
        self,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """認証付きHTTP requestを送りJSON objectを読む."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = Request(
            self.host + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                result: Any = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise RuntimeError(
                f"vLLM HTTP requestに失敗しました: {error.code}"
            ) from None
        except (URLError, OSError):
            raise RuntimeError("vLLMへの接続または応答読み取りに失敗しました") from None
        except (json.JSONDecodeError, UnicodeError):
            raise ValueError("vLLM API応答が有効なJSONではありません") from None
        if not isinstance(result, dict):
            raise ValueError("vLLM API応答にはJSON objectが必要です")
        return result
