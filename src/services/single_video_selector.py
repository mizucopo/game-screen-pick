"""単一動画全体からブログ掲載用画像を選ぶproduction pipeline."""

from __future__ import annotations

import logging
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from ..models.video_selection import (
    FrameAssessment,
    FrameCandidate,
    SelectedFrame,
    VideoMetadata,
)
from ..models.video_selection_request import (
    MAXIMUM_OUTPUT_COUNT,
    MINIMUM_SAMPLE_INTERVAL_SECONDS,
    VideoSelectionRequest,
)
from ..utils.contact_sheet import (
    build_contact_sheet,
    context_frame_path,
)
from ..utils.video_selection_files import (
    file_sha256,
    is_valid_image,
    json_digest,
    read_json,
    sampled_file_sha256,
    write_json_atomic,
)
from .ollama_frame_assessor import (
    MODEL_OPTIONS,
    OllamaFrameAssessor,
    OllamaModelValidationError,
)
from .video_frame_extractor import VideoFrameExtractor

logger = logging.getLogger(__name__)

ALGORITHM_VERSION = "single-video-selection-v1"
PROMPT_VERSION = "blog-image-selection-v3"
DEFAULT_MAX_SAMPLE_INTERVAL_SECONDS = 10.0
MINIMUM_ENDPOINT_MARGIN_SECONDS = 0.05
MAXIMUM_RAW_CANDIDATES = 4_000
PRIMARY_CANDIDATE_MULTIPLIER = 12
SECONDARY_CANDIDATE_MULTIPLIER = 3
PRIMARY_BATCH_SIZE = 12
SECONDARY_BATCH_SIZE = 6
CONTEXT_OFFSET_SECONDS = 0.35
MAXIMUM_OUTPUT_DHASH_DISTANCE = 10
MINIMUM_DISTINCT_DHASH_DISTANCE = 5


class SingleVideoSelector:
    """フレーム抽出、Ollama評価、選定、成果物生成を順に実行する."""

    def __init__(
        self,
        request: VideoSelectionRequest,
        *,
        frame_extractor: VideoFrameExtractor | None = None,
        assessor: OllamaFrameAssessor | None = None,
    ) -> None:
        """実行リクエストと差し替え可能な外部境界を受け取る."""
        self.request = request
        self.frame_extractor = frame_extractor or VideoFrameExtractor()
        self._provided_assessor = assessor
        self.assessor: OllamaFrameAssessor | None = None
        self.video = Path()
        self.output_dir = Path()
        self.work_dir = Path()
        self.game_title = ""
        self.metadata = VideoMetadata(0, 0, 0, "", "")
        self.end_margin_seconds = MINIMUM_ENDPOINT_MARGIN_SECONDS
        self.timestamps: tuple[float, ...] = ()
        self.model_metadata: dict[str, dict[str, Any]] = {}
        self._live_validated_models: set[str] = set()
        self.manifest_digest = ""

    def run(self) -> Path:
        """選定を実行し、人間確認用コンタクトシートのパスを返す."""
        self._prepare_run()
        if self._verify_completion():
            contact_sheet = self.output_dir / "selected-contact-sheet.jpg"
            logger.info("完了済み成果物を検証しました: %s", contact_sheet)
            return contact_sheet

        candidates = self._extract_candidates()
        primary_candidates = self._preselect_candidates(candidates)
        primary_assessments = self._assess_candidates(
            model=self.request.primary_model,
            stage="primary",
            candidates=primary_candidates,
        )
        primary_eligible = [
            candidate
            for candidate in primary_candidates
            if not primary_assessments[candidate.frame_id].is_transition
        ]
        secondary_count = min(
            self.request.output_count * SECONDARY_CANDIDATE_MULTIPLIER,
            len(primary_eligible),
        )
        if secondary_count < self.request.output_count:
            raise RuntimeError(
                f"一次評価の有効候補{secondary_count}件が"
                f"選択枚数{self.request.output_count}件を下回りました"
            )
        secondary_candidates = select_diverse_candidates(
            primary_candidates,
            primary_assessments,
            secondary_count,
        )
        write_json_atomic(
            self.work_dir / "secondary-candidates.json",
            [candidate_to_json(candidate) for candidate in secondary_candidates],
        )
        self._extract_context_frames(secondary_candidates)
        secondary_assessments = self._assess_candidates(
            model=self.request.secondary_model,
            stage="secondary",
            candidates=secondary_candidates,
        )
        selected = select_final_frames(
            secondary_candidates,
            primary_assessments,
            secondary_assessments,
            self.request.output_count,
        )
        artifacts = self._write_selected_artifacts(selected)
        self._write_completion(artifacts)
        contact_sheet = self.output_dir / "selected-contact-sheet.jpg"
        logger.info("画像選定が完了しました: %s", contact_sheet)
        return contact_sheet

    def _prepare_run(self) -> None:
        """入力・モデル・manifestを検証して実行状態を確定する."""
        if self.request.output_count <= 0:
            raise ValueError("選択枚数は正の整数で指定してください")
        if self.request.output_count > MAXIMUM_OUTPUT_COUNT:
            raise ValueError(f"選択枚数は{MAXIMUM_OUTPUT_COUNT}以下で指定してください")
        if not 1 <= self.request.ffmpeg_workers <= 4:
            raise ValueError("ffmpeg workersは1から4で指定してください")
        if (
            self.request.sample_interval_seconds is not None
            and self.request.sample_interval_seconds < MINIMUM_SAMPLE_INTERVAL_SECONDS
        ):
            raise ValueError(
                f"sample intervalは{MINIMUM_SAMPLE_INTERVAL_SECONDS}秒以上で"
                "指定してください"
            )

        self.video = Path(self.request.input_video).expanduser().resolve()
        if not self.video.is_file():
            raise FileNotFoundError(f"入力動画が見つかりません: {self.video}")
        self.output_dir = Path(self.request.output_dir).expanduser().resolve()
        self.work_dir = self.output_dir / ".game-screen-pick"
        self._preflight_output_dir()
        self.game_title = (
            self.request.game_title.strip()
            if self.request.game_title and self.request.game_title.strip()
            else infer_game_title(self.video)
        )
        self.metadata = self.frame_extractor.probe(self.video)
        self.end_margin_seconds = max(
            MINIMUM_ENDPOINT_MARGIN_SECONDS,
            frame_interval_seconds(self.metadata.average_frame_rate),
        )
        self.timestamps = make_timestamps(
            self.metadata.duration_seconds,
            self.request.output_count,
            self.request.sample_interval_seconds,
            minimum_end_margin_seconds=self.end_margin_seconds,
        )
        if len(self.timestamps) < self.request.output_count:
            raise ValueError(
                f"抽出可能な候補{len(self.timestamps)}件が"
                f"選択枚数{self.request.output_count}件を下回ります"
            )
        has_existing_manifest = self._restore_existing_manifest()
        if has_existing_manifest and (self.work_dir / "completion.json").is_file():
            return

        host = self.request.ollama_host or os.environ.get(
            "OLLAMA_HOST", "127.0.0.1:11434"
        )
        self.assessor = self._provided_assessor or OllamaFrameAssessor(
            host,
            timeout_seconds=self.request.ollama_timeout,
            require_gpu=not self.request.allow_cpu,
        )
        if has_existing_manifest:
            return
        self.model_metadata = self.assessor.fetch_model_metadata(
            {self.request.primary_model, self.request.secondary_model}
        )
        self._live_validated_models.update(self.model_metadata)
        manifest = self._build_manifest()
        self.manifest_digest = json_digest(manifest)
        manifest["manifest_digest"] = self.manifest_digest
        self._prepare_output_dir(manifest)

    def _preflight_output_dir(self) -> None:
        """再開不能なoutputを外部処理より前に拒否する."""
        if not self.output_dir.exists():
            return
        if not self.output_dir.is_dir():
            raise RuntimeError(f"出力先がフォルダではありません: {self.output_dir}")
        manifest_path = self.work_dir / "run-manifest.json"
        if any(self.output_dir.iterdir()) and not manifest_path.is_file():
            raise RuntimeError(
                f"出力フォルダが空ではなく、再開manifestもありません: {self.output_dir}"
            )

    def _restore_existing_manifest(self) -> bool:
        """保存済みmanifestを外部接続なしで検証しmodel情報を復元する."""
        manifest_path = self.work_dir / "run-manifest.json"
        if not manifest_path.is_file():
            return False

        existing = read_json(manifest_path)
        if not isinstance(existing, dict):
            raise RuntimeError("再開manifestが不正です")
        raw_models = existing.get("models")
        if not isinstance(raw_models, dict):
            raise RuntimeError("再開manifestのmodelsが不正です")

        model_metadata: dict[str, dict[str, Any]] = {}
        requested_by_stage = {
            "primary": self.request.primary_model,
            "secondary": self.request.secondary_model,
        }
        for stage, requested_model in requested_by_stage.items():
            raw_model = raw_models.get(stage)
            if (
                not isinstance(raw_model, dict)
                or raw_model.get("name") != requested_model
            ):
                raise RuntimeError(
                    "既存の実行条件が今回と異なります。"
                    "新しい出力フォルダを指定してください"
                )
            metadata = {key: value for key, value in raw_model.items() if key != "name"}
            previous = model_metadata.get(requested_model)
            if previous is not None and previous != metadata:
                raise RuntimeError("再開manifestのmodel metadataが不正です")
            model_metadata[requested_model] = metadata

        self.model_metadata = model_metadata
        manifest = self._build_manifest()
        self.manifest_digest = json_digest(manifest)
        manifest["manifest_digest"] = self.manifest_digest
        if existing != manifest:
            raise RuntimeError(
                "既存の実行条件が今回と異なります。新しい出力フォルダを指定してください"
            )
        return True

    def _validate_live_model_metadata(self, model: str) -> None:
        """未評価batchの実行前に保存済みmodelとの同一性を確認する."""
        if model in self._live_validated_models:
            return
        if self.assessor is None:
            raise RuntimeError("Ollama assessorが初期化されていません")
        live_metadata = self.assessor.fetch_model_metadata({model})
        if live_metadata.get(model) != self.model_metadata.get(model):
            raise OllamaModelValidationError(
                f"Ollama model metadataが再開manifestと一致しません: {model}"
            )
        self._live_validated_models.add(model)

    def _build_manifest(self) -> dict[str, Any]:
        """結果に影響する入力だけを含む再開manifestを作る."""
        stat = self.video.stat()
        return {
            "algorithm_version": ALGORITHM_VERSION,
            "prompt_version": PROMPT_VERSION,
            "input": {
                "path": str(self.video),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sampled_sha256": sampled_file_sha256(self.video),
                "duration_seconds": self.metadata.duration_seconds,
                "width": self.metadata.width,
                "height": self.metadata.height,
                "codec_name": self.metadata.codec_name,
                "average_frame_rate": self.metadata.average_frame_rate,
                "video_stream_index": self.metadata.video_stream_index,
            },
            "game_title": self.game_title,
            "game_context": self.request.game_context.strip(),
            "output_count": self.request.output_count,
            "timestamps": list(self.timestamps),
            "models": {
                "primary": {
                    "name": self.request.primary_model,
                    **self.model_metadata[self.request.primary_model],
                },
                "secondary": {
                    "name": self.request.secondary_model,
                    **self.model_metadata[self.request.secondary_model],
                },
            },
            "candidate_multipliers": {
                "primary": PRIMARY_CANDIDATE_MULTIPLIER,
                "secondary": SECONDARY_CANDIDATE_MULTIPLIER,
            },
            "batch_sizes": {
                "primary": PRIMARY_BATCH_SIZE,
                "secondary": SECONDARY_BATCH_SIZE,
            },
            "context_offset_seconds": CONTEXT_OFFSET_SECONDS,
            "end_margin_seconds": self.end_margin_seconds,
            "model_options": MODEL_OPTIONS,
            "require_gpu": not self.request.allow_cpu,
        }

    def _prepare_output_dir(self, manifest: dict[str, Any]) -> None:
        """新規outputまたは同一manifestの再開先だけを受け入れる."""
        manifest_path = self.work_dir / "run-manifest.json"
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            if not manifest_path.is_file():
                raise RuntimeError(
                    "出力フォルダが空ではなく、再開manifestもありません: "
                    f"{self.output_dir}"
                )
            existing = read_json(manifest_path)
            if existing != manifest:
                raise RuntimeError(
                    "既存の実行条件が今回と異なります。新しい出力フォルダを指定してください"
                )
            return
        self.work_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(manifest_path, manifest)

    def _verify_completion(self) -> bool:
        """完了記録と全成果物のhashが一致するか検証する."""
        completion_path = self.work_dir / "completion.json"
        if not completion_path.is_file():
            return False
        payload = read_json(completion_path)
        if not isinstance(payload, dict):
            raise RuntimeError("完了記録が不正です")
        if payload.get("manifest_digest") != self.manifest_digest:
            raise RuntimeError("完了記録のmanifestが今回の実行と一致しません")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list):
            raise RuntimeError("完了記録のartifactsが不正です")
        output_root = self.output_dir.resolve()
        for item in artifacts:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise RuntimeError("完了記録のartifactが不正です")
            artifact = (self.output_dir / item["path"]).resolve()
            try:
                artifact.relative_to(output_root)
            except ValueError as error:
                message = "成果物パスが出力フォルダ外を指しています"
                raise RuntimeError(message) from error
            if (
                not artifact.is_file()
                or artifact.stat().st_size != item.get("size")
                or file_sha256(artifact) != item.get("sha256")
            ):
                raise RuntimeError(f"完了済み成果物が変更されています: {artifact}")
        return True

    def _extract_candidates(self) -> list[FrameCandidate]:
        """動画全体の等間隔位置から縮小候補フレームを抽出する."""
        candidate_dir = self.work_dir / "candidate-frames"
        candidates = [
            FrameCandidate(
                frame_id=f"f{index:05d}",
                timestamp_seconds=timestamp,
                path=str(candidate_dir / f"f{index:05d}.jpg"),
            )
            for index, timestamp in enumerate(self.timestamps, start=1)
        ]
        pending = [
            candidate
            for candidate in candidates
            if not is_valid_image(Path(candidate.path))
        ]
        if pending:
            logger.info(
                "候補フレームを抽出します: %d/%d件", len(pending), len(candidates)
            )
        executor = ThreadPoolExecutor(max_workers=self.request.ffmpeg_workers)
        try:
            futures = [
                executor.submit(self._extract_candidate, candidate)
                for candidate in pending
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                future.result()
                if completed % 50 == 0 or completed == len(futures):
                    logger.info("候補フレーム抽出: %d/%d件", completed, len(futures))
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown()
        return candidates

    def _extract_candidate(self, candidate: FrameCandidate) -> None:
        """候補フレームを一枚抽出する."""
        self.frame_extractor.extract_frame(
            self.video,
            candidate.timestamp_seconds,
            Path(candidate.path),
            max_width=960,
            video_stream_index=self.metadata.video_stream_index,
        )

    def _preselect_candidates(
        self,
        candidates: Sequence[FrameCandidate],
    ) -> list[FrameCandidate]:
        """機械的品質と時間分散で一次Ollama評価候補を絞る."""
        logger.info("候補フレームを機械評価します: %d件", len(candidates))
        executor = ThreadPoolExecutor(
            max_workers=min(8, self.request.ffmpeg_workers * 2)
        )
        try:
            measured = list(executor.map(measure_candidate, candidates))
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown()
        usable = [candidate for candidate in measured if candidate is not None]
        if len(usable) < self.request.output_count:
            raise RuntimeError(
                f"有効候補{len(usable)}件が選択枚数"
                f"{self.request.output_count}件を下回りました"
            )

        target = min(
            self.request.output_count * PRIMARY_CANDIDATE_MULTIPLIER,
            len(usable),
        )
        bin_count = min(60, target)
        bins: list[list[FrameCandidate]] = [[] for _ in range(bin_count)]
        span = max(self.metadata.duration_seconds, 0.001)
        for candidate in usable:
            bin_index = min(
                bin_count - 1,
                int(candidate.timestamp_seconds / span * bin_count),
            )
            bins[bin_index].append(candidate)
        for bucket in bins:
            bucket.sort(key=lambda item: (-item.quality_score, item.timestamp_seconds))

        selected: list[FrameCandidate] = []
        selected_ids: set[str] = set()
        selected_by_bin: list[list[FrameCandidate]] = [[] for _ in range(bin_count)]
        while len(selected) < target:
            progressed = False
            for bin_index, bucket in enumerate(bins):
                remaining = [
                    candidate
                    for candidate in bucket
                    if candidate.frame_id not in selected_ids
                ]
                if not remaining:
                    continue
                diverse = next(
                    (
                        candidate
                        for candidate in remaining
                        if all(
                            difference_hash_distance(
                                candidate.difference_hash,
                                chosen.difference_hash,
                            )
                            >= MINIMUM_DISTINCT_DHASH_DISTANCE
                            for chosen in selected_by_bin[bin_index]
                        )
                    ),
                    None,
                )
                chosen = diverse or remaining[0]
                selected.append(chosen)
                selected_ids.add(chosen.frame_id)
                selected_by_bin[bin_index].append(chosen)
                progressed = True
                if len(selected) == target:
                    break
            if not progressed:
                break

        if len(selected) < target:
            remaining = sorted(
                (
                    candidate
                    for candidate in usable
                    if candidate.frame_id not in selected_ids
                ),
                key=lambda item: (-item.quality_score, item.timestamp_seconds),
            )
            selected.extend(remaining[: target - len(selected)])
        result = sorted(selected[:target], key=lambda item: item.timestamp_seconds)
        write_json_atomic(
            self.work_dir / "primary-candidates.json",
            [candidate_to_json(candidate) for candidate in result],
        )
        logger.info("一次評価候補を絞りました: %d/%d件", len(result), len(usable))
        return result

    def _extract_context_frames(
        self,
        candidates: Sequence[FrameCandidate],
    ) -> None:
        """二次評価候補の直前・直後フレームを抽出する."""
        context_dir = self.work_dir / "context-frames"
        jobs: list[tuple[FrameCandidate, str, float]] = []
        for candidate in candidates:
            before = max(0.05, candidate.timestamp_seconds - CONTEXT_OFFSET_SECONDS)
            after = min(
                self.metadata.duration_seconds - self.end_margin_seconds,
                candidate.timestamp_seconds + CONTEXT_OFFSET_SECONDS,
            )
            for position, timestamp in (("before", before), ("after", after)):
                if not is_valid_image(
                    context_frame_path(context_dir, candidate, position)
                ):
                    jobs.append((candidate, position, timestamp))
        if jobs:
            logger.info("遷移判定用フレームを抽出します: %d件", len(jobs))
        executor = ThreadPoolExecutor(max_workers=self.request.ffmpeg_workers)
        try:
            futures = [
                executor.submit(
                    self.frame_extractor.extract_frame,
                    self.video,
                    timestamp,
                    context_frame_path(context_dir, candidate, position),
                    max_width=960,
                    video_stream_index=self.metadata.video_stream_index,
                )
                for candidate, position, timestamp in jobs
            ]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown()

    def _assess_candidates(
        self,
        *,
        model: str,
        stage: str,
        candidates: Sequence[FrameCandidate],
    ) -> dict[str, FrameAssessment]:
        """候補をbatch単位で評価し、完了batchを再開時に再利用する."""
        if self.assessor is None:
            raise RuntimeError("Ollama assessorが初期化されていません")
        cache_key = self._assessment_cache_key(model, stage, candidates)
        state_path = self.work_dir / f"assessments-{stage}.json"
        state = load_assessment_state(state_path, cache_key)
        batch_size = PRIMARY_BATCH_SIZE if stage == "primary" else SECONDARY_BATCH_SIZE
        batches = [
            candidates[index : index + batch_size]
            for index in range(0, len(candidates), batch_size)
        ]
        context_dir = self.work_dir / "context-frames" if stage == "secondary" else None
        for batch_index, batch in enumerate(batches, start=1):
            batch_ids = {candidate.frame_id for candidate in batch}
            if batch_ids.issubset(state):
                continue
            self._validate_live_model_metadata(model)
            sheet_path = (
                self.work_dir
                / "contact-sheets"
                / stage
                / (
                    f"batch-{batch_index:04d}-"
                    f"{batch[0].frame_id}-{batch[-1].frame_id}.jpg"
                )
            )
            build_contact_sheet(batch, sheet_path, context_dir=context_dir)
            last_error: Exception | None = None
            for attempt in range(1, 4):
                started = time.monotonic()
                try:
                    assessments = self.assessor.assess(
                        model=str(self.model_metadata[model]["resolved_name"]),
                        model_digest=str(self.model_metadata[model]["digest"]),
                        prompt=self._model_prompt(stage, batch),
                        candidates=batch,
                        contact_sheet=sheet_path,
                    )
                    state.update(
                        {assessment.frame_id: assessment for assessment in assessments}
                    )
                    save_assessment_state(state_path, cache_key, state)
                    self._write_gpu_evidence()
                    logger.info(
                        "%s評価: %d/%d batch (%.1f秒)",
                        stage,
                        batch_index,
                        len(batches),
                        time.monotonic() - started,
                    )
                    last_error = None
                    break
                except OllamaModelValidationError:
                    raise
                except Exception as error:
                    last_error = error
                    logger.warning(
                        "%s評価batch %dの試行%dが失敗しました: %s",
                        stage,
                        batch_index,
                        attempt,
                        error,
                    )
                    if attempt < 3:
                        time.sleep(2 ** (attempt - 1))
            if last_error is not None:
                raise last_error
        return {
            candidate.frame_id: state[candidate.frame_id] for candidate in candidates
        }

    def _assessment_cache_key(
        self,
        model: str,
        stage: str,
        candidates: Sequence[FrameCandidate],
    ) -> str:
        """評価入力とmodelを含むcache keyを返す."""
        context_dir = self.work_dir / "context-frames"
        return json_digest(
            {
                "manifest_digest": self.manifest_digest,
                "prompt_version": PROMPT_VERSION,
                "stage": stage,
                "model": model,
                "model_digest": self.model_metadata[model]["digest"],
                "candidates": [
                    {
                        "id": candidate.frame_id,
                        "timestamp_seconds": candidate.timestamp_seconds,
                        "difference_hash": f"{candidate.difference_hash:016x}",
                        "image_sha256": file_sha256(Path(candidate.path)),
                        **(
                            {
                                "before_sha256": file_sha256(
                                    context_frame_path(context_dir, candidate, "before")
                                ),
                                "after_sha256": file_sha256(
                                    context_frame_path(context_dir, candidate, "after")
                                ),
                            }
                            if stage == "secondary"
                            else {}
                        ),
                    }
                    for candidate in candidates
                ],
                "batch_size": (
                    PRIMARY_BATCH_SIZE if stage == "primary" else SECONDARY_BATCH_SIZE
                ),
                "context_offset_seconds": (
                    0 if stage == "primary" else CONTEXT_OFFSET_SECONDS
                ),
                "model_options": MODEL_OPTIONS,
            }
        )

    def _model_prompt(
        self,
        stage: str,
        candidates: Sequence[FrameCandidate],
    ) -> str:
        """検証済みの汎用選定方針をOllama向けpromptにする."""
        ids = ", ".join(candidate.frame_id for candidate in candidates)
        if stage == "primary":
            stage_note = "動画全体を時間分散と機械的品質で絞った一次候補です。"
            context_note = ""
        else:
            stage_note = "一次評価上位から最終候補へ絞る厳しい再評価です。"
            context_note = (
                "各IDは左から直前・選定対象・直後の3コマです。"
                "中央だけを採点し、前後を画面遷移の判定に使ってください。"
            )
        game_context = (
            f"\nゲーム補足: {self.request.game_context.strip()}"
            if self.request.game_context.strip()
            else ""
        )
        duration_label = format_duration(self.metadata.duration_seconds)
        return f"""ゲーム『{self.game_title}』の{duration_label}の全編録画から、
ブログへ実際に掲載する画像を{self.request.output_count}枚選びます。{stage_note}
{context_note}{game_context}
contact sheet内の対象ID: {ids}

各画像について次を判定してください。
- blog_score: ブログ掲載価値を0から100で厳しく評価
- transition: 暗転、白飛び、ロード中、フェード、画面遷移途中ならtrue
- scene: ジャンルを限定しない短い日本語の場面名。同種には同じ語を使う
- reason: 判断理由を短い日本語で記載

鮮明さ、構図、場面の分かりやすさ、UIや字幕の状態、物語やゲーム性の
説明価値、最終画像全体のバラエティへの貢献を重視してください。
そのゲームでプレイヤーが普段繰り返し見る通常進行画面（移動、探索、
戦闘、会話、推理、パズルなど）が、イベントムービー、タイトル、メニュー、
結果画面などの特別画面より最終画像に少し多く残るよう採点してください。
ただし通常進行画面だけに偏らせず、物語上重要な特別場面やジャンル固有の
画面も残してください。タイトル固有のルールは設けないでください。
ゲームジャンルに存在しない場面を無理に仮定しないでください。
全IDを1回ずつ含め、JSON以外は返さないでください。
形式: {{"frames":[{{"id":"f00001","blog_score":80,"transition":false,
"scene":"探索","reason":"人物とフィールドが明瞭"}}]}}"""

    def _write_gpu_evidence(self) -> None:
        """実行中に確認したGPU利用証跡を保存する."""
        if self.assessor is not None and self.assessor.gpu_evidence:
            write_json_atomic(
                self.work_dir / "gpu-evidence.json",
                dict(sorted(self.assessor.gpu_evidence.items())),
            )

    def _write_selected_artifacts(
        self,
        selected: Sequence[SelectedFrame],
    ) -> list[Path]:
        """full resolution画像、report、一覧sheetを出力する."""
        width = max(2, len(str(self.request.output_count)))
        report_items: list[dict[str, Any]] = []
        contact_candidates: list[FrameCandidate] = []
        selected_paths: list[Path] = []
        for rank, selected_frame in enumerate(selected, start=1):
            output_path = self.output_dir / f"selected-{rank:0{width}d}.jpg"
            self.frame_extractor.extract_frame(
                self.video,
                selected_frame.candidate.timestamp_seconds,
                output_path,
                max_width=None,
                video_stream_index=self.metadata.video_stream_index,
            )
            with Image.open(output_path) as image:
                output_hash = image_difference_hash(image)
            hash_distance = difference_hash_distance(
                selected_frame.candidate.difference_hash,
                output_hash,
            )
            if hash_distance > MAXIMUM_OUTPUT_DHASH_DISTANCE:
                raise RuntimeError(
                    "出力画像が評価候補と一致しません: "
                    f"{selected_frame.candidate.frame_id}, distance={hash_distance}"
                )
            selected_paths.append(output_path)
            report_items.append(
                {
                    "rank": rank,
                    "output_path": output_path.name,
                    "frame_id": selected_frame.candidate.frame_id,
                    "timestamp_seconds": round(
                        selected_frame.candidate.timestamp_seconds, 6
                    ),
                    "aggregate_score": round(selected_frame.aggregate_score, 2),
                    "candidate_output_dhash_distance": hash_distance,
                    "primary": asdict(selected_frame.primary_assessment),
                    "secondary": asdict(selected_frame.secondary_assessment),
                }
            )
            contact_candidates.append(
                FrameCandidate(
                    frame_id=f"{rank:0{width}d}",
                    timestamp_seconds=selected_frame.candidate.timestamp_seconds,
                    path=str(output_path),
                )
            )

        report_path = self.output_dir / "report.json"
        write_json_atomic(
            report_path,
            {
                "manifest_digest": self.manifest_digest,
                "video": str(self.video),
                "game_title": self.game_title,
                "game_context": self.request.game_context.strip(),
                "output_count": self.request.output_count,
                "sample_count": len(self.timestamps),
                "models": {
                    "primary": {
                        "name": self.request.primary_model,
                        "resolved_name": self.model_metadata[
                            self.request.primary_model
                        ]["resolved_name"],
                        "digest": self.model_metadata[self.request.primary_model][
                            "digest"
                        ],
                    },
                    "secondary": {
                        "name": self.request.secondary_model,
                        "resolved_name": self.model_metadata[
                            self.request.secondary_model
                        ]["resolved_name"],
                        "digest": self.model_metadata[self.request.secondary_model][
                            "digest"
                        ],
                    },
                },
                "selected": report_items,
            },
        )
        contact_sheet_path = self.output_dir / "selected-contact-sheet.jpg"
        build_contact_sheet(contact_candidates, contact_sheet_path)
        return [report_path, contact_sheet_path, *selected_paths]

    def _write_completion(self, artifacts: Sequence[Path]) -> None:
        """人が使う全成果物のhashを完了記録へ保存する."""
        write_json_atomic(
            self.work_dir / "completion.json",
            {
                "manifest_digest": self.manifest_digest,
                "artifacts": [self._artifact_record(path) for path in artifacts],
            },
        )

    def _artifact_record(self, path: Path) -> dict[str, Any]:
        """出力root相対の成果物記録を返す."""
        relative = path.resolve().relative_to(self.output_dir.resolve())
        return {
            "path": str(relative),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }


def infer_game_title(video: Path) -> str:
    """一般的なPart/連番suffixを除き、動画名からゲーム名を推測する."""
    title = video.stem.strip()
    patterns = (
        r"\s+part\s*\d+.*$",
        r"\s+#\d+.*$",
        r"\s+パート\s*\d+.*$",
    )
    for pattern in patterns:
        shortened = re.sub(pattern, "", title, flags=re.IGNORECASE).strip()
        if shortened != title:
            return shortened or title
    return title


def make_timestamps(
    duration_seconds: float,
    output_count: int,
    requested_interval_seconds: float | None,
    *,
    minimum_end_margin_seconds: float = MINIMUM_ENDPOINT_MARGIN_SECONDS,
) -> tuple[float, ...]:
    """動画のほぼ先頭から末尾までを等間隔で覆う時刻列を返す."""
    if requested_interval_seconds is not None:
        if (
            not math.isfinite(requested_interval_seconds)
            or requested_interval_seconds <= 0
        ):
            raise ValueError("sample intervalは正の数で指定してください")
        if requested_interval_seconds < MINIMUM_SAMPLE_INTERVAL_SECONDS:
            raise ValueError(
                f"sample intervalは{MINIMUM_SAMPLE_INTERVAL_SECONDS}秒以上で"
                "指定してください"
            )
    if not math.isfinite(minimum_end_margin_seconds) or minimum_end_margin_seconds < 0:
        raise ValueError("end marginは0以上の有限値で指定してください")
    minimum_start_margin = MINIMUM_ENDPOINT_MARGIN_SECONDS
    minimum_end_margin = max(
        MINIMUM_ENDPOINT_MARGIN_SECONDS,
        minimum_end_margin_seconds,
    )
    default_start_margin = min(0.5, duration_seconds / 4.0)
    default_end_margin = max(default_start_margin, minimum_end_margin)
    required_output_span = max(0, output_count - 1) * MINIMUM_SAMPLE_INTERVAL_SECONDS
    available_margin = max(0.0, duration_seconds - required_output_span)
    minimum_total_margin = minimum_start_margin + minimum_end_margin
    default_total_margin = default_start_margin + default_end_margin
    target_total_margin = max(
        minimum_total_margin,
        min(default_total_margin, available_margin),
    )
    if minimum_total_margin >= duration_seconds:
        start = 0.0
        end_margin = min(duration_seconds, minimum_end_margin)
    else:
        start = min(default_start_margin, target_total_margin / 2.0)
        end_margin = target_total_margin - start
        if start < minimum_start_margin:
            start = minimum_start_margin
            end_margin = target_total_margin - start
        if end_margin < minimum_end_margin:
            end_margin = minimum_end_margin
            start = target_total_margin - end_margin
    end = max(start, duration_seconds - end_margin)
    span = end - start
    if requested_interval_seconds is not None:
        interval = requested_interval_seconds
        sample_count = max(
            math.ceil(span / interval) + 1,
            output_count,
        )
        if sample_count > MAXIMUM_RAW_CANDIDATES:
            raise ValueError("指定したsample intervalでは候補数が上限4,000件を超えます")
    else:
        base_count = (
            math.ceil(duration_seconds / DEFAULT_MAX_SAMPLE_INTERVAL_SECONDS) + 1
        )
        desired_count = max(
            output_count * PRIMARY_CANDIDATE_MULTIPLIER * 3,
            120,
        )
        sample_count = max(base_count, desired_count)
    maximum_by_interval = (
        math.floor((span + 1e-9) / MINIMUM_SAMPLE_INTERVAL_SECONDS) + 1
    )
    sample_count = min(
        sample_count,
        MAXIMUM_RAW_CANDIDATES,
        maximum_by_interval,
    )
    sample_count = max(1, sample_count)
    if sample_count == 1:
        return (round((start + end) / 2.0, 6),)
    timestamps = np.linspace(start, end, sample_count, dtype=np.float64)
    return tuple(round(float(timestamp), 6) for timestamp in timestamps)


def frame_interval_seconds(average_frame_rate: str) -> float:
    """ffprobeの平均frame rateから1frame分の秒数を返す."""
    try:
        if "/" in average_frame_rate:
            numerator_text, denominator_text = average_frame_rate.split("/", maxsplit=1)
            frames_per_second = float(numerator_text) / float(denominator_text)
        else:
            frames_per_second = float(average_frame_rate)
    except (ValueError, ZeroDivisionError):
        return 0.0
    if not math.isfinite(frames_per_second) or frames_per_second <= 0:
        return 0.0
    return 1.0 / frames_per_second


def measure_candidate(candidate: FrameCandidate) -> FrameCandidate | None:
    """単色・暗転を除き、機械的品質とdifference hashを付与する."""
    image = cv2.imread(candidate.path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    entropy = image_entropy(gray)
    if (
        (brightness < 7 and contrast < 4)
        or (brightness > 248 and contrast < 4)
        or (contrast < 2 and sharpness < 1)
    ):
        return None

    exposure_score = max(0.0, 1.0 - abs(brightness - 125.0) / 125.0)
    contrast_score = min(1.0, contrast / 70.0)
    sharpness_score = min(1.0, math.log1p(sharpness) / math.log1p(800.0))
    entropy_score = min(1.0, entropy / 7.5)
    quality_score = 100.0 * (
        exposure_score * 0.10
        + contrast_score * 0.25
        + sharpness_score * 0.35
        + entropy_score * 0.30
    )
    with Image.open(candidate.path) as source:
        difference_hash = image_difference_hash(source)
    return FrameCandidate(
        frame_id=candidate.frame_id,
        timestamp_seconds=candidate.timestamp_seconds,
        path=candidate.path,
        quality_score=quality_score,
        difference_hash=difference_hash,
    )


def image_difference_hash(image: Image.Image) -> int:
    """64-bit difference hashを返す."""
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    result = 0
    for bit in bits.flatten():
        result = (result << 1) | int(bit)
    return result


def image_entropy(gray: np.ndarray[Any, Any]) -> float:
    """8-bit grayscale画像のShannon entropyを返す."""
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(histogram.sum())
    if total <= 0:
        return 0.0
    probabilities = histogram[histogram > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def difference_hash_distance(left: int, right: int) -> int:
    """二つの64-bit hashのHamming distanceを返す."""
    return (left ^ right).bit_count()


def candidate_to_json(candidate: FrameCandidate) -> dict[str, Any]:
    """difference hashをhex化したJSON recordへ変換する."""
    payload = asdict(candidate)
    payload["difference_hash"] = f"{candidate.difference_hash:016x}"
    return payload


def select_diverse_candidates(
    candidates: Sequence[FrameCandidate],
    assessments: dict[str, FrameAssessment],
    count: int,
) -> list[FrameCandidate]:
    """一次評価から場面・見た目・時刻が分散した二次候補を選ぶ."""
    eligible = [
        candidate
        for candidate in candidates
        if not assessments[candidate.frame_id].is_transition
    ]
    if len(eligible) < count:
        raise RuntimeError(f"非遷移候補{len(eligible)}件が要求{count}件未満です")
    selected: list[FrameCandidate] = []
    remaining = list(eligible)
    scene_counts: Counter[str] = Counter()
    timeline_span = max(
        1.0,
        max(item.timestamp_seconds for item in eligible)
        - min(item.timestamp_seconds for item in eligible),
    )
    time_scale = max(60.0, timeline_span / max(1, count))
    while len(selected) < count:

        def utility(candidate: FrameCandidate) -> tuple[float, float, float]:
            assessment = assessments[candidate.frame_id]
            visual_distance, time_distance = _nearest_distances(
                candidate, selected, time_scale
            )
            scene_key = normalize_scene(assessment)
            near_duplicate_penalty = (
                30.0
                if visual_distance < MINIMUM_DISTINCT_DHASH_DISTANCE
                and time_distance < 60
                else 0.0
            )
            total = (
                assessment.blog_score
                + 14.0 / (scene_counts[scene_key] + 1)
                + min(32, visual_distance) * 0.60
                + min(time_scale, time_distance) / time_scale * 8.0
                - near_duplicate_penalty
            )
            return total, assessment.blog_score, -candidate.timestamp_seconds

        chosen = max(remaining, key=utility)
        selected.append(chosen)
        scene_counts[normalize_scene(assessments[chosen.frame_id])] += 1
        remaining.remove(chosen)
    return selected


def select_final_frames(
    candidates: Sequence[FrameCandidate],
    primary: dict[str, FrameAssessment],
    secondary: dict[str, FrameAssessment],
    count: int,
) -> list[SelectedFrame]:
    """遷移を除外し、重複・特殊画面の偏りを抑えて最終選定する."""
    eligible = [
        candidate
        for candidate in candidates
        if not primary[candidate.frame_id].is_transition
        and not secondary[candidate.frame_id].is_transition
    ]
    if len(eligible) < count:
        raise RuntimeError(f"非遷移候補{len(eligible)}件が要求{count}件未満です")

    def aggregate_score(candidate: FrameCandidate) -> float:
        return (
            primary[candidate.frame_id].blog_score * 0.4
            + secondary[candidate.frame_id].blog_score * 0.6
        )

    soft_caps = {
        "title": 1,
        "map": max(1, math.ceil(count * 0.10)),
        "menu": max(2, math.ceil(count * 0.15)),
    }
    selected: list[FrameCandidate] = []
    remaining = list(eligible)
    scene_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    timeline_span = max(
        1.0,
        max(item.timestamp_seconds for item in eligible)
        - min(item.timestamp_seconds for item in eligible),
    )
    time_scale = max(60.0, timeline_span / max(1, count))

    while len(selected) < count:
        preferred = [
            candidate
            for candidate in remaining
            if screen_family(secondary[candidate.frame_id]) not in soft_caps
            or family_counts[screen_family(secondary[candidate.frame_id])]
            < soft_caps[screen_family(secondary[candidate.frame_id])]
        ]
        pool = preferred or remaining
        visually_distinct = [
            candidate
            for candidate in pool
            if all(
                difference_hash_distance(
                    candidate.difference_hash,
                    chosen.difference_hash,
                )
                >= MINIMUM_DISTINCT_DHASH_DISTANCE
                for chosen in selected
            )
        ]
        pool = visually_distinct or pool

        def utility(candidate: FrameCandidate) -> tuple[float, float, float]:
            assessment = secondary[candidate.frame_id]
            visual_distance, time_distance = _nearest_distances(
                candidate, selected, time_scale
            )
            scene_key = normalize_scene(assessment)
            family = screen_family(assessment)
            family_penalty = family_counts[family] * 1.5 if family != "content" else 0.0
            near_duplicate_penalty = (
                35.0
                if visual_distance < MINIMUM_DISTINCT_DHASH_DISTANCE
                and time_distance < 60
                else 0.0
            )
            total = (
                aggregate_score(candidate)
                + 14.0 / (scene_counts[scene_key] + 1)
                + min(32, visual_distance) * 0.65
                + min(time_scale, time_distance) / time_scale * 9.0
                - family_penalty
                - near_duplicate_penalty
            )
            return total, aggregate_score(candidate), -candidate.timestamp_seconds

        chosen = max(pool, key=utility)
        selected.append(chosen)
        assessment = secondary[chosen.frame_id]
        scene_counts[normalize_scene(assessment)] += 1
        family_counts[screen_family(assessment)] += 1
        remaining.remove(chosen)

    result = [
        SelectedFrame(
            candidate=candidate,
            aggregate_score=aggregate_score(candidate),
            primary_assessment=primary[candidate.frame_id],
            secondary_assessment=secondary[candidate.frame_id],
        )
        for candidate in selected
    ]
    return sorted(
        result,
        key=lambda item: (-item.aggregate_score, item.candidate.timestamp_seconds),
    )


def _nearest_distances(
    candidate: FrameCandidate,
    selected: Sequence[FrameCandidate],
    default_time_distance: float,
) -> tuple[int, float]:
    """既選択候補への最小visual/time距離を返す."""
    if not selected:
        return 64, default_time_distance
    return (
        min(
            difference_hash_distance(
                candidate.difference_hash,
                chosen.difference_hash,
            )
            for chosen in selected
        ),
        min(
            abs(candidate.timestamp_seconds - chosen.timestamp_seconds)
            for chosen in selected
        ),
    )


def normalize_scene(assessment: FrameAssessment) -> str:
    """scene名を集計用の安定keyへ変換する."""
    normalized = "".join(
        character for character in assessment.scene.casefold() if character.isalnum()
    )
    return normalized[:48] or "その他"


def screen_family(assessment: FrameAssessment) -> str:
    """過剰選択を抑える特殊画面familyを返す."""
    scene = assessment.scene.casefold().strip()
    exact_families = {
        "title": {"タイトル", "オープニング", "opening", "title"},
        "map": {"マップ", "地図", "map", "world map"},
        "menu": {"メニュー", "インベントリ", "menu", "inventory"},
    }
    for family, names in exact_families.items():
        if scene in names:
            return family
    phrase_families = {
        "title": ("タイトル画面", "オープニングタイトル", "title screen"),
        "map": ("世界地図", "ワールドマップ", "マップ画面", "map screen"),
        "menu": (
            "メニュー画面",
            "装備画面",
            "設定画面",
            "ステータス画面",
            "settings screen",
        ),
    }
    for family, phrases in phrase_families.items():
        if any(phrase in scene for phrase in phrases):
            return family
    return "content"


def load_assessment_state(
    path: Path,
    expected_cache_key: str,
) -> dict[str, FrameAssessment]:
    """同じcache keyの評価済み状態を読む."""
    if not path.is_file():
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("cache_key") != expected_cache_key:
        raise RuntimeError(f"評価cacheが今回の実行と一致しません: {path}")
    raw_assessments = payload.get("assessments")
    if not isinstance(raw_assessments, dict):
        raise RuntimeError(f"評価cacheが不正です: {path}")
    state: dict[str, FrameAssessment] = {}
    for frame_id, item in raw_assessments.items():
        if not isinstance(frame_id, str) or not isinstance(item, dict):
            raise RuntimeError(f"評価cacheが不正です: {path}")
        score = item.get("blog_score")
        transition = item.get("is_transition")
        scene = item.get("scene")
        reason = item.get("reason")
        if (
            not isinstance(score, int | float)
            or isinstance(score, bool)
            or not isinstance(transition, bool)
            or not isinstance(scene, str)
            or not isinstance(reason, str)
        ):
            raise RuntimeError(f"評価cacheが不正です: {path}")
        state[frame_id] = FrameAssessment(
            frame_id=frame_id,
            blog_score=float(score),
            is_transition=transition,
            scene=scene,
            reason=reason,
        )
    return state


def save_assessment_state(
    path: Path,
    cache_key: str,
    state: dict[str, FrameAssessment],
) -> None:
    """評価済み状態をbatchごとにatomic保存する."""
    write_json_atomic(
        path,
        {
            "cache_key": cache_key,
            "assessments": {
                frame_id: asdict(assessment)
                for frame_id, assessment in sorted(state.items())
            },
        },
    )


def format_duration(duration_seconds: float) -> str:
    """動画時間をprompt用の日本語へ整形する."""
    total_minutes = round(duration_seconds / 60)
    hours, minutes = divmod(total_minutes, 60)
    if hours:
        return f"約{hours}時間{minutes}分"
    return f"約{minutes}分"
