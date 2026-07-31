"""Canonical report provenance構築のtest。"""

from pathlib import Path

from src.video_selection.models.completed_stage import CompletedStage
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.progress_event import ProgressEvent
from src.video_selection.models.stage_fingerprint import StageFingerprint
from src.video_selection.models.vision_inference_diagnostics import (
    VisionInferenceDiagnostics,
)
from src.video_selection.services.build_report_provenance import (
    build_report_provenance,
)
from src.video_selection.services.stage_version import stage_version


def test_catalog_cache_reuse_does_not_increase_inference_attempt_count(
    tmp_path: Path,
) -> None:
    """同じCatalogのbatch間cache reuseが推論試行数へ加算されないこと。

    Arrange:
        - 1回推論されたCatalogと後続batchでの1回のcache reuseが用意される
    Act:
        - Completed Stageからcanonical provenanceが構築される
    Assert:
        - cache hit/missは各1件、推論試行数は診断どおり1回になること
    """
    # Arrange
    fingerprint = StageFingerprint("a" * 64)
    completed = CompletedStage(
        stage=ProcessingStage.BUILD_SCENE_CATALOG,
        fingerprint=fingerprint,
        semantic_input={
            "stage_contract_version": "scene-catalog-v1",
            "model": {
                "runtime_identity": "ollama:0.32.3",
                "num_ctx": 32768,
            },
        },
    )
    events = (
        _catalog_event(
            fingerprint,
            cache_hit_count=0,
            cache_miss_count=1,
            reuse_count=0,
            recompute_count=1,
        ),
        _catalog_event(
            fingerprint,
            cache_hit_count=1,
            cache_miss_count=0,
            reuse_count=1,
            recompute_count=0,
        ),
    )
    diagnostics = VisionInferenceDiagnostics(
        request_fingerprint="b" * 64,
        model_name="qwen3-vl:8b-instruct",
        model_identity="ollama:sha256:" + "c" * 64,
        runtime_identity="ollama:0.32.3",
        prompt_version="catalog-prompt-v1",
        schema_version="catalog-schema-v1",
        stage_contract_version="scene-catalog-v1",
        retry_policy_version="retry-v1",
        cache_hit=False,
        attempt_count=1,
        validation_code=None,
        image_count=1,
        context_cue_count=0,
        duration_seconds=1.0,
        prompt_eval_count=10,
        eval_count=5,
        done_reason="stop",
    )
    configuration = EffectiveConfiguration(
        video_input_folder=tmp_path / "input",
        output_folder=tmp_path / "output",
    )

    # Act
    provenance = build_report_provenance(
        (completed,),
        events,
        configuration,
        {fingerprint.value: diagnostics},
    )

    # Assert
    stage = provenance.stages[0]
    assert stage.cache_hits == 1
    assert stage.cache_misses == 1
    assert stage.recomputed_items == 1
    assert stage.attempt_count == 1
    assert stage.effective_settings["engine_version"] == stage_version(
        ProcessingStage.BUILD_SCENE_CATALOG
    )


def test_video_scan_parallelism_diagnostics_are_recorded_outside_stage_identity(
    tmp_path: Path,
) -> None:
    """動的worker診断がStage identityと分離されたruntimeへ記録されること。

    Arrange:
        - 一つのCompleted Stageとprivacy-safeなworker診断が用意される
    Act:
        - canonical report provenanceが構築される
    Assert:
        - worker診断がruntimeへ記録されsemantic inputは変更されないこと
    """
    # Arrange
    fingerprint = StageFingerprint("a" * 64)
    semantic_input = {
        "scan_algorithm": "video-scan-v2",
        "media_runtime_identity": {
            "ffmpeg_version": "6.1.1",
            "ffprobe_version": "6.1.1",
            "build_capability_sha256": "b" * 64,
        },
    }
    completed = CompletedStage(
        stage=ProcessingStage.SCAN_VIDEO,
        fingerprint=fingerprint,
        semantic_input=semantic_input,
    )
    event = ProgressEvent(
        kind="stage_completed",
        severity="info",
        stage=ProcessingStage.SCAN_VIDEO,
        stage_fingerprint=fingerprint.value,
        cache_miss_count=1,
        recompute_count=1,
        elapsed_seconds=1.0,
    )
    diagnostics = {
        "mode": "auto",
        "initial_workers": 6,
        "final_workers": 5,
        "changes": [
            {
                "from_workers": 6,
                "to_workers": 5,
                "reason": "cpu_pressure",
                "metrics": {"cpu_percent": 94.0},
            }
        ],
    }
    configuration = EffectiveConfiguration(
        video_input_folder=tmp_path / "input",
        output_folder=tmp_path / "output",
    )

    # Act
    provenance = build_report_provenance(
        (completed,),
        (event,),
        configuration,
        {},
        video_scan_parallelism=diagnostics,
    )

    # Assert
    assert provenance.runtime["video_scan_parallelism"] == diagnostics
    assert completed.semantic_input == semantic_input


def test_unused_speech_runtime_is_omitted_from_context_provenance(
    tmp_path: Path,
) -> None:
    """Context入力がない場合に未使用STT runtimeがprovenanceへ混入しないこと。

    Arrange:
        - subtitle・audio依存を持たないContext Completed Stageが用意される
    Act:
        - 異なるSpeech Runtime Identityでprovenanceが構築される
    Assert:
        - 両provenanceが一致し、STT toolとruntime identityが省略されること
    """
    # Arrange
    fingerprint = StageFingerprint("c" * 64)
    completed = CompletedStage(
        stage=ProcessingStage.COLLECT_CONTEXT,
        fingerprint=fingerprint,
        semantic_input={"policy_version": "context-collection-v1"},
    )
    event = _context_event(fingerprint)
    configuration = EffectiveConfiguration(
        video_input_folder=tmp_path / "input",
        output_folder=tmp_path / "output",
    )

    # Act
    first = build_report_provenance(
        (completed,),
        (event,),
        configuration,
        {},
    )
    second = build_report_provenance(
        (completed,),
        (event,),
        configuration,
        {},
    )

    # Assert
    assert first == second
    assert first.runtime == {"application": "video_selection"}
    assert first.tools == {"video_selection": "application-v1"}
    assert first.stages[0].tool_refs == ("video_selection",)


def test_used_speech_runtime_is_recorded_in_context_provenance(
    tmp_path: Path,
) -> None:
    """音声Contextで使用されたSTT・FFmpeg identityが記録されること。

    Arrange:
        - mediaとSTT依存を持つContext Completed Stageが用意される
    Act:
        - provenanceが構築される
    Assert:
        - runtime、tool registry、Stage参照へ使用identityが記録されること
    """
    # Arrange
    fingerprint = StageFingerprint("d" * 64)
    completed = CompletedStage(
        stage=ProcessingStage.COLLECT_CONTEXT,
        fingerprint=fingerprint,
        semantic_input={
            "policy_version": "context-collection-v1",
            "media_runtime_identity": {
                "ffmpeg_version": "6.1.1",
                "ffprobe_version": "6.1.1",
                "build_capability_sha256": "e" * 64,
            },
            "speech_runtime_identity": "speech-runtime-a",
        },
    )
    event = _context_event(fingerprint)
    configuration = EffectiveConfiguration(
        video_input_folder=tmp_path / "input",
        output_folder=tmp_path / "output",
    )

    # Act
    provenance = build_report_provenance(
        (completed,),
        (event,),
        configuration,
        {},
    )

    # Assert
    assert provenance.runtime["speech_runtime_identity"] == "speech-runtime-a"
    assert provenance.tools["speech_to_text"] == "speech-runtime-a"
    assert "ffmpeg" in provenance.tools
    assert provenance.stages[0].tool_refs == ("ffmpeg", "speech_to_text")


def _catalog_event(
    fingerprint: StageFingerprint,
    *,
    cache_hit_count: int,
    cache_miss_count: int,
    reuse_count: int,
    recompute_count: int,
) -> ProgressEvent:
    """一つのCatalog完了eventを返す。"""
    return ProgressEvent(
        kind="stage_completed",
        severity="info",
        stage=ProcessingStage.BUILD_SCENE_CATALOG,
        stage_fingerprint=fingerprint.value,
        cache_hit_count=cache_hit_count,
        cache_miss_count=cache_miss_count,
        reuse_count=reuse_count,
        recompute_count=recompute_count,
        elapsed_seconds=1.0,
    )


def _context_event(fingerprint: StageFingerprint) -> ProgressEvent:
    """一つのContext完了eventを返す。"""
    return ProgressEvent(
        kind="stage_completed",
        severity="info",
        stage=ProcessingStage.COLLECT_CONTEXT,
        stage_fingerprint=fingerprint.value,
        cache_miss_count=1,
        recompute_count=1,
        elapsed_seconds=1.0,
    )
