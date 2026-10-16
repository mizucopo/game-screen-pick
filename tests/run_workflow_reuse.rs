//! Completed output reuse precedes context and semantic prerequisites, even when
//! the whole Rust cache namespace is gone. Fixed callbacks prove orchestration
//! contracts only; no live model capability is claimed.
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use anyhow::Result;
use game_screen_pick::cache::NAMESPACE;
use game_screen_pick::media::{Interrupted, MediaTools};
use game_screen_pick::pipeline::{AssessmentBatch, ModelConditions, PipelineOptions};
use game_screen_pick::run::{
    CacheConditions, ResolvedGameContext, RunPaths, SemanticPlan, Session,
};
use game_screen_pick::selection::{Assessment, SemanticEvent};
use serde_json::{Value, json};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/videos")
        .join(name)
}

fn paths(root: &Path) -> RunPaths {
    let root = root.canonicalize().unwrap();
    let input = root.join("input");
    fs::create_dir(&input).unwrap();
    fs::copy(fixture("01-blocks.mkv"), input.join("01-blocks.mkv")).unwrap();
    RunPaths {
        input_directory: input,
        output_directory: root.join("output"),
        cache_directory: root.join("cache"),
        config: None,
    }
}

fn context() -> String {
    let responses: Value = serde_json::from_str(include_str!("fixtures/responses.json")).unwrap();
    responses["game_context"].as_str().unwrap().into()
}

fn model() -> ModelConditions {
    ModelConditions {
        backend: "vllm".into(),
        model: "fixture-model".into(),
        inference_level: "none".into(),
        timeout_seconds: 30,
        cache_revision: "fixture-model-v1".into(),
    }
}

fn conditions(context: ResolvedGameContext) -> CacheConditions {
    CacheConditions {
        service_url: "http://localhost:8000/v1".into(),
        prompt_revision: "fixture-prompt-v1".into(),
        response_schema_revision: "fixture-schema-v1".into(),
        media_revision: "fixture-media-v1".into(),
        resolved_game_context: context,
    }
}

fn events() -> Vec<SemanticEvent> {
    let responses: Value = serde_json::from_str(include_str!("fixtures/responses.json")).unwrap();
    responses["semantic_video"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|source| source["source"] == "01-blocks.mkv")
        .flat_map(|source| {
            source["events"]
                .as_array()
                .unwrap()
                .iter()
                .map(|event| SemanticEvent {
                    source: "01-blocks.mkv".into(),
                    start_seconds: event["start_seconds"].as_f64().unwrap(),
                    end_seconds: event["end_seconds"].as_f64().unwrap(),
                    representative_seconds: event["representative_seconds"].as_f64().unwrap(),
                    importance: event["importance"].as_f64().unwrap(),
                    scene: event["scene"].as_str().unwrap().into(),
                    reason: event["reason"].as_str().unwrap().into(),
                })
        })
        .collect()
}

fn options(semantic: bool, context: &ResolvedGameContext) -> PipelineOptions {
    PipelineOptions {
        count: 1,
        samples_per_video: 6,
        context_delta_seconds: 0.25,
        semantic_events: semantic.then(events),
        model: model(),
        game_context: Some(context.text().into()),
    }
}

fn workflow(semantic: bool) -> Value {
    json!({
        "model":model(), "service_url":"http://localhost:8000/v1",
        "protocol_revision":"fixture-protocol-v1",
        "context":{"kind":"title","input_digest":"nonsecret-title-hash","search_revision":"brave-v1"},
        "prompt_revision":"fixture-prompt-v1","schema_revision":"fixture-schema-v1",
        "media_revision":"fixture-media-v1",
        "selection":{"method":if semantic {"semantic_video"} else {"sampled_frames"},
            "count":1,"samples":6,"context_delta_seconds":0.25,
            "requested_chunk_seconds":30.0,"effective_chunk_seconds":10.0}
    })
}

fn assess(batch: &AssessmentBatch<'_>) -> Result<Vec<Assessment>> {
    Ok(batch
        .candidates
        .iter()
        .map(|candidate| Assessment {
            id: candidate.id.clone(),
            blog_score: 80.0,
            transition: false,
            scene: format!("fixture progression {}", candidate.actual_seconds.floor()),
            reason: "fixed contract fixture".into(),
            special: false,
        })
        .collect())
}

fn publish(paths: &RunPaths, semantic: bool, workflow: Value) -> Value {
    let cancelled = AtomicBool::new(false);
    let mut session = Session::prepare(paths, &cancelled).unwrap();
    assert!(
        session
            .prepare_workflow(workflow, &cancelled)
            .unwrap()
            .is_none()
    );
    let resolved = session
        .generated_context(
            "fixture game",
            "search-v1",
            &conditions(ResolvedGameContext::direct("")),
            &model(),
            true,
            &cancelled,
            || Ok(context()),
        )
        .unwrap();
    let conditions = conditions(resolved.clone());
    let mut options = options(semantic, &resolved);
    if semantic {
        options.semantic_events = Some(
            session
                .semantic_plan(10.0, &model(), &conditions, &cancelled, |_| {
                    Ok(SemanticPlan {
                        durations: BTreeMap::from([("01-blocks.mkv".into(), 6.0)]),
                        events: events(),
                    })
                })
                .unwrap(),
        );
    }
    session
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &cancelled,
            assess,
        )
        .unwrap()
        .report
}

fn snapshot(directory: &Path) -> BTreeMap<String, Vec<u8>> {
    fs::read_dir(directory)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().to_str().unwrap().into(),
                fs::read(entry.path()).unwrap(),
            )
        })
        .collect()
}

#[test]
fn cache_namespace_loss_reuses_completed_output_before_generated_context_and_semantic_plan() {
    for semantic in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let paths = paths(root.path());
        let requested = workflow(semantic);
        let report = publish(&paths, semantic, requested.clone());
        assert_eq!(report["cache"]["workflow"]["conditions"], requested);
        assert!(report["cache"]["workflow"]["key"].as_str().unwrap().len() == 64);
        let original = snapshot(&paths.output_directory);
        fs::remove_dir_all(paths.cache_directory.join(NAMESPACE)).unwrap();
        let cancelled = AtomicBool::new(false);
        let mut session = Session::prepare(&paths, &cancelled).unwrap();
        let warm = session
            .prepare_workflow(requested, &cancelled)
            .unwrap()
            .expect("owned output must precede cache-only prerequisite failures");
        assert!(warm.warm);
        assert_eq!(warm.report, report);
        assert_eq!(warm.metrics.cache.input_hashes, 1);
        assert_eq!(warm.metrics.probe_calls, 0);
        assert_eq!(warm.metrics.context_calls, 0);
        assert_eq!(warm.metrics.semantic_calls, 0);
        assert_eq!(warm.metrics.assessment_calls, 0);
        assert_eq!(warm.metrics.assessment_bytes_read, 0);
        assert_eq!(snapshot(&paths.output_directory), original);
        assert!(
            !paths
                .cache_directory
                .join(NAMESPACE)
                .join("contexts")
                .exists()
        );
        assert!(
            !paths
                .cache_directory
                .join(NAMESPACE)
                .join("semantic-plans")
                .exists()
        );
        assert!(!paths.cache_directory.join(NAMESPACE).join("probe").exists());
    }
}

#[test]
fn complete_requested_conditions_and_named_input_content_invalidate_workflow_reuse() {
    let root = tempfile::tempdir().unwrap();
    let paths = paths(root.path());
    let requested = workflow(true);
    publish(&paths, true, requested.clone());
    let original = snapshot(&paths.output_directory);
    let cancelled = AtomicBool::new(false);
    for variant in 0..7 {
        let mut changed = requested.clone();
        match variant {
            0 => changed["model"]["cache_revision"] = "v2".into(),
            1 => changed["context"]["input_digest"] = "other-title-hash".into(),
            2 => changed["context"]["search_revision"] = "brave-v2".into(),
            3 => changed["selection"]["count"] = 2.into(),
            4 => changed["selection"]["requested_chunk_seconds"] = 60.0.into(),
            5 => changed["selection"]["effective_chunk_seconds"] = 5.0.into(),
            6 => changed["protocol_revision"] = "protocol-v2".into(),
            _ => unreachable!(),
        }
        let mut session = Session::prepare(&paths, &cancelled).unwrap();
        assert!(
            session
                .prepare_workflow(changed, &cancelled)
                .unwrap()
                .is_none()
        );
        assert_eq!(snapshot(&paths.output_directory), original);
    }
    fs::copy(
        fixture("02-mirrored.mkv"),
        paths.input_directory.join("01-blocks.mkv"),
    )
    .unwrap();
    {
        let mut session = Session::prepare(&paths, &cancelled).unwrap();
        assert!(
            session
                .prepare_workflow(requested.clone(), &cancelled)
                .unwrap()
                .is_none()
        );
    }
    fs::copy(
        fixture("01-blocks.mkv"),
        paths.input_directory.join("01-blocks.mkv"),
    )
    .unwrap();
    fs::rename(
        paths.input_directory.join("01-blocks.mkv"),
        paths.input_directory.join("renamed.mkv"),
    )
    .unwrap();
    let mut session = Session::prepare(&paths, &cancelled).unwrap();
    assert!(
        session
            .prepare_workflow(requested, &cancelled)
            .unwrap()
            .is_none()
    );
    assert_eq!(snapshot(&paths.output_directory), original);
}

#[test]
fn changed_generation_conditions_publish_new_workflow_even_when_final_run_key_matches() {
    let root = tempfile::tempdir().unwrap();
    let paths = paths(root.path());
    let requested = workflow(false);
    let first = publish(&paths, false, requested.clone());
    let mut changed = requested;
    changed["context"]["search_revision"] = "brave-v2".into();
    let second = publish(&paths, false, changed.clone());
    assert_eq!(first["cache"]["key"], second["cache"]["key"]);
    assert_ne!(
        first["cache"]["workflow"]["key"],
        second["cache"]["workflow"]["key"]
    );
    assert_eq!(second["cache"]["workflow"]["conditions"], changed);
    let mut session = Session::prepare(&paths, &AtomicBool::new(false)).unwrap();
    assert!(
        session
            .prepare_workflow(changed, &AtomicBool::new(false))
            .unwrap()
            .is_some()
    );
}

#[test]
fn workflow_metadata_remains_covered_by_output_ownership_verification() {
    let root = tempfile::tempdir().unwrap();
    let paths = paths(root.path());
    let mut report = publish(&paths, false, workflow(false));
    report["cache"]["workflow"]["conditions"]["selection"]["count"] = 99.into();
    fs::write(
        paths.output_directory.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    let edited = snapshot(&paths.output_directory);
    assert!(Session::prepare(&paths, &AtomicBool::new(false)).is_err());
    assert_eq!(snapshot(&paths.output_directory), edited);
}

#[test]
fn credential_fields_are_rejected_without_echo_and_cancelled_reuse_preserves_output() {
    let root = tempfile::tempdir().unwrap();
    let paths = paths(root.path());
    publish(&paths, false, workflow(false));
    let original = snapshot(&paths.output_directory);
    let mut session = Session::prepare(&paths, &AtomicBool::new(false)).unwrap();
    for name in ["api_key", "authorization", "password", "access_token"] {
        let secret = "WORKFLOW_CREDENTIAL_MUST_NOT_PERSIST";
        let mut requested = workflow(false);
        requested["nested"] = json!([{name:secret}]);
        let error = session
            .prepare_workflow(requested, &AtomicBool::new(false))
            .unwrap_err();
        assert!(!format!("{error:#}\n{error:?}").contains(secret));
    }
    let error = session
        .prepare_workflow(workflow(false), &AtomicBool::new(true))
        .unwrap_err();
    assert!(error.is::<Interrupted>());
    assert_eq!(snapshot(&paths.output_directory), original);
}

#[test]
fn existing_run_callers_keep_the_original_owned_output_shortcut_without_workflow_metadata() {
    let root = tempfile::tempdir().unwrap();
    let paths = paths(root.path());
    let cancelled = AtomicBool::new(false);
    let resolved = ResolvedGameContext::direct(context());
    let options = options(false, &resolved);
    let conditions = conditions(resolved);
    let first = Session::prepare(&paths, &cancelled)
        .unwrap()
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &cancelled,
            assess,
        )
        .unwrap();
    assert!(first.report["cache"]["workflow"].is_null());
    fs::remove_dir_all(paths.cache_directory.join(NAMESPACE)).unwrap();
    let warm = Session::prepare(&paths, &cancelled)
        .unwrap()
        .run(
            &MediaTools {
                ffmpeg: "missing-ffmpeg".into(),
                ffprobe: "missing-ffprobe".into(),
            },
            &options,
            &conditions,
            &cancelled,
            |_| panic!("completed output must keep warm reuse"),
        )
        .unwrap();
    assert!(warm.warm);
    assert_eq!(warm.report, first.report);
}
