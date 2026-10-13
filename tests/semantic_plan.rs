use std::collections::BTreeMap;
use std::fs;
use std::sync::atomic::AtomicBool;

use game_screen_pick::pipeline::ModelConditions;
use game_screen_pick::run::{
    CacheConditions, ResolvedGameContext, RunPaths, SemanticPlan, Session,
};
use game_screen_pick::selection::SemanticEvent;

fn paths(root: &std::path::Path) -> RunPaths {
    let root = root.canonicalize().unwrap();
    let input = root.join("input");
    fs::create_dir(&input).unwrap();
    fs::write(
        input.join("recording.mkv"),
        b"identity only; no media decoding in this cache contract test",
    )
    .unwrap();
    RunPaths {
        input_directory: input,
        output_directory: root.join("output"),
        cache_directory: root.join("cache"),
        config: None,
    }
}

fn dependencies() -> (ModelConditions, CacheConditions) {
    (
        ModelConditions {
            backend: "vllm".into(),
            model: "fixture".into(),
            inference_level: "none".into(),
            timeout_seconds: 30,
            cache_revision: "v1".into(),
        },
        CacheConditions {
            service_url: "http://localhost:8000/v1".into(),
            prompt_revision: "events-v1".into(),
            response_schema_revision: "events-v1".into(),
            media_revision: "mov-v1".into(),
            resolved_game_context: ResolvedGameContext::direct("colored block progression"),
        },
    )
}

fn plan() -> SemanticPlan {
    SemanticPlan {
        durations: BTreeMap::from([("recording.mkv".into(), 3.0)]),
        events: vec![SemanticEvent {
            source: "recording.mkv".into(),
            start_seconds: 0.0,
            end_seconds: 3.0,
            representative_seconds: 1.5,
            importance: 80.0,
            scene: "colored blocks".into(),
            reason: "blocks progress in visible temporal order".into(),
        }],
    }
}

#[test]
fn completed_plan_is_small_and_reused_without_frame_metadata() {
    let root = tempfile::tempdir().unwrap();
    let paths = paths(root.path());
    let (model, conditions) = dependencies();
    let cancelled = AtomicBool::new(false);
    for cold in [true, false] {
        let mut session = Session::prepare(&paths, &cancelled).unwrap();
        let events = session
            .semantic_plan(30.0, &model, &conditions, &cancelled, |_| {
                assert!(cold, "completed plan must bypass generator");
                Ok(plan())
            })
            .unwrap();
        assert_eq!(events.len(), 1);
    }
}

#[test]
fn completed_empty_plan_is_valid_and_reused_without_invented_events() {
    let root = tempfile::tempdir().unwrap();
    let paths = paths(root.path());
    let (model, conditions) = dependencies();
    let cancelled = AtomicBool::new(false);
    for cold in [true, false] {
        let mut session = Session::prepare(&paths, &cancelled).unwrap();
        let events = session
            .semantic_plan(30.0, &model, &conditions, &cancelled, |_| {
                assert!(cold, "empty completed plan must bypass generator");
                let mut plan = plan();
                plan.events.clear();
                Ok(plan)
            })
            .unwrap();
        assert!(events.is_empty());
    }
}

#[test]
fn generation_conditions_and_input_changes_invalidate_only_the_plan() {
    let root = tempfile::tempdir().unwrap();
    let paths = paths(root.path());
    let (mut model, mut conditions) = dependencies();
    let cancelled = AtomicBool::new(false);
    let mut generated = 0;
    for change in 0..5 {
        let chunk = if change == 4 { 15.0 } else { 30.0 };
        match change {
            1 => model.cache_revision = "v2".into(),
            2 => conditions.resolved_game_context = ResolvedGameContext::direct("new context"),
            3 => fs::write(
                paths.input_directory.join("recording.mkv"),
                b"changed input identity",
            )
            .unwrap(),
            _ => {}
        }
        let mut session = Session::prepare(&paths, &cancelled).unwrap();
        session
            .semantic_plan(chunk, &model, &conditions, &cancelled, |_| {
                generated += 1;
                Ok(plan())
            })
            .unwrap();
    }
    assert_eq!(generated, 5);
}

#[test]
fn invalid_source_times_or_duplicates_are_never_checkpointed() {
    for case in 0..4 {
        let root = tempfile::tempdir().unwrap();
        let paths = paths(root.path());
        let (model, conditions) = dependencies();
        let cancelled = AtomicBool::new(false);
        let mut session = Session::prepare(&paths, &cancelled).unwrap();
        let mut invalid = plan();
        match case {
            0 => invalid.events[0].source = "unknown.mkv".into(),
            1 => invalid.events[0].end_seconds = 4.0,
            2 => invalid.events.push(invalid.events[0].clone()),
            3 => invalid.durations.clear(),
            _ => unreachable!(),
        }
        assert!(
            session
                .semantic_plan(30.0, &model, &conditions, &cancelled, |_| Ok(invalid))
                .is_err()
        );
        assert!(
            !paths
                .cache_directory
                .join(game_screen_pick::cache::NAMESPACE)
                .join("semantic-plans")
                .exists()
        );
    }
}
