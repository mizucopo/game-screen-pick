use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use game_screen_pick::cache::NAMESPACE;
use game_screen_pick::media::{ChunkArtifact, Interrupted, MediaTools};
use game_screen_pick::pipeline::{
    AssessmentBatch, AssessmentPhase, ModelConditions, PipelineOptions,
};
use game_screen_pick::run::{CacheConditions, ResolvedGameContext, RunPaths, Session};
use game_screen_pick::selection::{Assessment, SemanticEvent};
use image::RgbImage;
use serde_json::Value;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/videos")
        .join(name)
}

fn responses() -> Value {
    serde_json::from_str(include_str!("fixtures/responses.json")).unwrap()
}

fn setup_paths(root: &Path, multiple: bool) -> RunPaths {
    let root = root.canonicalize().unwrap();
    let input = root.join("input");
    let cache = root.join("cache");
    fs::create_dir(&input).unwrap();
    fs::create_dir(&cache).unwrap();
    fs::copy(fixture("01-blocks.mkv"), input.join("01-blocks.mkv")).unwrap();
    if multiple {
        fs::copy(fixture("02-mirrored.mkv"), input.join("02-mirrored.mkv")).unwrap();
    }
    RunPaths {
        input_directory: input,
        output_directory: root.join("output"),
        cache_directory: cache,
        config: None,
    }
}

fn options(count: usize, semantic: bool, multiple: bool) -> PipelineOptions {
    let fixture = responses();
    let events = semantic.then(|| {
        fixture["semantic_video"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|source| multiple || source["source"] == "01-blocks.mkv")
            .flat_map(|source| {
                source["events"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|event| SemanticEvent {
                        source: source["source"].as_str().unwrap().to_owned(),
                        start_seconds: event["start_seconds"].as_f64().unwrap(),
                        end_seconds: event["end_seconds"].as_f64().unwrap(),
                        representative_seconds: event["representative_seconds"].as_f64().unwrap(),
                        importance: event["importance"].as_f64().unwrap(),
                        scene: event["scene"].as_str().unwrap().to_owned(),
                        reason: event["reason"].as_str().unwrap().to_owned(),
                    })
            })
            .collect()
    });
    PipelineOptions {
        count,
        samples_per_video: 6,
        context_delta_seconds: 0.25,
        semantic_events: events,
        model: ModelConditions {
            backend: "strata".to_owned(),
            model: "fixture-vision".to_owned(),
            inference_level: "high".to_owned(),
            timeout_seconds: 60,
            cache_revision: "test".to_owned(),
        },
        game_context: Some(fixture["game_context"].as_str().unwrap().to_owned()),
    }
}

fn conditions() -> CacheConditions {
    CacheConditions {
        service_url: "http://127.0.0.1:8123/v1".to_owned(),
        prompt_revision: "fixture-prompt-v1".to_owned(),
        response_schema_revision: "fixture-assessment-v1".to_owned(),
        media_revision: "ffmpeg-pts-png-v1".to_owned(),
        resolved_game_context: ResolvedGameContext::direct(
            responses()["game_context"].as_str().unwrap(),
        ),
    }
}

fn unavailable_tools() -> MediaTools {
    MediaTools {
        ffmpeg: "missing-ffmpeg-for-cache-test".into(),
        ffprobe: "missing-ffprobe-for-cache-test".into(),
    }
}

// The fixture recipe is independent of extraction, cache and the fixed ratings.
fn expected(source: &str, seconds: f64) -> RgbImage {
    let rows = include_str!("fixtures/video-tiles.tsv")
        .lines()
        .map(|line| {
            line.split('\t')
                .skip(2)
                .map(|value| value.parse::<u8>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mirrored = source == "02-mirrored.mkv";
    let rotated = ["03-multitrack-rotated.mov", "00-before.mov"].contains(&source);
    let (width, height) = if rotated { (96, 160) } else { (160, 96) };
    RgbImage::from_fn(width, height, |x, y| {
        let (x, y) = if rotated { (159 - y, x) } else { (x, y) };
        let tile = if mirrored { 9 - x / 16 } else { x / 16 };
        image::Rgb([rows[seconds.floor() as usize * 6 + y as usize / 16][tile as usize]; 3])
    })
}

fn assess(batch: &AssessmentBatch<'_>) -> Result<Vec<Assessment>> {
    assess_recipe(batch, None)
}

fn assess_recipe(batch: &AssessmentBatch<'_>, recipe: Option<&str>) -> Result<Vec<Assessment>> {
    assert!(!batch.candidates.is_empty());
    assert!(batch.sheets.iter().all(|sheet| image::open(sheet).is_ok()));
    for candidate in batch.candidates {
        assert_eq!(
            image::open(&candidate.image).unwrap().to_rgb8(),
            expected(
                recipe.unwrap_or(&candidate.source),
                candidate.actual_seconds
            ),
            "response requires this source's actual full-resolution frame"
        );
    }
    match batch.phase {
        AssessmentPhase::Primary => assert!(batch.contexts.is_empty()),
        AssessmentPhase::Secondary => {
            assert_eq!(batch.contexts.len(), batch.candidates.len());
            for (candidate, context) in batch.candidates.iter().zip(&batch.contexts) {
                assert_eq!(context.id, candidate.id);
                assert_eq!(context.frames[1].actual_seconds, candidate.actual_seconds);
                for frame in &context.frames {
                    assert_eq!(frame.source, candidate.source);
                    assert_eq!(
                        image::open(&frame.image).unwrap().to_rgb8(),
                        expected(recipe.unwrap_or(&frame.source), frame.actual_seconds),
                        "response requires the actual ordered before/center/after media"
                    );
                }
            }
        }
    }
    let mut ratings = batch
        .candidates
        .iter()
        .map(|candidate| Assessment {
            id: candidate.id.clone(),
            blog_score: 80.0,
            transition: false,
            scene: format!("通常進行 {}", candidate.actual_seconds.floor()),
            reason: "既知の入力模様と実 PTS・前後の原画像を照合".to_owned(),
            special: false,
        })
        .collect::<Vec<_>>();
    ratings.reverse();
    Ok(ratings)
}

fn snapshot(output: &Path) -> BTreeMap<String, Vec<u8>> {
    fs::read_dir(output)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().into_string().unwrap(),
                fs::read(entry.path()).unwrap(),
            )
        })
        .collect()
}

fn verify_outputs(output: &Path, report: &Value, count: usize) {
    assert_eq!(report["requested_count"], count);
    assert_eq!(report["selected_count"], count);
    let selected = report["selected"].as_array().unwrap();
    assert_eq!(selected.len(), count);
    assert_eq!(fs::read_dir(output).unwrap().count(), count + 2);
    for (ordinal, frame) in selected.iter().enumerate() {
        assert_eq!(frame["rank"], ordinal + 1);
        assert_eq!(frame["id"], frame["assessment"]["id"]);
        assert_eq!(frame["id"], frame["primary_assessment"]["id"]);
        let source = frame["source"].as_str().unwrap();
        let image = image::open(output.join(frame["image"].as_str().unwrap()))
            .unwrap()
            .to_rgb8();
        let expected = expected(source, frame["actual_seconds"].as_f64().unwrap());
        assert_eq!(image.dimensions(), expected.dimensions());
        let error = image
            .as_raw()
            .iter()
            .zip(expected.as_raw())
            .map(|(actual, wanted)| f64::from(actual.abs_diff(*wanted)))
            .sum::<f64>()
            / image.as_raw().len() as f64;
        assert!(error < 4.0, "published JPEG differs from source: {error}");
    }
    assert!(image::open(output.join("selected-contact-sheet.jpg")).is_ok());
    let stored: Value =
        serde_json::from_slice(&fs::read(output.join("report.json")).unwrap()).unwrap();
    assert_eq!(&stored, report);
}

#[test]
fn cold_and_warm_both_methods_publish_identical_actual_artifacts_without_media_calls() {
    for semantic in [false, true] {
        for multiple in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let paths = setup_paths(root.path(), multiple);
            let options = options(2, semantic, multiple);
            let conditions = conditions();
            let cold = Session::prepare(&paths, &AtomicBool::new(false))
                .unwrap()
                .run(
                    &MediaTools::default(),
                    &options,
                    &conditions,
                    &AtomicBool::new(false),
                    assess,
                )
                .unwrap();
            assert!(!cold.warm);
            assert_eq!(cold.metrics.probe_calls, if multiple { 2 } else { 1 });
            assert_eq!(
                cold.metrics.cache.input_hashes,
                if multiple { 2 } else { 1 }
            );
            assert!(cold.metrics.assessment_calls > 0);
            verify_outputs(&paths.output_directory, &cold.report, 2);
            let completed = snapshot(&paths.output_directory);
            let warm = Session::prepare(&paths, &AtomicBool::new(false))
                .unwrap()
                .run(
                    &unavailable_tools(),
                    &options,
                    &conditions,
                    &AtomicBool::new(false),
                    |_| panic!("warm must not invoke inference, search or runtime callbacks"),
                )
                .unwrap();
            assert!(warm.warm);
            assert_eq!(warm.metrics.probe_calls, 0);
            assert_eq!(warm.metrics.assessment_calls, 0);
            assert_eq!(warm.metrics.context_calls, 0);
            assert_eq!(warm.metrics.semantic_calls, 0);
            assert_eq!(warm.metrics.cache.input_hashes, 0);
            assert_eq!(warm.report, cold.report);
            assert_eq!(snapshot(&paths.output_directory), completed);
            let published_bytes = completed
                .values()
                .map(|bytes| bytes.len() as u64)
                .sum::<u64>();
            assert!(
                cold.metrics.output_bytes_read > published_bytes,
                "cold metrics must include staging reads as well as verification of published output"
            );
            assert!(
                warm.metrics.output_bytes_read >= published_bytes * 2,
                "warm metrics must include both preflight and current-run ownership verification"
            );
            assert!(warm.metrics.cache.bytes_read < 128 * 1024);
            assert!(warm.metrics.elapsed_seconds.is_finite());
            eprintln!(
                "warm semantic={semantic} multiple={multiple}: cache={}B output={}B elapsed={}s probe={} assessment={} input_hash={}",
                warm.metrics.cache.bytes_read,
                warm.metrics.output_bytes_read,
                warm.metrics.elapsed_seconds,
                warm.metrics.probe_calls,
                warm.metrics.assessment_calls,
                warm.metrics.cache.input_hashes,
            );
            if let Some(qa_root) = std::env::var_os("GSP_QA_DIR") {
                let destination = PathBuf::from(qa_root).join(format!(
                    "cached-{}-{}",
                    if semantic { "semantic" } else { "sampled" },
                    if multiple { "multiple" } else { "single" },
                ));
                fs::create_dir_all(&destination).unwrap();
                for (name, bytes) in completed {
                    fs::write(destination.join(name), bytes).unwrap();
                }
                fs::write(
                    destination.join("warm-metrics.json"),
                    serde_json::to_vec_pretty(&warm.metrics).unwrap(),
                )
                .unwrap();
            }
        }
    }
}

#[test]
fn metrics_include_prerun_context_semantic_and_preflight_work() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let mut options = completed_semantic_workflow(&paths);
    let previous = snapshot(&paths.output_directory);
    for kind in ["contexts", "inputs", "probe", "semantic"] {
        fs::remove_dir_all(paths.cache_directory.join(NAMESPACE).join(kind)).unwrap();
    }
    let mut conditions = conditions();
    let cancelled = AtomicBool::new(false);
    let source = paths.input_directory.join("01-blocks.mkv");
    let mut session = Session::prepare(&paths, &AtomicBool::new(false)).unwrap();
    let before_run = Instant::now();
    let mut context_calls = 0;
    conditions.resolved_game_context = session
        .generated_context(
            "合成ゲーム",
            "brave-fixed-query-v1",
            &conditions,
            &options.model,
            true,
            &cancelled,
            || {
                context_calls += 1;
                std::thread::sleep(Duration::from_millis(20));
                Ok(conditions.resolved_game_context.text().to_owned())
            },
        )
        .unwrap();
    let mut semantic_calls = 0;
    options.semantic_events = Some(
        session
            .semantic_chunk(
                &MediaTools::default(),
                &source,
                0.0,
                1.0,
                &options.model,
                &conditions,
                &cancelled,
                |chunk| {
                    semantic_calls += 1;
                    understand(chunk)
                },
            )
            .unwrap(),
    );
    let prerun_seconds = before_run.elapsed().as_secs_f64();
    let result = session
        .run(
            &unavailable_tools(),
            &options,
            &conditions,
            &cancelled,
            |_| panic!("identical completed output must skip assessment"),
        )
        .unwrap();
    assert!(result.warm);
    assert_eq!(context_calls, 1);
    assert_eq!(semantic_calls, 1);
    assert_eq!(result.metrics.context_calls, context_calls);
    assert_eq!(result.metrics.semantic_calls, semantic_calls);
    assert_eq!(result.metrics.cache.input_hashes, 1);
    assert_eq!(result.metrics.probe_calls, 1);
    assert_eq!(result.metrics.assessment_calls, 0);
    assert!(result.metrics.cache.misses >= 4);
    assert!(result.metrics.cache.bytes_read >= fs::metadata(&source).unwrap().len());
    assert!(result.metrics.elapsed_seconds >= prerun_seconds);
    assert_eq!(
        result.metrics.cache.elapsed_seconds,
        result.metrics.elapsed_seconds
    );
    let published_bytes = previous
        .values()
        .map(|bytes| bytes.len() as u64)
        .sum::<u64>();
    assert!(result.metrics.output_bytes_read >= 2 * published_bytes);
    assert_eq!(snapshot(&paths.output_directory), previous);
}

#[test]
fn metrics_include_warm_prerun_checkpoint_reads_without_callbacks() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let mut options = completed_semantic_workflow(&paths);
    let previous = snapshot(&paths.output_directory);
    let mut conditions = conditions();
    let cancelled = AtomicBool::new(false);
    let mut session = Session::prepare(&paths, &AtomicBool::new(false)).unwrap();
    conditions.resolved_game_context = session
        .generated_context(
            "合成ゲーム",
            "brave-fixed-query-v1",
            &conditions,
            &options.model,
            true,
            &cancelled,
            || panic!("warm context must not invoke generation"),
        )
        .unwrap();
    options.semantic_events = Some(
        session
            .semantic_chunk(
                &unavailable_tools(),
                &paths.input_directory.join("01-blocks.mkv"),
                0.0,
                1.0,
                &options.model,
                &conditions,
                &cancelled,
                |_| panic!("warm chunk must not invoke understanding"),
            )
            .unwrap(),
    );
    let result = session
        .run(
            &unavailable_tools(),
            &options,
            &conditions,
            &cancelled,
            |_| panic!("warm selection must not invoke assessment"),
        )
        .unwrap();
    assert!(result.warm);
    assert_eq!(result.metrics.probe_calls, 0);
    assert_eq!(result.metrics.assessment_calls, 0);
    assert_eq!(result.metrics.cache.input_hashes, 0);
    assert_eq!(result.metrics.context_calls, 0);
    assert_eq!(result.metrics.semantic_calls, 0);
    assert_eq!(result.metrics.cache.misses, 0);
    assert!(result.metrics.cache.hits >= 4);
    assert!(result.metrics.cache.bytes_read > 0);
    assert!(result.metrics.cache.bytes_read < 128 * 1024);
    let published_bytes = previous
        .values()
        .map(|bytes| bytes.len() as u64)
        .sum::<u64>();
    assert!(result.metrics.output_bytes_read >= 2 * published_bytes);
    assert_eq!(snapshot(&paths.output_directory), previous);
}

#[test]
fn metrics_accumulate_across_cold_and_warm_runs_in_one_session() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let options = options(1, false, false);
    let conditions = conditions();
    let cancelled = AtomicBool::new(false);
    let mut session = Session::prepare(&paths, &AtomicBool::new(false)).unwrap();
    let cold = session
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &cancelled,
            assess,
        )
        .unwrap();
    assert!(!cold.warm);
    assert_eq!(cold.metrics.cache.input_hashes, 1);
    assert_eq!(cold.metrics.probe_calls, 1);
    assert!(cold.metrics.assessment_calls > 0);
    assert!(cold.metrics.assessment_bytes_read > 0);
    let warm = session
        .run(
            &unavailable_tools(),
            &options,
            &conditions,
            &cancelled,
            |_| panic!("second run must reuse completed output"),
        )
        .unwrap();
    assert!(warm.warm);
    assert_eq!(warm.report, cold.report);
    assert_eq!(
        warm.metrics.cache.input_hashes,
        cold.metrics.cache.input_hashes
    );
    assert_eq!(warm.metrics.probe_calls, cold.metrics.probe_calls);
    assert_eq!(warm.metrics.assessment_calls, cold.metrics.assessment_calls);
    assert_eq!(
        warm.metrics.assessment_bytes_read,
        cold.metrics.assessment_bytes_read
    );
    assert_eq!(warm.metrics.context_calls, cold.metrics.context_calls);
    assert_eq!(warm.metrics.semantic_calls, cold.metrics.semantic_calls);
    assert!(warm.metrics.cache.bytes_read >= cold.metrics.cache.bytes_read);
    assert!(warm.metrics.cache.hits >= cold.metrics.cache.hits);
    assert!(warm.metrics.cache.misses >= cold.metrics.cache.misses);
    assert!(warm.metrics.output_bytes_read > cold.metrics.output_bytes_read);
    assert!(warm.metrics.elapsed_seconds >= cold.metrics.elapsed_seconds);
}

fn completed_semantic_workflow(paths: &RunPaths) -> PipelineOptions {
    let mut conditions = conditions();
    let cancelled = AtomicBool::new(false);
    let mut options = options(1, true, false);
    let mut session = Session::prepare(paths, &AtomicBool::new(false)).unwrap();
    conditions.resolved_game_context = session
        .generated_context(
            "合成ゲーム",
            "brave-fixed-query-v1",
            &conditions,
            &options.model,
            true,
            &cancelled,
            || Ok(conditions.resolved_game_context.text().to_owned()),
        )
        .unwrap();
    options.semantic_events = Some(
        session
            .semantic_chunk(
                &MediaTools::default(),
                &paths.input_directory.join("01-blocks.mkv"),
                0.0,
                1.0,
                &options.model,
                &conditions,
                &cancelled,
                understand,
            )
            .unwrap(),
    );
    session
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &cancelled,
            assess,
        )
        .unwrap();
    options
}

#[test]
fn completed_primary_and_secondary_batches_survive_immediate_interruption() {
    for interrupted_phase in [AssessmentPhase::Primary, AssessmentPhase::Secondary] {
        let root = tempfile::tempdir().unwrap();
        let paths = setup_paths(root.path(), true);
        let options = options(2, false, true);
        let conditions = conditions();
        let cancelled = AtomicBool::new(false);
        let mut completed = Vec::new();
        let error = Session::prepare(&paths, &AtomicBool::new(false))
            .unwrap()
            .run(
                &MediaTools::default(),
                &options,
                &conditions,
                &cancelled,
                |batch| {
                    let response = assess(batch)?;
                    completed.push((
                        batch.phase,
                        batch
                            .candidates
                            .iter()
                            .map(|candidate| candidate.id.clone())
                            .collect::<Vec<_>>(),
                    ));
                    if batch.phase == interrupted_phase {
                        cancelled.store(true, Ordering::Relaxed);
                    }
                    Ok(response)
                },
            )
            .err()
            .unwrap();
        assert!(error.downcast_ref::<Interrupted>().is_some());
        assert!(!paths.output_directory.exists());
        cancelled.store(false, Ordering::Relaxed);
        let resumed = Session::prepare(&paths, &AtomicBool::new(false))
            .unwrap()
            .run(
                &MediaTools::default(),
                &options,
                &conditions,
                &cancelled,
                |batch| {
                    let batch_ids = batch
                        .candidates
                        .iter()
                        .map(|candidate| candidate.id.clone())
                        .collect::<Vec<_>>();
                    assert!(
                        !completed.contains(&(batch.phase, batch_ids)),
                        "a completed validated batch must not repeat inference"
                    );
                    assess(batch)
                },
            )
            .unwrap();
        assert_eq!(resumed.metrics.probe_calls, 0);
        assert_eq!(resumed.metrics.cache.input_hashes, 0);
        verify_outputs(&paths.output_directory, &resumed.report, 2);
        let clean_root = tempfile::tempdir().unwrap();
        let clean_paths = setup_paths(clean_root.path(), true);
        let clean = Session::prepare(&clean_paths, &AtomicBool::new(false))
            .unwrap()
            .run(
                &MediaTools::default(),
                &options,
                &conditions,
                &AtomicBool::new(false),
                assess,
            )
            .unwrap();
        assert_eq!(resumed.report, clean.report);
        assert_eq!(
            snapshot(&paths.output_directory),
            snapshot(&clean_paths.output_directory)
        );
    }
}

#[test]
fn count_change_reuses_primary_assessment_and_probed_input() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let mut options = options(1, false, false);
    let conditions = conditions();
    Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &AtomicBool::new(false),
            assess,
        )
        .unwrap();
    options.count = 2;
    let changed = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &AtomicBool::new(false),
            |batch| {
                assert_eq!(
                    batch.phase,
                    AssessmentPhase::Secondary,
                    "count is not a primary evaluation dependency"
                );
                assess(batch)
            },
        )
        .unwrap();
    assert!(!changed.warm);
    assert_eq!(changed.metrics.probe_calls, 0);
    assert_eq!(changed.metrics.cache.input_hashes, 0);
    verify_outputs(&paths.output_directory, &changed.report, 2);
}

#[test]
fn input_directory_move_preserves_warm_and_added_video_only_probes_new_content() {
    let root = tempfile::tempdir().unwrap();
    let mut paths = setup_paths(root.path(), true);
    let options = options(2, false, true);
    let conditions = conditions();
    let first = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &AtomicBool::new(false),
            assess,
        )
        .unwrap();
    let moved = paths
        .input_directory
        .parent()
        .unwrap()
        .join("moved recordings");
    fs::rename(&paths.input_directory, &moved).unwrap();
    paths.input_directory = moved;
    let moved = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &unavailable_tools(),
            &options,
            &conditions,
            &AtomicBool::new(false),
            |_| panic!("moving the input directory must preserve warm"),
        )
        .unwrap();
    assert!(moved.warm);
    assert_eq!(moved.report, first.report);
    assert_eq!(moved.metrics.cache.input_hashes, 0);
    fs::copy(
        fixture("03-multitrack-rotated.mov"),
        paths.input_directory.join("00-before.mov"),
    )
    .unwrap();
    let added = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &AtomicBool::new(false),
            assess,
        )
        .unwrap();
    assert!(!added.warm);
    assert_eq!(added.metrics.probe_calls, 1);
    assert_eq!(added.metrics.cache.input_hashes, 1);
    assert_eq!(
        added.report["sources"],
        serde_json::json!(["00-before.mov", "01-blocks.mkv", "02-mirrored.mkv"])
    );
    for selected in added.report["selected"].as_array().unwrap() {
        let prefix = match selected["source"].as_str().unwrap() {
            "00-before.mov" => "V0001-",
            "01-blocks.mkv" => "V0002-",
            "02-mirrored.mkv" => "V0003-",
            source => panic!("unexpected source {source}"),
        };
        assert!(selected["id"].as_str().unwrap().starts_with(prefix));
    }
    verify_outputs(&paths.output_directory, &added.report, 2);
}

#[test]
fn semantic_effective_condition_changes_invalidate_only_dependent_results() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let initial_options = options(1, false, false);
    let initial_conditions = conditions();
    Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &MediaTools::default(),
            &initial_options,
            &initial_conditions,
            &AtomicBool::new(false),
            assess,
        )
        .unwrap();
    for changed in [
        "model", "backend", "level", "revision", "service", "prompt", "schema", "media", "context",
    ] {
        let mut options = initial_options.clone();
        let mut conditions = conditions();
        match changed {
            "model" => options.model.model = "other-fixture-model".to_owned(),
            "backend" => options.model.backend = "vllm".to_owned(),
            "level" => options.model.inference_level = "low".to_owned(),
            "revision" => options.model.cache_revision = "other-fixture-revision".to_owned(),
            "service" => conditions.service_url = "http://127.0.0.1:8124/custom-service".to_owned(),
            "prompt" => conditions.prompt_revision = "fixture-prompt-v2".to_owned(),
            "schema" => conditions.response_schema_revision = "fixture-assessment-v2".to_owned(),
            "media" => conditions.media_revision = "other-media-processing-v2".to_owned(),
            "context" => {
                conditions.resolved_game_context =
                    ResolvedGameContext::direct("別の解決済みゲーム説明");
                options.game_context = None;
            }
            _ => unreachable!(),
        }
        let mut calls = 0;
        let result = Session::prepare(&paths, &AtomicBool::new(false))
            .unwrap()
            .run(
                &MediaTools::default(),
                &options,
                &conditions,
                &AtomicBool::new(false),
                |batch| {
                    calls += 1;
                    assess(batch)
                },
            )
            .unwrap();
        assert!(!result.warm, "{changed} changes evaluation meaning");
        assert!(
            calls > 0,
            "{changed} must not reuse old evaluation responses"
        );
        assert_eq!(
            result.metrics.cache.input_hashes, 0,
            "{changed} does not change input content"
        );
        if changed != "media" {
            assert_eq!(
                result.metrics.probe_calls, 0,
                "{changed} does not change media probing"
            );
        }
        if changed == "context" {
            assert!(
                result.report.get("game_context").is_none(),
                "resolved context key dependency must not imply permission to publish it"
            );
        }
        verify_outputs(&paths.output_directory, &result.report, 1);
    }
}

#[test]
fn preflight_rejects_unowned_edited_overlapping_and_competing_outputs_without_callbacks() {
    let root = tempfile::tempdir().unwrap();
    let mut paths = setup_paths(root.path(), false);
    fs::create_dir(&paths.output_directory).unwrap();
    fs::write(
        paths.output_directory.join("notes.txt"),
        b"preserve user notes",
    )
    .unwrap();
    assert!(Session::prepare(&paths, &AtomicBool::new(false)).is_err());
    assert_eq!(
        fs::read(paths.output_directory.join("notes.txt")).unwrap(),
        b"preserve user notes"
    );
    fs::remove_file(paths.output_directory.join("notes.txt")).unwrap();
    fs::write(
        paths.output_directory.join("report.json"),
        br#"{"selected_count":1,"legacy_format":true}"#,
    )
    .unwrap();
    let old_report = fs::read(paths.output_directory.join("report.json")).unwrap();
    assert!(Session::prepare(&paths, &AtomicBool::new(false)).is_err());
    assert_eq!(
        fs::read(paths.output_directory.join("report.json")).unwrap(),
        old_report
    );
    fs::remove_dir_all(&paths.output_directory).unwrap();
    let original_output = paths.output_directory.clone();
    paths.output_directory = paths.input_directory.join("overlap");
    assert!(Session::prepare(&paths, &AtomicBool::new(false)).is_err());
    paths.output_directory = original_output;
    let session = Session::prepare(&paths, &AtomicBool::new(false)).unwrap();
    assert!(
        Session::prepare(&paths, &AtomicBool::new(false)).is_err(),
        "same output/cache cannot have competing sessions"
    );
    drop(session);
    let result = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &MediaTools::default(),
            &options(1, false, false),
            &conditions(),
            &AtomicBool::new(false),
            assess,
        )
        .unwrap();
    let mut report = result.report;
    report["selected"][0]["source"] = "edited-source.mkv".into();
    fs::write(
        paths.output_directory.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    let edited = snapshot(&paths.output_directory);
    assert!(
        Session::prepare(&paths, &AtomicBool::new(false)).is_err(),
        "report edits are not owned output"
    );
    assert_eq!(snapshot(&paths.output_directory), edited);
}

#[test]
fn valid_jpeg_edit_is_preserved_and_cache_loss_does_not_destroy_owned_output() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let options = options(1, false, false);
    let conditions = conditions();
    Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &AtomicBool::new(false),
            assess,
        )
        .unwrap();
    let valid = snapshot(&paths.output_directory);
    fs::remove_dir_all(paths.cache_directory.join(NAMESPACE)).unwrap();
    let recovered = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &unavailable_tools(),
            &options,
            &conditions,
            &AtomicBool::new(false),
            |_| panic!("owned output must survive cache removal without repeating inference"),
        )
        .unwrap();
    assert!(recovered.warm);
    assert_eq!(recovered.metrics.probe_calls, 0);
    assert_eq!(recovered.metrics.cache.input_hashes, 1);
    assert_eq!(snapshot(&paths.output_directory), valid);
    let error = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &unavailable_tools(),
            &options,
            &conditions,
            &AtomicBool::new(true),
            |_| panic!("cancelled run must not infer"),
        )
        .err()
        .unwrap();
    assert!(error.downcast_ref::<Interrupted>().is_some());
    assert_eq!(
        snapshot(&paths.output_directory),
        valid,
        "cache loss and cancellation retain normal owned output"
    );
    let selected = paths.output_directory.join("selected-01.jpg");
    let mut image = image::open(&selected).unwrap().to_rgb8();
    image.put_pixel(0, 0, image::Rgb([0, 255, 0]));
    image.save(&selected).unwrap();
    assert!(image::open(&selected).is_ok());
    let edited = snapshot(&paths.output_directory);
    assert!(
        Session::prepare(&paths, &AtomicBool::new(false)).is_err(),
        "even a valid same-name JPEG edit must not be overwritten"
    );
    assert_eq!(snapshot(&paths.output_directory), edited);
}

#[cfg(unix)]
#[test]
fn symlinks_hardlink_aliases_and_special_inputs_fail_before_media() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let source = paths.input_directory.join("01-blocks.mkv");
    let alias = paths.input_directory.join("alias.mkv");
    symlink(&source, &alias).unwrap();
    assert!(Session::prepare(&paths, &AtomicBool::new(false)).is_err());
    fs::remove_file(&alias).unwrap();
    fs::hard_link(&source, &alias).unwrap();
    assert!(
        Session::prepare(&paths, &AtomicBool::new(false)).is_err(),
        "path aliases must fail in preflight before probing"
    );
    fs::remove_file(&alias).unwrap();
    let fifo = paths.input_directory.join("special.mkv");
    let fifo_name = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(fifo_name.as_ptr(), 0o600) }, 0);
    assert!(Session::prepare(&paths, &AtomicBool::new(false)).is_err());
    fs::remove_file(&fifo).unwrap();
    symlink(&paths.input_directory, &paths.output_directory).unwrap();
    assert!(Session::prepare(&paths, &AtomicBool::new(false)).is_err());
    assert!(source.is_file(), "unsafe preflight must preserve the input");
}

fn checkpoints(paths: &RunPaths, kind: &str) -> Vec<PathBuf> {
    fs::read_dir(paths.cache_directory.join(NAMESPACE).join(kind))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect()
}

#[test]
fn malformed_misreferenced_and_invalid_batch_records_are_recomputed_when_used() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let mut options = options(1, false, false);
    let conditions = conditions();
    Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &AtomicBool::new(false),
            assess,
        )
        .unwrap();
    let primary_record = checkpoints(&paths, "assessments")
        .into_iter()
        .find(|path| {
            let value: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
            value["payload"].as_array().unwrap().len() > 3
        })
        .unwrap();
    for damage in ["json", "key", "assessment_id"] {
        let mut value: Value = serde_json::from_slice(&fs::read(&primary_record).unwrap()).unwrap();
        match damage {
            "json" => fs::write(&primary_record, b"{incomplete-checkpoint").unwrap(),
            "key" => {
                value["key"] = "0".repeat(64).into();
                fs::write(&primary_record, serde_json::to_vec(&value).unwrap()).unwrap();
            }
            "assessment_id" => {
                value["payload"][0]["id"] = "another-input-frame".into();
                value["payload_digest"] = game_screen_pick::storage::sha256(
                    &serde_json::to_vec(&value["payload"]).unwrap(),
                )
                .into();
                fs::write(&primary_record, serde_json::to_vec(&value).unwrap()).unwrap();
            }
            _ => unreachable!(),
        }
        if damage == "json" {
            let warm = Session::prepare(&paths, &AtomicBool::new(false))
                .unwrap()
                .run(
                    &unavailable_tools(),
                    &options,
                    &conditions,
                    &AtomicBool::new(false),
                    |_| panic!("warm must not inspect unused batch payloads"),
                )
                .unwrap();
            assert!(
                warm.warm,
                "unused checkpoints must not invalidate valid completed output"
            );
        }
        options.count = if options.count == 1 { 2 } else { 1 };
        let mut primary_calls = 0;
        let result = Session::prepare(&paths, &AtomicBool::new(false))
            .unwrap()
            .run(
                &MediaTools::default(),
                &options,
                &conditions,
                &AtomicBool::new(false),
                |batch| {
                    if batch.phase == AssessmentPhase::Primary {
                        primary_calls += 1;
                    }
                    assess(batch)
                },
            )
            .unwrap();
        assert_eq!(
            primary_calls, 1,
            "{damage} must not be a usable assessment hit"
        );
        assert_eq!(result.metrics.probe_calls, 0);
        assert_eq!(result.metrics.cache.input_hashes, 0);
        if damage == "assessment_id" {
            assert_eq!(result.metrics.cache.hits, 3);
            assert_eq!(
                result.metrics.cache.misses, 1,
                "a correctly encoded assessment with an unknown ID is a miss, not a hit"
            );
        }
        verify_outputs(&paths.output_directory, &result.report, options.count);
    }
}

#[test]
fn invalid_settings_and_context_fail_before_probe_or_any_assessment() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    for invalid in [
        "count",
        "interval",
        "backend",
        "context",
        "source",
        "credentials",
        "empty_resolved_context",
    ] {
        let mut options = options(1, false, false);
        let mut conditions = conditions();
        let reason = match invalid {
            "count" => {
                options.count = 0;
                "selected count"
            }
            "interval" => {
                options.context_delta_seconds = f64::NAN;
                "context interval"
            }
            "backend" => {
                options.model.backend = "unknown".to_owned();
                "backend"
            }
            "context" => {
                options.game_context = Some("different from resolved context".to_owned());
                "context"
            }
            "source" => {
                options.semantic_events = Some(vec![SemanticEvent {
                    source: "absent-source.mkv".to_owned(),
                    start_seconds: 0.0,
                    end_seconds: 1.0,
                    representative_seconds: 0.5,
                    importance: 80.0,
                    scene: "探索".to_owned(),
                    reason: "存在しない入力".to_owned(),
                }]);
                "unknown video source"
            }
            "credentials" => {
                conditions.service_url = "http://fixture-credential@127.0.0.1:8123/v1".to_owned();
                "credentials"
            }
            "empty_resolved_context" => {
                conditions.resolved_game_context = ResolvedGameContext::direct("");
                "cache conditions"
            }
            _ => unreachable!(),
        };
        let error = Session::prepare(&paths, &AtomicBool::new(false))
            .unwrap()
            .run(
                &unavailable_tools(),
                &options,
                &conditions,
                &AtomicBool::new(false),
                |_| panic!("invalid {invalid} must fail before inference"),
            )
            .err()
            .unwrap();
        assert!(
            error.to_string().contains(reason),
            "expected {reason}, got {error:#}"
        );
        assert!(!format!("{error:#}").contains("fixture-credential"));
        assert!(!paths.output_directory.exists());
    }
    assert!(
        !paths
            .cache_directory
            .join(NAMESPACE)
            .join("inputs")
            .exists(),
        "settings validation precedes even cold input hashing"
    );
    let mut paths = paths;
    let config = paths.cache_directory.parent().unwrap().join("invalid.toml");
    fs::write(&config, b"[ai\napi_key = 'never-print-fixture-key'").unwrap();
    paths.config = Some(config);
    let error = Session::prepare(&paths, &AtomicBool::new(false))
        .err()
        .unwrap();
    assert!(!format!("{error:#}").contains("never-print-fixture-key"));
}

fn understand(chunk: &ChunkArtifact) -> Result<Vec<SemanticEvent>> {
    understand_recipe(chunk, chunk.source.file_name().unwrap().to_str().unwrap())
}

fn understand_recipe(chunk: &ChunkArtifact, recipe: &str) -> Result<Vec<SemanticEvent>> {
    let tools = MediaTools::default();
    let cancelled = AtomicBool::new(false);
    let video = tools.probe(&chunk.path(), &cancelled)?;
    let source = chunk.source.file_name().unwrap().to_str().unwrap();
    for frame in [video.frames.first().unwrap(), video.frames.last().unwrap()] {
        let extracted = tools.extract(&video, frame.seconds, &cancelled)?;
        assert_eq!(
            image::open(extracted.path()).unwrap().to_rgb8(),
            expected(recipe, chunk.actual_start_seconds + frame.seconds),
            "semantic response requires the actual chunk's first/last frames and original times"
        );
    }
    assert_eq!((video.width, video.height), (chunk.width, chunk.height));
    Ok(vec![SemanticEvent {
        source: source.to_owned(),
        start_seconds: chunk.actual_start_seconds,
        end_seconds: chunk.actual_end_seconds,
        representative_seconds: (chunk.actual_start_seconds + chunk.actual_end_seconds) / 2.0,
        importance: 80.0,
        scene: "探索".to_owned(),
        reason: "chunk 原画像と元入力内の開始・終了時刻を照合".to_owned(),
    }])
}

#[test]
fn completed_semantic_chunk_survives_interruption_and_only_missing_chunk_repeats() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let source = paths.input_directory.join("01-blocks.mkv");
    let options = options(1, true, false);
    let conditions = conditions();
    let cancelled = AtomicBool::new(false);
    let error = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .semantic_chunk(
            &MediaTools::default(),
            &source,
            0.0,
            1.0,
            &options.model,
            &conditions,
            &cancelled,
            |chunk| {
                let events = understand(chunk)?;
                cancelled.store(true, Ordering::Relaxed);
                Ok(events)
            },
        )
        .err()
        .unwrap();
    assert!(error.downcast_ref::<Interrupted>().is_some());
    cancelled.store(false, Ordering::Relaxed);
    let mut session = Session::prepare(&paths, &AtomicBool::new(false)).unwrap();
    let first = session
        .semantic_chunk(
            &unavailable_tools(),
            &source,
            0.0,
            1.0,
            &options.model,
            &conditions,
            &cancelled,
            |_| panic!("completed semantic chunk must not repeat inference"),
        )
        .unwrap();
    let second = session
        .semantic_chunk(
            &MediaTools::default(),
            &source,
            4.0,
            5.0,
            &options.model,
            &conditions,
            &cancelled,
            understand,
        )
        .unwrap();
    assert_eq!(first[0].start_seconds, 0.0);
    assert_eq!(second[0].start_seconds, 4.0);
    drop(session);
    let missing = checkpoints(&paths, "semantic")
        .into_iter()
        .find(|path| {
            let record: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
            record["payload"][0]["start_seconds"] == 4.0
        })
        .unwrap();
    fs::remove_file(&missing).unwrap();
    let mut session = Session::prepare(&paths, &AtomicBool::new(false)).unwrap();
    session
        .semantic_chunk(
            &unavailable_tools(),
            &source,
            0.0,
            1.0,
            &options.model,
            &conditions,
            &cancelled,
            |_| panic!("unaffected chunk must remain cached"),
        )
        .unwrap();
    let mut calls = 0;
    let repeated = session
        .semantic_chunk(
            &MediaTools::default(),
            &source,
            4.0,
            5.0,
            &options.model,
            &conditions,
            &cancelled,
            |chunk| {
                calls += 1;
                understand(chunk)
            },
        )
        .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(
        serde_json::to_value(repeated).unwrap(),
        serde_json::to_value(second).unwrap()
    );
    session
        .semantic_chunk(
            &unavailable_tools(),
            &source,
            4.0,
            5.0,
            &options.model,
            &conditions,
            &cancelled,
            |_| panic!("repaired chunk must be reusable"),
        )
        .unwrap();
}

#[test]
fn generated_context_requires_persistence_permission_and_generation_dependency_match() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let model = options(1, false, false).model;
    let mut conditions = conditions();
    let cancelled = AtomicBool::new(false);
    let mut session = Session::prepare(&paths, &AtomicBool::new(false)).unwrap();
    for _ in 0..2 {
        let text = session
            .generated_context(
                "合成ゲーム",
                "brave-fixed-query-v1",
                &conditions,
                &model,
                false,
                &cancelled,
                || Ok("transient-context-not-approved-for-storage".to_owned()),
            )
            .unwrap();
        assert_eq!(text.text(), "transient-context-not-approved-for-storage");
    }
    assert!(
        !paths
            .cache_directory
            .join(NAMESPACE)
            .join("contexts")
            .exists()
    );
    let saved = session
        .generated_context(
            "合成ゲーム",
            "brave-fixed-query-v1",
            &conditions,
            &model,
            true,
            &cancelled,
            || Ok("保存可能と確認済みのゲーム説明".to_owned()),
        )
        .unwrap();
    let cached = session
        .generated_context(
            "合成ゲーム",
            "brave-fixed-query-v1",
            &conditions,
            &model,
            true,
            &cancelled,
            || panic!("approved identical generation should reuse saved context"),
        )
        .unwrap();
    assert_eq!(cached, saved);
    let mut denied_calls = 0;
    let transient = session
        .generated_context(
            "合成ゲーム",
            "brave-fixed-query-v1",
            &conditions,
            &model,
            false,
            &cancelled,
            || {
                denied_calls += 1;
                Ok("transient-new-context-without-read-or-write".to_owned())
            },
        )
        .unwrap();
    assert_eq!(
        denied_calls, 1,
        "permission false must not read even an existing approved checkpoint"
    );
    assert_ne!(transient, saved);
    assert_eq!(
        session
            .generated_context(
                "合成ゲーム",
                "brave-fixed-query-v1",
                &conditions,
                &model,
                true,
                &cancelled,
                || panic!("a denied run must not replace the approved checkpoint")
            )
            .unwrap(),
        saved
    );
    conditions.prompt_revision = "generation-search-prompt-v2".to_owned();
    let revised = session
        .generated_context(
            "合成ゲーム",
            "brave-fixed-query-v1",
            &conditions,
            &model,
            true,
            &cancelled,
            || Ok("新しい生成条件による説明".to_owned()),
        )
        .unwrap();
    assert_ne!(revised, saved);
    let other_title = session
        .generated_context(
            "別の合成ゲーム",
            "brave-fixed-query-v1",
            &conditions,
            &model,
            true,
            &cancelled,
            || Ok("別タイトルの説明".to_owned()),
        )
        .unwrap();
    assert_ne!(other_title, revised);
    let other_search = session
        .generated_context(
            "合成ゲーム",
            "brave-fixed-query-v2",
            &conditions,
            &model,
            true,
            &cancelled,
            || Ok("検索条件の変更後の説明".to_owned()),
        )
        .unwrap();
    assert_ne!(other_search, revised);
    for record in checkpoints(&paths, "contexts") {
        let text = fs::read_to_string(record).unwrap();
        assert!(!text.contains("transient-context-not-approved-for-storage"));
        assert!(!text.contains("transient-new-context-without-read-or-write"));
    }
}

fn sparse_vfr_source(paths: &RunPaths) -> PathBuf {
    let source = paths.input_directory.join("sparse-VFR.mkv");
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(fixture("01-blocks.mkv"))
        .args([
            "-vf",
            "select='eq(n,0)+eq(n,4)+eq(n,12)+eq(n,20)',setpts='if(eq(N,0),3,if(eq(N,1),3.25,if(eq(N,2),4.5,5.75)))/TB'",
            "-fps_mode", "passthrough", "-c:v", "ffv1",
        ])
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    source
}

fn inspect_sparse_chunk(chunk: &ChunkArtifact) {
    assert_eq!(chunk.source.file_name().unwrap(), "sparse-VFR.mkv");
    assert_eq!(
        (chunk.requested_start_seconds, chunk.requested_end_seconds),
        (0.1, 1.0)
    );
    assert_eq!(
        (chunk.actual_start_seconds, chunk.actual_end_seconds),
        (0.25, 1.5)
    );
    let tools = MediaTools::default();
    let cancelled = AtomicBool::new(false);
    let decoded = tools.probe(&chunk.path(), &cancelled).unwrap();
    assert_eq!(decoded.frames.len(), 1);
    assert_eq!(decoded.duration_seconds, 1.25);
    let frame = tools.extract(&decoded, 0.0, &cancelled).unwrap();
    assert_eq!(
        image::open(frame.path()).unwrap().to_rgb8(),
        expected("01-blocks.mkv", 1.0)
    );
}

fn sparse_event(start: f64, end: f64) -> Vec<SemanticEvent> {
    vec![SemanticEvent {
        source: "sparse-VFR.mkv".to_owned(),
        start_seconds: start,
        end_seconds: end,
        representative_seconds: (start + end) / 2.0,
        importance: 80.0,
        scene: "探索".to_owned(),
        reason: "実際の chunk 表示区間を元入力内の秒で確認".to_owned(),
    }]
}

#[test]
fn semantic_chunk_rejects_fresh_event_before_actual_start_inside_requested_range() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let source = sparse_vfr_source(&paths);
    let model = options(1, true, false).model;
    let result = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .semantic_chunk(
            &MediaTools::default(),
            &source,
            0.1,
            1.0,
            &model,
            &conditions(),
            &AtomicBool::new(false),
            |chunk| {
                inspect_sparse_chunk(chunk);
                Ok(sparse_event(0.1, 0.2))
            },
        );
    assert!(
        result.is_err(),
        "no pixels before actual start 0.25 were submitted for semantic inference"
    );
}

#[test]
fn semantic_chunk_rejects_cached_event_before_actual_start_and_recomputes_that_chunk() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let source = sparse_vfr_source(&paths);
    let model = options(1, true, false).model;
    Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .semantic_chunk(
            &MediaTools::default(),
            &source,
            0.1,
            1.0,
            &model,
            &conditions(),
            &AtomicBool::new(false),
            |chunk| {
                inspect_sparse_chunk(chunk);
                Ok(sparse_event(0.25, 0.75))
            },
        )
        .unwrap();
    let checkpoint = checkpoints(&paths, "semantic").pop().unwrap();
    let mut envelope: Value = serde_json::from_slice(&fs::read(&checkpoint).unwrap()).unwrap();
    envelope["payload"] = serde_json::to_value(sparse_event(0.1, 0.2)).unwrap();
    envelope["payload_digest"] =
        game_screen_pick::storage::sha256(&serde_json::to_vec(&envelope["payload"]).unwrap())
            .into();
    fs::write(&checkpoint, serde_json::to_vec(&envelope).unwrap()).unwrap();
    let mut calls = 0;
    let events = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .semantic_chunk(
            &MediaTools::default(),
            &source,
            0.1,
            1.0,
            &model,
            &conditions(),
            &AtomicBool::new(false),
            |chunk| {
                calls += 1;
                inspect_sparse_chunk(chunk);
                Ok(sparse_event(0.25, 0.75))
            },
        )
        .unwrap();
    assert_eq!(
        calls, 1,
        "a valid payload digest does not prove an event lies in the actual submitted chunk"
    );
    assert_eq!(events[0].start_seconds, 0.25);
}

#[test]
fn semantic_chunk_accepts_and_reuses_actual_trailing_display_beyond_requested_end() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let source = sparse_vfr_source(&paths);
    let model = options(1, true, false).model;
    let conditions = conditions();
    let events = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .semantic_chunk(
            &MediaTools::default(),
            &source,
            0.1,
            1.0,
            &model,
            &conditions,
            &AtomicBool::new(false),
            |chunk| {
                inspect_sparse_chunk(chunk);
                Ok(sparse_event(1.0, 1.5))
            },
        )
        .unwrap();
    assert_eq!(
        events[0].representative_seconds, 1.25,
        "events use original video seconds rather than chunk-relative timestamps"
    );
    let cached = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .semantic_chunk(
            &unavailable_tools(),
            &source,
            0.1,
            1.0,
            &model,
            &conditions,
            &AtomicBool::new(false),
            |_| panic!("the complete trailing display event should be a cache hit"),
        )
        .unwrap();
    assert_eq!(
        serde_json::to_value(cached).unwrap(),
        serde_json::to_value(events).unwrap()
    );
}

#[test]
fn generated_context_cold_empty_resolved_text_then_returned_text_reuses_generation() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let model = options(1, false, false).model;
    let mut conditions = conditions();
    conditions.resolved_game_context = ResolvedGameContext::direct("");
    let mut session = Session::prepare(&paths, &AtomicBool::new(false)).unwrap();
    let context = session
        .generated_context(
            "合成ゲーム",
            "brave-fixed-query-v1",
            &conditions,
            &model,
            true,
            &AtomicBool::new(false),
            || Ok("保存可能な生成結果。まだ生成前にこの値は存在しない".to_owned()),
        )
        .unwrap();
    conditions.resolved_game_context = context.clone();
    let mut calls = 0;
    let cached = session
        .generated_context(
            "合成ゲーム",
            "brave-fixed-query-v1",
            &conditions,
            &model,
            true,
            &AtomicBool::new(false),
            || {
                calls += 1;
                Ok("不要な二度目の生成".to_owned())
            },
        )
        .unwrap();
    assert_eq!(
        calls, 0,
        "generation's own output cannot be an input dependency of its cache key"
    );
    assert_eq!(cached, context);
}

#[test]
fn generated_context_returned_text_does_not_invalidate_generation_key() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let model = options(1, false, false).model;
    let mut conditions = conditions();
    let mut session = Session::prepare(&paths, &AtomicBool::new(false)).unwrap();
    let saved = session
        .generated_context(
            "合成ゲーム",
            "brave-fixed-query-v1",
            &conditions,
            &model,
            true,
            &AtomicBool::new(false),
            || Ok("以前の説明と異なる生成済みの説明".to_owned()),
        )
        .unwrap();
    conditions.resolved_game_context = saved.clone();
    let mut calls = 0;
    let reused = session
        .generated_context(
            "合成ゲーム",
            "brave-fixed-query-v1",
            &conditions,
            &model,
            true,
            &AtomicBool::new(false),
            || {
                calls += 1;
                Ok("生成結果を条件へ反映しても再生成不要".to_owned())
            },
        )
        .unwrap();
    assert_eq!(
        calls, 0,
        "resolved context only influences generation's consumers"
    );
    assert_eq!(reused, saved);
    for dependency in ["title", "search", "model", "prompt", "schema", "media"] {
        let mut changed_conditions = conditions.clone();
        let mut changed_model = model.clone();
        let mut title = "合成ゲーム";
        let mut search = "brave-fixed-query-v1";
        match dependency {
            "title" => title = "変更後のタイトル",
            "search" => search = "brave-fixed-query-v2",
            "model" => changed_model.model = "changed-generation-model".to_owned(),
            "prompt" => changed_conditions.prompt_revision = "changed-generation-prompt".to_owned(),
            "schema" => {
                changed_conditions.response_schema_revision = "changed-generation-schema".to_owned()
            }
            "media" => changed_conditions.media_revision = "changed-generation-media".to_owned(),
            _ => unreachable!(),
        }
        let mut calls = 0;
        let generated = session
            .generated_context(
                title,
                search,
                &changed_conditions,
                &changed_model,
                true,
                &AtomicBool::new(false),
                || {
                    calls += 1;
                    Ok(format!("生成条件の {dependency} 変更後の説明"))
                },
            )
            .unwrap();
        assert_eq!(
            calls, 1,
            "{dependency} changes generation input and must invalidate the old result"
        );
        changed_conditions.resolved_game_context = generated.clone();
        let warm = session.generated_context(
            title, search, &changed_conditions, &changed_model, true, &AtomicBool::new(false),
            || panic!("{dependency} changed result must become warm independently of returned text"),
        ).unwrap();
        assert_eq!(warm, generated);
    }
}

fn replace_with_mirrored_video(source: &Path) {
    let replacement = source.with_extension("replacement");
    fs::copy(fixture("02-mirrored.mkv"), &replacement).unwrap();
    fs::rename(replacement, source).unwrap();
}

#[test]
fn workflow_rejects_replaced_video_between_semantic_chunks_before_new_inference() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let source = paths.input_directory.join("01-blocks.mkv");
    let model = options(1, true, false).model;
    let conditions = conditions();
    let cancelled = AtomicBool::new(false);
    let mut session = Session::prepare(&paths, &AtomicBool::new(false)).unwrap();
    session
        .semantic_chunk(
            &MediaTools::default(),
            &source,
            0.0,
            1.0,
            &model,
            &conditions,
            &cancelled,
            understand,
        )
        .unwrap();
    replace_with_mirrored_video(&source);
    let mut calls = 0;
    let result = session.semantic_chunk(
        &MediaTools::default(),
        &source,
        4.0,
        5.0,
        &model,
        &conditions,
        &cancelled,
        |chunk| {
            calls += 1;
            understand_recipe(chunk, "02-mirrored.mkv")
        },
    );
    assert!(
        result.is_err(),
        "one semantic workflow must not combine events from different video contents"
    );
    assert_eq!(
        calls, 0,
        "replacement is rejected before decoding or new inference"
    );
    let error = session
        .semantic_chunk(
            &unavailable_tools(),
            &source,
            4.0,
            5.0,
            &model,
            &conditions,
            &cancelled,
            |_| panic!("stale workflow must not invoke inference"),
        )
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("input changed"),
        "must reject before trying unavailable probe tools: {error:#}"
    );
    assert!(!paths.output_directory.exists());
}

#[test]
fn workflow_rejects_replaced_video_between_semantic_events_and_run_preserving_output() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let conditions = conditions();
    let cancelled = AtomicBool::new(false);
    let mut options = options(1, false, false);
    Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &cancelled,
            assess,
        )
        .unwrap();
    let existing = snapshot(&paths.output_directory);
    let source = paths.input_directory.join("01-blocks.mkv");
    let mut session = Session::prepare(&paths, &AtomicBool::new(false)).unwrap();
    let events = session
        .semantic_chunk(
            &MediaTools::default(),
            &source,
            0.0,
            1.0,
            &options.model,
            &conditions,
            &cancelled,
            understand,
        )
        .unwrap();
    replace_with_mirrored_video(&source);
    options.semantic_events = Some(events);
    let mut calls = 0;
    let result = session.run(
        &MediaTools::default(),
        &options,
        &conditions,
        &cancelled,
        |batch| {
            calls += 1;
            assess_recipe(batch, Some("02-mirrored.mkv"))
        },
    );
    assert!(
        result.is_err(),
        "stale semantic evidence must not be published under the replacement video's digest"
    );
    assert_eq!(calls, 0);
    assert_eq!(snapshot(&paths.output_directory), existing);
}

#[test]
fn workflow_rejects_video_added_after_prepare_in_cold_and_warm_runs_before_media() {
    for warm in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let paths = setup_paths(root.path(), false);
        let options = options(1, false, false);
        let conditions = conditions();
        let cancelled = AtomicBool::new(false);
        let previous = warm.then(|| {
            Session::prepare(&paths, &AtomicBool::new(false))
                .unwrap()
                .run(
                    &MediaTools::default(),
                    &options,
                    &conditions,
                    &cancelled,
                    assess,
                )
                .unwrap();
            snapshot(&paths.output_directory)
        });
        let mut session = Session::prepare(&paths, &AtomicBool::new(false)).unwrap();
        fs::copy(
            fixture("02-mirrored.mkv"),
            paths.input_directory.join("02-mirrored.mkv"),
        )
        .unwrap();
        let error = session
            .run(
                &unavailable_tools(),
                &options,
                &conditions,
                &cancelled,
                |_| panic!("changed input membership must fail before inference"),
            )
            .err()
            .unwrap();
        assert!(
            error.to_string().contains("input directory changed"),
            "membership must fail before warm completion or any probe: {error:#}"
        );
        if let Some(previous) = previous {
            assert_eq!(snapshot(&paths.output_directory), previous);
        } else {
            assert!(!paths.output_directory.exists());
        }
    }
}

#[test]
fn workflow_rejects_video_added_during_assessment_and_keeps_previous_artifacts() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let conditions = conditions();
    let cancelled = AtomicBool::new(false);
    let mut options = options(1, false, false);
    Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &cancelled,
            assess,
        )
        .unwrap();
    let previous = snapshot(&paths.output_directory);
    options.count = 2;
    options.model.cache_revision = "membership-injection".to_owned();
    let mut calls = 0;
    let result = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &cancelled,
            |batch| {
                calls += 1;
                let response = assess(batch)?;
                if calls == 1 {
                    fs::copy(
                        fixture("02-mirrored.mkv"),
                        paths.input_directory.join("02-mirrored.mkv"),
                    )
                    .unwrap();
                }
                Ok(response)
            },
        );
    assert!(
        result.is_err(),
        "newly added recordings cannot silently be omitted from published results"
    );
    assert_eq!(
        calls, 1,
        "the changed workflow stops before any further inference"
    );
    assert_eq!(snapshot(&paths.output_directory), previous);
}

#[test]
fn workflow_rejects_video_added_during_semantic_callback_before_checkpointing() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let source = paths.input_directory.join("01-blocks.mkv");
    let model = options(1, true, false).model;
    let result = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .semantic_chunk(
            &MediaTools::default(),
            &source,
            0.0,
            1.0,
            &model,
            &conditions(),
            &AtomicBool::new(false),
            |chunk| {
                let events = understand(chunk)?;
                fs::copy(
                    fixture("02-mirrored.mkv"),
                    paths.input_directory.join("02-mirrored.mkv"),
                )
                .unwrap();
                Ok(events)
            },
        );
    assert!(
        result.is_err(),
        "completed responses are only valid for the workflow's verified input set"
    );
    assert!(
        !paths
            .cache_directory
            .join(NAMESPACE)
            .join("semantic")
            .exists()
    );
    assert!(!paths.output_directory.exists());
}

fn recover_shortened_probe_duration(final_frame: bool) {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let source = paths.input_directory.join("01-blocks.mkv");
    let options = options(1, false, false);
    let conditions = conditions();
    let cancelled = AtomicBool::new(false);
    let (start, end, damaged_frame) = if final_frame {
        (5.0, 6.0, 23)
    } else {
        (0.0, 1.0, 3)
    };
    let expected_events = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .semantic_chunk(
            &MediaTools::default(),
            &source,
            start,
            end,
            &options.model,
            &conditions,
            &cancelled,
            understand,
        )
        .unwrap();
    let checkpoint = checkpoints(&paths, "probe").pop().unwrap();
    let mut envelope: Value = serde_json::from_slice(&fs::read(&checkpoint).unwrap()).unwrap();
    envelope["payload"]["frames"][damaged_frame]["duration_seconds"] = 0.125.into();
    envelope["payload_digest"] =
        game_screen_pick::storage::sha256(&serde_json::to_vec(&envelope["payload"]).unwrap())
            .into();
    fs::write(&checkpoint, serde_json::to_vec(&envelope).unwrap()).unwrap();
    let mut session = Session::prepare(&paths, &AtomicBool::new(false)).unwrap();
    let recovered_events = session.semantic_chunk(
        &MediaTools::default(), &source, start, end, &options.model, &conditions,
        &cancelled,
        |_| panic!("valid semantic evidence must remain reusable after repairing corrupt probe duration"),
    ).unwrap();
    assert_eq!(
        serde_json::to_value(recovered_events).unwrap(),
        serde_json::to_value(expected_events).unwrap(),
    );
    let repaired: Value = serde_json::from_slice(&fs::read(&checkpoint).unwrap()).unwrap();
    assert_eq!(
        repaired["payload"]["frames"][damaged_frame]["duration_seconds"],
        0.25
    );
    let result = session
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &cancelled,
            assess,
        )
        .unwrap();
    assert_eq!(
        result.metrics.probe_calls, 1,
        "invalid duration is a miss requiring re-probe"
    );
    assert_eq!(
        result.metrics.semantic_calls, 0,
        "the already completed understanding stays reusable"
    );
    verify_outputs(&paths.output_directory, &result.report, 1);
}

#[test]
fn probe_duration_gap_is_recomputed_before_reusing_semantic_chunk() {
    recover_shortened_probe_duration(false);
}

#[test]
fn probe_duration_final_frame_is_recomputed_before_reusing_semantic_chunk() {
    recover_shortened_probe_duration(true);
}

fn assessment_bytes(batch: &AssessmentBatch<'_>) -> u64 {
    batch
        .candidates
        .iter()
        .map(|candidate| &candidate.image)
        .chain(
            batch
                .contexts
                .iter()
                .flat_map(|context| context.frames.iter().map(|frame| &frame.image)),
        )
        .chain(batch.sheets.iter())
        .map(|path| fs::metadata(path).unwrap().len())
        .sum()
}

#[test]
fn metrics_count_actual_assessment_media_in_cold_partial_and_warm_runs() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let mut options = options(1, false, false);
    let conditions = conditions();
    let cancelled = AtomicBool::new(false);
    let mut cold_bytes = 0;
    let mut primary_bytes = 0;
    let cold = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &cancelled,
            |batch| {
                let result = assess(batch)?;
                let bytes = assessment_bytes(batch);
                cold_bytes += bytes;
                if batch.phase == AssessmentPhase::Primary {
                    primary_bytes += bytes;
                }
                Ok(result)
            },
        )
        .unwrap();
    assert!(primary_bytes > 0);
    assert!(
        cold_bytes > primary_bytes,
        "secondary contexts and sheets must also contribute"
    );
    assert_eq!(
        serde_json::to_value(&cold.metrics).unwrap()["assessment_bytes_read"].as_u64(),
        Some(cold_bytes)
    );
    options.count = 2;
    let mut secondary_bytes = 0;
    let partial = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &cancelled,
            |batch| {
                assert_eq!(
                    batch.phase,
                    AssessmentPhase::Secondary,
                    "unchanged primary assessments are cached"
                );
                let result = assess(batch)?;
                secondary_bytes += assessment_bytes(batch);
                Ok(result)
            },
        )
        .unwrap();
    assert_eq!(partial.metrics.probe_calls, 0);
    assert_eq!(partial.metrics.cache.input_hashes, 0);
    assert_eq!(
        serde_json::to_value(&partial.metrics).unwrap()["assessment_bytes_read"].as_u64(),
        Some(primary_bytes + secondary_bytes),
        "cached primary batches still read their exact media for key verification"
    );
    verify_outputs(&paths.output_directory, &partial.report, 2);
    let warm = Session::prepare(&paths, &AtomicBool::new(false))
        .unwrap()
        .run(
            &unavailable_tools(),
            &options,
            &conditions,
            &cancelled,
            |_| panic!("completed warm output must not load assessment media"),
        )
        .unwrap();
    assert!(warm.warm);
    assert_eq!(
        serde_json::to_value(&warm.metrics).unwrap()["assessment_bytes_read"].as_u64(),
        Some(0)
    );
}

#[test]
fn generation_provenance_changes_reassess_identical_text_and_then_reuse_completed_output() {
    for dependency in ["title", "search", "model"] {
        let root = tempfile::tempdir().unwrap();
        let paths = setup_paths(root.path(), false);
        let options = options(1, false, false);
        let mut conditions = conditions();
        let generation = conditions.clone();
        let text = conditions.resolved_game_context.text().to_owned();
        let cancelled = AtomicBool::new(false);
        let mut session = Session::prepare(&paths, &cancelled).unwrap();
        conditions.resolved_game_context = session
            .generated_context(
                "合成ゲーム",
                "search-v1",
                &generation,
                &options.model,
                true,
                &cancelled,
                || Ok(text.clone()),
            )
            .unwrap();
        let cold = session
            .run(
                &MediaTools::default(),
                &options,
                &conditions,
                &cancelled,
                assess,
            )
            .unwrap();
        drop(session);
        let mut title = "合成ゲーム";
        let mut search = "search-v1";
        let mut model = options.model.clone();
        match dependency {
            "title" => title = "新しいタイトル",
            "search" => search = "search-v2",
            "model" => model.model = "different-generation-model".to_owned(),
            _ => unreachable!(),
        }
        let persist = dependency != "search";
        let mut generation_calls = 0;
        let before_contexts = checkpoints(&paths, "contexts").len();
        let mut session = Session::prepare(&paths, &cancelled).unwrap();
        conditions.resolved_game_context = session
            .generated_context(
                title,
                search,
                &generation,
                &model,
                persist,
                &cancelled,
                || {
                    generation_calls += 1;
                    Ok(text.clone())
                },
            )
            .unwrap();
        assert_eq!(generation_calls, 1);
        if !persist {
            assert_eq!(checkpoints(&paths, "contexts").len(), before_contexts);
        }
        let changed = session
            .run(
                &MediaTools::default(),
                &options,
                &conditions,
                &cancelled,
                assess,
            )
            .unwrap();
        assert!(
            !changed.warm,
            "{dependency} changed generation input despite returning identical text"
        );
        assert_ne!(changed.report["cache"]["key"], cold.report["cache"]["key"]);
        assert!(changed.metrics.assessment_calls > 0);
        assert_eq!(changed.metrics.probe_calls, 0);
        assert_eq!(changed.metrics.cache.input_hashes, 0);
        verify_outputs(&paths.output_directory, &changed.report, 1);
        drop(session);
        let mut session = Session::prepare(&paths, &cancelled).unwrap();
        conditions.resolved_game_context = session
            .generated_context(
                title,
                search,
                &generation,
                &model,
                persist,
                &cancelled,
                || {
                    assert!(
                        !persist,
                        "approved identical generation must hit its checkpoint"
                    );
                    Ok(text.clone())
                },
            )
            .unwrap();
        let warm = session
            .run(
                &unavailable_tools(),
                &options,
                &conditions,
                &cancelled,
                |_| panic!("unchanged generation provenance must keep completed output warm"),
            )
            .unwrap();
        assert!(warm.warm);
        assert_eq!(warm.metrics.assessment_calls, 0);
        assert_eq!(warm.report, changed.report);
    }
}

#[test]
fn generation_provenance_changes_semantic_dependencies_without_reprobing_input() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let source = paths.input_directory.join("01-blocks.mkv");
    let options = options(1, false, false);
    let mut conditions = conditions();
    let generation = conditions.clone();
    let text = conditions.resolved_game_context.text().to_owned();
    let cancelled = AtomicBool::new(false);
    let mut session = Session::prepare(&paths, &cancelled).unwrap();
    conditions.resolved_game_context = session
        .generated_context(
            "合成ゲーム",
            "search-v1",
            &generation,
            &options.model,
            true,
            &cancelled,
            || Ok(text.clone()),
        )
        .unwrap();
    let original = session
        .semantic_chunk(
            &MediaTools::default(),
            &source,
            0.0,
            1.0,
            &options.model,
            &conditions,
            &cancelled,
            understand,
        )
        .unwrap();
    drop(session);
    let mut session = Session::prepare(&paths, &cancelled).unwrap();
    conditions.resolved_game_context = session
        .generated_context(
            "合成ゲーム",
            "search-v1",
            &generation,
            &options.model,
            true,
            &cancelled,
            || panic!("same generated context must hit"),
        )
        .unwrap();
    session
        .semantic_chunk(
            &unavailable_tools(),
            &source,
            0.0,
            1.0,
            &options.model,
            &conditions,
            &cancelled,
            |_| panic!("same provenance must reuse understanding"),
        )
        .unwrap();
    let mut generation_model = options.model.clone();
    generation_model.cache_revision = "generation-v2".to_owned();
    conditions.resolved_game_context = session
        .generated_context(
            "合成ゲーム",
            "search-v1",
            &generation,
            &generation_model,
            false,
            &cancelled,
            || Ok(text.clone()),
        )
        .unwrap();
    let mut calls = 0;
    let changed = session
        .semantic_chunk(
            &MediaTools::default(),
            &source,
            0.0,
            1.0,
            &options.model,
            &conditions,
            &cancelled,
            |chunk| {
                calls += 1;
                understand(chunk)
            },
        )
        .unwrap();
    assert_eq!(
        calls, 1,
        "changed generation provenance invalidates dependent understanding"
    );
    assert_eq!(
        serde_json::to_value(original).unwrap(),
        serde_json::to_value(changed).unwrap()
    );
    assert_eq!(checkpoints(&paths, "semantic").len(), 2);
    let result = session
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &cancelled,
            assess,
        )
        .unwrap();
    assert_eq!(result.metrics.probe_calls, 0);
    assert_eq!(result.metrics.cache.input_hashes, 0);
}

#[test]
fn direct_context_does_not_inherit_unused_generation_provenance() {
    let root = tempfile::tempdir().unwrap();
    let paths = setup_paths(root.path(), false);
    let source = paths.input_directory.join("01-blocks.mkv");
    let options = options(1, false, false);
    let mut conditions = conditions();
    let cancelled = AtomicBool::new(false);
    let mut session = Session::prepare(&paths, &cancelled).unwrap();
    let cold = session
        .run(
            &MediaTools::default(),
            &options,
            &conditions,
            &cancelled,
            assess,
        )
        .unwrap();
    let semantic = session
        .semantic_chunk(
            &MediaTools::default(),
            &source,
            0.0,
            1.0,
            &options.model,
            &conditions,
            &cancelled,
            understand,
        )
        .unwrap();
    let generated = session
        .generated_context(
            "合成ゲーム",
            "search-v1",
            &conditions,
            &options.model,
            false,
            &cancelled,
            || Ok(conditions.resolved_game_context.text().to_owned()),
        )
        .unwrap();
    // An explicitly direct value has no generated identity, even with equal text.
    conditions.resolved_game_context = ResolvedGameContext::direct(generated.text());
    session
        .generated_context(
            "無関係なゲーム",
            "search-v2",
            &conditions,
            &options.model,
            false,
            &cancelled,
            || Ok("無関係な説明".to_owned()),
        )
        .unwrap();
    let cached = session
        .semantic_chunk(
            &unavailable_tools(),
            &source,
            0.0,
            1.0,
            &options.model,
            &conditions,
            &cancelled,
            |_| panic!("unused generated values cannot invalidate direct semantic evidence"),
        )
        .unwrap();
    assert_eq!(
        serde_json::to_value(cached).unwrap(),
        serde_json::to_value(semantic).unwrap()
    );
    let warm = session
        .run(
            &unavailable_tools(),
            &options,
            &conditions,
            &cancelled,
            |_| panic!("direct context must retain completed output after unused generation"),
        )
        .unwrap();
    assert!(warm.warm);
    assert_eq!(warm.report, cold.report);
    assert!(
        !paths
            .cache_directory
            .join(NAMESPACE)
            .join("contexts")
            .exists()
    );
}

#[test]
fn cancelled_cold_and_warm_sessions_stop_before_preflight_or_reuse() {
    for warm in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let paths = setup_paths(root.path(), false);
        let options = options(1, false, false);
        let conditions = conditions();
        let cancelled = AtomicBool::new(false);
        if warm {
            Session::prepare(&paths, &cancelled)
                .unwrap()
                .run(
                    &MediaTools::default(),
                    &options,
                    &conditions,
                    &cancelled,
                    assess,
                )
                .unwrap();
        }
        let before = fs::read_dir(root.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<std::collections::BTreeSet<_>>();
        let previous = warm.then(|| snapshot(&paths.output_directory));
        cancelled.store(true, Ordering::Relaxed);
        let error = Session::prepare(&paths, &cancelled).err().unwrap();
        assert!(error.downcast_ref::<Interrupted>().is_some());
        let after = fs::read_dir(root.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            before, after,
            "cancelled preflight does not create lock/cache/output files"
        );
        if let Some(previous) = previous {
            assert_eq!(snapshot(&paths.output_directory), previous);
            cancelled.store(false, Ordering::Relaxed);
            let mut session = Session::prepare(&paths, &cancelled).unwrap();
            cancelled.store(true, Ordering::Relaxed);
            let error = session
                .run(
                    &unavailable_tools(),
                    &options,
                    &conditions,
                    &cancelled,
                    |_| panic!("cancelled warm reuse must not assess"),
                )
                .err()
                .unwrap();
            assert!(error.downcast_ref::<Interrupted>().is_some());
            assert_eq!(snapshot(&paths.output_directory), previous);
            cancelled.store(false, Ordering::Relaxed);
            assert!(
                session
                    .run(
                        &unavailable_tools(),
                        &options,
                        &conditions,
                        &cancelled,
                        |_| panic!("cancellation must preserve reusable completed output"),
                    )
                    .unwrap()
                    .warm
            );
        } else {
            assert!(!paths.cache_directory.join(NAMESPACE).exists());
            assert!(!paths.output_directory.exists());
        }
    }
}
