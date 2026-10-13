use std::fs;
use std::path::Path;
use std::sync::atomic::AtomicBool;

use game_screen_pick::media::MediaTools;
use game_screen_pick::output::Output;
use game_screen_pick::pipeline::{ModelConditions, PipelineOptions, run};
use game_screen_pick::selection::Assessment;
use serde_json::{Value, json};

#[test]
fn publication_rejects_report_edits_against_the_pipeline_record() {
    let workspace = tempfile::tempdir().unwrap();
    let root = workspace.path().canonicalize().unwrap();
    let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/videos/01-blocks.mkv");
    let cancelled = AtomicBool::new(false);
    let tools = MediaTools::default();
    let video = tools.probe(&input, &cancelled).unwrap();
    let options = PipelineOptions {
        count: 1,
        samples_per_video: 6,
        context_delta_seconds: 0.25,
        semantic_events: None,
        model: ModelConditions {
            backend: "strata".to_owned(),
            model: "fixture-vision".to_owned(),
            inference_level: "high".to_owned(),
            timeout_seconds: 60,
            cache_revision: "test".to_owned(),
        },
        game_context: Some("fixture context".to_owned()),
    };
    let artifacts = run(&tools, &[video], &options, &cancelled, |batch| {
        Ok(batch
            .candidates
            .iter()
            .map(|candidate| Assessment {
                id: candidate.id.clone(),
                blog_score: 80.0,
                transition: false,
                scene: "fixture scene".to_owned(),
                reason: "fixture reason".to_owned(),
                special: false,
            })
            .collect())
    })
    .unwrap();
    let report_path = artifacts.path().join("report.json");
    let original: Value = serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
    let mut output = Output::prepare(&root.join("result"), &[&input], &cancelled).unwrap();
    let owned = output.publish(&artifacts, &cancelled).unwrap();
    let before_report = fs::read(output.path().join("report.json")).unwrap();
    let before_image = fs::read(output.path().join("selected-01.jpg")).unwrap();
    for (pointer, replacement) in [
        ("/sources/0", json!("invented.mkv")),
        ("/selected/0/source", json!("invented.mkv")),
        ("/selected/0/requested_seconds", json!(9998.0)),
        ("/selected/0/actual_seconds", json!(9999.0)),
        (
            "/selected/0/primary_assessment/reason",
            json!("edited primary reason"),
        ),
        ("/selected/0/assessment/reason", json!("edited reason")),
        ("/selected/0/assessment/scene", json!("edited scene")),
        ("/selected/0/assessment/blog_score", json!(99.0)),
        ("/model/model", json!("invented model")),
        ("/game_context", json!("edited context")),
    ] {
        let mut edited = original.clone();
        *edited.pointer_mut(pointer).unwrap() = replacement;
        fs::write(&report_path, serde_json::to_vec(&edited).unwrap()).unwrap();
        let error = output.publish(&artifacts, &cancelled).unwrap_err();
        assert!(
            error.to_string().contains("artifact record"),
            "{pointer}: {error:#}"
        );
        assert_eq!(
            output
                .verify_existing(&cancelled)
                .unwrap()
                .unwrap()
                .fingerprint,
            owned.fingerprint
        );
        assert_eq!(
            fs::read(output.path().join("report.json")).unwrap(),
            before_report
        );
        assert_eq!(
            fs::read(output.path().join("selected-01.jpg")).unwrap(),
            before_image
        );
    }
    let mut with_cache = original;
    with_cache["cache"] = json!({"namespace": "game-screen-pick-rust-v1", "key": "fixture"});
    fs::write(
        &report_path,
        serde_json::to_vec_pretty(&with_cache).unwrap(),
    )
    .unwrap();
    let owned = output.publish(&artifacts, &cancelled).unwrap();
    assert_eq!(owned.report["cache"], with_cache["cache"]);
}

#[test]
fn output_lock_probe() {
    let Some(path) = std::env::var_os("GSP_OUTPUT_LOCK_PROBE") else {
        return;
    };
    let result = Output::prepare(Path::new(&path), &[], &AtomicBool::new(false));
    assert_eq!(
        result.is_ok(),
        std::env::var("GSP_OUTPUT_LOCK_EXPECTED").unwrap() == "available"
    );
}

#[test]
fn output_lock_excludes_a_separate_process_and_releases_on_drop() {
    let workspace = tempfile::tempdir().unwrap();
    let path = workspace.path().canonicalize().unwrap().join("result");
    let output = Output::prepare(&path, &[], &AtomicBool::new(false)).unwrap();
    let probe = |expected: &str| {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "output_lock_probe", "--nocapture"])
            .env("GSP_OUTPUT_LOCK_PROBE", &path)
            .env("GSP_OUTPUT_LOCK_EXPECTED", expected)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
    };
    probe("locked");
    drop(output);
    probe("available");
}

#[test]
fn real_pipeline_images_sheet_and_rank_metadata_survive_safe_publication() {
    let workspace = tempfile::tempdir().unwrap();
    let root = workspace.path().canonicalize().unwrap();
    let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/videos/01-blocks.mkv");
    let cancelled = AtomicBool::new(false);
    let tools = MediaTools::default();
    let mut output =
        Output::prepare(&root.join("result"), &[&input], &AtomicBool::new(false)).unwrap();
    let video = tools.probe(&input, &cancelled).unwrap();
    let options = PipelineOptions {
        count: 2,
        samples_per_video: 6,
        context_delta_seconds: 0.25,
        semantic_events: None,
        model: ModelConditions {
            backend: "strata".to_owned(),
            model: "fixture-vision".to_owned(),
            inference_level: "high".to_owned(),
            timeout_seconds: 60,
            cache_revision: "test".to_owned(),
        },
        game_context: Some("既知のゲーム画面の探索・会話".to_owned()),
    };
    let artifacts = run(&tools, &[video], &options, &cancelled, |batch| {
        Ok(batch
            .candidates
            .iter()
            .map(|candidate| {
                assert_eq!(candidate.source, "01-blocks.mkv");
                assert_eq!(
                    image::open(&candidate.image)
                        .unwrap()
                        .into_rgb8()
                        .dimensions(),
                    (160, 96)
                );
                Assessment {
                    id: candidate.id.clone(),
                    blog_score: 80.0,
                    transition: false,
                    scene: format!("通常進行 {}", candidate.actual_seconds.floor()),
                    reason: "合成ブロックが明瞭".to_owned(),
                    special: false,
                }
            })
            .collect())
    })
    .unwrap();
    let before = fs::read(artifacts.path().join("selected-contact-sheet.jpg")).unwrap();
    let owned = output.publish(&artifacts, &cancelled).unwrap();
    assert_eq!(owned.report["selected_count"], 2);
    assert_eq!(
        owned.report["publication"]["files"]
            .as_object()
            .unwrap()
            .len(),
        3
    );
    assert_eq!(
        fs::read(output.path().join("selected-contact-sheet.jpg")).unwrap(),
        before
    );
    for (index, frame) in owned.report["selected"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        assert_eq!(frame["rank"], index + 1);
        assert_eq!(frame["image"], format!("selected-{:02}.jpg", index + 1));
        let name = frame["image"].as_str().unwrap();
        assert_eq!(
            fs::read(output.path().join(name)).unwrap(),
            fs::read(artifacts.path().join(name)).unwrap()
        );
        let image = image::open(output.path().join(name)).unwrap().into_rgb8();
        assert_eq!(image.dimensions(), (160, 96));
        // Independent recipe: the upper-left interior tile is listed by second.
        let second = frame["actual_seconds"].as_f64().unwrap().floor() as usize;
        let expected = include_str!("fixtures/video-tiles.tsv")
            .lines()
            .filter(|line| !line.starts_with('#') && !line.starts_with("second"))
            .nth(second * 6)
            .unwrap()
            .split('\t')
            .nth(2)
            .unwrap()
            .parse::<u8>()
            .unwrap();
        assert!(image.get_pixel(8, 8)[0].abs_diff(expected) <= 18);
    }
    let fingerprint = owned.fingerprint;
    let path = output.path().to_owned();
    drop(output);
    drop(artifacts);
    let output = Output::prepare(&path, &[&input], &AtomicBool::new(false)).unwrap();
    assert!(output.bytes_read() > 0);
    assert_eq!(
        output
            .verify_existing(&AtomicBool::new(false))
            .unwrap()
            .unwrap()
            .fingerprint,
        fingerprint
    );
    if let Some(qa) = std::env::var_os("GSP_QA_DIR") {
        let qa = Path::new(&qa).join("publication");
        fs::create_dir_all(&qa).unwrap();
        for entry in fs::read_dir(output.path()).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), qa.join(entry.file_name())).unwrap();
        }
    }
}
