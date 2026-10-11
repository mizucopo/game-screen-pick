use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail};
use game_screen_pick::media::{Interrupted, MediaTools, Video};
use game_screen_pick::pipeline::{
    AssessmentBatch, AssessmentPhase, ModelConditions, PipelineOptions, run,
};
use game_screen_pick::selection::{Assessment, SemanticEvent};
use image::RgbImage;
use serde_json::Value;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/videos")
        .join(name)
}

fn script() -> Value {
    serde_json::from_str(include_str!("fixtures/responses.json")).unwrap()
}

fn options(count: usize, semantic: bool, multiple: bool) -> PipelineOptions {
    let fixture = script();
    let semantic_events = semantic.then(|| {
        fixture["semantic_video"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|item| multiple || item["source"] == "01-blocks.mkv")
            .flat_map(|item| {
                item["events"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|event| SemanticEvent {
                        source: item["source"].as_str().unwrap().to_owned(),
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
        semantic_events,
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

fn probe(multiple: bool) -> Vec<Video> {
    let tools = MediaTools::default();
    let cancelled = AtomicBool::new(false);
    let mut videos = vec![tools.probe(&fixture("01-blocks.mkv"), &cancelled).unwrap()];
    if multiple {
        videos.push(
            tools
                .probe(&fixture("02-mirrored.mkv"), &cancelled)
                .unwrap(),
        );
    }
    videos
}

// Input recipe, independent of extraction, assessment and final selection.
fn expected(source: &str, seconds: f64) -> RgbImage {
    let rows = include_str!("fixtures/video-tiles.tsv")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.starts_with("second"))
        .map(|line| {
            line.split('\t')
                .skip(2)
                .map(|number| number.parse::<u8>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mirrored = source == "02-mirrored.mkv";
    let rotated = source == "03-multitrack-rotated.mov";
    let (width, height) = if rotated { (96, 160) } else { (160, 96) };
    RgbImage::from_fn(width, height, |x, y| {
        let (x, y) = if rotated { (159 - y, x) } else { (x, y) };
        let tile = if mirrored { 9 - x / 16 } else { x / 16 };
        image::Rgb([rows[seconds.floor() as usize * 6 + y as usize / 16][tile as usize]; 3])
    })
}

fn assert_thumbnail(sheet: &RgbImage, origin: (u32, u32), size: (u32, u32), expected: &RgbImage) {
    let ratio = (f64::from(size.0) / f64::from(expected.width()))
        .min(f64::from(size.1) / f64::from(expected.height()));
    let width = (f64::from(expected.width()) * ratio).round() as u32;
    let height = (f64::from(expected.height()) * ratio).round() as u32;
    let x = origin.0 + (size.0 - width) / 2;
    let y = origin.1 + (size.1 - height) / 2;
    // Interior points of nine known blocks avoid resizing/JPEG boundary effects.
    for source_y in [8, expected.height() / 2 - 8, expected.height() - 8] {
        for source_x in [8, expected.width() / 2 - 8, expected.width() - 8] {
            let actual = sheet.get_pixel(
                x + source_x * width / expected.width(),
                y + source_y * height / expected.height(),
            );
            let wanted = expected.get_pixel(source_x, source_y);
            for channel in 0..3 {
                assert!(
                    actual[channel].abs_diff(wanted[channel]) <= 18,
                    "sheet must display matching source/time/context image at ({x},{y}), source point ({source_x},{source_y}): {actual:?} vs {wanted:?}"
                );
            }
        }
    }
}

fn verify_batch(batch: &AssessmentBatch<'_>) {
    assert!(!batch.candidates.is_empty());
    assert!(batch.candidates.len() <= 12);
    assert!(!batch.sheets.is_empty());
    for sheet in &batch.sheets {
        let image = image::open(sheet).unwrap().to_rgb8();
        assert!(image.width() > 160 && image.height() > 96);
    }
    for candidate in batch.candidates {
        assert_eq!(
            candidate.actual_seconds * 4.0,
            (candidate.actual_seconds * 4.0).round()
        );
        assert!(
            candidate.actual_seconds >= candidate.requested_seconds
                || candidate.actual_seconds == candidate.duration_seconds - 0.25
        );
        let ordinal = if candidate.source == "02-mirrored.mkv" {
            2
        } else {
            1
        };
        assert_eq!(
            candidate.id,
            format!(
                "V{ordinal:04}-F{:08}",
                (candidate.actual_seconds * 4.0) as usize
            )
        );
        assert_eq!(
            image::open(&candidate.image).unwrap().to_rgb8(),
            expected(&candidate.source, candidate.actual_seconds)
        );
        assert!(candidate.quality.rejection.is_none());
    }
    match batch.phase {
        AssessmentPhase::Primary => assert!(batch.contexts.is_empty()),
        AssessmentPhase::Secondary => {
            assert_eq!(batch.contexts.len(), batch.candidates.len());
            for (context, candidate) in batch.contexts.iter().zip(batch.candidates) {
                assert_eq!(context.id, candidate.id);
                let requested = [
                    (candidate.requested_seconds - 0.25).max(0.0),
                    candidate.requested_seconds,
                    (candidate.requested_seconds + 0.25).min(candidate.duration_seconds),
                ];
                for (frame, requested) in context.frames.iter().zip(requested) {
                    assert_eq!(frame.source, candidate.source);
                    assert_eq!(frame.requested_seconds, requested);
                    assert_eq!(
                        image::open(&frame.image).unwrap().to_rgb8(),
                        expected(&frame.source, frame.actual_seconds)
                    );
                }
                assert_eq!(context.frames[1].actual_seconds, candidate.actual_seconds);
            }
        }
    }
    // Verify the sheet's image ordering before returning scripted scores. Layout
    // comes from the bundled font metrics; short corpus labels occupy six lines.
    let font = fontdue::Font::from_bytes(
        include_bytes!("../assets/NotoSansJP-Regular.otf").as_slice(),
        fontdue::FontSettings::default(),
    )
    .unwrap();
    let line = font
        .horizontal_line_metrics(16.0)
        .unwrap()
        .new_line_size
        .ceil() as u32;
    let small_line = font
        .horizontal_line_metrics(14.0)
        .unwrap()
        .new_line_size
        .ceil() as u32;
    let context_height = if batch.phase == AssessmentPhase::Secondary {
        55 + small_line + 12
    } else {
        0
    };
    let stride = line * 6 + 180 + context_height + 12 + 24 + 12;
    let sheet = image::open(&batch.sheets[0]).unwrap().to_rgb8();
    for (index, candidate) in batch.candidates.iter().enumerate() {
        let x = 24 + index as u32 % 3 * 356;
        let y = 24 + line + index as u32 / 3 * stride;
        assert_thumbnail(
            &sheet,
            (x, y),
            (320, 180),
            &expected(&candidate.source, candidate.actual_seconds),
        );
        if batch.phase == AssessmentPhase::Secondary {
            for (ordinal, frame) in batch.contexts[index].frames.iter().enumerate() {
                assert_thumbnail(
                    &sheet,
                    (x + ordinal as u32 * 110, y + 180 + 12 + line * 5),
                    (98, 55),
                    &expected(&frame.source, frame.actual_seconds),
                );
            }
        }
    }
}

fn fixed_assessments(batch: &AssessmentBatch<'_>) -> Result<Vec<Assessment>> {
    // Verify the actual candidate/context media before issuing the fixed response.
    verify_batch(batch);
    let fixture = script();
    let phase = if batch.phase == AssessmentPhase::Primary {
        "primary"
    } else {
        "secondary"
    };
    let mut assessments = batch
        .candidates
        .iter()
        .map(|candidate| {
            let matching = fixture["candidates"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| {
                    item["source"] == candidate.source
                        && item["time_seconds"].as_f64().unwrap() == candidate.actual_seconds
                });
            let response = matching.and_then(|item| {
                fixture[phase]["frames"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|response| response["id"] == item["id"])
            });
            if let Some(response) = response {
                Assessment {
                    id: candidate.id.clone(),
                    blog_score: response["blog_score"].as_f64().unwrap(),
                    transition: response["transition"].as_bool().unwrap(),
                    scene: response["scene"].as_str().unwrap().to_owned(),
                    reason: response["reason"].as_str().unwrap().to_owned(),
                    special: false,
                }
            } else {
                Assessment {
                    id: candidate.id.clone(),
                    blog_score: 55.0,
                    transition: false,
                    scene: format!("通常進行 {}", candidate.actual_seconds.floor()),
                    reason: "既知の合成ブロックが明瞭".to_owned(),
                    special: false,
                }
            }
        })
        .collect::<Vec<_>>();
    assessments.reverse(); // A valid response order must not change ID/report mapping.
    Ok(assessments)
}

// Optional local visual QA evidence from the same verified product run.
fn retain_qa(workspace: &Path, case: &str) {
    let Some(root) = std::env::var_os("GSP_QA_DIR") else {
        return;
    };
    let destination = PathBuf::from(root).join(case);
    fs::create_dir_all(&destination).unwrap();
    for directory in fs::read_dir(workspace).unwrap() {
        let directory = directory.unwrap();
        if !directory.file_type().unwrap().is_dir() {
            continue;
        }
        for file in fs::read_dir(directory.path()).unwrap() {
            let file = file.unwrap();
            if file
                .path()
                .extension()
                .is_some_and(|extension| extension == "jpg" || extension == "json")
            {
                let target = destination.join(directory.file_name());
                fs::create_dir_all(&target).unwrap();
                fs::copy(file.path(), target.join(file.file_name())).unwrap();
            }
        }
    }
}

#[test]
fn both_methods_single_and_multiple_inputs_produce_real_ranked_artifacts() {
    let tools = MediaTools::default();
    for semantic in [false, true] {
        for multiple in [false, true] {
            let options = options(2, semantic, multiple);
            let videos = probe(multiple);
            let artifacts = run(
                &tools,
                &videos,
                &options,
                &AtomicBool::new(false),
                fixed_assessments,
            )
            .unwrap();
            let path = artifacts.path();
            retain_qa(
                path.parent().unwrap(),
                &format!(
                    "{}-{}",
                    if semantic { "semantic" } else { "sampled" },
                    if multiple { "multiple" } else { "single" }
                ),
            );
            let report: Value =
                serde_json::from_slice(&fs::read(path.join("report.json")).unwrap()).unwrap();
            assert_eq!(
                report["method"],
                if semantic {
                    "semantic_video"
                } else {
                    "sampled_frames"
                }
            );
            assert_eq!(report["requested_count"], 2);
            assert_eq!(report["selected_count"], 2);
            assert_eq!(report["model"]["model"], "fixture-vision");
            assert_eq!(report["game_context"], script()["game_context"]);
            assert_eq!(artifacts.report().selected.len(), 2);
            let mut sources = Vec::new();
            for (ordinal, selected) in artifacts.report().selected.iter().enumerate() {
                assert_eq!(selected.rank, ordinal + 1);
                assert_eq!(selected.image, format!("selected-{:02}.jpg", ordinal + 1));
                assert_eq!(selected.id, selected.assessment.id);
                assert_eq!(selected.id, selected.primary_assessment.id);
                assert_ne!(selected.actual_seconds.floor(), 2.0);
                assert_eq!(selected.event.is_some(), semantic);
                if let Some(event) = &selected.event {
                    assert_eq!(event.source, selected.source);
                    assert!(
                        (event.start_seconds..=event.end_seconds)
                            .contains(&selected.actual_seconds)
                    );
                    assert!(!event.reason.is_empty());
                }
                let image = image::open(path.join(&selected.image)).unwrap().to_rgb8();
                let expected = expected(&selected.source, selected.actual_seconds);
                assert_eq!(image.dimensions(), expected.dimensions());
                let mean_error = image
                    .as_raw()
                    .iter()
                    .zip(expected.as_raw())
                    .map(|(actual, expected)| f64::from(actual.abs_diff(*expected)))
                    .sum::<f64>()
                    / image.as_raw().len() as f64;
                assert!(
                    mean_error < 4.0,
                    "full-resolution JPEG must retain source content: {mean_error}"
                );
                sources.push(selected.source.as_str());
            }
            if multiple {
                assert_ne!(sources[0], sources[1]);
            }
            let names = fs::read_dir(&path)
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect::<Vec<_>>();
            assert_eq!(
                names.len(),
                4,
                "artifact directory must expose only completed outputs"
            );
            assert!(
                image::open(path.join("selected-contact-sheet.jpg"))
                    .unwrap()
                    .width()
                    > 160
            );
            let sheet = image::open(path.join("selected-contact-sheet.jpg"))
                .unwrap()
                .to_rgb8();
            for (index, selected) in artifacts.report().selected.iter().enumerate() {
                assert_thumbnail(
                    &sheet,
                    (24 + index as u32 * 356, 48),
                    (320, 180),
                    &expected(&selected.source, selected.actual_seconds),
                );
            }
            if !semantic {
                assert!(
                    artifacts
                        .report()
                        .exclusions
                        .iter()
                        .any(|item| item.reason.contains("local quality"))
                );
            }
            drop(artifacts);
            assert!(
                !path.exists(),
                "private artifacts are owned and removed on drop"
            );
        }
    }
}

#[test]
fn missing_unknown_duplicate_and_invalid_ratings_fail_without_artifacts() {
    let tools = MediaTools::default();
    let videos = probe(false);
    for mutation in ["missing", "unknown", "duplicate", "nan", "empty"] {
        let mut scratch = None;
        let error = run(
            &tools,
            &videos,
            &options(1, false, false),
            &AtomicBool::new(false),
            |batch| {
                let mut response = fixed_assessments(batch)?;
                scratch = Some(
                    batch.candidates[0]
                        .image
                        .parent()
                        .unwrap()
                        .parent()
                        .unwrap()
                        .to_path_buf(),
                );
                match mutation {
                    "missing" => {
                        response.pop();
                    }
                    "unknown" => response[0].id = "wrong-media".to_owned(),
                    "duplicate" => response.push(response[0].clone()),
                    "nan" => response[0].blog_score = f64::NAN,
                    "empty" => response[0].reason.clear(),
                    _ => unreachable!(),
                }
                Ok(response)
            },
        )
        .unwrap_err();
        assert!(!error.to_string().is_empty());
        assert!(
            !scratch.unwrap().exists(),
            "response failures must remove owned scratch files"
        );
    }
}

#[test]
fn callback_errors_and_cancellation_preserve_source_and_clean_private_files() {
    let tools = MediaTools::default();
    let videos = probe(false);
    let source = fs::read(&videos[0].source).unwrap();
    let cancelled = AtomicBool::new(false);
    let error = run(
        &tools,
        &videos,
        &options(1, false, false),
        &cancelled,
        |_| bail!("fixture assessment unavailable"),
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("fixture assessment unavailable"));
    let mut scratch = None;
    let error = run(
        &tools,
        &videos,
        &options(1, false, false),
        &cancelled,
        |batch| {
            let response = fixed_assessments(batch)?;
            scratch = Some(
                batch.candidates[0]
                    .image
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .to_path_buf(),
            );
            cancelled.store(true, Ordering::Relaxed);
            Ok(response)
        },
    )
    .unwrap_err();
    assert!(error.downcast_ref::<Interrupted>().is_some());
    assert!(!scratch.unwrap().exists());
    let mut calls = 0;
    assert!(
        run(
            &tools,
            &videos,
            &options(1, false, false),
            &cancelled,
            |_| {
                calls += 1;
                Ok(Vec::new())
            }
        )
        .is_err()
    );
    assert_eq!(calls, 0);
    assert_eq!(fs::read(&videos[0].source).unwrap(), source);
}

#[test]
fn semantic_validation_happens_before_extraction_or_assessment() {
    let videos = probe(false);
    let missing_tools = MediaTools {
        ffmpeg: "missing-ffmpeg".into(),
        ffprobe: "missing-ffprobe".into(),
    };
    for mutation in ["source", "time", "empty", "coverage"] {
        let mut options = options(1, true, false);
        let events = options.semantic_events.as_mut().unwrap();
        match mutation {
            "source" => events[0].source = "unknown.mkv".to_owned(),
            "time" => events[0].representative_seconds = 30.0,
            "empty" => events.clear(),
            "coverage" => events[0].reason.clear(),
            _ => unreachable!(),
        }
        let error = run(
            &missing_tools,
            &videos,
            &options,
            &AtomicBool::new(false),
            |_| panic!("invalid semantic event must not reach assessment"),
        )
        .unwrap_err();
        assert!(
            !error.to_string().contains("cannot start"),
            "must validate the plan before media decode"
        );
    }
}

#[test]
fn secondary_rejection_replenishes_actual_frames_and_keeps_exclusion_evidence() {
    let mut options = options(1, false, false);
    options.samples_per_video = 1; // initial center request is 3 seconds.
    let mut secondary_calls = 0;
    let artifacts = run(
        &MediaTools::default(),
        &probe(false),
        &options,
        &AtomicBool::new(false),
        |batch| {
            let mut response = fixed_assessments(batch)?;
            if batch.phase == AssessmentPhase::Secondary {
                secondary_calls += 1;
                for assessment in &mut response {
                    if batch.candidates.iter().any(|candidate| {
                        candidate.id == assessment.id && candidate.actual_seconds == 3.0
                    }) {
                        assessment.transition = true;
                        assessment.reason = "前後の合成模様が遷移中".to_owned();
                    }
                }
            }
            Ok(response)
        },
    )
    .unwrap();
    assert!(secondary_calls > 1);
    assert_ne!(artifacts.report().selected[0].actual_seconds, 3.0);
    assert!(
        artifacts
            .report()
            .notes
            .iter()
            .any(|note| note.contains("Replenished"))
    );
    assert!(
        artifacts
            .report()
            .exclusions
            .iter()
            .any(|exclusion| exclusion.id.ends_with("00000012")
                && exclusion.reason.contains("secondary"))
    );
}

#[test]
fn finite_actual_frame_exhaustion_never_fills_with_rejected_or_duplicate_images() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("01-blocks.mkv");
    let generated = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(fixture("01-blocks.mkv"))
        .args(["-frames:v", "4", "-c:v", "ffv1"])
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let tools = MediaTools::default();
    let cancelled = AtomicBool::new(false);
    let videos = [tools.probe(&source, &cancelled).unwrap()];
    let original = fs::read(&source).unwrap();
    let mut options = options(1, false, false);
    options.samples_per_video = 1;
    options.game_context = None;
    let mut primary_ids = Vec::new();
    let mut scratch = None;
    let error = run(&tools, &videos, &options, &cancelled, |batch| {
        let mut response = fixed_assessments(batch)?;
        scratch = Some(
            batch.candidates[0]
                .image
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf(),
        );
        if batch.phase == AssessmentPhase::Primary {
            primary_ids.extend(
                batch
                    .candidates
                    .iter()
                    .map(|candidate| candidate.id.clone()),
            );
        } else {
            for assessment in &mut response {
                assessment.transition = true;
                assessment.reason = "前後の模様が loading 中".to_owned();
            }
        }
        Ok(response)
    })
    .unwrap_err();
    assert!(error.to_string().contains("after 4 actual frames"));
    assert!(error.to_string().contains("transition"));
    primary_ids.sort();
    primary_ids.dedup();
    assert_eq!(
        primary_ids.len(),
        4,
        "each actual PTS may be assessed only once"
    );
    assert!(!scratch.unwrap().exists());
    assert_eq!(fs::read(source).unwrap(), original);
}

#[test]
fn overlapping_semantic_events_are_deterministic_when_input_order_changes() {
    let mut options = options(1, true, false);
    let mut event = options.semantic_events.as_ref().unwrap()[0].clone();
    event.scene = "aaa".to_owned();
    let mut other = event.clone();
    other.scene = "bbb".to_owned();
    options.semantic_events = Some(vec![other, event]);
    let tools = MediaTools::default();
    let videos = probe(false);
    let first = run(
        &tools,
        &videos,
        &options,
        &AtomicBool::new(false),
        fixed_assessments,
    )
    .unwrap();
    options.semantic_events.as_mut().unwrap().reverse();
    let second = run(
        &tools,
        &videos,
        &options,
        &AtomicBool::new(false),
        fixed_assessments,
    )
    .unwrap();
    assert_eq!(
        first.report().selected[0].event.as_ref().unwrap().scene,
        "aaa"
    );
    assert_eq!(
        serde_json::to_value(first.report()).unwrap(),
        serde_json::to_value(second.report()).unwrap()
    );
}

#[test]
fn final_portrait_jpeg_preserves_selected_stream_rotation_and_context_privacy() {
    let tools = MediaTools::default();
    let cancelled = AtomicBool::new(false);
    let videos = [tools
        .probe(&fixture("03-multitrack-rotated.mov"), &cancelled)
        .unwrap()];
    let mut options = options(1, true, false);
    options.semantic_events.as_mut().unwrap().truncate(1);
    options.semantic_events.as_mut().unwrap()[0].source = "03-multitrack-rotated.mov".to_owned();
    options.game_context = None;
    let artifacts = run(&tools, &videos, &options, &cancelled, fixed_assessments).unwrap();
    let selected = &artifacts.report().selected[0];
    assert_eq!((selected.width, selected.height), (96, 160));
    let image = image::open(artifacts.path().join(&selected.image))
        .unwrap()
        .to_rgb8();
    let expected = expected(&selected.source, selected.actual_seconds);
    assert_eq!(image.dimensions(), expected.dimensions());
    let error = image
        .as_raw()
        .iter()
        .zip(expected.as_raw())
        .map(|(actual, expected)| f64::from(actual.abs_diff(*expected)))
        .sum::<f64>()
        / image.as_raw().len() as f64;
    assert!(
        error < 4.0,
        "selected portrait must retain the source display rotation"
    );
    assert!(
        serde_json::to_value(artifacts.report())
            .unwrap()
            .get("game_context")
            .is_none()
    );
}
