//! Independent fixture pixels/PTS, with rendered sheets used only to check wiring.
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use game_screen_pick::contact_sheet::{
    ContextImage, SheetEntry, render_candidate_sheets, render_selected_sheet,
};
use image::RgbImage;
use serde_json::Value;

pub const SCENE: &str = "colored gameplay blocks";

pub fn expected(source: &str, seconds: f64) -> Result<RgbImage> {
    ensure!(
        seconds.is_finite() && (0.0..6.0).contains(&seconds),
        "invalid fixture time"
    );
    let (mirrored, rotated) = match source {
        "blocks.mkv" | "quiet.mkv" => (false, false),
        "mirrored.mkv" => (true, false),
        "rotated.mov" => (false, true),
        _ => anyhow::bail!("unknown fixture source"),
    };
    let rows = include_str!("../fixtures/video-tiles.tsv")
        .lines()
        .map(|line| {
            line.split('\t')
                .skip(2)
                .map(|v| v.parse::<u8>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (width, height) = if rotated { (96, 160) } else { (160, 96) };
    Ok(RgbImage::from_fn(width, height, |x, y| {
        let (x, y) = if rotated { (159 - y, x) } else { (x, y) };
        let tile = if mirrored { 9 - x / 16 } else { x / 16 };
        image::Rgb([rows[seconds.floor() as usize * 6 + y as usize / 16][tile as usize]; 3])
    }))
}

fn data(part: &Value) -> Result<Vec<u8>> {
    let url = part["image_url"]["url"]
        .as_str()
        .or_else(|| part["video_url"]["url"].as_str())
        .context("missing media")?;
    Ok(STANDARD.decode(url.split_once(',').context("missing inline bytes")?.1)?)
}

fn meta(part: &Value) -> Option<Value> {
    serde_json::from_str(part["text"].as_str()?.split_once('\n')?.1).ok()
}

pub fn metadata(body: &Value, kind: &str) -> Result<Value> {
    body["messages"][0]["content"]
        .as_array()
        .context("missing content")?
        .iter()
        .filter_map(meta)
        .find(|value| value["kind"] == kind)
        .context("missing metadata")
}

pub fn verify_request(body: &Value) -> Result<()> {
    let parts = body["messages"][0]["content"]
        .as_array()
        .context("missing content")?;
    if let Some(video) = parts.iter().find(|p| p["type"] == "video_url") {
        let chunk = metadata(body, "semantic_chunk_metadata")?;
        let source = chunk["source"].as_str().context("source")?;
        let start = chunk["actual_start_seconds"].as_f64().context("start")?;
        let end = chunk["actual_end_seconds"].as_f64().context("end")?;
        ensure!(
            chunk["chunk_time_zero_source_seconds"] == chunk["actual_start_seconds"],
            "wrong chunk time zero"
        );
        let recipe = expected(source, start)?;
        ensure!(
            chunk["width"] == recipe.width() && chunk["height"] == recipe.height(),
            "wrong chunk dimensions"
        );
        let scratch = tempfile::tempdir()?;
        let path = scratch.path().join("chunk.mov");
        fs::write(&path, data(video)?)?;
        // Decode all frames, independently of the product extractor and ratings.
        let decoded = Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(path)
            .args([
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ])
            .output()?;
        ensure!(decoded.status.success(), "cannot decode supplied chunk");
        let size = recipe.as_raw().len();
        ensure!(
            decoded.stdout.len() == ((end - start) * 4.0).round() as usize * size,
            "wrong chunk frame count"
        );
        for (index, pixels) in decoded.stdout.chunks_exact(size).enumerate() {
            ensure!(
                pixels == expected(source, start + index as f64 / 4.0)?.as_raw(),
                "wrong chunk source/time/order pixels"
            );
        }
    } else if parts.iter().any(|p| p["type"] == "image_url") {
        let assessment = metadata(body, "assessment_metadata")?;
        let candidates = assessment["candidates"].as_array().context("candidates")?;
        let secondary = assessment["phase"] == "secondary";
        ensure!(
            parts
                .iter()
                .filter_map(meta)
                .filter(|value| value["kind"] == "transition_context_frame")
                .count()
                == if secondary { candidates.len() * 3 } else { 0 },
            "wrong total context inventory"
        );
        ensure!(
            parts
                .iter()
                .filter(|part| part["type"] == "image_url")
                .count()
                == candidates.len().div_ceil(12) + if secondary { candidates.len() * 3 } else { 0 },
            "wrong image inventory"
        );
        let scratch = tempfile::tempdir()?;
        let mut entries = Vec::new();
        for (index, candidate) in candidates.iter().enumerate() {
            let source = candidate["source"].as_str().context("source")?;
            let actual = candidate["actual_seconds"].as_f64().context("actual")?;
            let image = scratch.path().join(format!("center-{index}.png"));
            expected(source, actual)?.save(&image)?;
            let contexts = parts
                .iter()
                .enumerate()
                .filter_map(|(index, part)| {
                    let value = meta(part)?;
                    (value["kind"] == "transition_context_frame" && value["id"] == candidate["id"])
                        .then_some((index, value))
                })
                .collect::<Vec<_>>();
            ensure!(
                contexts.len() == if secondary { 3 } else { 0 },
                "wrong context inventory"
            );
            let mut frames = Vec::new();
            for ((part_index, frame), position) in
                contexts.iter().zip(["before", "center", "after"])
            {
                ensure!(
                    frame["position"] == position && frame["source"] == candidate["source"],
                    "wrong context source/order"
                );
                let seconds = frame["actual_seconds"].as_f64().context("context actual")?;
                if position == "center" {
                    ensure!(seconds == actual, "wrong center time");
                }
                let pixels = image::load_from_memory(&data(&parts[part_index + 1])?)?.to_rgb8();
                ensure!(
                    pixels == expected(source, seconds)?,
                    "wrong context pixels/time"
                );
                let path = scratch.path().join(format!("{index}-{position}.png"));
                pixels.save(&path)?;
                frames.push(ContextImage {
                    image: path,
                    actual_seconds: seconds,
                });
            }
            entries.push(SheetEntry {
                id: candidate["id"].as_str().context("id")?.into(),
                rank: None,
                source: source.into(),
                requested_seconds: candidate["requested_seconds"]
                    .as_f64()
                    .context("requested")?,
                actual_seconds: actual,
                scene: if secondary {
                    SCENE.into()
                } else {
                    String::new()
                },
                image,
                context: secondary.then(|| frames.try_into().unwrap()),
            });
        }
        let expected_sheets = render_candidate_sheets(&entries, &scratch.path().join("sheets"))?;
        let actual_sheets = parts
            .iter()
            .enumerate()
            .filter_map(|(index, part)| {
                let value = meta(part)?;
                (value["kind"] == "candidate_contact_sheet").then_some((index, value))
            })
            .collect::<Vec<_>>();
        ensure!(
            actual_sheets.len() == expected_sheets.len(),
            "wrong sheet inventory"
        );
        for (page, ((index, value), expected)) in
            actual_sheets.iter().zip(expected_sheets).enumerate()
        {
            let ids = candidates
                .iter()
                .skip(page * 12)
                .take(12)
                .map(|c| c["id"].clone())
                .collect::<Vec<_>>();
            ensure!(
                value["display_ids_in_order"] == Value::from(ids),
                "wrong sheet ID order"
            );
            ensure!(
                data(&parts[index + 1])? == fs::read(expected)?,
                "wrong sheet pixels/labels/source/time"
            );
        }
    }
    Ok(())
}

pub fn verify_output(output: &Path, report: &Value) -> Result<()> {
    let selected = report["selected"].as_array().context("selected")?;
    ensure!(
        report["selected_count"] == selected.len() && report["requested_count"] == selected.len(),
        "wrong count"
    );
    ensure!(
        fs::read_dir(output)?.count() == selected.len() + 2,
        "incomplete output inventory"
    );
    let mut ids = BTreeSet::new();
    let mut entries = Vec::new();
    for (index, frame) in selected.iter().enumerate() {
        ensure!(
            frame["rank"] == index + 1 && ids.insert(frame["id"].as_str().context("id")?),
            "wrong rank/duplicate ID"
        );
        ensure!(
            frame["id"] == frame["assessment"]["id"]
                && frame["id"] == frame["primary_assessment"]["id"],
            "wrong assessment ID"
        );
        let actual = frame["actual_seconds"].as_f64().context("time")?;
        ensure!(
            (actual * 4.0).fract() == 0.0 && !(2.0..3.0).contains(&actual),
            "wrong PTS or black frame"
        );
        let source = frame["source"].as_str().context("source")?;
        let image = output.join(frame["image"].as_str().context("image")?);
        let pixels = image::open(&image)?.to_rgb8();
        let wanted = expected(source, actual)?;
        ensure!(
            pixels.dimensions() == wanted.dimensions()
                && frame["width"] == wanted.width()
                && frame["height"] == wanted.height(),
            "wrong full-resolution orientation"
        );
        let error = pixels
            .as_raw()
            .iter()
            .zip(wanted.as_raw())
            .map(|(a, b)| f64::from(a.abs_diff(*b)))
            .sum::<f64>()
            / pixels.as_raw().len() as f64;
        ensure!(error < 4.0, "wrong selected source/time pixels");
        if report["method"] == "semantic_video" {
            let event = &frame["event"];
            ensure!(
                event["source"] == frame["source"]
                    && event["start_seconds"].as_f64().context("event start")? <= actual
                    && actual <= event["end_seconds"].as_f64().context("event end")?,
                "wrong semantic evidence mapping"
            );
            ensure!(
                !event["reason"].as_str().context("event reason")?.is_empty(),
                "missing event reason"
            );
        } else {
            ensure!(frame["event"].is_null(), "unexpected event");
        }
        entries.push(SheetEntry {
            id: frame["id"].as_str().unwrap().into(),
            rank: Some(index + 1),
            source: source.into(),
            requested_seconds: frame["requested_seconds"].as_f64().context("requested")?,
            actual_seconds: actual,
            scene: frame["assessment"]["scene"]
                .as_str()
                .context("scene")?
                .into(),
            image,
            context: None,
        });
    }
    let scratch = tempfile::tempdir()?;
    let sheet = render_selected_sheet(&entries, scratch.path())?;
    ensure!(
        fs::read(output.join("selected-contact-sheet.jpg"))? == fs::read(sheet)?,
        "sheet/report labels or order disagree"
    );
    Ok(())
}
