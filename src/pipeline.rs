//! Private selection artifacts. Transport and safe publication belong to #342/#341.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail, ensure};
use image::codecs::jpeg::JpegEncoder;
use serde::Serialize;
use tempfile::TempDir;

use crate::contact_sheet::{
    ContextImage, SheetEntry, render_candidate_sheets, render_selected_sheet,
};
use crate::media::{Interrupted, MediaTools, Video};
use crate::quality::{MAX_IMAGE_EDGE, MAX_IMAGE_PIXELS, analyze, read_image};
use crate::selection::{
    Assessment, Candidate, Exclusion, MAX_CANDIDATES, SemanticEvent, event_positions,
    sample_positions, select_up_to, validate_assessments,
};

const ASSESSMENT_BATCH: usize = 12;

#[derive(Debug, Clone, Serialize)]
pub struct ModelConditions {
    pub backend: String,
    pub model: String,
    pub inference_level: String,
    pub timeout_seconds: u64,
    pub cache_revision: String,
}

#[derive(Debug, Clone)]
pub struct PipelineOptions {
    pub count: usize,
    pub samples_per_video: usize,
    pub context_delta_seconds: f64,
    /// Supplied by a video-understanding caller; None chooses sampled_frames.
    pub semantic_events: Option<Vec<SemanticEvent>>,
    pub model: ModelConditions,
    /// Only supply context whose persistence has been approved by the caller.
    pub game_context: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssessmentPhase {
    Primary,
    Secondary,
}

#[derive(Debug, Clone)]
pub struct ContextFrame {
    pub source: String,
    pub image: PathBuf,
    pub requested_seconds: f64,
    pub actual_seconds: f64,
}

#[derive(Debug, Clone)]
pub struct CandidateContext {
    pub id: String,
    /// In before / center / after order, including clipped boundary requests.
    pub frames: [ContextFrame; 3],
}

/// A synchronous bounded batch. The callback must inspect these exact IDs and
/// images; any network transport and response retries are a caller concern.
pub struct AssessmentBatch<'a> {
    pub phase: AssessmentPhase,
    pub candidates: &'a [Candidate],
    pub sheets: Vec<PathBuf>,
    pub contexts: Vec<CandidateContext>,
}

#[derive(Debug, Serialize)]
pub struct SelectedFrame {
    pub rank: usize,
    pub id: String,
    pub source: String,
    pub requested_seconds: f64,
    pub actual_seconds: f64,
    pub image: String,
    pub width: u32,
    pub height: u32,
    pub quality_score: f64,
    pub primary_assessment: Assessment,
    pub assessment: Assessment,
    pub event: Option<SemanticEvent>,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub method: String,
    pub requested_count: usize,
    pub selected_count: usize,
    pub sources: Vec<String>,
    pub model: ModelConditions,
    pub samples_per_video: usize,
    pub context_delta_seconds: f64,
    pub max_candidates: usize,
    pub assessment_batch_size: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub game_context: Option<String>,
    pub selected: Vec<SelectedFrame>,
    pub exclusions: Vec<Exclusion>,
    pub notes: Vec<String>,
}

/// Owns all temporary files until dropped. path() contains only the completed
/// JPEGs, selected sheet and report for a future publisher to verify and copy.
#[derive(Debug)]
pub struct Artifacts {
    directory: TempDir,
    report: Report,
}

impl Artifacts {
    pub fn path(&self) -> PathBuf {
        self.directory.path().join("artifacts")
    }

    pub fn report(&self) -> &Report {
        &self.report
    }
}

struct Work<'a> {
    tools: &'a MediaTools,
    videos: &'a [Video],
    options: &'a PipelineOptions,
    cancelled: &'a AtomicBool,
    directory: TempDir,
    sources: Vec<String>,
    attempted: Vec<BTreeSet<usize>>,
    attempts: usize,
    candidates: Vec<Candidate>,
    primary: Vec<Assessment>,
    secondary_candidates: Vec<Candidate>,
    secondary: Vec<Assessment>,
    exclusions: BTreeMap<String, BTreeSet<String>>,
    notes: Vec<String>,
}

/// Process already-probed videos sequentially, with no live API or public output.
/// #341 can reuse the same probed metadata on resume without calling probe again.
pub fn run(
    tools: &MediaTools,
    videos: &[Video],
    options: &PipelineOptions,
    cancelled: &AtomicBool,
    mut assessor: impl FnMut(&AssessmentBatch<'_>) -> Result<Vec<Assessment>>,
) -> Result<Artifacts> {
    check_cancelled(cancelled)?;
    validate_options(videos, options)?;
    let sources = videos
        .iter()
        .map(|video| {
            video
                .source
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| !name.chars().any(char::is_control))
                .map(str::to_owned)
                .context("video source filename must be UTF-8")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        sources.iter().collect::<BTreeSet<_>>().len() == sources.len(),
        "video source filenames must be unique"
    );
    let plans = initial_plans(videos, &sources, options)?;
    let directory = tempfile::Builder::new()
        .prefix("game-screen-pick-selection-")
        .tempdir()
        .context("cannot create private selection workspace")?;
    fs::create_dir(directory.path().join("candidates"))?;
    fs::create_dir(directory.path().join("contexts"))?;
    fs::create_dir(directory.path().join("artifacts"))?;
    let mut work = Work {
        tools,
        videos,
        options,
        cancelled,
        directory,
        sources,
        attempted: vec![BTreeSet::new(); videos.len()],
        attempts: 0,
        candidates: Vec::new(),
        primary: Vec::new(),
        secondary_candidates: Vec::new(),
        secondary: Vec::new(),
        exclusions: BTreeMap::new(),
        notes: Vec::new(),
    };
    // Interleave sources, so a long first input cannot consume the entire budget.
    for ordinal in 0..plans.iter().map(Vec::len).max().unwrap_or(0) {
        for (source, plan) in plans.iter().enumerate() {
            if let Some((seconds, event)) = plan.get(ordinal) {
                work.extract(source, *seconds, event.clone())?;
            }
        }
    }
    loop {
        work.assess_primary(&mut assessor)?;
        work.assess_secondary(&mut assessor)?;
        let selected = if work.secondary_candidates.is_empty() {
            None
        } else {
            Some(select_up_to(
                &work.secondary_candidates,
                &work.secondary,
                options.count,
            )?)
        };
        if let Some(selection) = selected {
            work.record_exclusions("secondary", selection.exclusions);
            if selection.indices.len() == options.count {
                work.notes.extend(selection.notes);
                return work.finish(&selection.indices);
            }
        }
        let added = work.replenish()?;
        if added == 0 {
            let reasons = work
                .exclusions
                .values()
                .flatten()
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
                .join("; ");
            bail!(
                "insufficient distinct usable candidates for {} images after {} actual frames (limit {}): {}",
                options.count,
                work.attempts,
                MAX_CANDIDATES,
                if reasons.is_empty() {
                    "no eligible assessments"
                } else {
                    &reasons
                }
            );
        }
        work.notes.push(format!("Replenished {added} actual frames because secondary selection could not fill {} images.", options.count));
    }
}

fn validate_options(videos: &[Video], options: &PipelineOptions) -> Result<()> {
    ensure!(
        (1..=999).contains(&options.count),
        "selected count must be within 1..=999"
    );
    ensure!(
        !videos.is_empty() && videos.len() <= MAX_CANDIDATES,
        "video count must be within 1..={MAX_CANDIDATES}"
    );
    ensure!(
        (1..=MAX_CANDIDATES).contains(&options.samples_per_video),
        "sample count must be within 1..={MAX_CANDIDATES}"
    );
    ensure!(
        options.semantic_events.is_some()
            || options.samples_per_video <= MAX_CANDIDATES / videos.len(),
        "initial samples exceed the {MAX_CANDIDATES} candidate limit"
    );
    ensure!(
        options.context_delta_seconds.is_finite() && options.context_delta_seconds > 0.0,
        "context interval must be finite and positive"
    );
    ensure!(
        ["strata", "vllm"].contains(&options.model.backend.as_str()),
        "unsupported assessment backend"
    );
    ensure!(
        ["none", "low", "medium", "high"].contains(&options.model.inference_level.as_str()),
        "unsupported inference level"
    );
    ensure!(
        !options.model.model.trim().is_empty() && !options.model.cache_revision.trim().is_empty(),
        "assessment model and cache revision must not be blank"
    );
    ensure!(
        (1..=3600).contains(&options.model.timeout_seconds),
        "assessment timeout must be within 1..=3600 seconds"
    );
    ensure!(
        options
            .game_context
            .as_ref()
            .is_none_or(|context| !context.trim().is_empty()),
        "persisted game context must not be blank"
    );
    for video in videos {
        ensure!(
            !video.frames.is_empty()
                && video.duration_seconds.is_finite()
                && video.duration_seconds > 0.0,
            "probed video must have valid frame times and duration"
        );
        ensure!(
            video.width >= 16
                && video.height >= 16
                && video.width <= MAX_IMAGE_EDGE
                && video.height <= MAX_IMAGE_EDGE
                && u64::from(video.width) * u64::from(video.height) <= MAX_IMAGE_PIXELS,
            "video display dimensions exceed the image processing limits"
        );
    }
    Ok(())
}

type InitialPlan = Vec<(f64, Option<SemanticEvent>)>;

fn initial_plans(
    videos: &[Video],
    sources: &[String],
    options: &PipelineOptions,
) -> Result<Vec<InitialPlan>> {
    if let Some(events) = &options.semantic_events {
        ensure!(
            !events.is_empty() && events.len() <= MAX_CANDIDATES,
            "semantic events must be nonempty and within the candidate limit"
        );
        let mut plans = vec![Vec::new(); videos.len()];
        let mut total = 0;
        let mut ordered = events.iter().collect::<Vec<_>>();
        ordered.sort_by(|left, right| event_preference(left, right));
        for event in ordered {
            let index = sources
                .iter()
                .position(|source| source == &event.source)
                .context("semantic event refers to an unknown video source")?;
            let positions = event_positions(event, videos[index].duration_seconds)?;
            total += positions.len();
            ensure!(
                total <= MAX_CANDIDATES,
                "semantic event windows exceed the {MAX_CANDIDATES} candidate limit"
            );
            plans[index].extend(
                positions
                    .into_iter()
                    .map(|seconds| (seconds, Some(event.clone()))),
            );
        }
        ensure!(
            plans.iter().all(|plan| !plan.is_empty()),
            "semantic events must cover every input video"
        );
        Ok(plans)
    } else {
        videos
            .iter()
            .map(|video| {
                Ok(
                    sample_positions(video.duration_seconds, options.samples_per_video)?
                        .into_iter()
                        .map(|seconds| (seconds, None))
                        .collect(),
                )
            })
            .collect()
    }
}

impl Work<'_> {
    fn extract(
        &mut self,
        source: usize,
        seconds: f64,
        event: Option<SemanticEvent>,
    ) -> Result<bool> {
        check_cancelled(self.cancelled)?;
        if self.attempts == MAX_CANDIDATES {
            return Ok(false);
        }
        let video = &self.videos[source];
        let frame = video
            .frames
            .partition_point(|frame| frame.seconds < seconds)
            .min(video.frames.len() - 1);
        if !self.attempted[source].insert(frame) {
            return Ok(false);
        }
        self.attempts += 1;
        let artifact = self.tools.extract(video, seconds, self.cancelled)?;
        let id = format!(
            "V{:04}-F{:08}",
            source + 1,
            video.frames[frame].decode_index
        );
        if event.as_ref().is_some_and(|event| {
            artifact.actual_seconds < event.start_seconds
                || artifact.actual_seconds > event.end_seconds
        }) {
            self.exclusions
                .entry(id)
                .or_default()
                .insert("extracted PTS falls outside semantic event window".to_owned());
            return Ok(true);
        }
        let quality = analyze(&artifact.path())?;
        if let Some(reason) = &quality.rejection {
            self.exclusions
                .entry(id)
                .or_default()
                .insert(format!("local quality: {reason}"));
            return Ok(true);
        }
        let image = self
            .directory
            .path()
            .join("candidates")
            .join(format!("{id}.png"));
        fs::copy(artifact.path(), &image).context("cannot retain private candidate frame")?;
        self.candidates.push(Candidate {
            id,
            source: self.sources[source].clone(),
            requested_seconds: artifact.requested_seconds,
            actual_seconds: artifact.actual_seconds,
            duration_seconds: video.duration_seconds,
            image,
            quality,
            event,
        });
        Ok(true)
    }

    fn assess_primary(
        &mut self,
        assessor: &mut impl FnMut(&AssessmentBatch<'_>) -> Result<Vec<Assessment>>,
    ) -> Result<()> {
        let mut offset = self.primary.len();
        while offset < self.candidates.len() {
            check_cancelled(self.cancelled)?;
            let end = (offset + ASSESSMENT_BATCH).min(self.candidates.len());
            let candidates = &self.candidates[offset..end];
            let scratch = self.directory.path().join(format!("primary-{offset:04}"));
            fs::create_dir(&scratch)?;
            let entries = candidates
                .iter()
                .map(|candidate| sheet_entry(candidate, None, None, ""))
                .collect::<Vec<_>>();
            let sheets = render_candidate_sheets(&entries, &scratch)?;
            let batch = AssessmentBatch {
                phase: AssessmentPhase::Primary,
                candidates,
                sheets,
                contexts: Vec::new(),
            };
            let assessments = assessor(&batch).context("primary assessment failed")?;
            check_cancelled(self.cancelled)?;
            validate_assessments(candidates, &assessments)?;
            // Keep assessments aligned with candidate order even when model output is shuffled.
            self.primary.extend(aligned(candidates, assessments));
            offset = end;
        }
        Ok(())
    }

    fn assess_secondary(
        &mut self,
        assessor: &mut impl FnMut(&AssessmentBatch<'_>) -> Result<Vec<Assessment>>,
    ) -> Result<()> {
        let checked = self
            .secondary
            .iter()
            .map(|item| item.id.as_str())
            .collect::<BTreeSet<_>>();
        let unchecked = self
            .candidates
            .iter()
            .zip(&self.primary)
            .filter(|(candidate, _)| !checked.contains(candidate.id.as_str()))
            .map(|(candidate, assessment)| (candidate.clone(), assessment.clone()))
            .collect::<Vec<_>>();
        if unchecked.is_empty() {
            return Ok(());
        }
        let (candidates, assessments): (Vec<_>, Vec<_>) = unchecked.into_iter().unzip();
        let pool = select_up_to(
            &candidates,
            &assessments,
            (self.options.count * 3).min(MAX_CANDIDATES),
        )?;
        self.record_exclusions("primary", pool.exclusions);
        for chunk in pool.indices.chunks(ASSESSMENT_BATCH) {
            check_cancelled(self.cancelled)?;
            let candidates = chunk
                .iter()
                .map(|&index| candidates[index].clone())
                .collect::<Vec<_>>();
            let contexts = candidates
                .iter()
                .map(|candidate| self.context(candidate))
                .collect::<Result<Vec<_>>>()?;
            let entries = candidates
                .iter()
                .zip(&contexts)
                .map(|(candidate, context)| {
                    let images = context.frames.each_ref().map(|frame| ContextImage {
                        image: frame.image.clone(),
                        actual_seconds: frame.actual_seconds,
                    });
                    let scene = assessments
                        .iter()
                        .find(|item| item.id == candidate.id)
                        .expect("validated assessment")
                        .scene
                        .as_str();
                    sheet_entry(candidate, None, Some(images), scene)
                })
                .collect::<Vec<_>>();
            let scratch = self
                .directory
                .path()
                .join(format!("secondary-{:04}", self.secondary.len()));
            fs::create_dir(&scratch)?;
            let sheets = render_candidate_sheets(&entries, &scratch)?;
            let batch = AssessmentBatch {
                phase: AssessmentPhase::Secondary,
                candidates: &candidates,
                sheets,
                contexts,
            };
            let assessments = assessor(&batch).context("secondary assessment failed")?;
            check_cancelled(self.cancelled)?;
            validate_assessments(&candidates, &assessments)?;
            self.secondary.extend(aligned(&candidates, assessments));
            self.secondary_candidates.extend(candidates);
        }
        Ok(())
    }

    fn context(&self, candidate: &Candidate) -> Result<CandidateContext> {
        let source = self
            .sources
            .iter()
            .position(|source| source == &candidate.source)
            .expect("candidate source came from this work");
        let artifacts = self.tools.extract_context(
            &self.videos[source],
            candidate.requested_seconds,
            self.options.context_delta_seconds,
            self.cancelled,
        )?;
        let mut frames = Vec::new();
        for (index, artifact) in artifacts.into_iter().enumerate() {
            let image = self
                .directory
                .path()
                .join("contexts")
                .join(format!("{}-{index}.png", candidate.id));
            fs::copy(artifact.path(), &image).context("cannot retain private context frame")?;
            frames.push(ContextFrame {
                source: candidate.source.clone(),
                image,
                requested_seconds: artifact.requested_seconds,
                actual_seconds: artifact.actual_seconds,
            });
        }
        Ok(CandidateContext {
            id: candidate.id.clone(),
            frames: frames.try_into().expect("three context frames"),
        })
    }

    fn replenish(&mut self) -> Result<usize> {
        let before = self.attempts;
        let target = (self.options.count * 2)
            .max(self.videos.len())
            .min(MAX_CANDIDATES - before);
        while self.attempts - before < target {
            let mut progress = false;
            for source in 0..self.videos.len() {
                if self.attempts - before == target {
                    break;
                }
                if let Some(frame) = next_unattempted(
                    &self.videos[source],
                    &self.attempted[source],
                    self.options.semantic_events.as_deref(),
                    &self.sources[source],
                ) {
                    let seconds = self.videos[source].frames[frame].seconds;
                    let event = self
                        .options
                        .semantic_events
                        .as_ref()
                        .and_then(|events| {
                            events
                                .iter()
                                .filter(|event| {
                                    event.source == self.sources[source]
                                        && seconds >= event.start_seconds
                                        && seconds <= event.end_seconds
                                })
                                .min_by(|left, right| event_preference(left, right))
                        })
                        .cloned();
                    progress |= self.extract(source, seconds, event)?;
                }
            }
            if !progress {
                break;
            }
        }
        Ok(self.attempts - before)
    }

    fn record_exclusions(&mut self, phase: &str, exclusions: Vec<Exclusion>) {
        for exclusion in exclusions {
            self.exclusions
                .entry(exclusion.id)
                .or_default()
                .insert(format!("{phase}: {}", exclusion.reason));
        }
    }

    fn finish(mut self, indices: &[usize]) -> Result<Artifacts> {
        let output = self.directory.path().join("artifacts");
        let mut selected = Vec::new();
        let mut entries = Vec::new();
        for (ordinal, &index) in indices.iter().enumerate() {
            check_cancelled(self.cancelled)?;
            let candidate = &self.secondary_candidates[index];
            let assessment = &self.secondary[index];
            let rank = ordinal + 1;
            let name = format!("selected-{rank:02}.jpg");
            let path = output.join(&name);
            let image = read_image(&candidate.image)?;
            JpegEncoder::new_with_quality(File::create(&path)?, 95)
                .encode_image(&image)
                .context("cannot encode selected full-resolution JPEG")?;
            let primary = self
                .primary
                .iter()
                .find(|item| item.id == candidate.id)
                .expect("validated primary assessment");
            selected.push(SelectedFrame {
                rank,
                id: candidate.id.clone(),
                source: candidate.source.clone(),
                requested_seconds: candidate.requested_seconds,
                actual_seconds: candidate.actual_seconds,
                image: name,
                width: image.width(),
                height: image.height(),
                quality_score: candidate.quality.score,
                primary_assessment: primary.clone(),
                assessment: assessment.clone(),
                event: candidate.event.clone(),
            });
            let mut entry = sheet_entry(candidate, Some(rank), None, &assessment.scene);
            entry.image = path;
            entries.push(entry);
            self.exclusions.remove(&candidate.id);
        }
        check_cancelled(self.cancelled)?;
        render_selected_sheet(&entries, &output)?;
        let selected_ids = selected
            .iter()
            .map(|item| item.id.as_str())
            .collect::<BTreeSet<_>>();
        for candidate in &self.candidates {
            if !selected_ids.contains(candidate.id.as_str()) {
                self.exclusions
                    .entry(candidate.id.clone())
                    .or_default()
                    .insert(
                        "not selected after quality, assessment and diversity ranking".to_owned(),
                    );
            }
        }
        let report = Report {
            method: if self.options.semantic_events.is_some() {
                "semantic_video"
            } else {
                "sampled_frames"
            }
            .to_owned(),
            requested_count: self.options.count,
            selected_count: selected.len(),
            sources: self.sources,
            model: self.options.model.clone(),
            samples_per_video: self.options.samples_per_video,
            context_delta_seconds: self.options.context_delta_seconds,
            max_candidates: MAX_CANDIDATES,
            assessment_batch_size: ASSESSMENT_BATCH,
            game_context: self.options.game_context.clone(),
            selected,
            exclusions: self
                .exclusions
                .into_iter()
                .map(|(id, reasons)| Exclusion {
                    id,
                    reason: reasons.into_iter().collect::<Vec<_>>().join("; "),
                })
                .collect(),
            notes: self.notes,
        };
        check_cancelled(self.cancelled)?;
        serde_json::to_writer_pretty(File::create(output.join("report.json"))?, &report)
            .context("cannot write selection report")?;
        check_cancelled(self.cancelled)?;
        Ok(Artifacts {
            directory: self.directory,
            report,
        })
    }
}

fn aligned(candidates: &[Candidate], assessments: Vec<Assessment>) -> Vec<Assessment> {
    let mut by_id = assessments
        .into_iter()
        .map(|item| (item.id.clone(), item))
        .collect::<HashMap<_, _>>();
    candidates
        .iter()
        .map(|candidate| {
            by_id
                .remove(&candidate.id)
                .expect("validated assessment ID")
        })
        .collect()
}

fn sheet_entry(
    candidate: &Candidate,
    rank: Option<usize>,
    context: Option<[ContextImage; 3]>,
    scene: &str,
) -> SheetEntry {
    SheetEntry {
        id: candidate.id.clone(),
        rank,
        source: candidate.source.clone(),
        requested_seconds: candidate.requested_seconds,
        actual_seconds: candidate.actual_seconds,
        scene: scene.to_owned(),
        image: candidate.image.clone(),
        context,
    }
}

/// Choose the midpoint of the widest remaining time gap. Only the bounded set
/// of attempted ordinals is stored; Video already owns probe metadata.
fn next_unattempted(
    video: &Video,
    attempted: &BTreeSet<usize>,
    events: Option<&[SemanticEvent]>,
    source: &str,
) -> Option<usize> {
    let mut best = None;
    let mut width = -1.0;
    let mut consider = |begin: usize, limit: usize| {
        let mut start = begin;
        for end in attempted
            .range(begin..limit)
            .copied()
            .chain(std::iter::once(limit))
        {
            if start < end {
                let first = video.frames[start].seconds;
                let last = video.frames[end - 1].seconds;
                let gap = last - first;
                if gap > width {
                    let midpoint = (first + last) / 2.0;
                    let index =
                        video.frames[start..end].partition_point(|frame| frame.seconds < midpoint);
                    best = Some((start + index).min(end - 1));
                    width = gap;
                }
            }
            start = end.saturating_add(1);
        }
    };
    if let Some(events) = events {
        for event in events.iter().filter(|event| event.source == source) {
            let begin = video
                .frames
                .partition_point(|frame| frame.seconds < event.start_seconds);
            let limit = video
                .frames
                .partition_point(|frame| frame.seconds <= event.end_seconds);
            consider(begin, limit);
        }
    } else {
        consider(0, video.frames.len());
    }
    best
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<()> {
    if cancelled.load(Ordering::Relaxed) {
        Err(Interrupted.into())
    } else {
        Ok(())
    }
}

fn event_preference(left: &SemanticEvent, right: &SemanticEvent) -> std::cmp::Ordering {
    right
        .importance
        .total_cmp(&left.importance)
        .then_with(|| left.source.cmp(&right.source))
        .then_with(|| left.start_seconds.total_cmp(&right.start_seconds))
        .then_with(|| left.end_seconds.total_cmp(&right.end_seconds))
        .then_with(|| {
            left.representative_seconds
                .total_cmp(&right.representative_seconds)
        })
        .then_with(|| left.scene.cmp(&right.scene))
        .then_with(|| left.reason.cmp(&right.reason))
}
