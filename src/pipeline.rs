//! Private selection artifacts. Transport and safe publication belong to #342/#341.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail, ensure};
use image::codecs::jpeg::JpegEncoder;
use serde::Serialize;
use tempfile::TempDir;

use crate::contact_sheet::{
    ContextImage, SheetEntry, render_candidate_sheets, render_selected_sheet,
};
use crate::media::{FrameArtifact, Interrupted, MediaTools, Video};
use crate::metrics::{StageSeconds, timed};
use crate::quality::{
    MAX_IMAGE_EDGE, MAX_IMAGE_PIXELS, MAX_SELECTED_JPEG_BYTES, analyze, read_image,
};
use crate::selection::{
    Assessment, Candidate, Exclusion, MAX_CANDIDATES, SemanticEvent, event_positions, select_up_to,
    validate_assessments,
};

pub const ASSESSMENT_BATCH: usize = 12;

struct SelectedJpegWriter {
    file: BufWriter<File>,
    bytes: u64,
}

impl Write for SelectedJpegWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if buffer.len() as u64 > MAX_SELECTED_JPEG_BYTES - self.bytes {
            return Err(io::Error::other(
                "selected JPEG exceeds the encoded byte limit",
            ));
        }
        let written = self.file.write(buffer)?;
        self.bytes += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

pub(crate) fn encode_selected_jpeg(image: &image::RgbImage, path: &Path) -> Result<()> {
    let mut writer = SelectedJpegWriter {
        file: BufWriter::new(File::create(path)?),
        bytes: 0,
    };
    JpegEncoder::new_with_quality(&mut writer, 95)
        .encode_image(image)
        .context("cannot encode selected full-resolution JPEG")?;
    writer
        .flush()
        .context("cannot flush selected full-resolution JPEG")
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelConditions {
    pub backend: String,
    pub model: String,
    pub inference_level: String,
    pub timeout_seconds: u64,
    pub cache_revision: String,
}

#[derive(Debug, Clone, Serialize)]
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
/// Candidate, sheet and context paths exist only during this callback; copy QA evidence
/// before returning. Completed artifacts retain none of this assessment scratch.
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
    metrics: &'a mut StageSeconds,
    directory: TempDir,
    sources: Vec<String>,
    events: Option<EventIndex<'a>>,
    attempted: Vec<BTreeSet<usize>>,
    attempts: usize,
    next_source: usize,
    candidates: Vec<Candidate>,
    primary: Vec<Assessment>,
    secondary_pending: BTreeSet<usize>,
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
    assessor: impl FnMut(&AssessmentBatch<'_>) -> Result<Vec<Assessment>>,
) -> Result<Artifacts> {
    run_measured(
        tools,
        videos,
        options,
        cancelled,
        &mut StageSeconds::default(),
        assessor,
    )
}

/// Session retains these timings even if extraction or a later callback fails.
pub(crate) fn run_measured(
    tools: &MediaTools,
    videos: &[Video],
    options: &PipelineOptions,
    cancelled: &AtomicBool,
    metrics: &mut StageSeconds,
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
    let plans = initial_plans(
        videos,
        &sources,
        options.semantic_events.as_deref(),
        options.samples_per_video,
        cancelled,
    )?;
    let directory = tempfile::Builder::new()
        .prefix("game-screen-pick-selection-")
        .tempdir()
        .context("cannot create private selection workspace")?;
    fs::create_dir(directory.path().join("candidates"))?;
    fs::create_dir(directory.path().join("artifacts"))?;
    let mut work = Work {
        tools,
        videos,
        options,
        cancelled,
        metrics,
        directory,
        sources,
        events: plans.events,
        attempted: vec![BTreeSet::new(); videos.len()],
        attempts: 0,
        next_source: 0,
        candidates: Vec::new(),
        primary: Vec::new(),
        secondary_pending: BTreeSet::new(),
        secondary_candidates: Vec::new(),
        secondary: Vec::new(),
        exclusions: BTreeMap::new(),
        notes: Vec::new(),
    };
    // Interleave sources, so a long first input cannot consume the entire budget.
    for ordinal in 0..plans.positions.iter().map(Vec::len).max().unwrap_or(0) {
        for (source, plan) in plans.positions.iter().enumerate() {
            if let Some(seconds) = plan.get(ordinal) {
                work.produce(source, *seconds, &mut assessor)?;
            }
        }
    }
    loop {
        work.assess_primary(&mut assessor)?;
        let assessed_before = work.secondary.len();
        let primary_pool_exclusions = work.assess_secondary(&mut assessor)?;
        let newly_assessed = work.secondary.len() - assessed_before;
        let selected = if work.secondary_candidates.is_empty() {
            None
        } else {
            Some(select_up_to(
                &work.secondary_candidates,
                &work.secondary,
                options.count,
            )?)
        };
        let mut secondary_exclusions = Vec::new();
        if let Some(selection) = selected {
            if selection.indices.len() == options.count {
                work.record_exclusions("secondary", selection.exclusions);
                work.notes.extend(selection.notes);
                return work.finish(&selection.indices);
            }
            secondary_exclusions = selection.exclusions;
        }
        // Try the remaining eligible primary candidates before decoding more
        // frames or declaring exhaustion. Every pass consumes distinct IDs.
        if newly_assessed > 0 {
            continue;
        }
        let added = work.replenish(&mut assessor)?;
        if added == 0 {
            let reasons = work
                .exclusions
                .values()
                .flatten()
                .cloned()
                .chain(
                    primary_pool_exclusions
                        .into_iter()
                        .map(|item| format!("primary: {}", item.reason)),
                )
                .chain(
                    secondary_exclusions
                        .into_iter()
                        .map(|item| format!("secondary: {}", item.reason)),
                )
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
    validate_parameters(options)?;
    validate_videos(videos)
}

fn validate_videos(videos: &[Video]) -> Result<()> {
    ensure!(
        !videos.is_empty() && videos.len() <= MAX_CANDIDATES,
        "video count must be within 1..={MAX_CANDIDATES}"
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

pub(crate) fn validate_parameters(options: &PipelineOptions) -> Result<()> {
    ensure!(
        (1..=999).contains(&options.count),
        "selected count must be within 1..=999"
    );
    validate_sample_count(options.samples_per_video)?;
    ensure!(
        options.context_delta_seconds.is_finite() && options.context_delta_seconds > 0.0,
        "context interval must be finite and positive"
    );
    validate_model(&options.model)?;
    ensure!(
        options
            .game_context
            .as_ref()
            .is_none_or(|context| !context.trim().is_empty()),
        "persisted game context must not be blank"
    );
    Ok(())
}

fn validate_sample_count(samples_per_video: usize) -> Result<()> {
    ensure!(
        (1..=MAX_CANDIDATES).contains(&samples_per_video),
        "sample count must be within 1..={MAX_CANDIDATES}"
    );
    Ok(())
}

/// Check the real initial sampling budget before search or context generation.
/// This shares the exact first-grid-request/actual-frame deduplication used by
/// selection, and does not require model options, decode frames or perform I/O.
pub(crate) fn preflight_sampled_plans(
    videos: &[Video],
    samples_per_video: usize,
    cancelled: &AtomicBool,
) -> Result<()> {
    check_cancelled(cancelled)?;
    validate_sample_count(samples_per_video)?;
    validate_videos(videos)?;
    // Source labels are needed only when indexing semantic events.
    initial_plans(videos, &[], None, samples_per_video, cancelled)?;
    Ok(())
}

pub(crate) fn validate_model(model: &ModelConditions) -> Result<()> {
    ensure!(
        ["strata", "vllm"].contains(&model.backend.as_str()),
        "unsupported assessment backend"
    );
    ensure!(
        ["none", "low", "medium", "high"].contains(&model.inference_level.as_str()),
        "unsupported inference level"
    );
    ensure!(
        !model.model.trim().is_empty() && !model.cache_revision.trim().is_empty(),
        "assessment model and cache revision must not be blank"
    );
    ensure!(
        (1..=3600).contains(&model.timeout_seconds),
        "assessment timeout must be within 1..=3600 seconds"
    );
    Ok(())
}

type EventIndex<'a> = Vec<BTreeSet<OrderedEvent<'a>>>;

#[derive(Debug, Clone, Copy)]
struct OrderedEvent<'a>(&'a SemanticEvent);

impl PartialEq for OrderedEvent<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for OrderedEvent<'_> {}
impl PartialOrd for OrderedEvent<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedEvent<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        event_preference(self.0, other.0)
    }
}

#[derive(Debug)]
struct InitialPlans<'a> {
    positions: Vec<Vec<f64>>,
    events: Option<EventIndex<'a>>,
}

fn index_events<'a>(
    videos: &[Video],
    sources: &[String],
    events: Option<&'a [SemanticEvent]>,
    cancelled: &AtomicBool,
) -> Result<Option<EventIndex<'a>>> {
    let Some(events) = events else {
        return Ok(None);
    };
    ensure!(!events.is_empty(), "semantic events must be nonempty");
    let sources = sources
        .iter()
        .enumerate()
        .map(|(index, source)| (source.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut indexed = vec![BTreeSet::new(); videos.len()];
    for event in events {
        check_cancelled(cancelled)?;
        let source = *sources
            .get(event.source.as_str())
            .context("semantic event refers to an unknown video source")?;
        event_positions(event, videos[source].duration_seconds)?;
        indexed[source].insert(OrderedEvent(event));
        check_cancelled(cancelled)?;
    }
    Ok(Some(indexed))
}

fn initial_plans<'a>(
    videos: &[Video],
    sources: &[String],
    semantic_events: Option<&'a [SemanticEvent]>,
    samples_per_video: usize,
    cancelled: &AtomicBool,
) -> Result<InitialPlans<'a>> {
    let events = index_events(videos, sources, semantic_events, cancelled)?;
    let mut plans = vec![Vec::new(); videos.len()];
    let mut planned = vec![BTreeSet::new(); videos.len()];
    let mut total = 0;
    let mut add = |source: usize, seconds: f64| -> Result<()> {
        check_cancelled(cancelled)?;
        if planned[source].insert(frame_ordinal(&videos[source], seconds)) {
            total += 1;
            ensure!(
                total <= MAX_CANDIDATES,
                "initial plans exceed the {MAX_CANDIDATES} actual-frame candidate limit"
            );
            plans[source].push(seconds);
        }
        Ok(())
    };
    if let Some(events) = &events {
        for (source, events) in events.iter().enumerate() {
            check_cancelled(cancelled)?;
            for event in events {
                for seconds in event_positions(event.0, videos[source].duration_seconds)? {
                    add(source, seconds)?;
                }
            }
        }
    } else {
        for (source, video) in videos.iter().enumerate() {
            for seconds in sampled_requests(video, samples_per_video) {
                add(source, seconds)?;
            }
        }
    }
    Ok(InitialPlans {
        positions: plans,
        events,
    })
}

fn frame_ordinal(video: &Video, seconds: f64) -> usize {
    video
        .frames
        .partition_point(|frame| frame.seconds < seconds)
        .min(video.frames.len() - 1)
}

/// Preserve the first original grid request for each actual frame. Skip grids
/// inside VFR gaps without materializing or visiting every duplicate request.
fn sampled_requests(video: &Video, count: usize) -> impl Iterator<Item = f64> + '_ {
    let position = move |index: usize| {
        if count == 1 {
            video.duration_seconds / 2.0
        } else {
            video.duration_seconds * index as f64 / (count - 1) as f64
        }
    };
    let mut next = 0;
    std::iter::from_fn(move || {
        if next == count {
            return None;
        }
        let seconds = position(next);
        let frame = frame_ordinal(video, seconds);
        if frame == video.frames.len() - 1 {
            next = count;
        } else {
            let mut begin = next + 1;
            let mut end = count;
            while begin < end {
                let midpoint = begin + (end - begin) / 2;
                if position(midpoint) <= video.frames[frame].seconds {
                    begin = midpoint + 1;
                } else {
                    end = midpoint;
                }
            }
            next = begin;
        }
        Some(seconds)
    })
}

impl Work<'_> {
    fn produce(
        &mut self,
        source: usize,
        seconds: f64,
        assessor: &mut impl FnMut(&AssessmentBatch<'_>) -> Result<Vec<Assessment>>,
    ) -> Result<bool> {
        let progress = self.extract(source, seconds)?;
        if self.candidates.len() - self.primary.len() == ASSESSMENT_BATCH {
            self.assess_primary(assessor)?;
        }
        Ok(progress)
    }

    fn extract(&mut self, source: usize, seconds: f64) -> Result<bool> {
        check_cancelled(self.cancelled)?;
        if self.attempts == MAX_CANDIDATES {
            return Ok(false);
        }
        let video = &self.videos[source];
        let frame = frame_ordinal(video, seconds);
        if !self.attempted[source].insert(frame) {
            return Ok(false);
        }
        self.attempts += 1;
        let artifact = timed(&mut self.metrics.extraction, || {
            self.tools.extract(video, seconds, self.cancelled)
        })?;
        let id = format!(
            "V{:04}-F{:08}",
            source + 1,
            video.frames[frame].decode_index
        );
        let event = self.semantic_event(source, artifact.actual_seconds)?;
        if self.options.semantic_events.is_some() && event.is_none() {
            self.exclusions
                .entry(id)
                .or_default()
                .insert("extracted PTS falls outside every semantic event window".to_owned());
            return Ok(true);
        }
        let quality = timed(&mut self.metrics.machine_quality, || {
            analyze(&artifact.path())
        })?;
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
            let scratch = tempfile::Builder::new()
                .prefix("primary-")
                .tempdir_in(self.directory.path())?;
            let entries = candidates
                .iter()
                .map(|candidate| sheet_entry(candidate, None, None, ""))
                .collect::<Vec<_>>();
            let sheets = render_candidate_sheets(&entries, scratch.path())?;
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
            let assessments = aligned(candidates, assessments);
            for (index, assessment) in assessments.iter().enumerate() {
                let index = offset + index;
                let candidate = &self.candidates[index];
                if let Some(reason) = assessment.rejection() {
                    self.exclusions
                        .entry(candidate.id.clone())
                        .or_default()
                        .insert(format!("primary: {reason}"));
                } else {
                    self.secondary_pending.insert(index);
                }
                fs::remove_file(&candidate.image)
                    .context("cannot remove completed primary image")?;
            }
            self.primary.extend(assessments);
            offset = end;
        }
        Ok(())
    }

    fn assess_secondary(
        &mut self,
        assessor: &mut impl FnMut(&AssessmentBatch<'_>) -> Result<Vec<Assessment>>,
    ) -> Result<Vec<Exclusion>> {
        let pending = self.secondary_pending.iter().copied().collect::<Vec<_>>();
        if pending.is_empty() {
            return Ok(Vec::new());
        }
        let (candidates, assessments): (Vec<_>, Vec<_>) = pending
            .iter()
            .map(|&index| (self.candidates[index].clone(), self.primary[index].clone()))
            .unzip();
        let pool = select_up_to(
            &candidates,
            &assessments,
            (self.options.count * 3).min(MAX_CANDIDATES),
        )?;
        for chunk in pool.indices.chunks(ASSESSMENT_BATCH) {
            check_cancelled(self.cancelled)?;
            let mut candidates = chunk
                .iter()
                .map(|&index| candidates[index].clone())
                .collect::<Vec<_>>();
            let scratch = tempfile::Builder::new()
                .prefix("secondary-")
                .tempdir_in(self.directory.path())?;
            let contexts = candidates
                .iter()
                .map(|candidate| self.context(candidate, scratch.path()))
                .collect::<Result<Vec<_>>>()?;
            for (candidate, context) in candidates.iter_mut().zip(&contexts) {
                candidate.image = context.frames[1].image.clone();
            }
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
            let sheets = render_candidate_sheets(&entries, scratch.path())?;
            let batch = AssessmentBatch {
                phase: AssessmentPhase::Secondary,
                candidates: &candidates,
                sheets,
                contexts,
            };
            let assessments = assessor(&batch).context("secondary assessment failed")?;
            check_cancelled(self.cancelled)?;
            validate_assessments(&candidates, &assessments)?;
            for &index in chunk {
                self.secondary_pending.remove(&pending[index]);
            }
            self.secondary.extend(aligned(&candidates, assessments));
            self.secondary_candidates.extend(candidates);
        }
        Ok(pool.exclusions)
    }

    fn context(&mut self, candidate: &Candidate, scratch: &Path) -> Result<CandidateContext> {
        let source = self
            .sources
            .iter()
            .position(|source| source == &candidate.source)
            .expect("candidate source came from this work");
        let artifacts = timed(&mut self.metrics.extraction, || {
            self.tools.extract_context(
                &self.videos[source],
                candidate.actual_seconds,
                self.options.context_delta_seconds,
                self.cancelled,
            )
        })?;
        verify_frame(&self.videos[source], candidate, &artifacts[1])?;
        let mut frames = Vec::new();
        for (index, artifact) in artifacts.into_iter().enumerate() {
            let image = scratch.join(format!("{}-{index}.png", candidate.id));
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

    fn replenish(
        &mut self,
        assessor: &mut impl FnMut(&AssessmentBatch<'_>) -> Result<Vec<Assessment>>,
    ) -> Result<usize> {
        let before = self.attempts;
        let target = (self.options.count * 2)
            .max(self.videos.len())
            .min(MAX_CANDIDATES - before);
        let mut skipped = 0;
        while self.attempts - before < target && skipped < self.videos.len() {
            check_cancelled(self.cancelled)?;
            let source = self.next_source;
            self.next_source = (source + 1) % self.videos.len();
            let frame = next_unattempted(
                &self.videos[source],
                &self.attempted[source],
                self.events.as_ref().map(|events| &events[source]),
                self.cancelled,
            )?;
            let progress = if let Some(frame) = frame {
                self.produce(source, self.videos[source].frames[frame].seconds, assessor)?
            } else {
                false
            };
            skipped = if progress { 0 } else { skipped + 1 };
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

    fn semantic_event(&self, source: usize, seconds: f64) -> Result<Option<SemanticEvent>> {
        check_cancelled(self.cancelled)?;
        if let Some(events) = &self.events {
            for event in &events[source] {
                #[cfg(test)]
                tests::record_event_work();
                check_cancelled(self.cancelled)?;
                if (event.0.start_seconds..=event.0.end_seconds).contains(&seconds) {
                    return Ok(Some(event.0.clone()));
                }
            }
        }
        Ok(None)
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
            let name = selected_filename(rank, indices.len());
            let path = output.join(&name);
            let source = self
                .sources
                .iter()
                .position(|source| source == &candidate.source)
                .expect("candidate source came from this work");
            let artifact = timed(&mut self.metrics.extraction, || {
                self.tools.extract(
                    &self.videos[source],
                    candidate.actual_seconds,
                    self.cancelled,
                )
            })?;
            verify_frame(&self.videos[source], candidate, &artifact)?;
            let image = read_image(&artifact.path())?;
            encode_selected_jpeg(&image, &path)?;
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
        fs::remove_dir_all(self.directory.path().join("candidates"))
            .context("cannot remove completed candidate scratch")?;
        check_cancelled(self.cancelled)?;
        Ok(Artifacts {
            directory: self.directory,
            report,
        })
    }
}

fn verify_frame(video: &Video, candidate: &Candidate, artifact: &FrameArtifact) -> Result<()> {
    ensure!(
        artifact.source == video.source
            && artifact.actual_seconds == candidate.actual_seconds
            && (artifact.width, artifact.height)
                == (candidate.quality.width, candidate.quality.height),
        "re-extracted frame source, actual PTS or dimensions differ from the assessed candidate"
    );
    Ok(())
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
    events: Option<&BTreeSet<OrderedEvent<'_>>>,
    cancelled: &AtomicBool,
) -> Result<Option<usize>> {
    check_cancelled(cancelled)?;
    let mut best = None;
    let mut width = -1.0;
    let mut consider = |begin: usize, limit: usize| -> Result<()> {
        let mut start = begin;
        for end in attempted
            .range(begin..limit)
            .copied()
            .chain(std::iter::once(limit))
        {
            check_cancelled(cancelled)?;
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
        Ok(())
    };
    if let Some(events) = events {
        for event in events {
            #[cfg(test)]
            tests::EVENT_VISITS.with(|visits| visits.set(visits.get() + 1));
            let event = event.0;
            let begin = video
                .frames
                .partition_point(|frame| frame.seconds < event.start_seconds);
            let limit = video
                .frames
                .partition_point(|frame| frame.seconds <= event.end_seconds);
            consider(begin, limit)?;
        }
    } else {
        consider(0, video.frames.len())?;
    }
    Ok(best)
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<()> {
    if cancelled.load(Ordering::Relaxed) {
        Err(Interrupted.into())
    } else {
        Ok(())
    }
}

fn event_preference(left: &SemanticEvent, right: &SemanticEvent) -> std::cmp::Ordering {
    #[cfg(test)]
    tests::record_event_work();
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

fn selected_filename(rank: usize, count: usize) -> String {
    let width = if count < 100 { 2 } else { 3 };
    format!("selected-{rank:0width$}.jpg")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::FrameTime;
    use crate::selection::sample_positions;
    thread_local! {
        pub(super) static EVENT_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        static CANCEL_EVENT_WORK: std::cell::RefCell<Option<(std::sync::Arc<AtomicBool>, usize, usize)>> = const { std::cell::RefCell::new(None) };
    }

    pub(super) fn record_event_work() {
        CANCEL_EVENT_WORK.with(|probe| {
            if let Some((cancelled, work, limit)) = probe.borrow_mut().as_mut() {
                *work += 1;
                if *work == *limit {
                    cancelled.store(true, Ordering::Relaxed);
                }
            }
        });
    }

    #[test]
    fn interrupted_event_canonicalization_stops_before_processing_the_whole_input() {
        let videos = [video(&[0.0, 0.25, 0.5, 0.75], 1.0)];
        let sources = ["synthetic.mkv".to_owned()];
        let events = (0..4097)
            .map(|index| SemanticEvent {
                source: sources[0].clone(),
                start_seconds: 0.0,
                end_seconds: 1.0,
                representative_seconds: 0.5,
                importance: 80.0,
                scene: "exploration".to_owned(),
                reason: format!("canonical event {index:04}"),
            })
            .collect::<Vec<_>>();
        let cancelled = std::sync::Arc::new(AtomicBool::new(false));
        CANCEL_EVENT_WORK.with(|probe| *probe.borrow_mut() = Some((cancelled.clone(), 0, 64)));
        let result = index_events(&videos, &sources, Some(&events), &cancelled);
        let (_, work, _) = CANCEL_EVENT_WORK.with(|probe| probe.borrow_mut().take().unwrap());
        assert!(result.unwrap_err().downcast_ref::<Interrupted>().is_some());
        assert!(
            (64..=128).contains(&work),
            "canonicalization must stop within one bounded insertion after cancellation, rather than finishing all comparisons: {work}"
        );
    }

    #[test]
    fn interrupted_event_lookup_stops_before_scanning_the_whole_source() {
        let videos = [video(&[0.0, 0.25, 0.5, 0.75], 1.0)];
        let sources = vec!["synthetic.mkv".to_owned()];
        let mut events = (0..4097)
            .map(|index| SemanticEvent {
                source: sources[0].clone(),
                start_seconds: 0.0,
                end_seconds: 0.25,
                representative_seconds: 0.125,
                importance: 80.0,
                scene: "exploration".to_owned(),
                reason: format!("narrow event {index:04}"),
            })
            .collect::<Vec<_>>();
        events.push(SemanticEvent {
            end_seconds: 1.0,
            representative_seconds: 0.5,
            importance: 0.0,
            reason: "last containing event".to_owned(),
            ..events[0].clone()
        });
        let cancelled = std::sync::Arc::new(AtomicBool::new(false));
        let indexed = index_events(&videos, &sources, Some(&events), &cancelled).unwrap();
        let tools = MediaTools::default();
        let options = options(1);
        let mut metrics = StageSeconds::default();
        let work = Work {
            tools: &tools,
            videos: &videos,
            options: &options,
            cancelled: &cancelled,
            metrics: &mut metrics,
            directory: tempfile::tempdir().unwrap(),
            sources,
            events: indexed,
            attempted: vec![BTreeSet::new()],
            attempts: 0,
            next_source: 0,
            candidates: Vec::new(),
            primary: Vec::new(),
            secondary_pending: BTreeSet::new(),
            secondary_candidates: Vec::new(),
            secondary: Vec::new(),
            exclusions: BTreeMap::new(),
            notes: Vec::new(),
        };
        CANCEL_EVENT_WORK.with(|probe| *probe.borrow_mut() = Some((cancelled.clone(), 0, 64)));
        let result = work.semantic_event(0, 0.75);
        let (_, inspected, _) = CANCEL_EVENT_WORK.with(|probe| probe.borrow_mut().take().unwrap());
        let error =
            result.expect_err("lookup must interrupt before returning the last matching event");
        assert!(error.downcast_ref::<Interrupted>().is_some());
        assert_eq!(inspected, 64);
    }

    #[test]
    fn inactive_sources_do_not_inspect_unrelated_semantic_events() {
        let video = video(&[0.0, 0.25, 0.5, 0.75], 1.0);
        let event = SemanticEvent {
            source: "synthetic.mkv".to_owned(),
            start_seconds: 0.0,
            end_seconds: 1.0,
            representative_seconds: 0.5,
            importance: 80.0,
            scene: "exploration".to_owned(),
            reason: "indexed active source".to_owned(),
        };
        let events = vec![event; 4096];
        let sources = std::iter::once("synthetic.mkv".to_owned())
            .chain((1..MAX_CANDIDATES).map(|source| format!("inactive-{source:04}.mkv")))
            .collect::<Vec<_>>();
        assert_eq!(sources.len(), MAX_CANDIDATES);
        let videos = vec![video.clone(); sources.len()];
        let cancelled = AtomicBool::new(false);
        let indexed = index_events(&videos, &sources, Some(&events), &cancelled)
            .unwrap()
            .unwrap();
        assert_eq!(
            indexed[0].len(),
            1,
            "identical raw events must share one canonical indexed entry"
        );
        EVENT_VISITS.with(|visits| visits.set(0));
        for events in &indexed[1..] {
            assert!(
                next_unattempted(&video, &BTreeSet::new(), Some(events), &cancelled)
                    .unwrap()
                    .is_none()
            );
        }
        EVENT_VISITS.with(|visits| {
            assert_eq!(
                visits.get(),
                0,
                "inactive source traversal must inspect no raw semantic events"
            )
        });
        assert_eq!(
            next_unattempted(&video, &BTreeSet::new(), Some(&indexed[0]), &cancelled).unwrap(),
            Some(2)
        );
        EVENT_VISITS.with(|visits| assert_eq!(visits.get(), 1));
        cancelled.store(true, Ordering::Relaxed);
        let error =
            next_unattempted(&video, &BTreeSet::new(), Some(&indexed[1]), &cancelled).unwrap_err();
        assert!(
            error.downcast_ref::<Interrupted>().is_some(),
            "inactive source traversal must still observe cancellation"
        );
        let error = index_events(&videos, &sources, Some(&events), &cancelled).unwrap_err();
        assert!(error.downcast_ref::<Interrupted>().is_some());
    }

    fn options(samples_per_video: usize) -> PipelineOptions {
        PipelineOptions {
            count: 1,
            samples_per_video,
            context_delta_seconds: 0.25,
            semantic_events: None,
            model: ModelConditions {
                backend: "strata".to_owned(),
                model: "fixture".to_owned(),
                inference_level: "none".to_owned(),
                timeout_seconds: 60,
                cache_revision: "test".to_owned(),
            },
            game_context: None,
        }
    }

    fn video(times: &[f64], duration_seconds: f64) -> Video {
        Video {
            source: "synthetic.mkv".into(),
            stream_index: 0,
            start_seconds: 0.0,
            duration_seconds,
            width: 16,
            height: 16,
            frames: times
                .iter()
                .enumerate()
                .map(|(decode_index, &seconds)| FrameTime {
                    decode_index,
                    seconds,
                    duration_seconds: 0.25,
                })
                .collect(),
        }
    }

    #[test]
    fn sampled_plans_preserve_the_first_grid_request_for_each_actual_frame() {
        for video in [
            video(&[0.0], 0.25),
            video(&[0.0, 0.25, 0.5, 0.75], 1.0),
            video(&[0.0, 10.0, 11.0, 12.0], 12.25),
        ] {
            for count in [1, 2, 3, 5, 4096] {
                let mut ordinals = BTreeSet::new();
                let expected = sample_positions(video.duration_seconds, count)
                    .unwrap()
                    .into_iter()
                    .filter(|seconds| {
                        let frame = video
                            .frames
                            .iter()
                            .position(|frame| frame.seconds >= *seconds)
                            .unwrap_or(video.frames.len() - 1);
                        ordinals.insert(frame)
                    })
                    .map(f64::to_bits)
                    .collect::<Vec<_>>();
                let plans = initial_plans(
                    std::slice::from_ref(&video),
                    &["synthetic.mkv".to_owned()],
                    None,
                    count,
                    &AtomicBool::new(false),
                )
                .unwrap();
                assert_eq!(
                    plans.positions[0]
                        .iter()
                        .copied()
                        .map(f64::to_bits)
                        .collect::<Vec<_>>(),
                    expected,
                    "must preserve exact grid values at midpoint, first/final frames and VFR gaps for count {count}"
                );
            }
        }
    }

    #[test]
    fn distinct_initial_frames_over_the_budget_fail_before_decode() {
        let times = (0..MAX_CANDIDATES)
            .map(|index| index as f64 * 0.25)
            .collect::<Vec<_>>();
        let videos = [
            video(&times, MAX_CANDIDATES as f64 * 0.25),
            video(&times, MAX_CANDIDATES as f64 * 0.25),
        ];
        let result = initial_plans(
            &videos,
            &["one.mkv".to_owned(), "two.mkv".to_owned()],
            None,
            MAX_CANDIDATES,
            &AtomicBool::new(false),
        );
        assert!(
            result.is_err(),
            "the actual candidate budget must apply to all sources before any decoder or callback runs"
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("4096 actual-frame candidate limit")
        );

        let times = (0..8196)
            .map(|index| index as f64 * 0.5)
            .collect::<Vec<_>>();
        let videos = [video(&times, 4098.0)];
        let mut options = options(1);
        options.semantic_events = Some(
            (0..1366)
                .map(|index| SemanticEvent {
                    source: "synthetic.mkv".to_owned(),
                    start_seconds: index as f64 * 3.0,
                    representative_seconds: index as f64 * 3.0 + 1.0,
                    end_seconds: index as f64 * 3.0 + 2.0,
                    importance: 80.0,
                    scene: "exploration".to_owned(),
                    reason: "distinct actual-frame limit".to_owned(),
                })
                .collect(),
        );
        let error = initial_plans(
            &videos,
            &["synthetic.mkv".to_owned()],
            options.semantic_events.as_deref(),
            options.samples_per_video,
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("4096 actual-frame candidate limit")
        );
    }

    #[test]
    fn maximal_sparse_inputs_keep_only_one_request_per_actual_frame() {
        let sources = (0..MAX_CANDIDATES)
            .map(|index| format!("source-{index:04}.mkv"))
            .collect::<Vec<_>>();
        let videos = sources
            .iter()
            .map(|source| {
                let mut video = video(&[0.0], 0.25);
                video.source = source.into();
                video
            })
            .collect::<Vec<_>>();
        preflight_sampled_plans(&videos, MAX_CANDIDATES, &AtomicBool::new(false)).unwrap();
        let plans = initial_plans(
            &videos,
            &sources,
            None,
            MAX_CANDIDATES,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(plans.positions.len(), MAX_CANDIDATES);
        assert!(
            plans.positions.iter().all(|plan| plan == &[0.0]),
            "4096 samples for each of 4096 one-frame sources must retain only 4096 first requests, without materializing the 16M duplicate grid"
        );
    }

    #[test]
    fn sampled_preflight_shares_the_exact_global_actual_frame_boundary() {
        let times = (0..MAX_CANDIDATES / 2)
            .map(|index| index as f64 * 0.25)
            .collect::<Vec<_>>();
        let duration = MAX_CANDIDATES as f64 / 2.0 * 0.25;
        let mut videos = vec![video(&times, duration), video(&times, duration)];
        videos[0].source = "one.mkv".into();
        videos[1].source = "two.mkv".into();
        let cancelled = AtomicBool::new(false);
        preflight_sampled_plans(&videos, MAX_CANDIDATES, &cancelled).unwrap();
        let plans = initial_plans(&videos, &[], None, MAX_CANDIDATES, &cancelled).unwrap();
        assert_eq!(
            plans.positions.iter().map(Vec::len).sum::<usize>(),
            MAX_CANDIDATES
        );
        let mut extra = video(&[0.0], 0.25);
        extra.source = "extra.mkv".into();
        videos.push(extra);
        let error = preflight_sampled_plans(&videos, MAX_CANDIDATES, &cancelled).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("4096 actual-frame candidate limit")
        );
    }

    #[test]
    fn sampled_preflight_does_not_count_dense_grid_requests_twice_in_vfr_gaps() {
        let mut videos = [
            video(&[0.0, 0.25, 10.0, 11.0, 12.0], 12.25),
            video(&[0.0, 0.25, 10.0, 11.0, 12.0], 12.25),
        ];
        videos[0].source = "one.mkv".into();
        videos[1].source = "two.mkv".into();
        let cancelled = AtomicBool::new(false);
        // 8192 grid requests resolve to only ten actual frames across sources.
        preflight_sampled_plans(&videos, MAX_CANDIDATES, &cancelled).unwrap();
        let plans = initial_plans(&videos, &[], None, MAX_CANDIDATES, &cancelled).unwrap();
        assert_eq!(plans.positions.iter().map(Vec::len).sum::<usize>(), 10);
        for (video, positions) in videos.iter().zip(plans.positions) {
            let ordinals: BTreeSet<_> = positions
                .iter()
                .map(|seconds| frame_ordinal(video, *seconds))
                .collect();
            assert_eq!(ordinals.len(), positions.len());
        }
    }

    #[test]
    fn sampled_preflight_rejects_invalid_inputs_and_observes_cancellation() {
        let cancelled = AtomicBool::new(false);
        assert!(preflight_sampled_plans(&[], 1, &cancelled).is_err());
        for samples in [0, MAX_CANDIDATES + 1] {
            assert!(preflight_sampled_plans(&[video(&[0.0], 0.25)], samples, &cancelled).is_err());
        }
        for invalid in [
            video(&[], 0.25),
            video(&[0.0], 0.0),
            video(&[0.0], f64::NAN),
            Video {
                width: 0,
                ..video(&[0.0], 0.25)
            },
            Video {
                height: MAX_IMAGE_EDGE + 1,
                ..video(&[0.0], 0.25)
            },
        ] {
            assert!(preflight_sampled_plans(&[invalid], 1, &cancelled).is_err());
        }
        cancelled.store(true, Ordering::Relaxed);
        let error = preflight_sampled_plans(&[video(&[0.0], 0.25)], 1, &cancelled).unwrap_err();
        assert!(error.downcast_ref::<Interrupted>().is_some());
    }

    #[test]
    fn selected_filenames_sort_by_rank_across_99_100_and_999_boundaries() {
        for count in [99, 100, 999] {
            let ranked = (1..=count)
                .map(|rank| selected_filename(rank, count))
                .collect::<Vec<_>>();
            let mut sorted = ranked.clone();
            sorted.sort();
            assert_eq!(
                sorted, ranked,
                "filenames must preserve numeric ranks for {count} selections"
            );
            assert_eq!(
                ranked[98],
                if count == 99 {
                    "selected-99.jpg"
                } else {
                    "selected-099.jpg"
                }
            );
            if count >= 100 {
                assert_eq!(ranked[99], "selected-100.jpg");
            }
            if count == 999 {
                assert_eq!(ranked[998], "selected-999.jpg");
            }
        }
    }
}
