//! Live use of the cached Rust selection path. Transports start only on misses.
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};

use crate::ai::OpenAiClient;
use crate::brave::{self, BraveClient};
use crate::config::Config;
use crate::domain::{GameInput, SelectionMethod};
use crate::media::{Interrupted, MediaTools, Video};
use crate::pipeline::PipelineOptions;
use crate::run::{ResolvedGameContext, RunOutcome, SemanticPlan, Session};
use crate::runtime::{RuntimeCompletion, RuntimeManager};

pub struct Request<'a> {
    pub game: &'a GameInput,
    pub count: usize,
    pub cache_only: bool,
}

struct InferenceSession<'a> {
    config: &'a Config,
    cancelled: &'a AtomicBool,
    cache_only: bool,
    client: Option<OpenAiClient>,
    runtime: RuntimeManager,
}

impl InferenceSession<'_> {
    fn client(&mut self) -> Result<&OpenAiClient> {
        check_cancelled(self.cancelled)?;
        ensure!(
            !self.cache_only,
            "cache-only: required inference checkpoint is missing"
        );
        if self.client.is_none() {
            self.client = Some(OpenAiClient::new(&self.config.ai)?);
        }
        let client = self.client.as_ref().expect("initialized inference client");
        self.runtime.ensure_ready(client, self.cancelled)?;
        Ok(client)
    }
}

/// Call only after Session::prepare has established input/output/cache safety.
/// Stop failures are returned separately so they never replace the main error.
pub fn run(
    session: &mut Session,
    config: &Config,
    request: &Request<'_>,
    tools: &MediaTools,
    cancelled: &AtomicBool,
) -> Result<RuntimeCompletion<RunOutcome>> {
    let Request {
        game,
        count,
        cache_only,
    } = *request;
    let semantic = config.selection.method == SelectionMethod::SemanticVideo;
    let prompt_revision = format!(
        "{}:{}",
        crate::inference::ASSESSMENT_PROMPT_REVISION,
        if semantic {
            crate::inference::VIDEO_PROMPT_REVISION
        } else {
            "sampled-pts-v1"
        }
    );
    let schema_revision = format!(
        "{}:{}",
        crate::inference::ASSESSMENT_SCHEMA_REVISION,
        if semantic {
            crate::inference::VIDEO_SCHEMA_REVISION
        } else {
            "sampled-pts-v1"
        }
    );
    let mut inference = InferenceSession {
        config,
        cancelled,
        cache_only,
        client: None,
        runtime: RuntimeManager::new(config.runtime.as_ref(), &config.ai)?,
    };
    let result = (|| {
        check_cancelled(cancelled)?;
        let protocol = config
            .ai
            .protocol
            .as_ref()
            .context("ai.protocol is required for selection")?;
        protocol.validate(config.ai.backend, config.ai.inference_level)?;
        let media = protocol
            .media
            .as_ref()
            .context("image selection requires ai.protocol.media vision settings")?;
        ensure!(
            media.max_images > 0,
            "image selection is not supported by the configured model contract"
        );
        if config.selection.method == SelectionMethod::SemanticVideo {
            ensure!(
                media.max_video_bytes > 0 && media.max_video_seconds > 0.0,
                "semantic_video is not supported by the configured backend/model contract"
            );
        }
        let model = config.ai.model_conditions();
        let (resolved, persist) = match game {
            GameInput::Context(text) => (ResolvedGameContext::direct(text), true),
            GameInput::Title(title) => {
                let search = config.brave.as_ref().context("--game-title requires an explicit Brave search contract; use --game-context or configure [brave]")?;
                search.validate_search_contract()?;
                let conditions = config.ai.cache_conditions(
                    brave::CONTEXT_PROMPT_REVISION,
                    brave::CONTEXT_SCHEMA_REVISION,
                    brave::SEARCH_REVISION,
                    ResolvedGameContext::direct(""),
                )?;
                let context = session.generated_context(title, &search.search_conditions()?, &conditions, &model,
                    search.persist_context(), cancelled, || {
                        check_cancelled(cancelled)?;
                        ensure!(!cache_only, "cache-only: generated Game Context checkpoint is missing or persistence is disabled");
                        let evidence = BraveClient::new(search)?.search(title)?;
                        check_cancelled(cancelled)?;
                        brave::generate_context(inference.client()?, title, &evidence)
                    })?;
                (context, search.persist_context())
            }
        };
        let conditions = config.ai.cache_conditions(
            &prompt_revision,
            &schema_revision,
            crate::inference::MEDIA_REVISION,
            resolved.clone(),
        )?;
        let events = if config.selection.method == SelectionMethod::SemanticVideo {
            let events = session.semantic_plan(
                config.selection.semantic_chunk_seconds,
                &model,
                &conditions,
                cancelled,
                |session| {
                    let videos = session.videos(tools, cancelled)?;
                    let mut plan = SemanticPlan {
                        durations: Default::default(),
                        events: Vec::new(),
                    };
                    let mut chunks = 0;
                    for video in videos {
                        let name = video
                            .source
                            .file_name()
                            .and_then(|name| name.to_str())
                            .context("semantic source name must be UTF-8")?;
                        plan.durations
                            .insert(name.to_owned(), video.duration_seconds);
                        let ranges = semantic_ranges(
                            &video,
                            config
                                .selection
                                .semantic_chunk_seconds
                                .min(media.max_video_seconds),
                        )?;
                        chunks += ranges.len();
                        ensure!(
                            chunks <= crate::selection::MAX_CANDIDATES,
                            "semantic plan exceeds the chunk limit"
                        );
                        for (index, (start, end)) in ranges.into_iter().enumerate() {
                            check_cancelled(cancelled)?;
                            eprintln!(
                                "semantic_video: chunk {} (checkpoint reuse or inference)",
                                index + 1
                            );
                            plan.events.extend(session.semantic_chunk(
                                tools,
                                &video.source,
                                start,
                                end,
                                &model,
                                &conditions,
                                cancelled,
                                |chunk| {
                                    crate::inference::understand(
                                        inference.client()?,
                                        resolved.text(),
                                        chunk,
                                    )
                                },
                            )?);
                            ensure!(
                                plan.events.len() <= crate::selection::MAX_CANDIDATES,
                                "semantic plan exceeds the event limit"
                            );
                        }
                    }
                    Ok(plan)
                },
            )?;
            Some(events)
        } else {
            None
        };
        let options = PipelineOptions {
            count,
            samples_per_video: config.selection.samples_per_video,
            context_delta_seconds: config.selection.context_delta_seconds,
            semantic_events: events,
            model,
            game_context: persist.then(|| resolved.text().to_owned()),
        };
        session.run(tools, &options, &conditions, cancelled, |batch| {
            eprintln!(
                "{}: {:?} assessment, {} candidates (cache miss)",
                config.selection.method,
                batch.phase,
                batch.candidates.len()
            );
            crate::inference::assess(inference.client()?, resolved.text(), batch)
        })
    })();
    Ok(inference.runtime.finish(result))
}

/// Partition on real PTS boundaries; never create an empty or oversized chunk.
fn semantic_ranges(video: &Video, limit: f64) -> Result<Vec<(f64, f64)>> {
    ensure!(
        limit.is_finite() && limit > 0.0,
        "invalid semantic chunk duration limit"
    );
    let mut ranges = Vec::new();
    let mut first = 0;
    while first < video.frames.len() {
        let start = video.frames[first].seconds;
        let mut next = first;
        while next < video.frames.len() {
            let frame = &video.frames[next];
            let end = crate::media::probe_seconds(frame.seconds + frame.duration_seconds);
            if end - start > limit {
                break;
            }
            next += 1;
        }
        ensure!(
            next > first,
            "a video frame exceeds the configured semantic duration limit"
        );
        let end = video
            .frames
            .get(next)
            .map_or(video.duration_seconds, |frame| frame.seconds);
        ranges.push((start, end));
        ensure!(
            ranges.len() <= crate::selection::MAX_CANDIDATES,
            "semantic plan exceeds the chunk limit"
        );
        first = next;
    }
    ensure!(!ranges.is_empty(), "semantic input has no frames");
    Ok(ranges)
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<()> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(Interrupted.into());
    }
    Ok(())
}
