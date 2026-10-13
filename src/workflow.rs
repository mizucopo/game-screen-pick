//! Live use of the cached Rust selection path. Transports start only on misses.
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};
use serde_json::json;

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
    fn validate_transport(&mut self) -> Result<()> {
        check_cancelled(self.cancelled)?;
        ensure!(
            !self.cache_only,
            "cache-only: required inference checkpoint is missing"
        );
        if self.client.is_none() {
            self.client = Some(OpenAiClient::new(&self.config.ai)?);
        }
        Ok(())
    }

    fn client(&mut self) -> Result<&OpenAiClient> {
        self.validate_transport()?;
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
        ensure!(
            (1..=999).contains(&count),
            "selected count must be within 1..=999"
        );
        let media = protocol
            .media
            .as_ref()
            .context("image selection requires ai.protocol.media vision settings")?;
        let chunk_seconds = config
            .selection
            .semantic_chunk_seconds
            .min(media.max_video_seconds);
        ensure!(
            media.max_images as usize > 3 * (count * 3).min(crate::pipeline::ASSESSMENT_BATCH),
            "configured image capacity cannot cover the largest secondary assessment batch for this count"
        );
        if config.selection.method == SelectionMethod::SemanticVideo {
            ensure!(
                media.max_video_bytes > 0 && media.max_video_seconds > 0.0,
                "semantic_video is not supported by the configured backend/model contract"
            );
        }
        let model = config.ai.model_conditions();
        let game_conditions = match game {
            GameInput::Context(text) => {
                json!({"kind":"direct","context_digest":crate::storage::sha256(text.as_bytes())})
            }
            GameInput::Title(title) => {
                brave::validate_title(title)?;
                let search = config.brave.as_ref().context("--game-title requires an explicit Brave search contract; use --game-context or configure [brave]")?;
                search.validate_search_contract()?;
                json!({"kind":"generated","title_digest":crate::storage::sha256(title.as_bytes()),
                    "search_conditions_digest":crate::storage::sha256(search.search_conditions()?.as_bytes()),
                    "search":{"country":search.country,"search_lang":search.search_lang,"result_count":search.result_count},
                    "prompt_revision":brave::CONTEXT_PROMPT_REVISION,"schema_revision":brave::CONTEXT_SCHEMA_REVISION})
            }
        };
        let effective = json!({"revision":"rust-ai-workflow-v3","model":model,"service_url":config.ai.base_url.to_string(),
            "protocol":protocol,"game":game_conditions,"selection":{"method":config.selection.method.to_string(),
                "count":count,"samples_per_video":config.selection.samples_per_video,
                "context_delta_seconds":config.selection.context_delta_seconds,
                "semantic_chunk_seconds":semantic.then_some(config.selection.semantic_chunk_seconds),
                "effective_semantic_chunk_seconds":semantic.then_some(chunk_seconds)},
            "prompt_revision":prompt_revision,"schema_revision":schema_revision,"media_revision":crate::inference::MEDIA_REVISION});
        if let Some(outcome) = session.prepare_workflow(effective, cancelled)? {
            return Ok(outcome);
        }
        crate::contact_sheet::validate_candidate_sheet_limits(
            media.max_image_edge,
            media.max_image_pixels,
        )?;
        let mut title_geometry = None;
        let (resolved, persist) = match game {
            GameInput::Context(text) => {
                // Encoded context is a lower bound on every downstream request.
                // Reject an impossible budget without media/runtime work.
                ensure!(
                    serde_json::to_vec(text)?.len() as u64 <= protocol.max_request_bytes,
                    "direct game context exceeds inference request byte limit"
                );
                (ResolvedGameContext::direct(text), true)
            }
            GameInput::Title(title) => {
                let search = config.brave.as_ref().context("--game-title requires an explicit Brave search contract; use --game-context or configure [brave]")?;
                search.validate_search_contract()?;
                let conditions = config.ai.cache_conditions(
                    brave::CONTEXT_PROMPT_REVISION,
                    brave::CONTEXT_SCHEMA_REVISION,
                    brave::SEARCH_REVISION,
                    ResolvedGameContext::direct(""),
                )?;
                let context = session.generated_context_with_session(title, &search.search_conditions()?, &conditions, &model,
                    search.persist_context(), cancelled, |session| {
                        check_cancelled(cancelled)?;
                        ensure!(!cache_only, "cache-only: generated Game Context checkpoint is missing or persistence is disabled");
                        crate::inference::preflight_selection("", semantic, protocol.max_request_bytes)
                            .context("selection request exceeds configured byte limit")?;
                        let search_client = BraveClient::new(search)?;
                        inference.validate_transport()?;
                        brave::preflight_context(inference.client.as_ref().expect("validated transport"), title)?;
                        let videos = session.videos(tools, cancelled)?;
                        if semantic {
                            title_geometry = Some(plan_semantic_videos(videos, media, chunk_seconds, cancelled)?);
                        } else {
                            validate_image_geometry(&videos, media)?;
                            crate::pipeline::preflight_sampled_plans(&videos, config.selection.samples_per_video, cancelled)?;
                        }
                        let evidence = search_client.search_cancellable(title, cancelled)?;
                        check_cancelled(cancelled)?;
                        brave::generate_context_cancellable(inference.client()?, title, &evidence, cancelled)
                    })?;
                (context, search.persist_context())
            }
        };
        crate::inference::preflight_selection(
            resolved.text(),
            semantic,
            protocol.max_request_bytes,
        )
        .context("selection request exceeds configured byte limit")?;
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
                    let videos = match title_geometry.take() {
                        Some(videos) => videos,
                        None => plan_semantic_videos(
                            session.videos(tools, cancelled)?,
                            media,
                            chunk_seconds,
                            cancelled,
                        )?,
                    };
                    let mut plan = SemanticPlan {
                        durations: Default::default(),
                        events: Vec::new(),
                    };
                    for PlannedVideo { video, ranges } in videos {
                        let name = video
                            .source
                            .file_name()
                            .and_then(|name| name.to_str())
                            .context("semantic source name must be UTF-8")?;
                        plan.durations
                            .insert(name.to_owned(), video.duration_seconds);
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
                                    crate::inference::understand_cancellable(
                                        inference.client()?,
                                        resolved.text(),
                                        chunk,
                                        cancelled,
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
            ensure!(
                !events.is_empty(),
                "semantic understanding found no usable events; not enough candidates"
            );
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
            crate::inference::assess_cancellable(
                inference.client()?,
                resolved.text(),
                batch,
                cancelled,
            )
        })
    })();
    Ok(inference.runtime.finish_cancellable(result, cancelled))
}

fn validate_image_geometry(videos: &[Video], limits: &crate::config::MediaConfig) -> Result<()> {
    ensure!(
        videos.iter().all(|video| {
            video.width <= limits.max_image_edge
                && video.height <= limits.max_image_edge
                && u64::from(video.width) * u64::from(video.height) <= limits.max_image_pixels
        }),
        "input frame geometry exceeds configured image limits"
    );
    Ok(())
}

struct PlannedVideo {
    video: Video,
    ranges: Vec<(f64, f64)>,
}

fn plan_semantic_videos(
    videos: Vec<Video>,
    limits: &crate::config::MediaConfig,
    limit: f64,
    cancelled: &AtomicBool,
) -> Result<Vec<PlannedVideo>> {
    validate_image_geometry(&videos, limits)?;
    let mut plan = Vec::with_capacity(videos.len());
    let mut chunks = 0;
    for video in videos {
        check_cancelled(cancelled)?;
        let ranges = semantic_ranges(&video, limit)?;
        chunks += ranges.len();
        ensure!(
            chunks <= crate::selection::MAX_CANDIDATES,
            "semantic plan exceeds the chunk limit"
        );
        plan.push(PlannedVideo { video, ranges });
    }
    Ok(plan)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::FrameTime;

    fn video(name: &str, count: usize) -> Video {
        Video {
            source: name.into(),
            stream_index: 0,
            start_seconds: 0.0,
            duration_seconds: count as f64 * 0.25,
            width: 16,
            height: 16,
            frames: (0..count)
                .map(|decode_index| FrameTime {
                    decode_index,
                    seconds: decode_index as f64 * 0.25,
                    duration_seconds: 0.25,
                })
                .collect(),
        }
    }

    #[test]
    fn all_input_geometry_is_bounded_before_any_chunk_can_be_inferred() {
        let cancelled = AtomicBool::new(false);
        let limits = crate::config::MediaConfig {
            vision_revision: "fixture".into(),
            max_images: 64,
            max_image_pixels: 256,
            max_image_edge: 16,
            max_video_bytes: 1024,
            max_video_seconds: 3600.0,
        };
        let acceptable = plan_semantic_videos(
            vec![video("a.mkv", 4000), video("b.mkv", 96)],
            &limits,
            0.25,
            &cancelled,
        )
        .unwrap();
        assert_eq!(
            acceptable
                .iter()
                .map(|video| video.ranges.len())
                .sum::<usize>(),
            4096
        );
        let rejected = plan_semantic_videos(
            vec![video("a.mkv", 4000), video("b.mkv", 97)],
            &limits,
            0.25,
            &cancelled,
        );
        assert!(rejected.err().unwrap().to_string().contains("chunk limit"));
    }
}
