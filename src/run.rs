//! Cached selection and publication; AI transports remain caller supplied.
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use url::Url;

use crate::cache::{self, Cache, CacheStats, InputIdentity};
use crate::media::{ChunkArtifact, Interrupted, MediaTools, Video, discover_inputs};
use crate::output::Output;
use crate::pipeline::{self, AssessmentBatch, AssessmentPhase, ModelConditions, PipelineOptions};
use crate::selection::{
    Assessment, MAX_CANDIDATES, SemanticEvent, event_positions, validate_assessments,
};
use crate::storage;

const ALGORITHM: &str = "rust-selection-v1";
const PROBE: &str = "ffprobe-pts-v1";
const WORKFLOW: &str = "rust-workflow-v1";

pub struct RunPaths {
    pub input_directory: PathBuf,
    pub output_directory: PathBuf,
    pub cache_directory: PathBuf,
    pub config: Option<PathBuf>,
}

/// A resolved description paired with the conditions that generated it.
/// Direct descriptions carry no generation provenance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ResolvedGameContext {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_key: Option<String>,
}

impl ResolvedGameContext {
    pub fn direct(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            generation_key: None,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

/// Effective dependencies, never credentials. Resolved context participates in
/// keys even when its text is not permitted in a report or context checkpoint.
#[derive(Clone, Debug, Serialize)]
pub struct CacheConditions {
    pub service_url: String,
    pub prompt_revision: String,
    pub response_schema_revision: String,
    pub media_revision: String,
    pub resolved_game_context: ResolvedGameContext,
}

impl CacheConditions {
    fn normalized(&self, require_context: bool) -> Result<Self> {
        let service = Url::parse(&self.service_url).context("invalid inference service URL")?;
        ensure!(
            ["http", "https"].contains(&service.scheme())
                && service.host_str().is_some()
                && service.username().is_empty()
                && service.password().is_none()
                && service.query().is_none()
                && service.fragment().is_none(),
            "inference service URL must be HTTP(S) without credentials, query or fragment"
        );
        ensure!(
            [
                &self.prompt_revision,
                &self.response_schema_revision,
                &self.media_revision
            ]
            .iter()
            .all(|value| !value.trim().is_empty()),
            "cache conditions must not be blank"
        );
        ensure!(
            !require_context || !self.resolved_game_context.text().trim().is_empty(),
            "cache conditions must not be blank"
        );
        Ok(Self {
            service_url: service.to_string(),
            ..self.clone()
        })
    }
}

/// Cumulative work since `Session::prepare`, including preceding methods/runs.
/// Callback counts record caller invocations; both elapsed fields share this
/// session interval. `RunOutcome::warm` describes the current output shortcut.
#[derive(Debug, Serialize)]
pub struct RunMetrics {
    pub cache: CacheStats,
    pub output_bytes_read: u64,
    /// Candidate images, context frames and sheets read for assessment keys.
    pub assessment_bytes_read: u64,
    pub probe_calls: u64,
    pub assessment_calls: u64,
    pub context_calls: u64,
    pub semantic_calls: u64,
    pub elapsed_seconds: f64,
}

#[derive(Debug)]
pub struct RunOutcome {
    pub warm: bool,
    pub report: Value,
    pub metrics: RunMetrics,
}

/// Bounded semantic result without all-frame probe metadata. Each event was
/// validated against its actual chunk before this aggregate was checkpointed.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticPlan {
    pub durations: BTreeMap<String, f64>,
    pub events: Vec<SemanticEvent>,
}

/// Holds both locks across preflight, callbacks, checkpoints and publication.
pub struct Session {
    input_directory: PathBuf,
    inputs: Vec<PathBuf>,
    identities: Option<Vec<InputIdentity>>,
    cache: Cache,
    output: Output,
    report_bytes_read: u64,
    assessment_bytes_read: u64,
    prepared_at: Instant,
    probe_calls: u64,
    assessment_calls: u64,
    context_calls: u64,
    semantic_calls: u64,
    workflow: Option<Value>,
}

impl Session {
    /// Check completed owned output before generating context, probing media or
    /// resolving semantic plans. Conditions must describe every requested
    /// dependency without credentials or raw context/search material; callers
    /// hash the original game input instead of persisting its text here.
    /// Input content identities are added internally and may be rehashed after
    /// cache loss. No external command or callback runs in this method.
    pub fn prepare_workflow(
        &mut self,
        conditions: Value,
        cancelled: &AtomicBool,
    ) -> Result<Option<RunOutcome>> {
        check_cancelled(cancelled)?;
        ensure!(
            conditions.is_object()
                && serde_json::to_vec(&conditions)?.len() <= 1024 * 1024
                && !has_credential_fields(&conditions),
            "workflow conditions must be a bounded non-secret object"
        );
        let identities = self.identify_inputs(cancelled)?;
        let inputs = self
            .inputs
            .iter()
            .zip(&identities)
            .map(|(path, identity)| Ok((source_name(path)?.to_owned(), &identity.content_digest)))
            .collect::<Result<Vec<_>>>()?;
        let key = cache::key(&json!({"workflow":WORKFLOW,"algorithm":ALGORITHM,
            "probe":PROBE,"inputs":inputs,"conditions":conditions}))?;
        self.workflow = Some(json!({"key":key,"revision":WORKFLOW,"conditions":conditions}));
        let previous = self.output.verify_existing(cancelled)?;
        if let Some(owned) = previous.filter(|owned| {
            owned.report["cache"]["namespace"] == cache::NAMESPACE
                && owned.report["cache"]["workflow"]["key"] == key
        }) {
            self.verify_inputs()?;
            check_cancelled(cancelled)?;
            return Ok(Some(RunOutcome {
                warm: true,
                report: owned.report,
                metrics: self.metrics(),
            }));
        }
        self.verify_inputs()?;
        check_cancelled(cancelled)?;
        Ok(None)
    }

    /// Reuse the small completed semantic plan before reading frame checkpoints.
    /// Individual chunk checkpoints still recover partial/missing plans.
    pub fn semantic_plan(
        &mut self,
        chunk_seconds: f64,
        model: &ModelConditions,
        conditions: &CacheConditions,
        cancelled: &AtomicBool,
        generate: impl FnOnce(&mut Self) -> Result<SemanticPlan>,
    ) -> Result<Vec<SemanticEvent>> {
        check_cancelled(cancelled)?;
        ensure!(
            chunk_seconds.is_finite() && (0.1..=3600.0).contains(&chunk_seconds),
            "invalid semantic plan chunk duration"
        );
        pipeline::validate_model(model)?;
        let conditions = conditions.normalized(true)?;
        let identities = self.identify_inputs(cancelled)?;
        let inputs = self
            .inputs
            .iter()
            .zip(&identities)
            .map(|(path, identity)| {
                Ok((
                    source_name(path)?.to_owned(),
                    identity.content_digest.clone(),
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let key = cache::key(&json!({"algorithm":"semantic-plan-v1","probe":PROBE,
            "inputs":inputs,"chunk_seconds":chunk_seconds,"model":model,"conditions":conditions}))?;
        let validate = |plan: &SemanticPlan| -> Result<()> {
            ensure!(
                plan.durations.len() == inputs.len()
                    && plan.durations.keys().eq(inputs.keys())
                    && plan
                        .durations
                        .values()
                        .all(|seconds| seconds.is_finite() && *seconds > 0.0)
                    && !plan.events.is_empty()
                    && plan.events.len() <= MAX_CANDIDATES,
                "semantic plan has invalid sources, duration or event count"
            );
            let mut intervals = BTreeSet::new();
            for event in &plan.events {
                let duration = plan
                    .durations
                    .get(&event.source)
                    .context("semantic plan contains an unknown source")?;
                event_positions(event, *duration)?;
                ensure!(
                    intervals.insert((
                        &event.source,
                        event.start_seconds.to_bits(),
                        event.end_seconds.to_bits(),
                        event.representative_seconds.to_bits()
                    )),
                    "semantic plan contains duplicate events"
                );
            }
            Ok(())
        };
        if let Some(plan) =
            self.cache
                .read_validated::<SemanticPlan>("semantic-plans", &key, |plan| {
                    validate(plan).is_ok()
                })?
        {
            self.verify_inputs()?;
            check_cancelled(cancelled)?;
            return Ok(plan.events);
        }
        self.verify_inputs()?;
        let plan = generate(self)?;
        validate(&plan)?;
        self.verify_inputs()?;
        self.cache.write("semantic-plans", &key, &plan)?;
        check_cancelled(cancelled)?;
        Ok(plan.events)
    }
    /// Uses the same validated probe checkpoints as selection and semantic chunks.
    /// A warm semantic plan does not decode inputs or contact inference servers.
    pub fn videos(&mut self, tools: &MediaTools, cancelled: &AtomicBool) -> Result<Vec<Video>> {
        let identities = self.identify_inputs(cancelled)?;
        let mut videos = Vec::with_capacity(self.inputs.len());
        for (path, identity) in self.inputs.iter().zip(&identities) {
            self.verify_inputs()?;
            videos.push(probe(
                &mut self.cache,
                tools,
                path,
                identity,
                cancelled,
                &mut self.probe_calls,
            )?);
            self.verify_inputs()?;
        }
        Ok(videos)
    }
    /// No external command, search, inference or runtime operation occurs here.
    pub fn prepare(paths: &RunPaths, cancelled: &AtomicBool) -> Result<Self> {
        let prepared_at = Instant::now();
        check_cancelled(cancelled)?;
        let input = storage::checked_path(&paths.input_directory)?;
        let output = storage::checked_path(&paths.output_directory)?;
        let cache_path = storage::checked_path(&paths.cache_directory)?;
        let config = paths
            .config
            .as_deref()
            .map(storage::checked_path)
            .transpose()?;
        let protected = [&input, &cache_path]
            .into_iter()
            .chain(config.as_ref())
            .collect::<Vec<_>>();
        ensure!(
            !input.starts_with(&cache_path) && !cache_path.starts_with(&input),
            "cache and input paths must not overlap"
        );
        if let Some(config) = &config {
            ensure!(
                fs::symlink_metadata(config)?.is_file(),
                "configuration must be an ordinary file"
            );
            ensure!(
                !config.starts_with(&cache_path),
                "cache must not contain configuration"
            );
            crate::config::Config::load(config)?;
        }
        let inputs = discover_inputs(&input)?;
        validate_inputs(&inputs)?;
        let output = Output::prepare(
            &output,
            &protected.iter().map(|p| p.as_path()).collect::<Vec<_>>(),
            cancelled,
        )?;
        check_cancelled(cancelled)?;
        if !cache_path.exists() {
            fs::create_dir(&cache_path)
                .context(storage::OperationalError("cannot create cache directory"))?;
        }
        let cache = Cache::open(&cache_path)?;
        check_cancelled(cancelled)?;
        Ok(Self {
            input_directory: input,
            inputs,
            identities: None,
            cache,
            output,
            report_bytes_read: 0,
            assessment_bytes_read: 0,
            prepared_at,
            probe_calls: 0,
            assessment_calls: 0,
            context_calls: 0,
            semantic_calls: 0,
            workflow: None,
        })
    }

    pub fn run(
        &mut self,
        tools: &MediaTools,
        options: &PipelineOptions,
        conditions: &CacheConditions,
        cancelled: &AtomicBool,
        mut assessor: impl FnMut(&AssessmentBatch<'_>) -> Result<Vec<Assessment>>,
    ) -> Result<RunOutcome> {
        check_cancelled(cancelled)?;
        let conditions = conditions.normalized(true)?;
        pipeline::validate_parameters(options)?;
        ensure!(
            options
                .game_context
                .as_ref()
                .is_none_or(|context| context == conditions.resolved_game_context.text()),
            "persisted context must match resolved inference context"
        );
        if let Some(events) = &options.semantic_events {
            ensure!(!events.is_empty(), "semantic events must be nonempty");
            let sources = self
                .inputs
                .iter()
                .map(|path| source_name(path))
                .collect::<Result<BTreeSet<_>>>()?;
            for event in events {
                check_cancelled(cancelled)?;
                ensure!(
                    sources.contains(event.source.as_str()),
                    "semantic event refers to an unknown video source"
                );
                event_positions(event, event.end_seconds)?;
            }
        }
        let identities = self.identify_inputs(cancelled)?;
        let named = self
            .inputs
            .iter()
            .zip(&identities)
            .map(|(path, identity)| {
                (
                    source_name(path).map(str::to_owned),
                    &identity.content_digest,
                )
            })
            .map(|(name, digest)| Ok((name?, digest)))
            .collect::<Result<Vec<_>>>()?;
        let run_key = cache::key(
            &json!({"algorithm":ALGORITHM,"probe":PROBE,"inputs":named,"options":options,"conditions":conditions}),
        )?;
        let previous = self.output.verify_existing(cancelled)?;
        if let Some(owned) = previous.filter(|owned| {
            owned.report["cache"]["namespace"] == cache::NAMESPACE
                && owned.report["cache"]["key"] == run_key
                && self.workflow.as_ref().is_none_or(|workflow| {
                    owned.report["cache"]["workflow"]["key"] == workflow["key"]
                })
        }) {
            self.verify_inputs()?;
            check_cancelled(cancelled)?;
            return Ok(RunOutcome {
                warm: true,
                report: owned.report,
                metrics: self.metrics(),
            });
        }
        let mut videos = Vec::with_capacity(self.inputs.len());
        for (path, identity) in self.inputs.iter().zip(&identities) {
            self.verify_inputs()?;
            videos.push(probe(
                &mut self.cache,
                tools,
                path,
                identity,
                cancelled,
                &mut self.probe_calls,
            )?);
            self.verify_inputs()?;
        }
        let by_source = self
            .inputs
            .iter()
            .zip(&identities)
            .map(|(path, identity)| {
                Ok((
                    source_name(path)?.to_owned(),
                    identity.content_digest.clone(),
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let artifacts = pipeline::run(tools, &videos, options, cancelled, |batch| {
            check_cancelled(cancelled)?;
            self.verify_inputs()?;
            let candidates = batch.candidates.iter().map(|candidate| Ok(json!({
                "id":candidate.id,"source":candidate.source,"input":by_source.get(&candidate.source).context("unknown assessment source")?,
                "requested":candidate.requested_seconds,"actual":candidate.actual_seconds,"duration":candidate.duration_seconds,
                "quality":candidate.quality,"event":candidate.event,"image":self.assessment_digest(&candidate.image, cancelled)?
            }))).collect::<Result<Vec<_>>>()?;
            let contexts = batch.contexts.iter().map(|context| {
                let frames = context.frames.iter().map(|frame| Ok(json!({"source":frame.source,"requested":frame.requested_seconds,"actual":frame.actual_seconds,"image":self.assessment_digest(&frame.image, cancelled)?}))).collect::<Result<Vec<_>>>()?;
                Ok(json!({"id":context.id,"frames":frames}))
            }).collect::<Result<Vec<_>>>()?;
            let sheets = batch
                .sheets
                .iter()
                .map(|path| self.assessment_digest(path, cancelled))
                .collect::<Result<Vec<_>>>()?;
            let key = cache::key(
                &json!({"algorithm":ALGORITHM,"phase":match batch.phase {AssessmentPhase::Primary=>"primary",AssessmentPhase::Secondary=>"secondary"},"model":options.model,"conditions":conditions,"candidates":candidates,"contexts":contexts,"sheets":sheets}),
            )?;
            if let Some(cached) =
                self.cache
                    .read_validated::<Vec<Assessment>>("assessments", &key, |cached| {
                        validate_assessments(batch.candidates, cached).is_ok()
                    })?
            {
                self.verify_inputs()?;
                check_cancelled(cancelled)?;
                return Ok(cached);
            }
            self.verify_inputs()?;
            self.assessment_calls += 1;
            let result = assessor(batch)?;
            validate_assessments(batch.candidates, &result)?;
            self.verify_inputs()?;
            // A complete response survives a signal received just after it.
            self.cache.write("assessments", &key, &result)?;
            check_cancelled(cancelled)?;
            Ok(result)
        })?;
        self.verify_inputs()?;
        check_cancelled(cancelled)?;
        let report_path = artifacts.path().canonicalize()?.join("report.json");
        let report_bytes = storage::read_regular(&report_path, 64 * 1024 * 1024)?;
        self.report_bytes_read += report_bytes.len() as u64;
        let mut report: Value = serde_json::from_slice(&report_bytes)?;
        report["cache"] = json!({"namespace":cache::NAMESPACE,"key":run_key,"algorithm":ALGORITHM,"probe":PROBE,
            "service_url":conditions.service_url,"prompt_revision":conditions.prompt_revision,
            "response_schema_revision":conditions.response_schema_revision,"media_revision":conditions.media_revision,
            "context_digest":storage::sha256(conditions.resolved_game_context.text().as_bytes())});
        if let Some(workflow) = &self.workflow {
            report["cache"]["workflow"] = workflow.clone();
        }
        storage::atomic_write(&report_path, &serde_json::to_vec_pretty(&report)?)?;
        let owned = self.output.publish_checked(&artifacts, cancelled, || {
            verify_workflow(&self.input_directory, &self.inputs, Some(&identities))
        })?;
        Ok(RunOutcome {
            warm: false,
            report: owned.report,
            metrics: self.metrics(),
        })
    }

    /// Checkpoint one validated semantic chunk; missing/corrupt chunks alone
    /// invoke the caller again. Events use source-relative original video times.
    #[allow(clippy::too_many_arguments)]
    pub fn semantic_chunk(
        &mut self,
        tools: &MediaTools,
        source: &Path,
        start: f64,
        end: f64,
        model: &ModelConditions,
        conditions: &CacheConditions,
        cancelled: &AtomicBool,
        mut understand: impl FnMut(&ChunkArtifact) -> Result<Vec<SemanticEvent>>,
    ) -> Result<Vec<SemanticEvent>> {
        check_cancelled(cancelled)?;
        let conditions = conditions.normalized(true)?;
        pipeline::validate_model(model)?;
        ensure!(
            start.is_finite() && end.is_finite() && start >= 0.0 && end > start,
            "invalid semantic chunk range"
        );
        let source = storage::checked_path(source)?;
        let index = self
            .inputs
            .iter()
            .position(|path| path == &source)
            .context("semantic chunk source is not an input")?;
        let identities = self.identify_inputs(cancelled)?;
        let identity = &identities[index];
        let video = probe(
            &mut self.cache,
            tools,
            &source,
            identity,
            cancelled,
            &mut self.probe_calls,
        )?;
        self.verify_inputs()?;
        ensure!(
            start.is_finite()
                && end.is_finite()
                && start >= 0.0
                && end > start
                && end <= video.duration_seconds,
            "invalid semantic chunk range"
        );
        let name = source_name(&source)?;
        let (first, last) = crate::media::chunk_frames(&video, start, end)?;
        let actual_start = first.seconds;
        let actual_end = crate::media::probe_seconds(last.seconds + last.duration_seconds);
        let key = cache::key(
            &json!({"algorithm":"semantic-chunk-v2","probe":PROBE,"input":identity.content_digest,"source":name,"stream":video.stream_index,"start":start,"end":end,"model":model,"conditions":conditions}),
        )?;
        let validate =
            |events: &Vec<SemanticEvent>, actual_start: f64, actual_end: f64| -> Result<()> {
                for event in events {
                    event_positions(event, video.duration_seconds)?;
                    ensure!(
                        event.source == name
                            && event.start_seconds >= actual_start
                            && event.end_seconds <= actual_end,
                        "semantic event is outside its source/chunk"
                    );
                }
                Ok(())
            };
        if let Some(events) =
            self.cache
                .read_validated::<Vec<SemanticEvent>>("semantic", &key, |events| {
                    validate(events, actual_start, actual_end).is_ok()
                })?
        {
            self.verify_inputs()?;
            check_cancelled(cancelled)?;
            return Ok(events);
        }
        self.verify_inputs()?;
        let chunk = tools.extract_chunk(&video, start, end, cancelled)?;
        self.verify_inputs()?;
        self.semantic_calls += 1;
        let events = understand(&chunk)?;
        validate(
            &events,
            chunk.actual_start_seconds,
            chunk.actual_end_seconds,
        )?;
        self.verify_inputs()?;
        self.cache.write("semantic", &key, &events)?;
        check_cancelled(cancelled)?;
        Ok(events)
    }

    /// Only call with persist=true after the search contract permits storing the
    /// generated text. Raw search responses have no persistence path here.
    #[allow(clippy::too_many_arguments)]
    pub fn generated_context(
        &mut self,
        title: &str,
        search_conditions: &str,
        generation_conditions: &CacheConditions,
        model: &ModelConditions,
        persist: bool,
        cancelled: &AtomicBool,
        mut generate: impl FnMut() -> Result<String>,
    ) -> Result<ResolvedGameContext> {
        check_cancelled(cancelled)?;
        ensure!(!title.trim().is_empty(), "game title must not be blank");
        ensure!(
            !search_conditions.trim().is_empty(),
            "search conditions must not be blank"
        );
        let conditions = generation_conditions.normalized(false)?;
        pipeline::validate_model(model)?;
        self.identify_inputs(cancelled)?;
        let key = cache::key(
            &json!({"title":title,"search":search_conditions,"generation":{
                "service_url":conditions.service_url,"prompt_revision":conditions.prompt_revision,
                "response_schema_revision":conditions.response_schema_revision,"media_revision":conditions.media_revision
            },"model":model}),
        )?;
        if persist
            && let Some(context) =
                self.cache
                    .read_validated::<String>("contexts", &key, |context| {
                        !context.trim().is_empty()
                    })?
        {
            self.verify_inputs()?;
            check_cancelled(cancelled)?;
            return Ok(ResolvedGameContext {
                text: context,
                generation_key: Some(key),
            });
        }
        self.verify_inputs()?;
        self.context_calls += 1;
        let context = generate()?;
        ensure!(
            !context.trim().is_empty(),
            "generated game context must not be blank"
        );
        self.verify_inputs()?;
        if persist {
            self.cache.write("contexts", &key, &context)?;
        }
        check_cancelled(cancelled)?;
        Ok(ResolvedGameContext {
            text: context,
            generation_key: Some(key),
        })
    }

    fn verify_inputs(&self) -> Result<()> {
        verify_workflow(
            &self.input_directory,
            &self.inputs,
            self.identities.as_deref(),
        )
    }

    fn identify_inputs(&mut self, cancelled: &AtomicBool) -> Result<Vec<InputIdentity>> {
        self.verify_inputs()?;
        if let Some(identities) = &self.identities {
            return Ok(identities.clone());
        }
        let identities = self.cache.identify(&self.inputs, cancelled)?;
        verify_workflow(&self.input_directory, &self.inputs, Some(&identities))?;
        self.identities = Some(identities.clone());
        Ok(identities)
    }

    fn assessment_digest(&mut self, path: &Path, cancelled: &AtomicBool) -> Result<String> {
        let (_, digest) = storage::digest_file_cancelled(path, cancelled, |bytes| {
            self.assessment_bytes_read += bytes;
        })?;
        Ok(digest)
    }

    fn metrics(&self) -> RunMetrics {
        let elapsed_seconds = self.prepared_at.elapsed().as_secs_f64();
        RunMetrics {
            cache: CacheStats {
                elapsed_seconds,
                ..self.cache.stats()
            },
            output_bytes_read: self.output.bytes_read() + self.report_bytes_read,
            assessment_bytes_read: self.assessment_bytes_read,
            probe_calls: self.probe_calls,
            assessment_calls: self.assessment_calls,
            context_calls: self.context_calls,
            semantic_calls: self.semantic_calls,
            elapsed_seconds,
        }
    }
}

fn has_credential_fields(value: &Value) -> bool {
    match value {
        Value::Object(object) => object.iter().any(|(name, value)| {
            matches!(
                name.to_ascii_lowercase().as_str(),
                "api_key"
                    | "apikey"
                    | "authorization"
                    | "password"
                    | "secret"
                    | "credentials"
                    | "token"
                    | "access_token"
                    | "refresh_token"
                    | "bearer_token"
            ) || has_credential_fields(value)
        }),
        Value::Array(values) => values.iter().any(has_credential_fields),
        _ => false,
    }
}

pub(crate) fn validate_inputs(inputs: &[PathBuf]) -> Result<()> {
    ensure!(inputs.len() <= MAX_CANDIDATES, "input count exceeds 4096");
    let mut aliases = BTreeSet::new();
    for path in inputs {
        ensure!(
            !source_name(path)?.chars().any(char::is_control),
            "video source names must not contain controls"
        );
        let metadata = fs::symlink_metadata(path)?;
        ensure!(
            aliases.insert((metadata.dev(), metadata.ino())),
            "input paths refer to the same file more than once"
        );
    }
    Ok(())
}

fn probe(
    cache: &mut Cache,
    tools: &MediaTools,
    path: &Path,
    identity: &InputIdentity,
    cancelled: &AtomicBool,
    calls: &mut u64,
) -> Result<Video> {
    let key = cache::key(&json!({"probe":PROBE,"input":identity.content_digest}))?;
    if let Some(mut video) = cache.read_validated::<Video>("probe", &key, valid_video)? {
        video.source = path.to_owned();
        return Ok(video);
    }
    *calls += 1;
    let mut video = tools.probe(path, cancelled)?;
    cache::verify_inputs(&[path.to_owned()], std::slice::from_ref(identity))?;
    ensure!(valid_video(&video), "invalid probed video checkpoint");
    let source = video.source.clone();
    video.source = PathBuf::new();
    cache.write("probe", &key, &video)?;
    video.source = source;
    Ok(video)
}

fn verify_workflow(
    directory: &Path,
    inputs: &[PathBuf],
    identities: Option<&[InputIdentity]>,
) -> Result<()> {
    let directory = storage::checked_path(directory)?;
    let current =
        discover_inputs(&directory).context("input directory changed during processing")?;
    ensure!(
        current == inputs,
        "input directory changed during processing"
    );
    if let Some(identities) = identities {
        cache::verify_inputs(inputs, identities)?;
    }
    Ok(())
}

fn valid_video(video: &Video) -> bool {
    video.start_seconds.is_finite()
        && video.duration_seconds.is_finite()
        && video.duration_seconds > 0.0
        && video.width >= 16
        && video.height >= 16
        && video.width <= crate::quality::MAX_IMAGE_EDGE
        && video.height <= crate::quality::MAX_IMAGE_EDGE
        && u64::from(video.width) * u64::from(video.height) <= crate::quality::MAX_IMAGE_PIXELS
        && !video.frames.is_empty()
        && video.frames[0].seconds == 0.0
        && video.frames.iter().all(|frame| {
            frame.seconds.is_finite()
                && frame.seconds >= 0.0
                && frame.duration_seconds.is_finite()
                && frame.duration_seconds > 0.0
                && frame.seconds + frame.duration_seconds <= video.duration_seconds + 0.000001
        })
        && video.frames.windows(2).all(|pair| {
            pair[0].seconds < pair[1].seconds
                && pair[0].decode_index < pair[1].decode_index
                && pair[0].duration_seconds
                    == crate::media::probe_seconds(pair[1].seconds - pair[0].seconds)
        })
        && video.frames.last().is_some_and(|last| {
            last.duration_seconds
                == crate::media::probe_seconds(video.duration_seconds - last.seconds)
        })
}

fn source_name(path: &Path) -> Result<&str> {
    path.file_name()
        .and_then(|name| name.to_str())
        .context("UTF-8 source filename required")
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

    fn context_test_paths(root: &Path) -> RunPaths {
        let root = root.canonicalize().unwrap();
        let input = root.join("input");
        fs::create_dir(&input).unwrap();
        fs::write(input.join("recording.mkv"), b"input").unwrap();
        RunPaths {
            input_directory: input,
            output_directory: root.join("output"),
            cache_directory: root.join("cache"),
            config: None,
        }
    }

    fn context_test_conditions() -> (CacheConditions, ModelConditions) {
        (
            CacheConditions {
                service_url: "http://localhost:8000".to_owned(),
                prompt_revision: "prompt-v1".to_owned(),
                response_schema_revision: "schema-v1".to_owned(),
                media_revision: "media-v1".to_owned(),
                resolved_game_context: ResolvedGameContext::direct(""),
            },
            ModelConditions {
                backend: "vllm".to_owned(),
                model: "test-model".to_owned(),
                inference_level: "none".to_owned(),
                timeout_seconds: 60,
                cache_revision: "model-v1".to_owned(),
            },
        )
    }

    #[test]
    fn generated_context_rejects_input_mutations_during_first_callback_without_checkpoint() {
        for persist in [false, true] {
            for replace in [false, true] {
                let directory = tempfile::tempdir().unwrap();
                let paths = context_test_paths(directory.path());
                let source = paths.input_directory.join("recording.mkv");
                let (conditions, model) = context_test_conditions();
                let cancelled = AtomicBool::new(false);
                let mut session = Session::prepare(&paths, &cancelled).unwrap();
                let mut calls = 0;
                let result = session.generated_context(
                    "test game",
                    "search-v1",
                    &conditions,
                    &model,
                    persist,
                    &cancelled,
                    || {
                        calls += 1;
                        if replace {
                            let replacement = source.with_extension("replacement");
                            fs::write(&replacement, b"input").unwrap();
                            fs::rename(replacement, &source).unwrap();
                        } else {
                            fs::write(&source, b"other").unwrap();
                        }
                        Ok("generated context".to_owned())
                    },
                );
                let error = result.unwrap_err();
                assert!(
                    error.to_string().contains("input changed"),
                    "persist={persist}, replace={replace}: {error:#}"
                );
                assert_eq!(calls, 1);
                assert!(
                    !paths
                        .cache_directory
                        .join(cache::NAMESPACE)
                        .join("contexts")
                        .exists(),
                    "a context generated while inputs change must not be checkpointed"
                );
                assert!(!paths.output_directory.exists());
            }
        }
    }

    #[test]
    fn generated_context_rejects_changed_input_before_warm_return_or_new_callback() {
        let directory = tempfile::tempdir().unwrap();
        let paths = context_test_paths(directory.path());
        let (conditions, model) = context_test_conditions();
        let cancelled = AtomicBool::new(false);
        let mut session = Session::prepare(&paths, &cancelled).unwrap();
        session
            .generated_context(
                "test game",
                "search-v1",
                &conditions,
                &model,
                true,
                &cancelled,
                || Ok("saved context".to_owned()),
            )
            .unwrap();
        fs::write(paths.input_directory.join("recording.mkv"), b"other").unwrap();
        for persist in [false, true] {
            let error = session
                .generated_context(
                    "test game",
                    "search-v1",
                    &conditions,
                    &model,
                    persist,
                    &cancelled,
                    || panic!("changed input must fail before generation"),
                )
                .unwrap_err();
            assert!(error.to_string().contains("input changed"), "{error:#}");
        }
        assert_eq!(session.metrics().context_calls, 1);
    }

    #[test]
    fn interrupted_assessment_hash_counts_partial_bytes_before_same_session_retry() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let input = root.join("input");
        fs::create_dir(&input).unwrap();
        fs::write(input.join("recording.mkv"), b"input").unwrap();
        let paths = RunPaths {
            input_directory: input,
            output_directory: root.join("output"),
            cache_directory: root.join("cache"),
            config: None,
        };
        let cancelled = AtomicBool::new(false);
        let mut session = Session::prepare(&paths, &cancelled).unwrap();
        let image = root.join("candidate.png");
        let contents = vec![7; 256 * 1024];
        fs::write(&image, &contents).unwrap();
        let interruption = storage::interrupt_digest_after(64 * 1024);
        let error = session.assessment_digest(&image, &cancelled).unwrap_err();
        drop(interruption);
        assert!(error.downcast_ref::<Interrupted>().is_some());
        assert_eq!(session.metrics().assessment_bytes_read, 64 * 1024);
        cancelled.store(false, Ordering::Relaxed);
        assert_eq!(
            session.assessment_digest(&image, &cancelled).unwrap(),
            storage::sha256(&contents)
        );
        assert_eq!(
            session.metrics().assessment_bytes_read,
            64 * 1024 + contents.len() as u64
        );
    }
}
