use std::{fmt, fs, path::Path};

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use url::Url;

use crate::domain::SelectionMethod;

#[derive(Clone, Debug)]
pub struct Config {
    pub selection: SelectionConfig,
    pub ai: AiConfig,
}

#[derive(Clone, Debug)]
pub struct SelectionConfig {
    pub method: SelectionMethod,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    Strata,
    Vllm,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Strata => "strata",
            Self::Vllm => "vllm",
        })
    }
}

/// Requested inference setting; backend/model capability checks belong to the client.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum InferenceLevel {
    None,
    Low,
    Medium,
    High,
}

impl fmt::Display for InferenceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::None => "none",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        })
    }
}

#[derive(Clone)]
pub struct AiConfig {
    pub backend: Backend,
    pub base_url: Url,
    pub model: String,
    pub inference_level: InferenceLevel,
    pub timeout_seconds: u64,
    pub cache_revision: String,
    pub protocol: Option<ProtocolConfig>,
    api_key: Option<String>,
}

/// Operator-declared server/model contract, not proof of live model capability.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolConfig {
    pub reasoning_format: ReasoningFormat,
    pub supported_inference_levels: Vec<InferenceLevel>,
    pub json_schema: bool,
    pub max_output_tokens: u32,
    pub max_request_bytes: u64,
    pub max_response_bytes: u64,
    pub max_attempts: u8,
    pub media: Option<MediaConfig>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningFormat {
    ReasoningEffort,
    ChatTemplate,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MediaConfig {
    pub vision_revision: String,
    pub max_images: u32,
    pub max_image_pixels: u64,
    pub max_image_edge: u32,
    pub max_video_bytes: u64,
    pub max_video_seconds: f64,
}

impl ProtocolConfig {
    pub fn validate(&self, backend: Backend, level: InferenceLevel) -> Result<()> {
        if self.supported_inference_levels.is_empty()
            || !self.supported_inference_levels.contains(&level)
            || self
                .supported_inference_levels
                .iter()
                .enumerate()
                .any(|(i, value)| self.supported_inference_levels[..i].contains(value))
        {
            bail!(
                "ai.protocol must explicitly support the selected inference level without duplicates"
            );
        }
        if !(1..=3).contains(&self.max_attempts)
            || !(1..=65536).contains(&self.max_output_tokens)
            || !(1024..=256 * 1024 * 1024).contains(&self.max_request_bytes)
            || !(1024..=16 * 1024 * 1024).contains(&self.max_response_bytes)
        {
            bail!("invalid ai.protocol request/response/token/attempt limits");
        }
        if let Some(media) = &self.media {
            if media.vision_revision.trim().is_empty()
                || media.max_images > 64
                || media.max_image_pixels == 0
                || media.max_image_pixels > 200_000_000
                || media.max_image_edge == 0
                || media.max_image_edge > 32768
                || media.max_video_bytes > 128 * 1024 * 1024
                || !media.max_video_seconds.is_finite()
                || media.max_video_seconds < 0.0
                || media.max_video_seconds > 3600.0
                || (media.max_video_bytes == 0) != (media.max_video_seconds == 0.0)
            {
                bail!("invalid ai.protocol.media vision revision or limits");
            }
            if backend == Backend::Strata && media.max_video_bytes != 0 {
                bail!("Strata semantic_video input is not supported");
            }
        }
        Ok(())
    }
}

impl fmt::Debug for AiConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AiConfig")
            .field("backend", &self.backend)
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("inference_level", &self.inference_level)
            .field("timeout_seconds", &self.timeout_seconds)
            .field("cache_revision", &self.cache_revision)
            .field("protocol", &self.protocol)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}

impl AiConfig {
    pub fn resolved_api_key(&self) -> Option<String> {
        let fallback = std::env::var("GAME_SCREEN_PICK_API_KEY").ok();
        self.effective_api_key(fallback.as_deref())
            .map(str::to_owned)
    }

    /// Supply GAME_SCREEN_PICK_API_KEY as fallback. Authentication stays out of
    /// Debug output and is not part of serializable configuration metadata.
    pub fn effective_api_key<'a>(&'a self, fallback: Option<&'a str>) -> Option<&'a str> {
        self.api_key
            .as_deref()
            .or_else(|| fallback.map(str::trim).filter(|value| !value.is_empty()))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    selection: RawSelectionConfig,
    ai: RawAiConfig,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSelectionConfig {
    method: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAiConfig {
    backend: String,
    base_url: String,
    model: String,
    inference_level: String,
    timeout_seconds: u64,
    cache_revision: String,
    api_key: Option<String>,
    protocol: Option<ProtocolConfig>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let source = fs::read_to_string(path)
            .map_err(|error| anyhow!("cannot read configuration file: {error}"))?;
        Self::parse(&source)
    }

    pub fn parse(source: &str) -> Result<Self> {
        let raw: RawConfig = toml::from_str(source).map_err(|error| {
            // The parser's Display includes source excerpts and its message can
            // quote arbitrary values. Neither is safe for configuration secrets.
            let reason = if error.message().starts_with("unknown field ") {
                "unknown configuration key"
            } else if error.message().starts_with("missing field ") {
                "missing required configuration field"
            } else {
                "invalid TOML syntax or configuration value type"
            };
            if let Some(span) = error.span() {
                let prefix = &source.as_bytes()[..span.start.min(source.len())];
                let line = prefix.iter().filter(|&&byte| byte == b'\n').count() + 1;
                let column = prefix
                    .iter()
                    .rev()
                    .take_while(|&&byte| byte != b'\n')
                    .count()
                    + 1;
                anyhow!("{reason} at line {line}, column {column}; see config.example.toml")
            } else {
                anyhow!("{reason}; see config.example.toml")
            }
        })?;

        let method = match raw.selection.method.as_str() {
            "sampled_frames" => SelectionMethod::SampledFrames,
            "semantic_video" => SelectionMethod::SemanticVideo,
            _ => bail!("selection.method must be sampled_frames or semantic_video"),
        };
        let backend = match raw.ai.backend.as_str() {
            "strata" => Backend::Strata,
            "vllm" => Backend::Vllm,
            _ => bail!("ai.backend must be strata or vllm"),
        };
        let inference_level = match raw.ai.inference_level.as_str() {
            "none" => InferenceLevel::None,
            "low" => InferenceLevel::Low,
            "medium" => InferenceLevel::Medium,
            "high" => InferenceLevel::High,
            _ => bail!("ai.inference_level must be none, low, medium, or high"),
        };
        if !(1..=3600).contains(&raw.ai.timeout_seconds) {
            bail!("ai.timeout_seconds must be an integer from 1 to 3600");
        }
        let model = nonblank(raw.ai.model, "ai.model must not be blank")?;
        let cache_revision =
            nonblank(raw.ai.cache_revision, "ai.cache_revision must not be blank")?;
        let base_url = parse_base_url(raw.ai.base_url.trim())?;
        if let Some(protocol) = &raw.ai.protocol {
            protocol.validate(backend, inference_level)?;
        }
        let api_key = raw
            .ai
            .api_key
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());

        Ok(Self {
            selection: SelectionConfig { method },
            ai: AiConfig {
                backend,
                base_url,
                model,
                inference_level,
                timeout_seconds: raw.ai.timeout_seconds,
                cache_revision,
                protocol: raw.ai.protocol,
                api_key,
            },
        })
    }
}

fn nonblank(value: String, message: &'static str) -> Result<String> {
    let value = value.trim();
    if value.is_empty() {
        bail!(message);
    }
    Ok(value.to_owned())
}

fn parse_base_url(value: &str) -> Result<Url> {
    let invalid = || anyhow!("ai.base_url must be an absolute HTTP(S) URL with a host");
    let Some((scheme, authority)) = value.split_once("://") else {
        return Err(invalid());
    };
    if !(scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https"))
        || value.chars().any(char::is_control)
    {
        return Err(invalid());
    }
    let mut url = Url::parse(value).map_err(|_| invalid())?;
    if url.host_str().is_none() {
        return Err(invalid());
    }
    if !url.username().is_empty()
        || url.password().is_some()
        || authority
            .split(['/', '?', '#'])
            .next()
            .is_some_and(|host| host.contains('@'))
        || url.query().is_some()
        || url.fragment().is_some()
    {
        bail!("ai.base_url must not contain authentication, a query, or a fragment");
    }
    // Preserve custom service prefixes; endpoint callers can append paths to
    // the normalized base rather than assuming every service lives at /v1.
    let path = url.path().trim_end_matches('/').to_owned();
    url.set_path(if path.is_empty() { "/" } else { &path });
    Ok(url)
}
