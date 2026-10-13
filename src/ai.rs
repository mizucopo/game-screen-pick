//! One bounded OpenAI-compatible JSON boundary. No search, runtime or cache I/O.
use std::io::Cursor;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderValue, LOCATION};
use serde_json::{Value, json};

use crate::config::{AiConfig, Backend, InferenceLevel, ProtocolConfig, ReasoningFormat};
use crate::http::{HttpClient, check_cancelled, connection_refused};
use crate::pipeline::ModelConditions;
use crate::run::{CacheConditions, ResolvedGameContext};

const TRANSPORT_REVISION: &str = "openai-json-inline-media-v1";

/// Parts retain their order. Only local inline media is accepted, so the server
/// cannot fetch a URL with an independent redirect/authentication policy.
pub enum Content {
    Text(String),
    Image {
        bytes: Vec<u8>,
        mime: String,
    },
    Video {
        bytes: Vec<u8>,
        mime: String,
        duration_seconds: f64,
    },
}

pub struct JsonRequest {
    pub prompt: String,
    pub schema: Value,
    pub content: Vec<Content>,
}

pub struct OpenAiClient {
    config: AiConfig,
    http: HttpClient,
    authorization: Option<HeaderValue>,
    secret: Option<String>,
}

struct PreparedRequest {
    schema: jsonschema::Validator,
    content: Vec<Value>,
    body: Vec<u8>,
}

impl AiConfig {
    /// Safe to compute before a cache hit: no credential resolution or I/O.
    pub fn model_conditions(&self) -> ModelConditions {
        ModelConditions {
            backend: self.backend.to_string(),
            model: self.model.clone(),
            inference_level: self.inference_level.to_string(),
            timeout_seconds: self.timeout_seconds,
            cache_revision: self.cache_revision.clone(),
        }
    }

    /// Pair this with model_conditions for every context/assessment/chunk key.
    /// The protocol digest includes vision, limits and the reasoning encoding;
    /// model weights/server changes additionally require cache_revision updates.
    pub fn cache_conditions(
        &self,
        prompt_revision: &str,
        response_schema_revision: &str,
        media_revision: &str,
        resolved_game_context: ResolvedGameContext,
    ) -> Result<CacheConditions> {
        let protocol = self.checked_protocol()?;
        ensure!(
            [prompt_revision, response_schema_revision, media_revision]
                .iter()
                .all(|value| !value.trim().is_empty()),
            "inference cache revisions must not be blank"
        );
        let digest = crate::cache::key(&json!({
            "transport": TRANSPORT_REVISION, "protocol": protocol, "media": media_revision
        }))?;
        Ok(CacheConditions {
            service_url: self.base_url.to_string(),
            prompt_revision: prompt_revision.to_owned(),
            response_schema_revision: response_schema_revision.to_owned(),
            media_revision: format!("{TRANSPORT_REVISION}:{digest}"),
            resolved_game_context,
        })
    }

    fn checked_protocol(&self) -> Result<&ProtocolConfig> {
        ensure!(
            ["http", "https"].contains(&self.base_url.scheme())
                && self.base_url.host_str().is_some()
                && self.base_url.username().is_empty()
                && self.base_url.password().is_none()
                && self.base_url.query().is_none()
                && self.base_url.fragment().is_none()
                && !self.model.trim().is_empty()
                && !self.cache_revision.trim().is_empty()
                && (1..=3600).contains(&self.timeout_seconds),
            "invalid inference connection settings"
        );
        let protocol = self.protocol.as_ref().ok_or_else(|| {
            anyhow!("ai.protocol must declare the server/model contract before inference")
        })?;
        protocol.validate(self.backend, self.inference_level)?;
        Ok(protocol)
    }
}

impl OpenAiClient {
    /// Construction never contacts /models or starts/stops a server. Callers
    /// should invoke this only inside cache-miss callbacks after output preflight.
    pub fn new(config: &AiConfig) -> Result<Self> {
        config.checked_protocol()?;
        let secret = config.resolved_api_key();
        let loopback = match config.base_url.host() {
            Some(url::Host::Ipv4(address)) => address.is_loopback(),
            Some(url::Host::Ipv6(address)) => address.is_loopback(),
            Some(url::Host::Domain(name)) => name == "localhost",
            None => false,
        };
        // Check the resolved key, including the environment fallback, before
        // constructing an authenticated client. Never trust DNS as loopback.
        ensure!(
            secret.is_none() || config.base_url.scheme() == "https" || loopback,
            "inference authentication requires HTTPS for non-loopback hosts"
        );
        let authorization = secret
            .as_ref()
            .map(|key| {
                let mut value = HeaderValue::from_str(&format!("Bearer {key}"))
                    .map_err(|_| anyhow!("invalid inference authentication header"))?;
                value.set_sensitive(true);
                Ok::<_, anyhow::Error>(value)
            })
            .transpose()?;
        let http =
            HttpClient::new().map_err(|_| anyhow!("cannot initialize inference HTTP client"))?;
        Ok(Self {
            config: config.clone(),
            http,
            authorization,
            secret,
        })
    }

    /// Runtime readiness uses this same authenticated transport. Only a refused
    /// connection or an explicit temporary-unavailability status means not ready;
    /// a live incompatible, unauthorized or malformed endpoint is an error.
    /// The caller supplies an absolute deadline so startup polling stays bounded.
    pub fn is_ready(&self, deadline: Instant) -> Result<bool> {
        self.is_ready_cancellable(deadline, &AtomicBool::new(false))
    }

    pub fn is_ready_cancellable(&self, deadline: Instant, cancelled: &AtomicBool) -> Result<bool> {
        check_cancelled(cancelled)?;
        let protocol = self.config.checked_protocol()?;
        let deadline =
            deadline.min(Instant::now() + Duration::from_secs(self.config.timeout_seconds));
        let mut url = self.config.base_url.clone();
        url.path_segments_mut()
            .map_err(|_| anyhow!("invalid inference service path"))?
            .pop_if_empty()
            .push("models");
        for redirect in 0..=3 {
            let mut request = self.http.get(url.clone()).timeout(remaining(deadline)?);
            if let Some(auth) = &self.authorization {
                request = request.header(AUTHORIZATION, auth);
            }
            let response = match self
                .http
                .send(request, deadline, cancelled, "inference readiness")
            {
                Ok(response) => response,
                Err(error) if connection_refused(&error) => return Ok(false),
                Err(error) => return Err(error),
            };
            let status = response.status();
            if status.is_redirection() {
                ensure!(redirect < 3, "inference readiness redirect limit exceeded");
                let next = response
                    .headers()
                    .get(LOCATION)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|location| url.join(location).ok())
                    .ok_or_else(|| anyhow!("invalid inference readiness redirect"))?;
                ensure!(
                    next.origin() == self.config.base_url.origin()
                        && next.username().is_empty()
                        && next.password().is_none()
                        && next.query().is_none()
                        && next.fragment().is_none(),
                    "inference readiness redirect must stay on the configured origin without credentials/query/fragment"
                );
                url = next;
                continue;
            }
            if [502, 503, 504].contains(&status.as_u16()) {
                return Ok(false);
            }
            ensure!(
                status.is_success(),
                "inference readiness HTTP status {}",
                status.as_u16()
            );
            ensure!(
                response
                    .headers()
                    .get(CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.split(';').next())
                    .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json")),
                "inference readiness response must be application/json"
            );
            ensure!(
                response
                    .content_length()
                    .is_none_or(|length| length <= protocol.max_response_bytes),
                "inference readiness response exceeds byte limit"
            );
            let bytes = self.http.read_body(
                response,
                protocol.max_response_bytes,
                deadline,
                cancelled,
                "inference readiness",
            )?;
            check_cancelled(cancelled)?;
            remaining(deadline)?;
            ensure!(
                bytes.len() as u64 <= protocol.max_response_bytes,
                "inference readiness response exceeds byte limit"
            );
            let models: Value = serde_json::from_slice(&bytes)
                .map_err(|_| anyhow!("invalid inference readiness response"))?;
            self.reject_secret(&models)?;
            let identities = models["data"]
                .as_array()
                .ok_or_else(|| anyhow!("invalid inference readiness model list"))?;
            let expected = protocol
                .response_model
                .as_deref()
                .unwrap_or(&self.config.model);
            ensure!(
                identities
                    .iter()
                    .all(|model| model["id"].as_str().is_some())
                    && identities
                        .iter()
                        .any(|model| model["id"].as_str() == Some(expected)),
                "inference readiness endpoint does not serve the declared model"
            );
            check_cancelled(cancelled)?;
            return Ok(true);
        }
        unreachable!("redirect limit fails before exhausting the loop")
    }

    /// Always validate JSON Schema locally, then the caller's domain contract
    /// (IDs, scores, media evidence, source/time bounds). Neither /models nor
    /// structural JSON can prove that a model actually understood the media.
    pub fn generate<T>(
        &self,
        request: &JsonRequest,
        validate: impl FnMut(&Value) -> Result<T>,
    ) -> Result<T> {
        self.generate_cancellable(request, &AtomicBool::new(false), validate)
    }

    /// Validate the actual initial request without HTTP or runtime I/O. Use
    /// this before spending search quota on a request that cannot be sent.
    /// Media decoding and complete inline/base64 payload sizing are identical
    /// to generation; later response-correction prompts are sized separately.
    pub fn validate_request(&self, request: &JsonRequest) -> Result<()> {
        self.prepare_request(request, self.config.checked_protocol()?)?;
        Ok(())
    }

    pub fn generate_cancellable<T>(
        &self,
        request: &JsonRequest,
        cancelled: &AtomicBool,
        mut validate: impl FnMut(&Value) -> Result<T>,
    ) -> Result<T> {
        check_cancelled(cancelled)?;
        let protocol = self.config.checked_protocol()?;
        let deadline = Instant::now() + Duration::from_secs(self.config.timeout_seconds);
        let PreparedRequest {
            schema,
            mut content,
            mut body,
        } = self.prepare_request(request, protocol)?;
        let mut failure = "invalid JSON";
        for attempt in 0..protocol.max_attempts {
            check_cancelled(cancelled)?;
            remaining(deadline)?;
            if attempt > 0 {
                content.push(json!({"type":"text", "text":format!(
                    "The previous response failed: {failure}. Correct it. Return only JSON matching the supplied schema and all requested IDs, media evidence and time/score bounds."
                )}));
                body = self.payload_body(request, protocol, &content)?;
            }
            let text = self.post(&body, protocol, deadline, cancelled)?;
            check_cancelled(cancelled)?;
            let value: Value = match serde_json::from_str(&text) {
                Ok(value) => value,
                Err(_) => {
                    failure = "invalid JSON";
                    continue;
                }
            };
            self.reject_secret(&value)?;
            if !schema.is_valid(&value) {
                failure = "JSON Schema violation";
                continue;
            }
            match validate(&value) {
                Ok(result) => {
                    check_cancelled(cancelled)?;
                    remaining(deadline)?;
                    return Ok(result);
                }
                Err(_) => {
                    failure = "application contract violation";
                }
            }
        }
        check_cancelled(cancelled)?;
        bail!(
            "inference validation failed after {} attempts: {failure}",
            protocol.max_attempts
        )
    }

    fn prepare_request(
        &self,
        request: &JsonRequest,
        protocol: &ProtocolConfig,
    ) -> Result<PreparedRequest> {
        ensure!(
            !request.prompt.trim().is_empty(),
            "inference prompt must not be blank"
        );
        let schema_bytes = serde_json::to_vec(&request.schema)
            .map_err(|_| anyhow!("cannot encode inference schema"))?;
        ensure!(
            schema_bytes.len() as u64 <= protocol.max_request_bytes,
            "inference schema exceeds request byte limit"
        );
        // Remote HTTP/file resolution is disabled in Cargo.toml. A bad schema
        // fails before any inference and its diagnostic is never echoed.
        let schema = jsonschema::options()
            .should_validate_formats(true)
            .build(&request.schema)
            .map_err(|_| anyhow!("invalid or nonlocal inference JSON Schema"))?;
        let content = self.content(request, protocol)?;
        let body = self.payload_body(request, protocol, &content)?;
        self.post_request(self.completion_url()?, Vec::new())
            .build()
            .map_err(|_| anyhow!("invalid inference HTTP request"))?;
        Ok(PreparedRequest {
            schema,
            content,
            body,
        })
    }

    fn payload_body(
        &self,
        request: &JsonRequest,
        protocol: &ProtocolConfig,
        content: &[Value],
    ) -> Result<Vec<u8>> {
        let mut payload = json!({
            "model":self.config.model, "stream":false, "max_tokens":protocol.max_output_tokens,
            "messages":[{"role":"user","content":content}]
        });
        match protocol.reasoning_format {
            ReasoningFormat::ReasoningEffort => {
                payload["reasoning_effort"] = json!(self.config.inference_level.to_string());
            }
            ReasoningFormat::ChatTemplate => {
                payload["chat_template_kwargs"] = if self.config.inference_level
                    == InferenceLevel::None
                {
                    json!({"enable_thinking":false})
                } else {
                    json!({"enable_thinking":true,"reasoning_effort":self.config.inference_level.to_string()})
                };
            }
        }
        if protocol.json_schema {
            payload["response_format"] = json!({"type":"json_schema", "json_schema":{
                "name":"response", "strict":true, "schema":request.schema
            }});
        }
        let body =
            serde_json::to_vec(&payload).map_err(|_| anyhow!("cannot encode inference request"))?;
        ensure!(
            body.len() as u64 <= protocol.max_request_bytes,
            "inference request exceeds byte limit"
        );
        Ok(body)
    }

    fn content(&self, request: &JsonRequest, protocol: &ProtocolConfig) -> Result<Vec<Value>> {
        let mut parts = vec![json!({"type":"text","text":format!(
            "{}\nReturn only JSON matching this schema: {}", request.prompt, request.schema
        )})];
        let mut images = 0;
        let mut videos = 0;
        let mut bytes_total = request.prompt.len() as u64;
        for part in &request.content {
            let bytes = match part {
                Content::Text(text) => text.len(),
                Content::Image { bytes, .. } | Content::Video { bytes, .. } => bytes.len(),
            };
            bytes_total = bytes_total
                .checked_add(bytes as u64)
                .ok_or_else(|| anyhow!("inference content exceeds request byte limit"))?;
            ensure!(
                bytes_total <= protocol.max_request_bytes,
                "inference content exceeds request byte limit"
            );
            match part {
                Content::Text(text) => parts.push(json!({"type":"text","text":text})),
                Content::Image { bytes, mime } => {
                    let media = protocol
                        .media
                        .as_ref()
                        .ok_or_else(|| anyhow!("image input requires explicit vision settings"))?;
                    images += 1;
                    ensure!(
                        images <= media.max_images,
                        "image count exceeds the declared vision limit"
                    );
                    validate_image(bytes, mime, media)?;
                    parts.push(
                        json!({"type":"image_url","image_url":{"url":data_url(mime, bytes)}}),
                    );
                }
                Content::Video {
                    bytes,
                    mime,
                    duration_seconds,
                } => {
                    ensure!(
                        self.config.backend != Backend::Strata,
                        "Strata semantic_video input is not supported"
                    );
                    let media = protocol.media.as_ref().ok_or_else(|| {
                        anyhow!("semantic_video requires explicit video/vision settings")
                    })?;
                    videos += 1;
                    ensure!(
                        videos == 1
                            && !bytes.is_empty()
                            && ["video/mp4", "video/quicktime"].contains(&mime.as_str())
                            && media.max_video_bytes > 0
                            && bytes.len() as u64 <= media.max_video_bytes
                            && has_video_header(bytes, mime)
                            && duration_seconds.is_finite()
                            && *duration_seconds > 0.0
                            && *duration_seconds <= media.max_video_seconds,
                        "semantic_video input exceeds or lacks the declared MP4/MOV/byte/time contract"
                    );
                    parts.push(
                        json!({"type":"video_url","video_url":{"url":data_url(mime, bytes)}}),
                    );
                }
            }
        }
        Ok(parts)
    }

    fn completion_url(&self) -> Result<url::Url> {
        let mut url = self.config.base_url.clone();
        url.path_segments_mut()
            .map_err(|_| anyhow!("invalid inference service path"))?
            .pop_if_empty()
            .push("chat")
            .push("completions");
        Ok(url)
    }

    fn post_request(&self, url: url::Url, body: Vec<u8>) -> reqwest::RequestBuilder {
        let mut request = self
            .http
            .post(url)
            .header(CONTENT_TYPE, "application/json")
            .body(body);
        if let Some(auth) = &self.authorization {
            request = request.header(AUTHORIZATION, auth);
        }
        request
    }

    fn post(
        &self,
        body: &[u8],
        protocol: &ProtocolConfig,
        deadline: Instant,
        cancelled: &AtomicBool,
    ) -> Result<String> {
        let mut url = self.completion_url()?;
        for redirect in 0..=3 {
            let request = self
                .post_request(url.clone(), body.to_owned())
                .timeout(remaining(deadline)?);
            let response = self.http.send(request, deadline, cancelled, "inference")?;
            let status = response.status();
            if status.is_redirection() {
                ensure!(
                    [307, 308].contains(&status.as_u16()) && redirect < 3,
                    "inference redirect must preserve POST and stay within the redirect limit"
                );
                let next = response
                    .headers()
                    .get(LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|location| url.join(location).ok())
                    .ok_or_else(|| anyhow!("invalid inference redirect"))?;
                ensure!(
                    next.origin() == self.config.base_url.origin()
                        && next.username().is_empty()
                        && next.password().is_none()
                        && next.query().is_none()
                        && next.fragment().is_none(),
                    "inference redirect must stay on the configured origin without credentials/query/fragment"
                );
                url = next;
                continue;
            }
            ensure!(
                status.is_success(),
                "inference HTTP status {}",
                status.as_u16()
            );
            ensure!(
                response
                    .headers()
                    .get(CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .is_some_and(|v| v
                        .split(';')
                        .next()
                        .is_some_and(|v| v.trim().eq_ignore_ascii_case("application/json"))),
                "inference response must be application/json"
            );
            ensure!(
                response
                    .content_length()
                    .is_none_or(|n| n <= protocol.max_response_bytes),
                "inference response exceeds byte limit"
            );
            let bytes = self.http.read_body(
                response,
                protocol.max_response_bytes,
                deadline,
                cancelled,
                "inference",
            )?;
            check_cancelled(cancelled)?;
            remaining(deadline)?;
            ensure!(
                bytes.len() as u64 <= protocol.max_response_bytes,
                "inference response exceeds byte limit"
            );
            let envelope: Value = serde_json::from_slice(&bytes)
                .map_err(|_| anyhow!("invalid inference response envelope"))?;
            let response_model = protocol
                .response_model
                .as_deref()
                .unwrap_or(&self.config.model);
            return completion_text(envelope, response_model);
        }
        unreachable!("redirect limit fails before exhausting the loop")
    }

    fn reject_secret(&self, value: &Value) -> Result<()> {
        if let Some(secret) = &self.secret {
            ensure!(
                !contains_secret(value, secret),
                "inference response contains authentication material"
            );
        }
        Ok(())
    }
}

fn contains_secret(value: &Value, secret: &str) -> bool {
    match value {
        Value::String(text) => text == secret || embedded_credential_echo(text, secret),
        Value::Array(values) => values.iter().any(|v| contains_secret(v, secret)),
        Value::Object(values) => values
            .iter()
            .any(|(key, v)| embedded_credential_echo(key, secret) || contains_secret(v, secret)),
        _ => false,
    }
}

fn embedded_credential_echo(text: &str, secret: &str) -> bool {
    // Local servers commonly use dummy credentials such as `a` or `test`.
    // Substrings cannot distinguish those from ordinary words or JSON fields.
    if secret.len() >= 16 && text.contains(secret) {
        return true;
    }
    let bytes = text.as_bytes();
    let marker = b"Bearer ";
    bytes
        .windows(marker.len())
        .enumerate()
        .any(|(start, prefix)| {
            if !prefix.eq_ignore_ascii_case(marker) {
                return false;
            }
            // An ASCII marker establishes UTF-8 boundaries before slicing.
            let framed = text[..start]
                .chars()
                .next_back()
                .is_none_or(|previous| !previous.is_alphanumeric() && previous != '_');
            if !framed {
                return false;
            }
            // Compare literal credential bytes before inspecting surrounding
            // punctuation, so a key such as `a.` is never shortened to `a`.
            text[start + marker.len()..]
                .strip_prefix(secret)
                .is_some_and(|tail| {
                    tail.chars().next().is_none_or(|next| {
                        !next.is_alphanumeric() && !matches!(next, '_' | '-' | '+' | '/' | '=')
                    })
                })
        })
}

fn remaining(deadline: Instant) -> Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|v| !v.is_zero())
        .ok_or_else(|| anyhow!("inference deadline exceeded"))
}

fn data_url(mime: &str, bytes: &[u8]) -> String {
    format!("data:{mime};base64,{}", STANDARD.encode(bytes))
}

/// Check container headers without probing or decoding. QuickTime permits
/// legacy movies without ftyp; the caller still verifies tracks and timing.
fn has_video_header(bytes: &[u8], mime: &str) -> bool {
    let Some((kind, size, header_bytes)) = atom_header(bytes) else {
        return false;
    };
    if kind == b"ftyp" {
        let required = header_bytes + 8; // major brand and minor version
        // A to-EOF ftyp box leaves no video tracks. Require a bounded box.
        return bytes[..4] != [0; 4] && size >= required && (size - required) % 4 == 0;
    }
    if mime != "video/quicktime" {
        return false;
    }

    let mut remaining = bytes;
    let mut movie = false;
    let mut media = false;
    while !remaining.is_empty() {
        let Some((kind, size, header_bytes)) = atom_header(remaining) else {
            return false;
        };
        match kind {
            // A later ftyp is not a legacy movie and must not bypass its check.
            b"ftyp" => return false,
            b"moov" => movie |= size > header_bytes,
            b"mdat" => media |= size > header_bytes,
            _ => {}
        }
        remaining = &remaining[size..];
    }
    movie && media
}

/// Return only complete, advancing atoms; size zero extends to EOF.
fn atom_header(bytes: &[u8]) -> Option<(&[u8], usize, usize)> {
    let header = bytes.get(..8)?;
    let short_size = u32::from_be_bytes(header[..4].try_into().expect("four size bytes"));
    let (size, header_bytes) = match short_size {
        0 => (bytes.len(), 8),
        1 => {
            let extended = bytes.get(8..16)?;
            (
                usize::try_from(u64::from_be_bytes(
                    extended.try_into().expect("eight size bytes"),
                ))
                .ok()?,
                16,
            )
        }
        size => (size as usize, 8),
    };
    (size >= header_bytes && size <= bytes.len()).then_some((&header[4..8], size, header_bytes))
}

fn validate_image(bytes: &[u8], mime: &str, media: &crate::config::MediaConfig) -> Result<()> {
    let format = match mime {
        "image/png" => ImageFormat::Png,
        "image/jpeg" => ImageFormat::Jpeg,
        _ => bail!("image input must be PNG or JPEG"),
    };
    let mut reader = ImageReader::with_format(Cursor::new(bytes), format);
    let mut limits = Limits::default();
    limits.max_image_width = Some(media.max_image_edge);
    limits.max_image_height = Some(media.max_image_edge);
    limits.max_alloc = Some(media.max_image_pixels * 8);
    reader.limits(limits);
    let decoder = reader
        .into_decoder()
        .map_err(|_| anyhow!("invalid inference image header or dimensions"))?;
    let (width, height) = decoder.dimensions();
    ensure!(
        width > 0 && height > 0 && u64::from(width) * u64::from(height) <= media.max_image_pixels,
        "inference image exceeds pixel limit"
    );
    DynamicImage::from_decoder(decoder).map_err(|_| anyhow!("corrupt inference image"))?;
    Ok(())
}

fn completion_text(envelope: Value, model: &str) -> Result<String> {
    ensure!(
        envelope.get("error").is_none() && envelope["model"].as_str() == Some(model),
        "inference response error or model mismatch"
    );
    let choices = envelope["choices"]
        .as_array()
        .ok_or_else(|| anyhow!("inference response must have one completion"))?;
    ensure!(
        choices.len() == 1 && choices[0]["index"].as_u64() == Some(0),
        "inference response must have one completion"
    );
    let choice = &choices[0];
    let message = &choice["message"];
    ensure!(
        choice["finish_reason"].as_str() == Some("stop")
            && message["role"].as_str() == Some("assistant")
            && message["refusal"].is_null()
            && message
                .get("tool_calls")
                .is_none_or(|v| v.is_null() || v.as_array().is_some_and(Vec::is_empty)),
        "inference response was refused, truncated, or requested unsupported tools"
    );
    let content = message["content"]
        .as_str()
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| anyhow!("inference response has no JSON content"))?;
    Ok(content.to_owned())
}
