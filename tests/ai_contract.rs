//! HTTP contract evidence; these mocks do not establish live model capability.
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use game_screen_pick::ai::{Content, JsonRequest, OpenAiClient};
use game_screen_pick::config::{AiConfig, Config, InferenceLevel};
use game_screen_pick::contact_sheet::{ContextImage, SheetEntry, render_candidate_sheets};
use game_screen_pick::media::MediaTools;
use serde_json::{Value, json};

const SECRET: &str = "CONTRACT_AUTH_DO_NOT_DISCLOSE";

#[derive(Clone, Debug)]
struct Received {
    path: String,
    headers: String,
    body: Value,
}

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    header_delay: Duration,
    body_delay: Duration,
    include_length: bool,
}

impl Reply {
    fn json(value: Value) -> Self {
        Self::bytes(serde_json::to_vec(&value).unwrap())
    }

    fn bytes(body: Vec<u8>) -> Self {
        Self {
            status: 200,
            headers: Vec::new(),
            body,
            header_delay: Duration::ZERO,
            body_delay: Duration::ZERO,
            include_length: true,
        }
    }

    fn redirect(status: u16, destination: String) -> Self {
        Self {
            status,
            headers: vec![("Location".to_owned(), destination)],
            ..Self::bytes(Vec::new())
        }
    }
}

struct Mock {
    url: String,
    received: Arc<Mutex<Vec<Received>>>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl Mock {
    fn new(mut handler: impl FnMut(&Received, usize) -> Reply + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let received = Arc::new(Mutex::new(Vec::new()));
        let recorded = received.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let worker = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(30);
            while !stopping.load(Ordering::Relaxed) && Instant::now() < deadline {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => panic!("mock accept failed: {error}"),
                };
                // macOS can inherit the listener's nonblocking mode on accept.
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let request = read_request(&mut stream);
                let index = {
                    let mut requests = recorded.lock().unwrap();
                    let index = requests.len();
                    requests.push(request.clone());
                    index
                };
                let reply = handler(&request, index);
                thread::sleep(reply.header_delay);
                let mut head = format!(
                    "HTTP/1.1 {} Test\r\nContent-Type: application/json\r\nConnection: close\r\n",
                    reply.status
                );
                if reply.include_length {
                    head.push_str(&format!("Content-Length: {}\r\n", reply.body.len()));
                }
                for (name, value) in reply.headers {
                    head.push_str(&format!("{name}: {value}\r\n"));
                }
                head.push_str("\r\n");
                // A deadline test may close the client connection before either write.
                if stream.write_all(head.as_bytes()).is_ok() {
                    thread::sleep(reply.body_delay);
                    let _ = stream.write_all(&reply.body);
                }
            }
        });
        Self {
            url,
            received,
            stop,
            worker: Some(worker),
        }
    }

    fn requests(&self) -> Vec<Received> {
        self.received.lock().unwrap().clone()
    }
}

impl Drop for Mock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let result = self.worker.take().unwrap().join();
        if !thread::panicking() {
            result.expect("mock contract assertion failed");
        }
    }
}

fn read_request(stream: &mut TcpStream) -> Received {
    let started = Instant::now();
    let mut bytes = Vec::new();
    let mut buffer = [0; 4096];
    let header_end = loop {
        assert!(started.elapsed() < Duration::from_secs(3));
        assert!(bytes.len() < 2 * 1024 * 1024);
        let count = stream.read(&mut buffer).expect("bounded mock request read");
        assert_ne!(count, 0, "request ended before headers");
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(offset) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            break offset + 4;
        }
    };
    let headers = String::from_utf8(bytes[..header_end].to_vec()).unwrap();
    let length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .expect("JSON request must have a content length");
    assert!(length < 2 * 1024 * 1024);
    while bytes.len() - header_end < length {
        assert!(started.elapsed() < Duration::from_secs(3));
        let count = stream.read(&mut buffer).expect("bounded mock body read");
        assert_ne!(count, 0, "request ended before body");
        bytes.extend_from_slice(&buffer[..count]);
    }
    let first = headers.lines().next().unwrap();
    assert!(first.starts_with("POST "), "unexpected method: {first}");
    Received {
        path: first.split_whitespace().nth(1).unwrap().to_owned(),
        headers,
        body: serde_json::from_slice(&bytes[header_end..header_end + length]).unwrap(),
    }
}

fn configuration(mock: &Mock, backend: &str) -> AiConfig {
    configuration_with_key(mock, backend, SECRET)
}

fn configuration_with_key(mock: &Mock, backend: &str, api_key: &str) -> AiConfig {
    Config::parse(&format!(
        r#"
[selection]
method = "sampled_frames"
[ai]
backend = "{backend}"
base_url = "{}/custom%20service/v1/"
model = "contract-vision"
inference_level = "none"
timeout_seconds = 2
cache_revision = "contract-v1"
api_key = "{api_key}"
[ai.protocol]
reasoning_format = "reasoning_effort"
supported_inference_levels = ["none", "low", "medium", "high"]
json_schema = true
max_output_tokens = 512
max_request_bytes = 1048576
max_response_bytes = 16384
max_attempts = 3
"#,
        mock.url
    ))
    .unwrap()
    .ai
}

fn enable_media(config: &mut AiConfig) {
    config.protocol.as_mut().unwrap().media = Some(game_screen_pick::config::MediaConfig {
        vision_revision: "verified-mock-pixels-v1".to_owned(),
        max_images: 8,
        max_image_pixels: 10_000_000,
        max_image_edge: 4096,
        max_video_bytes: if config.backend == game_screen_pick::config::Backend::Vllm {
            1024 * 1024
        } else {
            0
        },
        max_video_seconds: if config.backend == game_screen_pick::config::Backend::Vllm {
            10.0
        } else {
            0.0
        },
    });
}

fn request() -> JsonRequest {
    JsonRequest {
        prompt: "Evaluate exactly A01 and A02 from the supplied media.".to_owned(),
        schema: json!({"type":"object","additionalProperties":false,"required":["frames"],
            "properties":{"frames":{"type":"array","items":{"type":"object",
                "additionalProperties":false,"required":["id","score"],
                "properties":{"id":{"type":"string"},"score":{"type":"number","minimum":0,"maximum":100}}}}}}),
        content: Vec::new(),
    }
}

fn valid() -> Value {
    json!({"frames":[{"id":"A01","score":80},{"id":"A02","score":85}]})
}

fn envelope(content: String) -> Value {
    json!({"model":"contract-vision","choices":[{"index":0,"finish_reason":"stop",
        "message":{"role":"assistant","content":content,"refusal":null}}]})
}

fn success() -> Reply {
    Reply::json(envelope(valid().to_string()))
}

fn validate_ids(value: &Value) -> Result<Value> {
    let ids: Vec<_> = value["frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|frame| frame["id"].as_str().unwrap())
        .collect();
    ensure!(ids == ["A01", "A02"], "VALIDATOR_PRIVATE_MARKER");
    Ok(value.clone())
}

fn error(client: &OpenAiClient, request: &JsonRequest) -> String {
    let error = client.generate(request, validate_ids).unwrap_err();
    let text = format!("{error:#}\n{error:?}");
    assert!(!text.contains(SECRET), "authentication leaked into error");
    assert!(!text.contains("VALIDATOR_PRIVATE_MARKER"));
    assert!(!text.contains("RESPONSE_PRIVATE_MARKER"));
    text
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/videos")
        .join(name)
}

fn data_part(request: &Received, kind: &str) -> Vec<Vec<u8>> {
    request.body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|message| message["content"].as_array())
        .flat_map(|parts| parts.iter())
        .filter(|part| part["type"] == kind)
        .map(|part| {
            let url = part[kind]["url"].as_str().unwrap();
            let (prefix, data) = url.split_once(',').unwrap();
            assert!(prefix.starts_with("data:") && prefix.ends_with(";base64"));
            STANDARD.decode(data).unwrap()
        })
        .collect()
}

#[test]
fn constructor_has_no_http_and_requires_an_explicit_usable_protocol() {
    let mock = Mock::new(|_, _| success());
    let mut config = configuration(&mock, "vllm");
    OpenAiClient::new(&config).unwrap();
    assert!(!format!("{config:?}").contains(SECRET));
    config.protocol.as_mut().unwrap().supported_inference_levels = vec![InferenceLevel::High];
    assert!(OpenAiClient::new(&config).is_err());
    config.protocol = None;
    assert!(OpenAiClient::new(&config).is_err());
    assert!(mock.requests().is_empty());
}

#[test]
fn backend_and_reasoning_settings_switch_through_the_same_json_boundary() {
    for backend in ["strata", "vllm"] {
        let expected_backend = backend.to_owned();
        let mock = Mock::new(move |received, _| {
            assert_eq!(received.path, "/custom%20service/v1/chat/completions");
            assert_eq!(received.body["model"], "contract-vision");
            assert_eq!(received.body["stream"], false);
            assert_eq!(received.body["max_tokens"], 512);
            assert!(!received.body.to_string().contains(SECRET));
            assert!(received.headers.contains(&format!("Bearer {SECRET}")));
            if expected_backend == "strata" {
                assert_eq!(
                    received.body["chat_template_kwargs"]["enable_thinking"],
                    false
                );
                assert!(received.body.get("reasoning_effort").is_none());
                assert!(received.body.get("response_format").is_none());
                assert!(received.body.to_string().contains("additionalProperties"));
            } else {
                assert_eq!(received.body["reasoning_effort"], "high");
                assert_eq!(received.body["response_format"]["type"], "json_schema");
                let format = &received.body["response_format"]["json_schema"];
                assert_eq!(format["name"], "response");
                assert_eq!(format["strict"], true);
                assert_eq!(format["schema"], request().schema);
            }
            success()
        });
        let mut config = configuration(&mock, backend);
        if backend == "strata" {
            let protocol = config.protocol.as_mut().unwrap();
            protocol.reasoning_format = game_screen_pick::config::ReasoningFormat::ChatTemplate;
            protocol.json_schema = false;
        } else {
            config.inference_level = InferenceLevel::High;
        }
        let client = OpenAiClient::new(&config).unwrap();
        assert_eq!(client.generate(&request(), validate_ids).unwrap(), valid());
        assert_eq!(mock.requests().len(), 1);
    }
}

#[test]
fn served_model_alias_uses_an_explicit_canonical_response_identity() {
    let mock = Mock::new(|received, _| {
        assert_eq!(received.body["model"], "contract-vision-alias");
        success() // vLLM returns the first served name, contract-vision.
    });
    let mut config = configuration(&mock, "vllm");
    config.model = "contract-vision-alias".to_owned();
    config.protocol.as_mut().unwrap().response_model = Some("contract-vision".to_owned());
    let client = OpenAiClient::new(&config).unwrap();
    assert_eq!(client.generate(&request(), validate_ids).unwrap(), valid());

    config.protocol.as_mut().unwrap().response_model = Some("unexpected-model".to_owned());
    let client = OpenAiClient::new(&config).unwrap();
    assert!(error(&client, &request()).contains("model mismatch"));
    assert_eq!(mock.requests().len(), 2);
}

#[test]
fn actual_png_triptych_and_contact_sheet_bytes_keep_ids_times_and_order() {
    let scratch = tempfile::tempdir().unwrap();
    let tools = MediaTools::default();
    let cancelled = AtomicBool::new(false);
    let video = tools.probe(&fixture("01-blocks.mkv"), &cancelled).unwrap();
    let frames: Vec<_> = [0.5, 2.5, 4.5]
        .into_iter()
        .map(|time| tools.extract(&video, time, &cancelled).unwrap())
        .collect();
    let entries = [SheetEntry {
        id: "A01".to_owned(),
        rank: None,
        source: "01-blocks.mkv".to_owned(),
        requested_seconds: 2.5,
        actual_seconds: 2.5,
        scene: "契約検証".to_owned(),
        image: frames[1].path(),
        context: Some(std::array::from_fn(|index| ContextImage {
            image: frames[index].path(),
            actual_seconds: [0.5, 2.5, 4.5][index],
        })),
    }];
    let sheet = render_candidate_sheets(&entries, &scratch.path().join("sheets")).unwrap();
    let mut images: Vec<_> = frames
        .iter()
        .map(|frame| fs::read(frame.path()).unwrap())
        .collect();
    images.push(fs::read(&sheet[0]).unwrap());
    for backend in ["strata", "vllm"] {
        let expected = images.clone();
        let mock = Mock::new(move |received, _| {
            let actual = data_part(received, "image_url");
            assert_eq!(
                actual, expected,
                "mock checks exact attached media before responding"
            );
            for bytes in &actual[..3] {
                assert_eq!(image::load_from_memory(bytes).unwrap().width(), 160);
            }
            assert!(image::load_from_memory(&actual[3]).unwrap().width() > 320);
            let payload = received.body.to_string();
            assert!(payload.contains("A01 before=0.5 center=2.5 after=4.5"));
            success()
        });
        let mut config = configuration(&mock, backend);
        enable_media(&mut config);
        let client = OpenAiClient::new(&config).unwrap();
        let mut request = request();
        request.content.push(Content::Text(
            "A01 before=0.5 center=2.5 after=4.5".to_owned(),
        ));
        for (index, bytes) in images.iter().cloned().enumerate() {
            request.content.push(Content::Image {
                bytes,
                mime: if index == 3 {
                    "image/jpeg"
                } else {
                    "image/png"
                }
                .to_owned(),
            });
        }
        client.generate(&request, validate_ids).unwrap();
        assert_eq!(mock.requests().len(), 1);
    }
}

#[test]
fn actual_video_bytes_preserve_fixture_frames_and_times_at_the_http_boundary() {
    let source = fixture("03-multitrack-rotated.mov");
    let tools = MediaTools::default();
    let cancelled = AtomicBool::new(false);
    let video = tools.probe(&source, &cancelled).unwrap();
    let chunk = tools.extract_chunk(&video, 0.0, 6.0, &cancelled).unwrap();
    let mut legacy = fs::read(&source).unwrap();
    assert_eq!(&legacy[4..8], b"ftyp");
    // Turn the optional type atom into padding without moving chunk offsets.
    legacy[4..8].copy_from_slice(b"free");
    let directory = tempfile::tempdir().unwrap();
    let mut extended_legacy = legacy.clone();
    extended_legacy[..4].copy_from_slice(&1_u32.to_be_bytes());
    extended_legacy[8..16].copy_from_slice(&20_u64.to_be_bytes());
    let faststart_path = directory.path().join("faststart.mov");
    let output = std::process::Command::new(&tools.ffmpeg)
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(&source)
        .args(["-map", "0", "-c", "copy", "-movflags", "+faststart"])
        .arg(&faststart_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "ffmpeg must create a faststart MOV"
    );
    let mut movie_first = fs::read(&faststart_path).unwrap();
    movie_first[4..8].copy_from_slice(b"free");
    let mut media_to_eof = movie_first.clone();
    let media_type = media_to_eof
        .windows(4)
        .position(|part| part == b"mdat")
        .unwrap();
    media_to_eof[media_type - 4..media_type].copy_from_slice(&0_u32.to_be_bytes());
    for bytes in [
        fs::read(&source).unwrap(),
        fs::read(chunk.path()).unwrap(),
        legacy,
        extended_legacy,
        movie_first,
        media_to_eof,
    ] {
        // Verify every alternate container before the caller sends its bytes.
        let candidate_path = directory.path().join("candidate.mov");
        fs::write(&candidate_path, &bytes).unwrap();
        let candidate = tools.probe(&candidate_path, &cancelled).unwrap();
        assert_eq!(candidate.duration_seconds, video.duration_seconds);
        assert_eq!(
            serde_json::to_value(&candidate.frames).unwrap(),
            serde_json::to_value(&video.frames).unwrap()
        );
        let expected = bytes.clone();
        let mock = Mock::new(move |received, _| {
            let videos = data_part(received, "video_url");
            assert_eq!(videos.as_slice(), std::slice::from_ref(&expected));
            let directory = tempfile::tempdir().unwrap();
            let received_path = directory.path().join("received.mov");
            fs::write(&received_path, &videos[0]).unwrap();
            let tools = MediaTools::default();
            let cancelled = AtomicBool::new(false);
            let video = tools.probe(&received_path, &cancelled).unwrap();
            assert_eq!(video.duration_seconds, 6.0);
            assert_eq!(video.frames.len(), 24);
            assert_eq!((video.width, video.height), (96, 160));
            let tiles: Vec<Vec<u8>> = include_str!("fixtures/video-tiles.tsv")
                .lines()
                .map(|line| {
                    line.split('\t')
                        .skip(2)
                        .map(|value| value.parse().unwrap())
                        .collect()
                })
                .collect();
            for (time, second) in [(0.5, 0), (2.5, 2), (4.5, 4)] {
                let frame = tools.extract(&video, time, &cancelled).unwrap();
                assert_eq!(frame.actual_seconds, time);
                let pixels = image::open(frame.path()).unwrap().to_rgb8();
                for (y, row) in tiles[second * 6..second * 6 + 6].iter().enumerate() {
                    for (x, &luma) in row.iter().enumerate() {
                        assert_eq!(
                            pixels.get_pixel(y as u32 * 16 + 8, 159 - (x as u32 * 16 + 8))[0],
                            luma
                        );
                    }
                }
            }
            assert!(
                received
                    .body
                    .to_string()
                    .contains("source=03-multitrack-rotated.mov start=0 end=6")
            );
            success()
        });
        let mut config = configuration(&mock, "vllm");
        enable_media(&mut config);
        config.timeout_seconds = 10;
        let client = OpenAiClient::new(&config).unwrap();
        let mut request = request();
        request.content = vec![
            Content::Text("source=03-multitrack-rotated.mov start=0 end=6".to_owned()),
            Content::Video {
                bytes,
                mime: "video/quicktime".to_owned(),
                duration_seconds: 6.0,
            },
        ];
        client.generate(&request, validate_ids).unwrap();
    }
}

#[test]
fn json_schema_and_domain_failures_retry_with_safe_correction_and_finite_attempts() {
    let invalid = [
        "{RESPONSE_PRIVATE_MARKER".to_owned(),
        r#"{"frames":[{"id":"A01","score":NaN},{"id":"A02","score":85}]}"#.to_owned(),
        json!({"frames":[{"id":"A01","score":"RESPONSE_PRIVATE_MARKER"}]}).to_string(),
        json!({"frames":[{"id":"A01","score":101},{"id":"A02","score":85}]}).to_string(),
        json!({"frames":[{"id":"A01","score":-1},{"id":"A02","score":85}]}).to_string(),
        json!({"frames":[{"id":"UNKNOWN","score":80},{"id":"A02","score":85}]}).to_string(),
        json!({"frames":[{"id":"A01","score":80}]}).to_string(),
        json!({"frames":[{"id":"A01","score":80},{"id":"A01","score":85}]}).to_string(),
    ];
    for invalid in invalid {
        let image = include_bytes!("fixtures/images/rgb-grid.png").to_vec();
        let mock = Mock::new(move |received, index| {
            assert_eq!(
                data_part(received, "image_url").as_slice(),
                std::slice::from_ref(&image)
            );
            if index == 0 {
                Reply::json(envelope(invalid.clone()))
            } else {
                let body = received.body.to_string();
                assert!(!body.contains("RESPONSE_PRIVATE_MARKER"));
                assert!(!body.contains("VALIDATOR_PRIVATE_MARKER"));
                let texts = received.body["messages"][0]["content"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|part| part["type"] == "text")
                    .count();
                assert!(texts > 1, "retry must add a correction prompt");
                success()
            }
        });
        let mut config = configuration(&mock, "vllm");
        enable_media(&mut config);
        let client = OpenAiClient::new(&config).unwrap();
        let mut request = request();
        request.content.push(Content::Image {
            bytes: include_bytes!("fixtures/images/rgb-grid.png").to_vec(),
            mime: "image/png".to_owned(),
        });
        assert_eq!(client.generate(&request, validate_ids).unwrap(), valid());
        assert_eq!(mock.requests().len(), 2);
    }
    let mock = Mock::new(|_, _| Reply::json(envelope("{RESPONSE_PRIVATE_MARKER".to_owned())));
    let client = OpenAiClient::new(&configuration(&mock, "vllm")).unwrap();
    error(&client, &request());
    assert_eq!(mock.requests().len(), 3);
}

#[test]
fn refusal_truncation_tools_and_invalid_envelopes_fail_without_retry() {
    for mutation in [
        "refusal",
        "length",
        "tool_calls",
        "model",
        "role",
        "index",
        "choices",
        "content",
    ] {
        let mock = Mock::new(move |_, _| {
            let mut reply = envelope(valid().to_string());
            match mutation {
                "refusal" => reply["choices"][0]["message"]["refusal"] = json!(SECRET),
                "length" => reply["choices"][0]["finish_reason"] = json!("length"),
                "tool_calls" => {
                    reply["choices"][0]["message"]["tool_calls"] =
                        json!([{"id":"unrequested-tool"}])
                }
                "model" => reply["model"] = json!("another-model"),
                "role" => reply["choices"][0]["message"]["role"] = json!("user"),
                "index" => reply["choices"][0]["index"] = json!(1),
                "choices" => reply["choices"] = json!([]),
                "content" => {
                    reply["choices"][0]["message"]["content"] =
                        json!([{"type":"text","text":SECRET}])
                }
                _ => unreachable!(),
            }
            Reply::json(reply)
        });
        let client = OpenAiClient::new(&configuration(&mock, "vllm")).unwrap();
        error(&client, &request());
        assert_eq!(mock.requests().len(), 1, "{mutation} must not retry");
    }
}

#[test]
fn http_errors_and_server_echoed_authentication_never_escape_or_retry() {
    for status in [200, 401, 500] {
        let mock = Mock::new(move |_, _| {
            let mut value = valid();
            value["frames"][0]["id"] = json!(SECRET);
            let mut reply = Reply::json(envelope(value.to_string()));
            reply.status = status;
            reply
        });
        let config = configuration(&mock, "vllm");
        assert!(!format!("{config:?}").contains(SECRET));
        let client = OpenAiClient::new(&config).unwrap();
        error(&client, &request());
        assert_eq!(mock.requests().len(), 1);
    }
}

#[test]
fn short_dummy_credentials_do_not_match_ordinary_response_words() {
    for key in [
        "a",
        "test",
        "dummy",
        "assistant",
        "contract-vision",
        "stop",
        "id",
        "frames",
        "score",
        "reason",
    ] {
        let mock = Mock::new(|_, _| {
            let mut value = valid();
            value["frames"][0]["reason"] = json!("a tested gameplay scene");
            Reply::json(envelope(value.to_string()))
        });
        let client = OpenAiClient::new(&configuration_with_key(&mock, "vllm", key)).unwrap();
        let mut request = request();
        request.schema["properties"]["frames"]["items"]["properties"]["reason"] =
            json!({"type":"string"});
        let result = client.generate(&request, validate_ids).unwrap();
        assert_eq!(result["frames"][0]["reason"], "a tested gameplay scene");
        assert_eq!(mock.requests().len(), 1);
    }
}

#[test]
fn explicit_short_credential_echoes_are_still_rejected() {
    for (key, echo) in [
        ("a", "a"),
        ("a", "Bearer a"),
        ("a", "Authorization was \"Bearer a\"."),
        ("a", "'Bearer a'"),
        ("a", "`Bearer a`"),
        ("a", "認証表記「Bearer a」"),
        ("a.", "Bearer a."),
    ] {
        let mock = Mock::new(move |_, _| {
            let mut value = valid();
            value["frames"][0]["id"] = json!(echo);
            Reply::json(envelope(value.to_string()))
        });
        let client = OpenAiClient::new(&configuration_with_key(&mock, "vllm", key)).unwrap();
        let error = client.generate(&request(), validate_ids).unwrap_err();
        assert!(error.to_string().contains("authentication material"));
        assert_eq!(mock.requests().len(), 1);
    }
}

#[test]
fn media_mime_shape_count_support_and_request_limits_fail_before_http() {
    let png = include_bytes!("fixtures/images/rgb-grid.png").to_vec();
    for case in [
        "unset",
        "unset_video",
        "bad_png",
        "mime",
        "pixels",
        "edge",
        "count",
        "strata_video",
        "duration",
        "video_size",
        "video_header",
        "video_mime",
        "request_size",
    ] {
        let mock = Mock::new(|_, _| success());
        let mut config = configuration(
            &mock,
            if case == "strata_video" {
                "strata"
            } else {
                "vllm"
            },
        );
        if !["unset", "unset_video"].contains(&case) {
            enable_media(&mut config);
        }
        let mut request = request();
        request.content.push(Content::Image {
            bytes: png.clone(),
            mime: "image/png".to_owned(),
        });
        match case {
            "unset" => {}
            "bad_png" => {
                request.content = vec![Content::Image {
                    bytes: b"not a PNG".to_vec(),
                    mime: "image/png".to_owned(),
                }]
            }
            "mime" => {
                request.content = vec![Content::Image {
                    bytes: png.clone(),
                    mime: "image/jpeg".to_owned(),
                }]
            }
            "pixels" => {
                config
                    .protocol
                    .as_mut()
                    .unwrap()
                    .media
                    .as_mut()
                    .unwrap()
                    .max_image_pixels = 1
            }
            "edge" => {
                config
                    .protocol
                    .as_mut()
                    .unwrap()
                    .media
                    .as_mut()
                    .unwrap()
                    .max_image_edge = 1
            }
            "count" => {
                for _ in 0..8 {
                    request.content.push(Content::Image {
                        bytes: png.clone(),
                        mime: "image/png".to_owned(),
                    });
                }
            }
            "unset_video" | "strata_video" | "duration" | "video_size" | "video_header"
            | "video_mime" => {
                let mut bytes = fs::read(fixture("03-multitrack-rotated.mov")).unwrap();
                if case == "video_size" {
                    bytes.resize(1024 * 1024 + 1, 0);
                }
                if case == "video_header" {
                    // An unknown atom can be skipped in legacy QuickTime;
                    // use a genuinely truncated declared box instead.
                    bytes[..4].copy_from_slice(&u32::MAX.to_be_bytes());
                }
                request.content = vec![Content::Video {
                    bytes,
                    mime: if case == "video_mime" {
                        "video/webm"
                    } else {
                        "video/quicktime"
                    }
                    .to_owned(),
                    duration_seconds: if case == "duration" { f64::NAN } else { 6.0 },
                }]
            }
            "request_size" => {
                config.protocol.as_mut().unwrap().max_request_bytes = 1024;
                request.content = vec![Content::Text("x".repeat(2048))];
            }
            _ => unreachable!(),
        }
        let client = OpenAiClient::new(&config).unwrap();
        error(&client, &request);
        assert!(
            mock.requests().is_empty(),
            "{case} must fail before network"
        );
    }
}

#[test]
fn truncated_or_invalid_video_boxes_fail_before_http() {
    for bytes in [
        b"\0\0\0\0ftypisom\0\0\0\0".to_vec(),
        b"\0\0\0\x18ftyp".to_vec(),
        b"\0\0\0\x08ftypisom\0\0\0\0".to_vec(),
        b"\0\0\0\x18ftypisom\0\0\0\0".to_vec(),
        b"\0\0\0\x11ftypisom\0\0\0\0x".to_vec(),
        b"\0\0\0\x01ftyp\0\0\0\0\0\0\0\x20".to_vec(),
    ] {
        for mime in ["video/mp4", "video/quicktime"] {
            let mock = Mock::new(|_, _| success());
            let mut config = configuration(&mock, "vllm");
            enable_media(&mut config);
            let client = OpenAiClient::new(&config).unwrap();
            let mut request = request();
            request.content = vec![Content::Video {
                bytes: bytes.clone(),
                mime: mime.to_owned(),
                duration_seconds: 1.0,
            }];
            error(&client, &request);
            assert!(
                mock.requests().is_empty(),
                "invalid ftyp box must fail before HTTP"
            );
        }
    }
}

#[test]
fn malformed_legacy_mov_and_mp4_without_ftyp_fail_before_http() {
    let mut legacy = fs::read(fixture("03-multitrack-rotated.mov")).unwrap();
    legacy[4..8].copy_from_slice(b"free");
    let mut too_short = legacy.clone();
    too_short[..4].copy_from_slice(&4_u32.to_be_bytes());
    let mut too_long = legacy.clone();
    too_long[..4].copy_from_slice(&u32::MAX.to_be_bytes());
    let mut truncated = legacy.clone();
    truncated.pop();
    let mut no_movie = legacy.clone();
    let movie_type = no_movie
        .windows(4)
        .position(|part| part == b"moov")
        .unwrap();
    no_movie[movie_type..movie_type + 4].copy_from_slice(b"free");
    let mut no_media = legacy.clone();
    let media_type = no_media
        .windows(4)
        .position(|part| part == b"mdat")
        .unwrap();
    no_media[media_type..media_type + 4].copy_from_slice(b"free");
    for (bytes, mime) in [
        (too_short, "video/quicktime"),
        (too_long, "video/quicktime"),
        (truncated, "video/quicktime"),
        (no_movie, "video/quicktime"),
        (no_media, "video/quicktime"),
        (legacy, "video/mp4"),
    ] {
        let mock = Mock::new(|_, _| success());
        let mut config = configuration(&mock, "vllm");
        enable_media(&mut config);
        let client = OpenAiClient::new(&config).unwrap();
        let mut request = request();
        request.content = vec![Content::Video {
            bytes,
            mime: mime.to_owned(),
            duration_seconds: 6.0,
        }];
        error(&client, &request);
        assert!(mock.requests().is_empty());
    }
}

#[test]
fn only_bounded_same_origin_307_and_308_redirects_preserve_post_and_auth() {
    for status in [307, 308] {
        let mock = Mock::new(move |received, index| {
            if index == 0 {
                Reply::redirect(status, "/same-origin".to_owned())
            } else {
                assert_eq!(received.path, "/same-origin");
                assert!(received.headers.contains(&format!("Bearer {SECRET}")));
                assert_eq!(received.body["model"], "contract-vision");
                success()
            }
        });
        let client = OpenAiClient::new(&configuration(&mock, "vllm")).unwrap();
        client.generate(&request(), validate_ids).unwrap();
        assert_eq!(mock.requests().len(), 2);
    }
    let destination = Mock::new(|_, _| panic!("cross-origin destination must never receive auth"));
    let url = format!("{}/foreign", destination.url);
    let source = Mock::new(move |_, _| Reply::redirect(307, url.clone()));
    let client = OpenAiClient::new(&configuration(&source, "vllm")).unwrap();
    error(&client, &request());
    assert_eq!(source.requests().len(), 1);
    assert!(destination.requests().is_empty());
    let mock = Mock::new(|_, _| Reply::redirect(307, "/loop".to_owned()));
    let client = OpenAiClient::new(&configuration(&mock, "vllm")).unwrap();
    error(&client, &request());
    assert_eq!(mock.requests().len(), 4);
    let mock = Mock::new(|_, _| Reply::redirect(302, "/changed-method".to_owned()));
    let client = OpenAiClient::new(&configuration(&mock, "vllm")).unwrap();
    error(&client, &request());
    assert_eq!(mock.requests().len(), 1);
}

#[test]
fn header_body_and_redirect_chain_share_one_bounded_deadline() {
    for case in ["header", "body", "redirect_chain"] {
        let mock = Mock::new(move |_, index| {
            let mut reply = if case == "redirect_chain" && index == 0 {
                Reply::redirect(307, "/delayed".to_owned())
            } else {
                success()
            };
            if case == "body" {
                reply.body_delay = Duration::from_millis(1250);
            } else {
                reply.header_delay =
                    Duration::from_millis(if case == "redirect_chain" { 650 } else { 1250 });
            }
            reply
        });
        let mut config = configuration(&mock, "vllm");
        config.timeout_seconds = 1;
        let client = OpenAiClient::new(&config).unwrap();
        let started = Instant::now();
        error(&client, &request());
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{case} did not honor the whole-call deadline"
        );
        assert_eq!(
            mock.requests().len(),
            if case == "redirect_chain" { 2 } else { 1 }
        );
    }
}

#[test]
fn oversized_response_is_rejected_without_retry_or_body_diagnostics() {
    let mock = Mock::new(|_, _| Reply::bytes(format!("{SECRET}{}", "x".repeat(4096)).into_bytes()));
    let mut config = configuration(&mock, "vllm");
    config.protocol.as_mut().unwrap().max_response_bytes = 1024;
    let client = OpenAiClient::new(&config).unwrap();
    error(&client, &request());
    assert_eq!(mock.requests().len(), 1);
}

#[test]
fn response_without_content_length_still_rejects_one_byte_over_the_limit() {
    let mock = Mock::new(|_, _| {
        let mut reply = Reply::bytes(vec![b'x'; 1025]);
        reply.include_length = false;
        reply
    });
    let mut config = configuration(&mock, "vllm");
    config.protocol.as_mut().unwrap().max_response_bytes = 1024;
    let client = OpenAiClient::new(&config).unwrap();
    assert!(error(&client, &request()).contains("byte limit"));
    assert_eq!(mock.requests().len(), 1);
}

#[test]
fn external_schema_references_fail_without_any_http_or_domain_callback() {
    let mock = Mock::new(|_, _| panic!("schema resolution must not contact HTTP"));
    let client = OpenAiClient::new(&configuration(&mock, "vllm")).unwrap();
    for reference in [
        format!("{}/remote-schema", mock.url),
        "file:///tmp/private-schema.json".to_owned(),
    ] {
        let mut request = request();
        request.schema = json!({"$ref":reference});
        let result: Result<()> = client.generate(&request, |_| {
            panic!("invalid schema must fail before domain validation")
        });
        assert!(result.is_err());
    }
    assert!(mock.requests().is_empty());
}

#[test]
fn semantic_event_bounds_are_checked_by_the_domain_contract_and_corrected() {
    for invalid in [
        json!({"start":-1.0,"end":1.0,"representative":0.5}),
        json!({"start":5.0,"end":7.0,"representative":5.5}),
        json!({"start":2.0,"end":1.0,"representative":1.5}),
        json!({"start":0.0,"end":1.0,"representative":3.0}),
    ] {
        let mock = Mock::new(move |received, index| {
            if index == 0 {
                Reply::json(envelope(invalid.to_string()))
            } else {
                assert!(
                    !received
                        .body
                        .to_string()
                        .contains("VALIDATOR_PRIVATE_MARKER")
                );
                Reply::json(envelope(
                    json!({"start":0.0,"end":1.0,"representative":0.5}).to_string(),
                ))
            }
        });
        let client = OpenAiClient::new(&configuration(&mock, "vllm")).unwrap();
        let request = JsonRequest {
            prompt: "Return an event interval inside the six-second source chunk.".to_owned(),
            schema: json!({"type":"object","additionalProperties":false,
                "required":["start","end","representative"],"properties":{
                    "start":{"type":"number"},"end":{"type":"number"},
                    "representative":{"type":"number"}}}),
            content: Vec::new(),
        };
        let result = client
            .generate(&request, |value| {
                let start = value["start"].as_f64().unwrap();
                let end = value["end"].as_f64().unwrap();
                let representative = value["representative"].as_f64().unwrap();
                ensure!(
                    start >= 0.0
                        && end <= 6.0
                        && start < end
                        && (start..=end).contains(&representative),
                    "VALIDATOR_PRIVATE_MARKER"
                );
                Ok(value.clone())
            })
            .unwrap();
        assert_eq!(result["representative"], 0.5);
        assert_eq!(mock.requests().len(), 2);
    }
}

#[test]
fn persisted_context_cold_then_warm_calls_the_real_http_boundary_once() {
    use game_screen_pick::run::{ResolvedGameContext, RunPaths, Session};

    let mock = Mock::new(|_, _| {
        Reply::json(envelope(
            json!({"context":"Known synthetic fixture scenes"}).to_string(),
        ))
    });
    let config = configuration(&mock, "vllm");
    let client = OpenAiClient::new(&config).unwrap();
    let conditions = config
        .cache_conditions(
            "context-prompt-v1",
            "context-schema-v1",
            "context-media-v1",
            ResolvedGameContext::direct(""),
        )
        .unwrap();
    let model = config.model_conditions();
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let input = root.join("input");
    fs::create_dir(&input).unwrap();
    fs::copy(fixture("01-blocks.mkv"), input.join("01-blocks.mkv")).unwrap();
    let paths = RunPaths {
        input_directory: input,
        output_directory: root.join("output"),
        cache_directory: root.join("cache"),
        config: None,
    };
    let cancelled = AtomicBool::new(false);
    let context_request = JsonRequest {
        prompt: "Describe the public synthetic fixture.".to_owned(),
        schema: json!({"type":"object","additionalProperties":false,"required":["context"],
            "properties":{"context":{"type":"string","minLength":1}}}),
        content: Vec::new(),
    };
    let mut cold = Session::prepare(&paths, &cancelled).unwrap();
    let context = cold
        .generated_context(
            "Fixture title",
            "fixed-mock-search-v1",
            &conditions,
            &model,
            true,
            &cancelled,
            || {
                client.generate(&context_request, |value| {
                    Ok(value["context"].as_str().unwrap().to_owned())
                })
            },
        )
        .unwrap();
    assert_eq!(context.text(), "Known synthetic fixture scenes");
    assert_eq!(mock.requests().len(), 1);
    drop(cold);
    let mut warm = Session::prepare(&paths, &cancelled).unwrap();
    let cached = warm
        .generated_context(
            "Fixture title",
            "fixed-mock-search-v1",
            &conditions,
            &model,
            true,
            &cancelled,
            || panic!("warm cache must bypass HTTP construction and inference"),
        )
        .unwrap();
    assert_eq!(cached, context);
    assert_eq!(mock.requests().len(), 1, "warm must add zero HTTP calls");
    let records = paths
        .cache_directory
        .join(game_screen_pick::cache::NAMESPACE)
        .join("contexts");
    for entry in fs::read_dir(records).unwrap() {
        assert!(
            !fs::read_to_string(entry.unwrap().path())
                .unwrap()
                .contains(SECRET)
        );
    }
}
