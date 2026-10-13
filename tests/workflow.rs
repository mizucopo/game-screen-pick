//! Fixed HTTP answers verify live caller wiring; they do not prove model quality.
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::Command;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::thread;
use std::time::Duration;

use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};

const BINARY: &str = env!("CARGO_BIN_EXE_game-screen-pick");

struct Server {
    url: String,
    calls: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Server {
    fn new(mut answer: impl FnMut(&str, &Value) -> Value + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_calls = calls.clone();
        let thread_stop = stop.clone();
        let thread = thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let (header, body) = request(&mut stream);
                        thread_calls.fetch_add(1, Ordering::Relaxed);
                        let response = answer(&header, &body).to_string();
                        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).unwrap();
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5))
                    }
                    Err(error) => panic!("{error}"),
                }
            }
        });
        Self {
            url,
            calls,
            stop,
            thread: Some(thread),
        }
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let result = self.thread.take().unwrap().join();
        if !thread::panicking() {
            result.unwrap();
        }
    }
}

fn request(stream: &mut TcpStream) -> (String, Value) {
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let mut bytes = Vec::new();
    while !bytes.ends_with(b"\r\n\r\n") {
        let mut byte = [0];
        stream.read_exact(&mut byte).unwrap();
        bytes.push(byte[0]);
        assert!(bytes.len() < 65536);
    }
    let header = String::from_utf8(bytes).unwrap();
    let length = header
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    assert!(length < 32 * 1024 * 1024);
    let mut body = vec![0; length];
    stream.read_exact(&mut body).unwrap();
    (
        header,
        if length == 0 {
            Value::Null
        } else {
            serde_json::from_slice(&body).unwrap()
        },
    )
}

fn metadata(body: &Value, kind: &str) -> Value {
    body["messages"][0]["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|part| {
            let (_, text) = part["text"].as_str()?.split_once('\n')?;
            serde_json::from_str::<Value>(text).ok()
        })
        .find(|value| value["kind"] == kind)
        .unwrap()
}

fn ai() -> Server {
    Server::new(|header, body| {
        assert!(header.starts_with("POST /v1/chat/completions "));
        assert_eq!(body["model"], "fixture-vision");
        assert_eq!(body["reasoning_effort"], "none");
        let content = body["messages"][0]["content"].as_array().unwrap();
        let answer = if content.iter().any(|part| part["type"] == "video_url") {
            let value = metadata(body, "semantic_chunk_metadata");
            let url = content
                .iter()
                .find(|part| part["type"] == "video_url")
                .unwrap()["video_url"]["url"]
                .as_str()
                .unwrap();
            assert!(url.starts_with("data:video/quicktime;base64,"));
            let bytes = STANDARD.decode(url.split_once(',').unwrap().1).unwrap();
            assert!(bytes.len() > 100 && bytes.windows(4).any(|part| part == b"moov"));
            assert_eq!(value["width"], 160);
            assert_eq!(value["height"], 96);
            let start = value["actual_start_seconds"].as_f64().unwrap();
            let end = value["actual_end_seconds"].as_f64().unwrap();
            json!({"events":[{"source":value["source"], "start_seconds":start,"end_seconds":end,
                "representative_seconds":(start + end) / 2.0,"importance":90,
                "scene":"colored gameplay blocks","reason":"Colored blocks change in temporal order across the video"}]})
        } else if content.iter().any(|part| part["type"] == "image_url") {
            let value = metadata(body, "assessment_metadata");
            let mut images = 0;
            for part in content.iter().filter(|part| part["type"] == "image_url") {
                let url = part["image_url"]["url"].as_str().unwrap();
                let bytes = STANDARD.decode(url.split_once(',').unwrap().1).unwrap();
                let image = image::load_from_memory(&bytes).unwrap();
                assert!(image.width() >= 16 && image.height() >= 16);
                images += 1;
            }
            let candidates = value["candidates"].as_array().unwrap();
            assert_eq!(
                images,
                if value["phase"] == "secondary" {
                    1 + 3 * candidates.len()
                } else {
                    1
                }
            );
            json!({"assessments":candidates.iter().map(|candidate| json!({
                "id":candidate["id"],"blog_score":90,"transition":false,"special":false,
                "scene":"colored gameplay blocks","reason":"Colored blocks and readable gameplay elements are clearly visible"
            })).collect::<Vec<_>>()})
        } else {
            let text = content
                .iter()
                .filter_map(|part| part["text"].as_str())
                .collect::<String>();
            assert!(text.contains("Example Game") && text.contains("Gameplay evidence"));
            json!({"game_title":"Example Game","identification":"Example Game fixture game",
                "genre":"puzzle adventure","progression":["Explore changing colored blocks"],
                "representative_screens":["Colored blocks on a readable gameplay screen"],
                "selection_priorities":["Clear normal gameplay with changing colors"],
                "ambiguous":false,"contradictions":[],"source_urls":["https://example.test/game"]})
        };
        json!({"model":"fixture-vision","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":answer.to_string()}}]})
    })
}

fn brave() -> Server {
    Server::new(|header, body| {
        assert!(header.starts_with("GET /res/v1/web/search?"));
        assert!(header.contains("Example+Game") || header.contains("Example%20Game"));
        assert!(
            header
                .to_lowercase()
                .contains("x-subscription-token: search-secret")
        );
        assert!(body.is_null());
        json!({"web":{"results":[{"title":"Example Game","url":"https://example.test/game",
            "description":"Gameplay evidence for a puzzle adventure with colored block progression"}]}})
    })
}

fn setup(root: &Path, ai: &Server, brave: &Server, method: &str, persist: bool) {
    fs::create_dir(root.join("input")).unwrap();
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/videos/01-blocks.mkv"),
        root.join("input/blocks.mkv"),
    )
    .unwrap();
    let source = format!(
        r#"
[selection]
method = "{method}"
samples_per_video = 6
semantic_chunk_seconds = 1.5
[ai]
backend = "vllm"
base_url = "{}/v1"
model = "fixture-vision"
inference_level = "none"
timeout_seconds = 10
cache_revision = "fixture-v1"
[ai.protocol]
reasoning_format = "reasoning_effort"
supported_inference_levels = ["none"]
json_schema = false
max_output_tokens = 4096
max_request_bytes = 33554432
max_response_bytes = 1048576
max_attempts = 2
[ai.protocol.media]
vision_revision = "fixture-mock-not-live-capability"
max_images = 64
max_image_pixels = 32000000
max_image_edge = 16384
max_video_bytes = 134217728
max_video_seconds = 30.0
[brave]
endpoint = "{}/res/v1/web/search"
api_key = "search-secret"
quota_stop_confirmed = true
quota_contract_reference = "mock hard-stop test contract"
context_generation_confirmed = true
context_contract_reference = "mock inference-use test contract"
generated_context_persistence_allowed = {persist}
storage_contract_reference = "mock generated-context storage test contract"
"#,
        ai.url, brave.url
    );
    fs::write(root.join("config.toml"), source).unwrap();
}

fn invoke(root: &Path, generated: bool, cache_only: bool, no_tools: bool) -> std::process::Output {
    let mut command = Command::new(BINARY);
    command.current_dir(root).args([
        "--config",
        "config.toml",
        "--count",
        "2",
        if generated {
            "--game-title"
        } else {
            "--game-context"
        },
        if generated {
            "Example Game"
        } else {
            "Pick clear colored gameplay blocks"
        },
        "input",
        "output",
    ]);
    if cache_only {
        command.arg("--cache-only");
    }
    if no_tools {
        command.env("PATH", "");
    }
    command.output().unwrap()
}

#[test]
fn both_methods_use_live_adapters_and_warm_runs_make_no_external_calls() {
    for method in ["sampled_frames", "semantic_video"] {
        for generated in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let ai = ai();
            let brave = brave();
            setup(root.path(), &ai, &brave, method, true);
            let cold = invoke(root.path(), generated, false, false);
            assert!(
                cold.status.success(),
                "{method}/{generated}: {}",
                String::from_utf8_lossy(&cold.stderr)
            );
            let counts = (ai.calls(), brave.calls());
            assert!(counts.0 >= 2);
            assert_eq!(counts.1, if generated { 2 } else { 0 });
            let before = fs::read(root.path().join("output/report.json")).unwrap();
            // Management changes do not invalidate inference. A warm hit must
            // neither probe /models nor try these deliberately absent commands.
            let config = root.path().join("config.toml");
            let source = fs::read_to_string(&config).unwrap();
            fs::write(&config, format!("{source}\n[runtime]\nbackend = \"vllm\"\nstart_command = [\"/missing/runtime-start\", \"{{session}}\"]\nstop_command = [\"/missing/runtime-stop\", \"{{pid}}\", \"{{session}}\"]\n")).unwrap();
            let warm = invoke(root.path(), generated, true, true);
            assert!(
                warm.status.success(),
                "{}",
                String::from_utf8_lossy(&warm.stderr)
            );
            assert_eq!((ai.calls(), brave.calls()), counts);
            assert_eq!(
                fs::read(root.path().join("output/report.json")).unwrap(),
                before
            );
            assert!(String::from_utf8_lossy(&warm.stderr).contains("warm=true"));
        }
    }
}

#[test]
fn cache_only_misses_and_unapproved_context_storage_do_not_contact_services() {
    let root = tempfile::tempdir().unwrap();
    let ai = ai();
    let brave = brave();
    setup(root.path(), &ai, &brave, "sampled_frames", false);
    for generated in [false, true] {
        let miss = invoke(root.path(), generated, true, false);
        assert_eq!(miss.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&miss.stderr).contains("cache-only"));
        assert_eq!((ai.calls(), brave.calls()), (0, 0));
    }
    let cold = invoke(root.path(), true, false, false);
    assert!(
        cold.status.success(),
        "{}",
        String::from_utf8_lossy(&cold.stderr)
    );
    let report: Value =
        serde_json::from_slice(&fs::read(root.path().join("output/report.json")).unwrap()).unwrap();
    assert!(report.get("game_context").is_none());
    assert!(
        !root
            .path()
            .join(".game-screen-pick-cache")
            .join(game_screen_pick::cache::NAMESPACE)
            .join("contexts")
            .exists()
    );
    let counts = (ai.calls(), brave.calls());
    let miss = invoke(root.path(), true, true, true);
    assert_eq!(miss.status.code(), Some(1));
    assert_eq!((ai.calls(), brave.calls()), counts);
}

#[test]
fn unconfirmed_search_and_unowned_output_fail_before_all_external_operations() {
    let root = tempfile::tempdir().unwrap();
    let ai = ai();
    let brave = brave();
    setup(root.path(), &ai, &brave, "sampled_frames", true);
    let path = root.path().join("config.toml");
    let confirmed = fs::read_to_string(&path).unwrap();
    fs::write(
        &path,
        confirmed.replace(
            "quota_stop_confirmed = true",
            "quota_stop_confirmed = false",
        ),
    )
    .unwrap();
    let rejected = invoke(root.path(), true, false, true);
    assert_eq!(rejected.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("quota_stop_confirmed"));
    assert_eq!((ai.calls(), brave.calls()), (0, 0));
    // Direct context does not use even an unconfirmed Brave account.
    let direct = invoke(root.path(), false, false, false);
    assert!(
        direct.status.success(),
        "{}",
        String::from_utf8_lossy(&direct.stderr)
    );
    assert_eq!(brave.calls(), 0);
    let calls = ai.calls();
    fs::write(&path, confirmed).unwrap();
    fs::write(
        root.path().join("output/unknown.txt"),
        b"preserve user file",
    )
    .unwrap();
    let rejected = invoke(root.path(), true, false, true);
    assert_eq!(rejected.status.code(), Some(2));
    assert_eq!((ai.calls(), brave.calls()), (calls, 0));
    assert_eq!(
        fs::read(root.path().join("output/unknown.txt")).unwrap(),
        b"preserve user file"
    );
}
