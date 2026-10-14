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

#[path = "support/evidence.rs"]
mod evidence;

const BINARY: &str = env!("CARGO_BIN_EXE_game-screen-pick");

fn fixture(name: &str) -> std::path::PathBuf {
    std::env::var_os("GSP_ARTIFACT_FIXTURES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"))
        .join("videos")
        .join(name)
}

fn binary_command(root: &Path) -> Command {
    let mut command = if let Some(binary) = std::env::var_os("GSP_ARTIFACT_BINARY") {
        assert!(Path::new(&binary).is_absolute());
        let mut command = Command::new(binary);
        // The harness uses development tools; the unpacked CLI gets only its
        // declared video tools, a fresh home, and a directory without sources.
        command
            .env_clear()
            .env("PATH", std::env::var_os("GSP_ARTIFACT_TOOLS").unwrap());
        command.env("HOME", root).env("TMPDIR", root);
        command
    } else {
        Command::new(BINARY)
    };
    command.current_dir(root);
    command
}

struct Server {
    url: String,
    calls: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Server {
    fn new(mut answer: impl FnMut(&str, &Value) -> Value + Send + 'static) -> Self {
        Self::with_status(move |header, body| (200, answer(header, body)))
    }

    fn with_status(mut answer: impl FnMut(&str, &Value) -> (u16, Value) + Send + 'static) -> Self {
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
                        let (status, response) = answer(&header, &body);
                        let response = response.to_string();
                        // A cancellable client may close before the mock writes.
                        let _ = write!(
                            stream,
                            "HTTP/1.1 {status} Mock\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}",
                            response.len()
                        );
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
    ai_with_quiet_chunks(|_| false)
}

fn ai_with_quiet_chunks(quiet: impl Fn(&Value) -> bool + Send + 'static) -> Server {
    Server::new(move |header, body| fixed_answer(header, body, &quiet))
}

fn fixed_answer(header: &str, body: &Value, quiet: &impl Fn(&Value) -> bool) -> Value {
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
        assert!([(160, 96), (96, 160)].contains(&(
            value["width"].as_u64().unwrap(),
            value["height"].as_u64().unwrap()
        )));
        let start = value["actual_start_seconds"].as_f64().unwrap();
        let end = value["actual_end_seconds"].as_f64().unwrap();
        if quiet(&value) {
            json!({"events":[]})
        } else {
            json!({"events":[{"source":value["source"], "start_seconds":start,"end_seconds":end,
                "representative_seconds":(start + end) / 2.0,"importance":90,
                "scene":"colored gameplay blocks","reason":"Colored blocks change in temporal order across the video"}]})
        }
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
}

fn evidence_ai() -> Server {
    Server::new(|header, body| {
        evidence::verify_request(body).unwrap();
        fixed_answer(header, body, &|_| false)
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
    fs::copy(fixture("01-blocks.mkv"), root.join("input/blocks.mkv")).unwrap();
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
    invoke_with_path(
        root,
        generated,
        cache_only,
        no_tools.then_some(Path::new("")),
    )
}

fn invoke_with_path(
    root: &Path,
    generated: bool,
    cache_only: bool,
    tools_path: Option<&Path>,
) -> std::process::Output {
    invoke_count(root, generated, cache_only, tools_path, 2)
}

fn invoke_count(
    root: &Path,
    generated: bool,
    cache_only: bool,
    tools_path: Option<&Path>,
    count: usize,
) -> std::process::Output {
    let mut command = binary_command(root);
    command.env_remove("GAME_SCREEN_PICK_BRAVE_API_KEY");
    command.args([
        "--config",
        "config.toml",
        "--count",
        &count.to_string(),
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
    if let Some(path) = tools_path {
        command.env("PATH", path);
    }
    command.output().unwrap()
}

fn write_large_video(path: &Path, geometry: &str) {
    let input = format!("color=c=red:s={geometry}:r=24");
    let made = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            &input,
            "-frames:v",
            "6",
            "-c:v",
            "ffv1",
        ])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        made.status.success(),
        "{}",
        String::from_utf8_lossy(&made.stderr)
    );
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
            if method == "semantic_video" {
                let cache = root
                    .path()
                    .join(".game-screen-pick-cache")
                    .join(game_screen_pick::cache::NAMESPACE);
                // Completed-plan warm lookup must never touch per-frame probes
                // or enumerate chunk checkpoints, even when these are absent.
                fs::remove_dir_all(cache.join("probe")).unwrap();
                fs::remove_dir_all(cache.join("semantic")).unwrap();
                let plans = fs::read_dir(cache.join("semantic-plans"))
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                assert_eq!(plans.len(), 1);
                assert!(plans[0].metadata().unwrap().len() < 20000);
            }
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
            // Cache loss cannot force inference/search prerequisites when the
            // complete owned output still matches all requested conditions.
            fs::remove_dir_all(
                root.path()
                    .join(".game-screen-pick-cache")
                    .join(game_screen_pick::cache::NAMESPACE),
            )
            .unwrap();
            let recovered = invoke(root.path(), generated, true, true);
            assert!(
                recovered.status.success(),
                "{}",
                String::from_utf8_lossy(&recovered.stderr)
            );
            assert_eq!((ai.calls(), brave.calls()), counts);
            assert_eq!(
                fs::read(root.path().join("output/report.json")).unwrap(),
                before
            );
            let report: Value = serde_json::from_slice(&before).unwrap();
            if method == "semantic_video" {
                assert_eq!(
                    report["cache"]["workflow"]["conditions"]["selection"]["semantic_chunk_seconds"],
                    1.5
                );
                assert_eq!(
                    report["cache"]["workflow"]["conditions"]["selection"]["effective_semantic_chunk_seconds"],
                    1.5
                );
            }
        }
    }
}

fn completed_files(root: &Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    fs::read_dir(root.join("output"))
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().into_string().unwrap(),
                fs::read(entry.path()).unwrap(),
            )
        })
        .collect()
}

fn add_mirrored(root: &Path) {
    fs::copy(fixture("02-mirrored.mkv"), root.join("input/mirrored.mkv")).unwrap();
}

fn require_success(output: &std::process::Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    for secret in ["search-secret", "fixture-api-secret"] {
        assert!(!String::from_utf8_lossy(&output.stdout).contains(secret));
        assert!(!String::from_utf8_lossy(&output.stderr).contains(secret));
    }
    serde_json::from_str(
        String::from_utf8_lossy(&output.stderr)
            .lines()
            .find_map(|line| line.strip_prefix("metrics: "))
            .expect("successful CLI emits measurements"),
    )
    .unwrap()
}

#[test]
fn cli_help_version_validation_extraction_and_unowned_output() {
    let root = tempfile::tempdir().unwrap();
    for arguments in [["--help"], ["--version"]] {
        let output = binary_command(root.path())
            .env("PATH", "")
            .args(arguments)
            .output()
            .unwrap();
        assert!(output.status.success());
        let text = String::from_utf8(output.stdout).unwrap();
        if arguments == ["--version"] {
            assert_eq!(
                text.trim(),
                concat!("game-screen-pick ", env!("CARGO_PKG_VERSION"))
            );
        } else {
            assert!(text.contains("--config") && text.contains("--game-context"));
        }
    }
    let ai = evidence_ai();
    let brave = brave();
    setup(root.path(), &ai, &brave, "sampled_frames", true);
    let arguments = [
        "validate",
        "--config",
        "config.toml",
        "--count",
        "2",
        "--game-context",
        "Pick colored gameplay blocks",
        "input",
        "output",
    ];
    let valid = binary_command(root.path())
        .env("PATH", "")
        .args(arguments)
        .output()
        .unwrap();
    assert!(
        valid.status.success(),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );
    assert!(!root.path().join("output").exists());
    let original = fs::read(root.path().join("config.toml")).unwrap();
    let example = if let Some(binary) = std::env::var_os("GSP_ARTIFACT_BINARY") {
        fs::read(
            Path::new(&binary)
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("config.example.toml"),
        )
        .unwrap()
    } else {
        include_bytes!("../config.example.toml").to_vec()
    };
    fs::write(root.path().join("config.toml"), example).unwrap();
    let example_valid = binary_command(root.path())
        .env("PATH", "")
        .args(arguments)
        .output()
        .unwrap();
    assert!(
        example_valid.status.success(),
        "{}",
        String::from_utf8_lossy(&example_valid.stderr)
    );
    fs::write(
        root.path().join("config.toml"),
        "[selection]\nmethod='invalid'\n",
    )
    .unwrap();
    let invalid = binary_command(root.path())
        .env("PATH", "")
        .args(arguments)
        .output()
        .unwrap();
    assert_eq!(invalid.status.code(), Some(2));
    fs::write(root.path().join("config.toml"), original).unwrap();
    let extracted = binary_command(root.path())
        .args([
            "extract",
            "--at",
            "1",
            "--context-seconds",
            "0.25",
            "input/blocks.mkv",
            "extracted",
        ])
        .output()
        .unwrap();
    assert!(
        extracted.status.success(),
        "{}",
        String::from_utf8_lossy(&extracted.stderr)
    );
    let record: Value =
        serde_json::from_slice(&fs::read(root.path().join("extracted/extraction.json")).unwrap())
            .unwrap();
    assert_eq!(record["source"], "blocks.mkv");
    let frames = record["frames"].as_array().unwrap();
    assert_eq!(frames.len(), 3);
    for (frame, seconds) in frames.iter().zip([0.75, 1.0, 1.25]) {
        assert_eq!(frame["actual_seconds"], seconds);
        let pixels = image::open(
            root.path()
                .join("extracted")
                .join(frame["file"].as_str().unwrap()),
        )
        .unwrap()
        .to_rgb8();
        assert_eq!(
            pixels,
            evidence::expected("blocks.mkv", frame["actual_seconds"].as_f64().unwrap()).unwrap()
        );
    }
    fs::create_dir(root.path().join("output")).unwrap();
    fs::write(
        root.path().join("output/user-image.jpg"),
        b"keep existing user output",
    )
    .unwrap();
    let rejected = invoke(root.path(), false, false, false);
    assert_eq!(rejected.status.code(), Some(2));
    assert_eq!(
        fs::read(root.path().join("output/user-image.jpg")).unwrap(),
        b"keep existing user output"
    );
    assert_eq!((ai.calls(), brave.calls()), (0, 0));
}

#[test]
fn cli_evidence_matrix_checks_pixels_labels_events_and_zero_call_warm() {
    for method in ["sampled_frames", "semantic_video"] {
        for multiple in [false, true] {
            for generated in [false, true] {
                let root = tempfile::tempdir().unwrap();
                let ai = evidence_ai();
                let brave = brave();
                setup(root.path(), &ai, &brave, method, true);
                if multiple {
                    add_mirrored(root.path());
                }
                let cold = invoke(root.path(), generated, false, false);
                let cold_metrics = require_success(&cold);
                let files = completed_files(root.path());
                let report: Value = serde_json::from_slice(&files["report.json"]).unwrap();
                evidence::verify_output(&root.path().join("output"), &report).unwrap();
                assert_eq!(report["method"], method);
                assert_eq!(report["model"]["backend"], "vllm");
                assert_eq!(report["model"]["model"], "fixture-vision");
                assert_eq!(
                    report["sources"].as_array().unwrap().len(),
                    if multiple { 2 } else { 1 }
                );
                let selected = report["selected"].as_array().unwrap();
                if multiple {
                    assert_ne!(selected[0]["source"], selected[1]["source"]);
                } else {
                    assert_ne!(selected[0]["actual_seconds"], selected[1]["actual_seconds"]);
                }
                let calls = (ai.calls(), brave.calls());
                assert_eq!(calls.1, if generated { 2 } else { 0 });
                let warm = invoke(root.path(), generated, true, true);
                let warm_metrics = require_success(&warm);
                assert_eq!((ai.calls(), brave.calls()), calls);
                assert_eq!(completed_files(root.path()), files);
                for field in [
                    "extraction",
                    "machine_quality",
                    "inference_callbacks",
                    "media_probe",
                ] {
                    assert_eq!(
                        warm_metrics["stage_seconds"][field], 0.0,
                        "warm performs no {field}"
                    );
                    assert!(
                        cold_metrics["stage_seconds"][field]
                            .as_f64()
                            .unwrap()
                            .is_finite()
                    );
                }
                assert!(
                    cold_metrics["stage_seconds"]["cache_verification"]
                        .as_f64()
                        .unwrap()
                        > 0.0
                );
                assert!(
                    warm_metrics["stage_seconds"]["cache_verification"]
                        .as_f64()
                        .unwrap()
                        > 0.0
                );
                assert!(
                    cold_metrics["peak_memory"]["status"] == "measured"
                        || cold_metrics["peak_memory"]["status"] == "unavailable"
                );
                if cold_metrics["peak_memory"]["status"] == "measured" {
                    assert!(cold_metrics["peak_memory"]["bytes"].as_u64().unwrap() > 0);
                }
                if let Some(qa) = std::env::var_os("GSP_QA_DIR") {
                    let destination = std::path::PathBuf::from(qa).join(format!(
                        "cli-{method}-{}-{}",
                        if multiple { "multiple" } else { "single" },
                        if generated {
                            "generated-mock"
                        } else {
                            "direct"
                        }
                    ));
                    fs::create_dir_all(&destination).unwrap();
                    for (name, bytes) in &files {
                        fs::write(destination.join(name), bytes).unwrap();
                    }
                    fs::write(destination.join("measurements.json"), serde_json::to_vec_pretty(&json!({"evidence":"synthetic fixtures and loopback fixed responses; no live capability or human review", "cold":cold_metrics,"warm":warm_metrics,"external_calls":{"cold":{"ai":calls.0,"search":calls.1,"runtime":0},"warm":{"ai":0,"search":0,"runtime":0}}})).unwrap()).unwrap();
                }
            }
        }
    }
}

#[test]
fn cli_rotated_multitrack_artifacts_keep_full_resolution_display_orientation() {
    for method in ["sampled_frames", "semantic_video"] {
        let root = tempfile::tempdir().unwrap();
        let ai = evidence_ai();
        let brave = brave();
        setup(root.path(), &ai, &brave, method, true);
        fs::remove_file(root.path().join("input/blocks.mkv")).unwrap();
        fs::copy(
            fixture("03-multitrack-rotated.mov"),
            root.path().join("input/rotated.mov"),
        )
        .unwrap();
        require_success(&invoke(root.path(), false, false, false));
        let report: Value =
            serde_json::from_slice(&completed_files(root.path())["report.json"]).unwrap();
        evidence::verify_output(&root.path().join("output"), &report).unwrap();
        assert!(
            report["selected"]
                .as_array()
                .unwrap()
                .iter()
                .all(|frame| frame["width"] == 96 && frame["height"] == 160)
        );
    }
}

#[test]
fn cli_resume_reuses_committed_media_and_context_and_matches_clean_cold() {
    for method in ["sampled_frames", "semantic_video"] {
        for multiple in [false, true] {
            for generated in [false, true] {
                let root = tempfile::tempdir().unwrap();
                let failed_once = Arc::new(AtomicBool::new(false));
                let accepted = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
                let calls = accepted.clone();
                let failed = failed_once.clone();
                let mut media_calls = 0;
                let ai = Server::with_status(move |header, body| {
                    evidence::verify_request(body).unwrap();
                    let key = evidence::metadata(body, "semantic_chunk_metadata")
                        .or_else(|_| evidence::metadata(body, "assessment_metadata"));
                    if let Ok(key) = key {
                        media_calls += 1;
                        if media_calls == 2 {
                            failed.store(true, Ordering::Relaxed);
                            return (401, json!({"error":"fixture-api-secret"}));
                        }
                        calls.lock().unwrap().push(key);
                    }
                    (200, fixed_answer(header, body, &|_| false))
                });
                let brave = brave();
                setup(root.path(), &ai, &brave, method, true);
                if multiple {
                    add_mirrored(root.path());
                }
                let interrupted = invoke(root.path(), generated, false, false);
                assert_eq!(interrupted.status.code(), Some(1));
                assert!(failed_once.load(Ordering::Relaxed));
                assert!(!root.path().join("output").exists());
                assert!(
                    !String::from_utf8_lossy(&interrupted.stderr).contains("fixture-api-secret")
                );
                let first = accepted.lock().unwrap()[0].clone();
                let partial_calls = (ai.calls(), brave.calls());
                let resumed = invoke(root.path(), generated, false, false);
                let resumed_metrics = require_success(&resumed);
                assert_eq!(
                    accepted
                        .lock()
                        .unwrap()
                        .iter()
                        .filter(|key| **key == first)
                        .count(),
                    1,
                    "completed media must not repeat on resume"
                );
                assert_eq!(
                    brave.calls(),
                    partial_calls.1,
                    "completed generated context must not search again"
                );
                let files = completed_files(root.path());
                let report: Value = serde_json::from_slice(&files["report.json"]).unwrap();
                evidence::verify_output(&root.path().join("output"), &report).unwrap();
                let resume_calls = (
                    ai.calls() - partial_calls.0,
                    brave.calls() - partial_calls.1,
                );
                fs::remove_dir_all(root.path().join("output")).unwrap();
                fs::remove_dir_all(root.path().join(".game-screen-pick-cache")).unwrap();
                let before_clean = ai.calls();
                let clean = invoke(root.path(), generated, false, false);
                let clean_metrics = require_success(&clean);
                assert_eq!(
                    completed_files(root.path()),
                    files,
                    "resumed artifacts must match clean cold byte for byte"
                );
                assert!(resume_calls.0 < ai.calls() - before_clean);
                if let Some(qa) = std::env::var_os("GSP_QA_DIR") {
                    let destination = std::path::PathBuf::from(qa).join(format!(
                        "resume-{method}-{}-{}",
                        if multiple { "multiple" } else { "single" },
                        if generated {
                            "generated-mock"
                        } else {
                            "direct"
                        }
                    ));
                    fs::create_dir_all(&destination).unwrap();
                    fs::write(destination.join("measurements.json"), serde_json::to_vec_pretty(&json!({"evidence":"recoverable HTTP failure after a committed batch/chunk, not a signal interruption", "partial_external_calls":{"ai":partial_calls.0,"search":partial_calls.1,"runtime":0},"resume_external_calls":{"ai":resume_calls.0,"search":resume_calls.1,"runtime":0},"resume":resumed_metrics,"clean_cold":clean_metrics})).unwrap()).unwrap();
                }
            }
        }
    }
}

#[test]
fn cli_shortage_http_and_schema_failures_preserve_previous_complete_output() {
    for method in ["sampled_frames", "semantic_video"] {
        let root = tempfile::tempdir().unwrap();
        let mode = Arc::new(AtomicUsize::new(0));
        let current = mode.clone();
        let ai = Server::with_status(move |header, body| match current.load(Ordering::Relaxed) {
            1 => (401, json!({"error":"fixture-api-secret"})),
            2 => (
                200,
                json!({"model":"fixture-vision","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"{\"assessments\":[]}"}}]}),
            ),
            _ => {
                evidence::verify_request(body).unwrap();
                (200, fixed_answer(header, body, &|_| false))
            }
        });
        let brave = brave();
        setup(root.path(), &ai, &brave, method, true);
        require_success(&invoke(root.path(), true, false, false));
        let files = completed_files(root.path());
        let shortage = invoke_count(root.path(), true, false, None, 999);
        assert_eq!(shortage.status.code(), Some(1));
        assert!(
            String::from_utf8_lossy(&shortage.stderr).contains("insufficient"),
            "{}",
            String::from_utf8_lossy(&shortage.stderr)
        );
        assert_eq!(completed_files(root.path()), files);
        let config = root.path().join("config.toml");
        let source = fs::read_to_string(&config).unwrap().replace(
            "cache_revision = \"fixture-v1\"",
            "cache_revision = \"changed-for-failure\"",
        );
        fs::write(config, source).unwrap();
        for (failure_mode, expected_calls) in [(1, 1), (2, 2)] {
            mode.store(failure_mode, Ordering::Relaxed);
            let before = ai.calls();
            let rejected = invoke(root.path(), false, false, false);
            assert_eq!(rejected.status.code(), Some(1));
            assert_eq!(
                ai.calls() - before,
                expected_calls,
                "finite request attempts"
            );
            assert_eq!(completed_files(root.path()), files);
            assert!(!String::from_utf8_lossy(&rejected.stderr).contains("fixture-api-secret"));
        }
        mode.store(0, Ordering::Relaxed);
        require_success(&invoke(root.path(), false, false, false));
        evidence::verify_output(
            &root.path().join("output"),
            &serde_json::from_slice(&completed_files(root.path())["report.json"]).unwrap(),
        )
        .unwrap();
    }
}

#[cfg(unix)]
#[test]
fn cli_ffmpeg_failure_preserves_previous_complete_output_without_inference() {
    use std::os::unix::fs::PermissionsExt;
    for method in ["sampled_frames", "semantic_video"] {
        let root = tempfile::tempdir().unwrap();
        let ai = ai();
        let brave = brave();
        setup(root.path(), &ai, &brave, method, true);
        require_success(&invoke(root.path(), false, false, false));
        let files = completed_files(root.path());
        let counts = (ai.calls(), brave.calls());
        let config = root.path().join("config.toml");
        let source = fs::read_to_string(&config).unwrap().replace(
            "cache_revision = \"fixture-v1\"",
            "cache_revision = \"force-extraction-miss\"",
        );
        fs::write(config, source).unwrap();
        let tools = root.path().join("tools");
        fs::create_dir(&tools).unwrap();
        let ffprobe = std::env::split_paths(&std::env::var_os("PATH").unwrap())
            .map(|directory| directory.join("ffprobe"))
            .find(|path| path.is_file())
            .unwrap();
        std::os::unix::fs::symlink(ffprobe, tools.join("ffprobe")).unwrap();
        fs::write(tools.join("ffmpeg"), "#!/bin/sh\nexit 1\n").unwrap();
        fs::set_permissions(tools.join("ffmpeg"), fs::Permissions::from_mode(0o700)).unwrap();
        let rejected = invoke_with_path(root.path(), false, false, Some(&tools));
        assert_eq!(rejected.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&rejected.stderr).contains("FFmpeg"));
        assert_eq!((ai.calls(), brave.calls()), counts);
        assert_eq!(completed_files(root.path()), files);
        require_success(&invoke(root.path(), false, false, false));
    }
}

#[test]
fn evidence_rejects_wrong_pixels_times_labels_and_incomplete_artifacts() {
    let root = tempfile::tempdir().unwrap();
    let captured = Arc::new(std::sync::Mutex::new(None));
    let capture = captured.clone();
    let ai = Server::new(move |header, body| {
        evidence::verify_request(body).unwrap();
        if evidence::metadata(body, "assessment_metadata")
            .is_ok_and(|value| value["phase"] == "secondary")
        {
            *capture.lock().unwrap() = Some(body.clone());
        }
        fixed_answer(header, body, &|_| false)
    });
    let brave = brave();
    setup(root.path(), &ai, &brave, "sampled_frames", true);
    add_mirrored(root.path());
    require_success(&invoke(root.path(), false, false, false));
    let request = captured.lock().unwrap().take().unwrap();
    let parts = request["messages"][0]["content"].as_array().unwrap();
    let images = parts
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (value["type"] == "image_url").then_some(index))
        .collect::<Vec<_>>();
    let mut swapped = request.clone();
    let other = images
        .iter()
        .skip(2)
        .find(|index| parts[**index] != parts[images[1]])
        .expect("fixture has visually distinct contexts");
    swapped["messages"][0]["content"][images[1]] = parts[*other].clone();
    assert!(
        evidence::verify_request(&swapped).is_err(),
        "same-size context from another source/time must fail"
    );
    let mut wrong_label = request.clone();
    let index = parts
        .iter()
        .position(|part| {
            part["text"]
                .as_str()
                .is_some_and(|text| text.contains("assessment_metadata"))
        })
        .unwrap();
    let mut metadata = evidence::metadata(&request, "assessment_metadata").unwrap();
    metadata["candidates"][0]["actual_seconds"] = json!(5.25);
    wrong_label["messages"][0]["content"][index]["text"] =
        json!(format!("external data\n{metadata}"));
    assert!(
        evidence::verify_request(&wrong_label).is_err(),
        "media/time mismatch must fail"
    );
    let files = completed_files(root.path());
    let report: Value = serde_json::from_slice(&files["report.json"]).unwrap();
    for (pointer, value) in [
        ("/selected/0/actual_seconds", json!(5.25)),
        ("/selected/0/source", json!("unknown.mkv")),
        ("/selected/0/rank", json!(2)),
        ("/selected/0/assessment/scene", json!("wrong label")),
    ] {
        let mut wrong = report.clone();
        *wrong.pointer_mut(pointer).unwrap() = value;
        assert!(
            evidence::verify_output(&root.path().join("output"), &wrong).is_err(),
            "{pointer}"
        );
    }
    fs::write(
        root.path().join("output/selected-01.jpg"),
        &files["selected-02.jpg"],
    )
    .unwrap();
    assert!(evidence::verify_output(&root.path().join("output"), &report).is_err());
    fs::write(
        root.path().join("output/selected-01.jpg"),
        &files["selected-01.jpg"],
    )
    .unwrap();
    fs::remove_file(root.path().join("output/selected-02.jpg")).unwrap();
    assert!(evidence::verify_output(&root.path().join("output"), &report).is_err());
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
    let warm = invoke(root.path(), true, true, true);
    assert!(
        warm.status.success(),
        "{}",
        String::from_utf8_lossy(&warm.stderr)
    );
    assert_eq!((ai.calls(), brave.calls()), counts);
}

#[test]
fn unusable_capacity_request_schema_and_local_auth_fail_before_brave_or_inference() {
    for failure in ["images", "request", "auth"] {
        let root = tempfile::tempdir().unwrap();
        let ai = ai();
        let brave = brave();
        setup(root.path(), &ai, &brave, "sampled_frames", true);
        let path = root.path().join("config.toml");
        let source = fs::read_to_string(&path).unwrap();
        let source = match failure {
            "images" => source.replace("max_images = 64", "max_images = 1"),
            "request" => source.replace("max_request_bytes = 33554432", "max_request_bytes = 1024"),
            "auth" => source.replace(
                "[ai.protocol]",
                "api_key = \"invalid\\r\\nheader\"\n[ai.protocol]",
            ),
            _ => unreachable!(),
        };
        fs::write(&path, source).unwrap();
        let rejected = invoke(root.path(), true, false, false);
        assert_eq!(rejected.status.code(), Some(1));
        let diagnostic = String::from_utf8_lossy(&rejected.stderr);
        assert!(
            diagnostic.contains(match failure {
                "images" => "image capacity",
                "request" => "byte limit",
                "auth" => "authentication",
                _ => unreachable!(),
            }),
            "{failure}: {diagnostic}"
        );
        assert_eq!((ai.calls(), brave.calls()), (0, 0));
        assert!(!root.path().join("output").exists());
    }
}

#[test]
fn unusable_media_fails_before_title_search_context_or_runtime() {
    for failure in ["corrupt", "ffprobe"] {
        let root = tempfile::tempdir().unwrap();
        let ai = ai();
        let brave = brave();
        setup(root.path(), &ai, &brave, "sampled_frames", true);
        if failure == "corrupt" {
            fs::write(root.path().join("input/blocks.mkv"), b"invalid video").unwrap();
        }
        let config = root.path().join("config.toml");
        let source = fs::read_to_string(&config).unwrap();
        fs::write(&config, format!("{source}\n[runtime]\nbackend = \"vllm\"\nstart_command = [\"/missing/runtime-start\", \"{{session}}\"]\nstop_command = [\"/missing/runtime-stop\", \"{{pid}}\", \"{{session}}\"]\n")).unwrap();
        let rejected = invoke(root.path(), true, false, failure == "ffprobe");
        assert_eq!(rejected.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&rejected.stderr).contains("FFprobe"));
        assert_eq!((ai.calls(), brave.calls()), (0, 0));
        assert!(!root.path().join("output").exists());
    }
}

#[test]
fn probed_frames_outside_model_limits_fail_before_title_calls() {
    for method in ["sampled_frames", "semantic_video"] {
        for failure in ["edge", "pixels"] {
            let root = tempfile::tempdir().unwrap();
            let ai = ai();
            let brave = brave();
            setup(root.path(), &ai, &brave, method, true);
            write_large_video(&root.path().join("input/blocks.mkv"), "1536x256");
            let config = root.path().join("config.toml");
            let source = fs::read_to_string(&config).unwrap();
            let source = match failure {
                "edge" => source.replace("max_image_edge = 16384", "max_image_edge = 1535"),
                "pixels" => {
                    source.replace("max_image_pixels = 32000000", "max_image_pixels = 393215")
                }
                _ => unreachable!(),
            };
            fs::write(&config, source).unwrap();
            let rejected = invoke(root.path(), true, false, false);
            assert_eq!(rejected.status.code(), Some(1));
            assert!(String::from_utf8_lossy(&rejected.stderr).contains("input frame geometry"));
            assert_eq!((ai.calls(), brave.calls()), (0, 0));
            assert!(!root.path().join("output").exists());
        }
    }
}

#[test]
fn semantic_plan_misses_validate_all_frames_before_extraction_inference_or_runtime() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let ffprobe = Command::new("/usr/bin/which")
        .arg("ffprobe")
        .output()
        .unwrap();
    assert!(ffprobe.status.success(), "FFprobe is required");
    let ffprobe = String::from_utf8(ffprobe.stdout).unwrap();
    for generated in [false, true] {
        for failure in ["edge", "pixels"] {
            let root = tempfile::tempdir().unwrap();
            let ai = ai();
            let brave = brave();
            setup(root.path(), &ai, &brave, "sampled_frames", true);
            let config = root.path().join("config.toml");
            let source = fs::read_to_string(&config).unwrap();
            let source = match failure {
                "edge" => source.replace("max_image_edge = 16384", "max_image_edge = 2047"),
                "pixels" => {
                    source.replace("max_image_pixels = 32000000", "max_image_pixels = 2097151")
                }
                _ => unreachable!(),
            };
            fs::write(&config, &source).unwrap();
            let previous_output = if generated {
                // Prime a real persisted context under the same model/protocol.
                let cold = invoke(root.path(), true, false, false);
                assert!(
                    cold.status.success(),
                    "{}",
                    String::from_utf8_lossy(&cold.stderr)
                );
                assert_eq!(brave.calls(), 2);
                Some(fs::read(root.path().join("output/report.json")).unwrap())
            } else {
                None
            };
            let counts = (ai.calls(), brave.calls());
            let source = source
                .replace("method = \"sampled_frames\"", "method = \"semantic_video\"")
                .replace("api_key = \"search-secret\"\n", "");
            fs::write(&config, format!("{source}\n[runtime]\nbackend = \"vllm\"\nstart_command = [\"/bin/sh\", \"-c\", \": > runtime-started; exit 1\", \"{{session}}\"]\nstop_command = [\"/bin/sh\", \"-c\", \": > runtime-stopped; exit 1\", \"{{pid}}\", \"{{session}}\"]\n")).unwrap();
            // A valid first input must not cause extraction before the later
            // oversized input has been checked. The aggregate plan is absent.
            write_large_video(&root.path().join("input/z-large.mkv"), "2048x1024");
            let tools = root.path().join("tools");
            fs::create_dir(&tools).unwrap();
            symlink(ffprobe.trim(), tools.join("ffprobe")).unwrap();
            fs::write(
                tools.join("ffmpeg"),
                "#!/bin/sh\n: > ffmpeg-called\nexit 1\n",
            )
            .unwrap();
            fs::set_permissions(tools.join("ffmpeg"), fs::Permissions::from_mode(0o755)).unwrap();
            let rejected = invoke_with_path(root.path(), generated, false, Some(&tools));
            assert_eq!(rejected.status.code(), Some(1));
            let diagnostic = String::from_utf8_lossy(&rejected.stderr);
            assert!(
                diagnostic.contains("input frame geometry"),
                "{generated}/{failure}: {diagnostic}"
            );
            assert_eq!((ai.calls(), brave.calls()), counts);
            for marker in ["ffmpeg-called", "runtime-started", "runtime-stopped"] {
                assert!(!root.path().join(marker).exists(), "{marker}");
            }
            if let Some(before) = previous_output {
                assert_eq!(
                    fs::read(root.path().join("output/report.json")).unwrap(),
                    before
                );
            } else {
                assert!(!root.path().join("output").exists());
            }
        }
    }
}

#[test]
fn impossible_contact_sheet_limits_fail_before_probe_or_title_calls() {
    for method in ["sampled_frames", "semantic_video"] {
        let root = tempfile::tempdir().unwrap();
        let ai = ai();
        let brave = brave();
        setup(root.path(), &ai, &brave, method, true);
        let config = root.path().join("config.toml");
        let source = fs::read_to_string(&config)
            .unwrap()
            .replace("max_image_edge = 16384", "max_image_edge = 200");
        fs::write(&config, source).unwrap();
        let rejected = invoke(root.path(), true, false, true);
        assert_eq!(rejected.status.code(), Some(1));
        let diagnostic = String::from_utf8_lossy(&rejected.stderr);
        assert!(
            diagnostic.contains("candidate contact sheet"),
            "{diagnostic}"
        );
        assert!(!diagnostic.contains("FFprobe"));
        assert_eq!((ai.calls(), brave.calls()), (0, 0));
    }
}

#[test]
fn fixed_selection_request_budget_fails_before_media_or_runtime() {
    for method in ["sampled_frames", "semantic_video"] {
        let root = tempfile::tempdir().unwrap();
        let ai = ai();
        let brave = brave();
        setup(root.path(), &ai, &brave, method, true);
        let config = root.path().join("config.toml");
        let source = fs::read_to_string(&config)
            .unwrap()
            .replace("max_request_bytes = 33554432", "max_request_bytes = 1024");
        fs::write(&config, format!("{source}\n[runtime]\nbackend = \"vllm\"\nstart_command = [\"/missing/runtime-start\", \"{{session}}\"]\nstop_command = [\"/missing/runtime-stop\", \"{{pid}}\", \"{{session}}\"]\n")).unwrap();
        let rejected = invoke(root.path(), false, false, true);
        assert_eq!(rejected.status.code(), Some(1));
        let diagnostic = String::from_utf8_lossy(&rejected.stderr);
        assert!(
            diagnostic.contains("selection request exceeds configured byte limit"),
            "{diagnostic}"
        );
        assert!(!diagnostic.contains("FFprobe") && !diagnostic.contains("runtime command"));
        assert_eq!((ai.calls(), brave.calls()), (0, 0));
    }
}

#[test]
fn sampled_actual_frame_budget_fails_before_title_calls() {
    let root = tempfile::tempdir().unwrap();
    let ai = ai();
    let brave = brave();
    setup(root.path(), &ai, &brave, "sampled_frames", true);
    let made = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x96:r=1000",
            "-frames:v",
            "2050",
            "-c:v",
            "ffv1",
        ])
        .arg(root.path().join("input/blocks.mkv"))
        .output()
        .unwrap();
    assert!(
        made.status.success(),
        "{}",
        String::from_utf8_lossy(&made.stderr)
    );
    fs::copy(
        root.path().join("input/blocks.mkv"),
        root.path().join("input/second.mkv"),
    )
    .unwrap();
    let config = root.path().join("config.toml");
    let source = fs::read_to_string(&config)
        .unwrap()
        .replace("samples_per_video = 6", "samples_per_video = 4096");
    fs::write(&config, source).unwrap();
    let rejected = invoke(root.path(), true, false, false);
    assert_eq!(rejected.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("actual-frame candidate limit"));
    assert_eq!((ai.calls(), brave.calls()), (0, 0));
    assert!(!root.path().join("output").exists());
}

#[test]
fn missing_brave_credentials_fail_before_input_probe() {
    let root = tempfile::tempdir().unwrap();
    let ai = ai();
    let brave = brave();
    setup(root.path(), &ai, &brave, "sampled_frames", true);
    let config = root.path().join("config.toml");
    let source = fs::read_to_string(&config)
        .unwrap()
        .replace("api_key = \"search-secret\"\n", "");
    fs::write(&config, source).unwrap();
    let rejected = invoke(root.path(), true, false, true);
    assert_eq!(rejected.status.code(), Some(1));
    let diagnostic = String::from_utf8_lossy(&rejected.stderr);
    assert!(diagnostic.contains("Brave API key is required"));
    assert!(!diagnostic.contains("FFprobe"));
    assert_eq!((ai.calls(), brave.calls()), (0, 0));
    let invalid = Command::new(BINARY)
        .current_dir(root.path())
        .env("PATH", "")
        .env("GAME_SCREEN_PICK_BRAVE_API_KEY", "bad\r\nsecret")
        .args([
            "--config",
            "config.toml",
            "--count",
            "2",
            "--game-title",
            "Example Game",
            "input",
            "output",
        ])
        .output()
        .unwrap();
    assert_eq!(invalid.status.code(), Some(1));
    let diagnostic = String::from_utf8_lossy(&invalid.stderr);
    assert!(diagnostic.contains("invalid Brave authentication header"));
    assert!(!diagnostic.contains("FFprobe") && !diagnostic.contains("bad\r\nsecret"));
    assert_eq!((ai.calls(), brave.calls()), (0, 0));
}

#[test]
fn cached_title_context_does_not_require_brave_credentials_for_new_selection() {
    let root = tempfile::tempdir().unwrap();
    let ai = ai();
    let brave = brave();
    setup(root.path(), &ai, &brave, "sampled_frames", true);
    let cold = invoke(root.path(), true, false, false);
    assert!(
        cold.status.success(),
        "{}",
        String::from_utf8_lossy(&cold.stderr)
    );
    let counts = (ai.calls(), brave.calls());
    let config = root.path().join("config.toml");
    let source = fs::read_to_string(&config)
        .unwrap()
        .replace("api_key = \"search-secret\"\n", "")
        .replace("samples_per_video = 6", "samples_per_video = 7");
    fs::write(&config, source).unwrap();
    let changed = invoke(root.path(), true, false, false);
    assert!(
        changed.status.success(),
        "{}",
        String::from_utf8_lossy(&changed.stderr)
    );
    assert!(ai.calls() > counts.0);
    assert_eq!(brave.calls(), counts.1);
}

#[test]
fn oversized_direct_context_fails_before_media_or_runtime() {
    for text in ["x".repeat(2048), "\"".repeat(600)] {
        let root = tempfile::tempdir().unwrap();
        let ai = ai();
        let brave = brave();
        setup(root.path(), &ai, &brave, "sampled_frames", true);
        let config = root.path().join("config.toml");
        let source = fs::read_to_string(&config)
            .unwrap()
            .replace("max_request_bytes = 33554432", "max_request_bytes = 1024");
        fs::write(&config, format!("{source}\n[runtime]\nbackend = \"vllm\"\nstart_command = [\"/missing/runtime-start\", \"{{session}}\"]\nstop_command = [\"/missing/runtime-stop\", \"{{pid}}\", \"{{session}}\"]\n")).unwrap();
        let rejected = Command::new(BINARY)
            .current_dir(root.path())
            .env("PATH", "")
            .args([
                "--config",
                "config.toml",
                "--count",
                "2",
                "--game-context",
                &format!("{text}x"),
                "input",
                "output",
            ])
            .output()
            .unwrap();
        assert_eq!(rejected.status.code(), Some(1));
        assert!(
            String::from_utf8_lossy(&rejected.stderr)
                .contains("direct game context exceeds inference request byte limit")
        );
        assert_eq!((ai.calls(), brave.calls()), (0, 0));
        assert!(!root.path().join("output").exists());
    }
}

#[test]
fn contract_references_are_hashed_in_owned_report_and_still_invalidate_reuse() {
    const SECRET: &str = "CONTRACT_REFERENCE_SECRET_SENTINEL";
    let root = tempfile::tempdir().unwrap();
    let ai = ai();
    let brave = brave();
    setup(root.path(), &ai, &brave, "sampled_frames", true);
    let config = root.path().join("config.toml");
    let source = fs::read_to_string(&config)
        .unwrap()
        .replace("mock hard-stop test contract", SECRET);
    fs::write(&config, &source).unwrap();
    let cold = invoke(root.path(), true, false, false);
    assert!(
        cold.status.success(),
        "{}",
        String::from_utf8_lossy(&cold.stderr)
    );
    let before = fs::read_to_string(root.path().join("output/report.json")).unwrap();
    assert!(!before.contains(SECRET));
    let report: Value = serde_json::from_str(&before).unwrap();
    let game = &report["cache"]["workflow"]["conditions"]["game"];
    assert_eq!(game["search_conditions_digest"].as_str().unwrap().len(), 64);
    assert_eq!(game["search"].as_object().unwrap().len(), 3);
    let counts = (ai.calls(), brave.calls());
    fs::write(
        &config,
        source.replace(SECRET, "different approved reference"),
    )
    .unwrap();
    let changed = invoke(root.path(), true, false, false);
    assert!(
        changed.status.success(),
        "{}",
        String::from_utf8_lossy(&changed.stderr)
    );
    assert!(ai.calls() > counts.0);
    assert_eq!(brave.calls(), counts.1 + 2);
    let after = fs::read_to_string(root.path().join("output/report.json")).unwrap();
    assert_ne!(before, after);
    assert!(!after.contains("different approved reference"));
}

#[test]
fn quiet_semantic_sources_are_optional_and_an_empty_completed_plan_is_reused() {
    for all_quiet in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let ai =
            ai_with_quiet_chunks(move |metadata| all_quiet || metadata["source"] == "quiet.mkv");
        let brave = brave();
        setup(root.path(), &ai, &brave, "semantic_video", true);
        fs::copy(
            root.path().join("input/blocks.mkv"),
            root.path().join("input/quiet.mkv"),
        )
        .unwrap();
        let result = invoke(root.path(), false, false, false);
        if all_quiet {
            assert_eq!(result.status.code(), Some(1));
            assert!(
                String::from_utf8_lossy(&result.stderr)
                    .contains("no usable events; not enough candidates")
            );
            assert!(!root.path().join("output").exists());
            let counts = (ai.calls(), brave.calls());
            let repeated = invoke(root.path(), false, true, true);
            assert_eq!(repeated.status.code(), Some(1));
            assert!(
                String::from_utf8_lossy(&repeated.stderr)
                    .contains("no usable events; not enough candidates")
            );
            assert_eq!((ai.calls(), brave.calls()), counts);
        } else {
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let report = fs::read_to_string(root.path().join("output/report.json")).unwrap();
            let report: Value = serde_json::from_str(&report).unwrap();
            assert!(
                report["selected"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|entry| entry["source"] == "blocks.mkv")
            );
        }
    }
}

#[test]
fn invalid_brave_titles_are_input_errors_without_preparation_writes() {
    for title in ["x".repeat(201), "Example\nGame".into()] {
        let root = tempfile::tempdir().unwrap();
        let ai = ai();
        let brave = brave();
        setup(root.path(), &ai, &brave, "sampled_frames", true);
        for diagnostic in [false, true] {
            let mut command = Command::new(BINARY);
            command.current_dir(root.path()).env("PATH", "");
            if diagnostic {
                command.arg("validate");
            }
            let rejected = command
                .args([
                    "--config",
                    "config.toml",
                    "--count",
                    "2",
                    "--game-title",
                    &title,
                    "input",
                    "output",
                ])
                .output()
                .unwrap();
            assert_eq!(rejected.status.code(), Some(2));
        }
        assert!(!root.path().join(".game-screen-pick-cache").exists());
        assert!(!root.path().join(".gsp-output-locks").exists());
        assert_eq!((ai.calls(), brave.calls()), (0, 0));
    }
}

#[test]
fn sigint_during_live_http_promptly_cleans_the_owned_runtime() {
    sigint_during_live_http(false);
}

#[test]
fn sigint_also_cancels_a_hung_runtime_stop_command() {
    sigint_during_live_http(true);
}

fn sigint_during_live_http(hung_stop: bool) {
    use std::os::unix::fs::PermissionsExt;
    use std::time::Instant;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let started = root.join("started");
    let stopped = root.join("stopped");
    let requested = root.join("requested");
    let release = Arc::new(AtomicBool::new(false));
    let server_release = release.clone();
    let server_started = started.clone();
    let server_requested = requested.clone();
    let ai = Server::with_status(move |header, _| {
        if header.starts_with("GET /v1/models ") {
            if server_started.exists() {
                return (200, json!({"data":[{"id":"fixture-vision"}]}));
            }
            return (503, Value::Null);
        }
        assert!(header.starts_with("POST /v1/chat/completions "));
        fs::write(&server_requested, b"headers deliberately delayed").unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        while !server_release.load(Ordering::Relaxed) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        (200, Value::Null)
    });
    let brave = brave();
    setup(&root, &ai, &brave, "sampled_frames", true);
    let start = root.join("start.sh");
    let stop = root.join("stop.sh");
    fs::write(
        &start,
        format!(
            "#!/bin/sh\nprintf '%s' \"$1\" > '{}'\nexec /bin/sleep 60\n",
            started.display()
        ),
    )
    .unwrap();
    let stop_action = if hung_stop {
        "exec /bin/sleep 60"
    } else {
        "kill -TERM \"$1\""
    };
    fs::write(
        &stop,
        format!(
            "#!/bin/sh\nprintf '%s %s %s' \"$1\" \"$2\" \"$$\" > '{}'\n{stop_action}\n",
            stopped.display()
        ),
    )
    .unwrap();
    for path in [&start, &stop] {
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    }
    let path = root.join("config.toml");
    let source = fs::read_to_string(&path).unwrap();
    let timeout = if hung_stop { 300 } else { 3 };
    fs::write(&path,format!("{source}\n[runtime]\nbackend = \"vllm\"\nstart_command = [\"{}\", \"{{session}}\"]\nstop_command = [\"{}\", \"{{pid}}\", \"{{session}}\"]\nstartup_timeout_seconds = 10\ncommand_timeout_seconds = {timeout}\n",start.display(),stop.display())).unwrap();
    let mut child = Command::new(BINARY)
        .current_dir(&root)
        .args([
            "--config",
            "config.toml",
            "--count",
            "2",
            "--game-context",
            "colored gameplay",
            "input",
            "output",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !requested.exists() {
        assert!(Instant::now() < deadline, "inference did not start");
        thread::sleep(Duration::from_millis(10));
    }
    let interrupted = Instant::now();
    unsafe {
        assert_eq!(libc::kill(child.id() as i32, libc::SIGINT), 0);
    }
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if interrupted.elapsed() > Duration::from_secs(3) {
            release.store(true, Ordering::Relaxed);
            let _ = child.kill();
            let _ = child.wait();
            panic!("HTTP cancellation did not promptly return and clean runtime");
        }
        thread::sleep(Duration::from_millis(10));
    };
    release.store(true, Ordering::Relaxed);
    assert_eq!(status.code(), Some(130));
    assert!(stopped.exists());
    let stop_record = fs::read_to_string(&stopped).unwrap();
    let stop_pid: i32 = stop_record
        .split_whitespace()
        .nth(2)
        .unwrap()
        .parse()
        .unwrap();
    unsafe {
        assert_eq!(
            libc::kill(stop_pid, 0),
            -1,
            "owned stop command survived cancellation"
        );
    }
    let pid: i32 = stop_record
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap();
    unsafe {
        assert_eq!(
            libc::kill(pid, 0),
            -1,
            "owned runtime survived cancellation"
        );
    }
    assert!(!root.join("output").exists());
    assert_eq!(brave.calls(), 0);
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
