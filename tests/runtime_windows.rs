//! Windows-native process/HTTP lifecycle contracts, using compiled helpers.
#![cfg(windows)]

#[path = "support/process_helper.rs"]
mod process_helper;

use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::anyhow;
use game_screen_pick::ai::OpenAiClient;
use game_screen_pick::config::{AiConfig, Config};
use game_screen_pick::media::Interrupted;
use game_screen_pick::runtime::{RuntimeBackend, RuntimeConfig, RuntimeManager};
use process_helper::{Helper, assert_gone, wait_for};

struct Mock {
    url: String,
    requests: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<io::Result<()>>>,
}

impl Mock {
    fn new(mut reply: impl FnMut() -> (u16, Duration) + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(AtomicUsize::new(0));
        let recorded = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let worker = thread::spawn(move || -> io::Result<()> {
            while !stopping.load(Ordering::Relaxed) {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                if !read_request(&mut stream)? {
                    continue;
                }
                recorded.fetch_add(1, Ordering::Relaxed);
                let (status, delay) = reply();
                thread::sleep(delay);
                let body = r#"{"data":[{"id":"fixture-model"}]}"#;
                if let Err(error) = write!(
                    stream,
                    "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                ) && !peer_disconnected(&error)
                {
                    return Err(error);
                }
            }
            Ok(())
        });
        Self {
            url,
            requests,
            stop,
            worker: Some(worker),
        }
    }
}

impl Drop for Mock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let failure = match self.worker.take().unwrap().join() {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(format!("mock worker failed: {error}")),
            Err(payload) => Some(format!(
                "mock worker panicked: {}",
                panic_text(payload.as_ref())
            )),
        };
        if let Some(failure) = failure {
            if thread::panicking() {
                // Preserve the primary assertion and its captured diagnostics.
                // A second panic from Drop would abort the entire test process.
                let _ = writeln!(io::stderr(), "secondary mock cleanup failure: {failure}");
            } else {
                panic!("{failure}");
            }
        }
    }
}

fn peer_disconnected(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::BrokenPipe
            | io::ErrorKind::UnexpectedEof
    )
}

fn read_request(stream: &mut TcpStream) -> io::Result<bool> {
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    let mut bytes = Vec::new();
    while !bytes.ends_with(b"\r\n\r\n") {
        let mut byte = [0];
        match stream.read(&mut byte) {
            Ok(0) => return Ok(false),
            Ok(_) => bytes.push(byte[0]),
            Err(error) if peer_disconnected(&error) => return Ok(false),
            Err(error) => return Err(error),
        }
        if bytes.len() >= 16384 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "mock request headers exceed fixture limit",
            ));
        }
    }
    Ok(true)
}

fn ai_config(url: &str) -> AiConfig {
    Config::parse(&format!(
        r#"
[selection]
method = "sampled_frames"
[ai]
backend = "vllm"
base_url = "{url}"
model = "fixture-model"
inference_level = "none"
timeout_seconds = 10
cache_revision = "fixture-v1"
[ai.protocol]
reasoning_format = "reasoning_effort"
supported_inference_levels = ["none"]
json_schema = false
max_output_tokens = 1024
max_request_bytes = 1024
max_response_bytes = 1024
max_attempts = 1
"#
    ))
    .unwrap()
    .ai
}

fn config(helper: &Helper, start: &str, stop: &str) -> RuntimeConfig {
    RuntimeConfig {
        backend: RuntimeBackend::Vllm,
        start_command: vec![helper.executable(start).to_str().unwrap().to_owned()],
        stop_command: vec![
            helper.executable(stop).to_str().unwrap().to_owned(),
            "{pid}".into(),
            "{session}".into(),
        ],
        command_timeout_seconds: 1,
        startup_timeout_seconds: 1,
    }
}

fn assert_cleaned(helper: &Helper, stop_descendant: bool) {
    let session = wait_for(&helper.marker("started"));
    assert!(!std::path::Path::new(&session).exists());
    let stopped = wait_for(&helper.marker("stopped"));
    let values: Vec<_> = stopped.lines().collect();
    assert_eq!(values[1], session);
    assert_eq!(values[0], wait_for(&helper.marker("runtime-leader")));
    for marker in ["runtime-leader", "runtime-descendant"] {
        assert_gone(wait_for(&helper.marker(marker)).parse().unwrap());
    }
    if stop_descendant {
        assert_gone(wait_for(&helper.marker("stop-descendant")).parse().unwrap());
    }
}

struct Unrelated(Child);

impl Drop for Unrelated {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn mock_disconnected_and_incomplete_peers_do_not_count_as_requests() {
    let mock = Mock::new(|| (200, Duration::ZERO));
    let address = mock.url.strip_prefix("http://").unwrap();
    drop(TcpStream::connect(address).unwrap());
    let mut partial = TcpStream::connect(address).unwrap();
    partial.write_all(b"GET /unfinished").unwrap();
    partial.shutdown(Shutdown::Both).unwrap();
    drop(partial);

    let mut complete = TcpStream::connect(address).unwrap();
    complete
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    complete
        .write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .unwrap();
    let mut response = String::new();
    complete.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200 "));
    assert_eq!(mock.requests.load(Ordering::Relaxed), 1);
}

fn panicked_mock_worker() -> Mock {
    let entered = Arc::new(AtomicBool::new(false));
    let observed = entered.clone();
    let mock = Mock::new(move || {
        observed.store(true, Ordering::Relaxed);
        panic!("deliberate mock worker failure");
    });
    let mut stream = TcpStream::connect(mock.url.strip_prefix("http://").unwrap()).unwrap();
    stream
        .write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !entered.load(Ordering::Relaxed) {
        assert!(Instant::now() < deadline, "mock did not receive request");
        thread::sleep(Duration::from_millis(1));
    }
    mock
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-text panic")
}

#[test]
fn mock_worker_failure_still_fails_an_otherwise_successful_test() {
    let failure = std::panic::catch_unwind(|| drop(panicked_mock_worker())).unwrap_err();
    assert!(panic_text(failure.as_ref()).contains("deliberate mock worker failure"));
}

#[test]
fn mock_worker_failure_during_unwind_preserves_the_primary_failure() {
    let mock = panicked_mock_worker();
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _mock = mock;
        panic!("primary runtime assertion");
    }))
    .unwrap_err();
    assert_eq!(panic_text(failure.as_ref()), "primary runtime assertion");
}

#[test]
fn mock_invalid_request_failure_is_not_ignored() {
    let mock = Mock::new(|| unreachable!("oversized headers must be refused"));
    let mut stream = TcpStream::connect(mock.url.strip_prefix("http://").unwrap()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    stream.write_all(&vec![b'x'; 16384]).unwrap();
    match stream.read(&mut [0]) {
        Ok(count) => assert_eq!(count, 0, "invalid request received a response"),
        Err(error) => assert!(peer_disconnected(&error), "{error}"),
    }
    let failure =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(mock))).unwrap_err();
    assert!(panic_text(failure.as_ref()).contains("mock request headers exceed fixture limit"));
}

#[test]
fn windows_absent_management_and_healthy_existing_endpoints_are_not_owned() {
    let helper = Helper::new();
    let config = config(&helper, "runtime-start", "runtime-stop");
    let mock = Mock::new(|| (200, Duration::ZERO));
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let cancelled = AtomicBool::new(false);
    let mut unmanaged = RuntimeManager::new(None, &ai).unwrap();
    unmanaged.ensure_ready(&client, &cancelled).unwrap();
    assert!(unmanaged.finish(Ok(())).cleanup_error.is_none());
    assert_eq!(mock.requests.load(Ordering::Relaxed), 0);
    let mut managed = RuntimeManager::new(Some(&config), &ai).unwrap();
    managed.ensure_ready(&client, &cancelled).unwrap();
    managed.ensure_ready(&client, &cancelled).unwrap();
    assert!(managed.finish(Ok(())).cleanup_error.is_none());
    assert_eq!(mock.requests.load(Ordering::Relaxed), 1);
    assert!(!helper.marker("started").exists());
    assert!(!helper.marker("stopped").exists());
}

#[test]
fn windows_normal_shutdown_reaps_foreground_and_descendants() {
    let helper = Helper::new();
    let config = config(&helper, "runtime-start", "runtime-stop");
    let marker = helper.marker("started");
    let mock = Mock::new(move || (if marker.exists() { 200 } else { 503 }, Duration::ZERO));
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&config), &ai).unwrap();
    runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap();
    let completion = runtime.finish(Ok(42));
    assert_eq!(completion.result.unwrap(), 42);
    assert!(completion.cleanup_error.is_none());
    assert_cleaned(&helper, false);
}

#[test]
fn windows_exited_start_wrapper_retains_ownership_of_descendants() {
    let helper = Helper::new();
    let config = config(&helper, "runtime-exit", "runtime-stop");
    let mock = Mock::new(|| (503, Duration::ZERO));
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&config), &ai).unwrap();
    let result = runtime.ensure_ready(&client, &AtomicBool::new(false));
    assert!(result.is_err());
    let completion = runtime.finish(result);
    assert!(completion.result.is_err());
    assert!(completion.cleanup_error.is_none());
    assert_cleaned(&helper, false);
}

#[test]
fn windows_exited_foreground_without_descendants_does_not_block_cleanup() {
    let helper = Helper::new();
    let config = config(&helper, "runtime-exit-alone", "runtime-stop");
    let mock = Mock::new(|| (503, Duration::ZERO));
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&config), &ai).unwrap();
    let before = Instant::now();
    let result = runtime.ensure_ready(&client, &AtomicBool::new(false));
    assert!(result.is_err());
    let completion = runtime.finish(result);
    assert!(completion.result.is_err());
    assert!(completion.cleanup_error.is_none());
    assert!(before.elapsed() < Duration::from_secs(3));
    assert_gone(wait_for(&helper.marker("runtime-leader")).parse().unwrap());
    assert!(!std::path::Path::new(&wait_for(&helper.marker("started"))).exists());
}

#[test]
fn windows_inflight_http_cancellation_bounds_hung_stop_and_keeps_unrelated_process() {
    let helper = Helper::new();
    let mut config = config(&helper, "runtime-start", "runtime-stop-hang");
    config.command_timeout_seconds = 120;
    config.startup_timeout_seconds = 120;
    let mut unrelated = Unrelated(
        Command::new(helper.executable("unrelated"))
            .spawn()
            .unwrap(),
    );
    let marker = helper.marker("started");
    let inflight = Arc::new(AtomicBool::new(false));
    let observed = inflight.clone();
    let mock = Mock::new(move || {
        if marker.exists() {
            observed.store(true, Ordering::Relaxed);
            (200, Duration::from_secs(3))
        } else {
            (503, Duration::ZERO)
        }
    });
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&config), &ai).unwrap();
    let cancelled = AtomicBool::new(false);
    let elapsed = Mutex::new(None);
    thread::scope(|scope| {
        scope.spawn(|| {
            let deadline = Instant::now() + Duration::from_secs(10);
            while !inflight.load(Ordering::Relaxed) {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(10));
            }
            *elapsed.lock().unwrap() = Some(Instant::now());
            cancelled.store(true, Ordering::Relaxed);
        });
        let result = runtime.ensure_ready(&client, &cancelled);
        assert!(
            result
                .as_ref()
                .unwrap_err()
                .downcast_ref::<Interrupted>()
                .is_some()
        );
        let completion = runtime.finish_cancellable(result, &cancelled);
        assert!(
            completion
                .result
                .unwrap_err()
                .downcast_ref::<Interrupted>()
                .is_some()
        );
        assert!(
            completion
                .cleanup_error
                .unwrap()
                .downcast_ref::<Interrupted>()
                .is_some()
        );
    });
    assert!(elapsed.lock().unwrap().unwrap().elapsed() < Duration::from_secs(2));
    assert_cleaned(&helper, true);
    assert!(unrelated.0.try_wait().unwrap().is_none());
}

#[test]
fn windows_shutdown_failure_preserves_the_primary_error() {
    let helper = Helper::new();
    let config = config(&helper, "runtime-start", "runtime-stop-hang");
    let marker = helper.marker("started");
    let mock = Mock::new(move || (if marker.exists() { 200 } else { 503 }, Duration::ZERO));
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&config), &ai).unwrap();
    runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap();
    let completion = runtime.finish(Err::<(), _>(anyhow!("inference body failed")));
    assert_eq!(
        completion.result.unwrap_err().to_string(),
        "inference body failed"
    );
    assert!(completion.cleanup_error.is_some());
    assert_cleaned(&helper, true);
}
