//! Slow live HTTP tests: cancellation closes the owned socket, without waiting
//! for an otherwise valid one-hour inference timeout or launching a retry.
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use game_screen_pick::ai::{JsonRequest, OpenAiClient};
use game_screen_pick::config::{AiConfig, Config};
use game_screen_pick::media::Interrupted;
use serde_json::json;

#[derive(Clone, Copy)]
enum Stage {
    Headers,
    Body,
    Error,
}

struct StalledServer {
    url: String,
    established: Option<mpsc::Receiver<()>>,
    closed: mpsc::Receiver<()>,
    worker: Option<JoinHandle<()>>,
}

impl StalledServer {
    fn new(stage: Stage) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (establish, established) = mpsc::channel();
        let (close, closed) = mpsc::channel();
        let worker = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "HTTP request never arrived");
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("mock accept: {error}"),
                }
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            read_request(&mut stream);
            match stage {
                Stage::Headers => {}
                Stage::Body => {
                    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000\r\n\r\n{\"unfinished\":").unwrap();
                    stream.flush().unwrap();
                }
                Stage::Error => {
                    stream
                        .write_all(b"HTTP/1.1 503 Unavailable\r\nContent-Length: 1000\r\n\r\n")
                        .unwrap();
                    stream.flush().unwrap();
                }
            }
            let _ = establish.send(());
            let mut byte = [0];
            match stream.read(&mut byte) {
                Ok(0) => {}
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::BrokenPipe
                    ) => {}
                result => panic!("cancelled HTTP socket remained active: {result:?}"),
            }
            close.send(()).unwrap();
            assert!(
                matches!(listener.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock),
                "cancelled request was retried"
            );
        });
        Self {
            url,
            established: Some(established),
            closed,
            worker: Some(worker),
        }
    }

    fn cancel_when_stalled(&mut self, cancelled: Arc<AtomicBool>) -> JoinHandle<()> {
        let established = self.established.take().unwrap();
        thread::spawn(move || {
            established.recv_timeout(Duration::from_secs(5)).unwrap();
            thread::sleep(Duration::from_millis(100));
            cancelled.store(true, Ordering::Relaxed);
        })
    }

    fn verify_closed(&self) {
        self.closed
            .recv_timeout(Duration::from_secs(1))
            .expect("socket must close promptly after cancellation/response drop");
    }
}

impl Drop for StalledServer {
    fn drop(&mut self) {
        let result = self.worker.take().unwrap().join();
        if !thread::panicking() {
            result.unwrap();
        }
    }
}

fn read_request(stream: &mut TcpStream) {
    let mut bytes = Vec::new();
    let mut buffer = [0; 4096];
    let header_end = loop {
        let count = stream.read(&mut buffer).unwrap();
        assert!(count > 0 && bytes.len() < 1024 * 1024);
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
        .unwrap_or(0);
    while bytes.len() - header_end < length {
        let count = stream.read(&mut buffer).unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&buffer[..count]);
    }
}

fn config(server: &StalledServer) -> AiConfig {
    Config::parse(&format!(
        r#"
[selection]
method = "sampled_frames"
[ai]
backend = "vllm"
base_url = "{}/v1"
model = "slow-model"
inference_level = "none"
timeout_seconds = 3600
cache_revision = "cancellation-test-v1"
api_key = "LOCAL_CANCELLATION_TEST_SECRET"
[ai.protocol]
reasoning_format = "reasoning_effort"
supported_inference_levels = ["none"]
json_schema = true
max_output_tokens = 512
max_request_bytes = 1048576
max_response_bytes = 16384
max_attempts = 3
"#,
        server.url
    ))
    .unwrap()
    .ai
}

fn request() -> JsonRequest {
    JsonRequest {
        prompt: "Return the answer as JSON".to_owned(),
        schema: json!({"type":"object", "required":["answer"],
            "properties":{"answer":{"type":"string"}}}),
        content: Vec::new(),
    }
}

#[test]
fn inference_headers_and_body_cancel_promptly_and_close_the_socket() {
    for stage in [Stage::Headers, Stage::Body] {
        let mut server = StalledServer::new(stage);
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancellation = server.cancel_when_stalled(cancelled.clone());
        let client = OpenAiClient::new(&config(&server)).unwrap();
        let started = Instant::now();
        let error = client
            .generate_cancellable(&request(), &cancelled, |value| Ok(value.clone()))
            .unwrap_err();
        assert!(error.downcast_ref::<Interrupted>().is_some(), "{error:#}");
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(!format!("{error:#}").contains("LOCAL_CANCELLATION_TEST_SECRET"));
        cancellation.join().unwrap();
        server.verify_closed();
        // A already-cancelled caller stops before connecting or retrying.
        assert!(
            client
                .generate_cancellable(&request(), &cancelled, |value| Ok(value.clone()))
                .unwrap_err()
                .downcast_ref::<Interrupted>()
                .is_some()
        );
    }
}

#[test]
fn readiness_headers_and_body_are_cancellable_under_a_long_deadline() {
    for stage in [Stage::Headers, Stage::Body] {
        let mut server = StalledServer::new(stage);
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancellation = server.cancel_when_stalled(cancelled.clone());
        let client = OpenAiClient::new(&config(&server)).unwrap();
        let started = Instant::now();
        let error = client
            .is_ready_cancellable(started + Duration::from_secs(3600), &cancelled)
            .unwrap_err();
        assert!(error.downcast_ref::<Interrupted>().is_some(), "{error:#}");
        assert!(started.elapsed() < Duration::from_secs(1));
        cancellation.join().unwrap();
        server.verify_closed();
    }
}

#[test]
fn rejected_status_does_not_wait_for_a_slow_body_and_closes_the_socket() {
    let server = StalledServer::new(Stage::Error);
    let client = OpenAiClient::new(&config(&server)).unwrap();
    let started = Instant::now();
    let error = client
        .generate(&request(), |value| Ok(value.clone()))
        .unwrap_err();
    assert!(error.to_string().contains("503"));
    assert!(started.elapsed() < Duration::from_secs(1));
    server.verify_closed();
}
