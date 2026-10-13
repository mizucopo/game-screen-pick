//! Mock/script lifecycle contracts only; these do not prove live model quality.
#![cfg(unix)]

use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use game_screen_pick::ai::OpenAiClient;
use game_screen_pick::config::{AiConfig, Backend, Config};
use game_screen_pick::runtime::{RuntimeBackend, RuntimeConfig, RuntimeManager};
use serde_json::json;
use tempfile::TempDir;

const SECRET: &str = "RUNTIME_AUTH_NEVER_DISCLOSE";

struct Reply {
    status: u16,
    body: String,
    location: Option<String>,
    delay: Duration,
}

impl Reply {
    fn models(model: &str) -> Self {
        Self {
            status: 200,
            body: json!({"object":"list","data":[{"id":model}]}).to_string(),
            location: None,
            delay: Duration::ZERO,
        }
    }

    fn unavailable() -> Self {
        Self {
            status: 503,
            ..Self::models("unused")
        }
    }
}

struct Mock {
    url: String,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl Mock {
    fn new(mut handler: impl FnMut(&str) -> Reply + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}/custom", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let worker = thread::spawn(move || {
            while !stopping.load(Ordering::Relaxed) {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => panic!("mock failed: {error}"),
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let request = read_request(&mut stream);
                recorded.lock().unwrap().push(request.clone());
                let reply = handler(&request);
                thread::sleep(reply.delay);
                let mut response = format!(
                    "HTTP/1.1 {} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
                    reply.status,
                    reply.body.len()
                );
                if let Some(location) = reply.location {
                    response.push_str(&format!("Location: {location}\r\n"));
                }
                response.push_str("\r\n");
                response.push_str(&reply.body);
                let _ = stream.write_all(response.as_bytes());
            }
        });
        Self {
            url,
            requests,
            stop,
            worker: Some(worker),
        }
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for Mock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let result = self.worker.take().unwrap().join();
        if !thread::panicking() {
            result.unwrap();
        }
    }
}

fn read_request(stream: &mut TcpStream) -> String {
    let mut bytes = Vec::new();
    let mut byte = [0; 1];
    while !bytes.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).unwrap() == 0 {
            break;
        }
        bytes.push(byte[0]);
        assert!(bytes.len() < 16384);
    }
    String::from_utf8(bytes).unwrap()
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
timeout_seconds = 3
cache_revision = "fixture-v1"
api_key = "{SECRET}"
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

fn quote(path: &Path) -> String {
    format!("'{}'", path.to_str().unwrap().replace('\'', "'\\''"))
}

fn script(directory: &Path, name: &str, body: &str) -> String {
    let path = directory.join(name);
    fs::write(&path, format!("#!/bin/sh\nset -eu\n{body}\n")).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    path.to_str().unwrap().to_owned()
}

struct Scripts {
    _directory: TempDir,
    marker: PathBuf,
    stopped: PathBuf,
    config: RuntimeConfig,
}

impl Scripts {
    fn new(start_tail: &str, stop_tail: &str) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("started");
        let stopped = directory.path().join("stopped");
        let start = script(
            directory.path(),
            "start",
            &format!(
                "printf '%s\\n' \"$GAME_SCREEN_PICK_RUNTIME_SESSION\" > {}\n{start_tail}",
                quote(&marker)
            ),
        );
        let stop = script(
            directory.path(),
            "stop",
            &format!(
                "test \"$2\" = \"$GAME_SCREEN_PICK_RUNTIME_SESSION\"\nprintf '%s\\n%s\\n' \"$1\" \"$2\" > {}\n{stop_tail}",
                quote(&stopped)
            ),
        );
        let config = RuntimeConfig {
            backend: RuntimeBackend::Vllm,
            start_command: vec![start],
            stop_command: vec![stop, "{pid}".into(), "{session}".into()],
            command_timeout_seconds: 1,
            startup_timeout_seconds: 1,
        };
        Self {
            _directory: directory,
            marker,
            stopped,
            config,
        }
    }

    fn normal() -> Self {
        Self::new("while :; do sleep 0.1; done", "kill -TERM \"$1\"")
    }

    fn mock(&self) -> Mock {
        let marker = self.marker.clone();
        Mock::new(move |_| {
            if marker.exists() {
                Reply::models("fixture-model")
            } else {
                Reply::unavailable()
            }
        })
    }

    fn assert_stopped(&self) {
        let stopped = fs::read_to_string(&self.stopped).unwrap();
        let mut lines = stopped.lines();
        let pid: libc::pid_t = lines.next().unwrap().parse().unwrap();
        let token = lines.next().unwrap();
        assert_eq!(token, fs::read_to_string(&self.marker).unwrap().trim());
        assert!(
            !Path::new(token).exists(),
            "private session must be removed"
        );
        assert_eq!(
            unsafe { libc::kill(pid, 0) },
            -1,
            "own child must be reaped"
        );
    }
}

#[test]
fn absent_or_commandless_config_does_zero_management_or_readiness() {
    let mock = Mock::new(|_| panic!("unmanaged runtime must not probe"));
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let config: RuntimeConfig = toml::from_str("backend = 'vllm'").unwrap();
    for config in [None, Some(&config)] {
        let mut runtime = RuntimeManager::new(config, &ai).unwrap();
        runtime
            .ensure_ready(&client, &AtomicBool::new(false))
            .unwrap();
        let completion = runtime.finish(Ok(42));
        assert_eq!(completion.result.unwrap(), 42);
        assert!(completion.cleanup_error.is_none());
    }
    assert!(mock.requests().is_empty());
}

#[test]
fn config_rejects_backend_mismatch_unpaired_and_unscoped_commands() {
    let scripts = Scripts::normal();
    for variant in 0..6 {
        let mut config = scripts.config.clone();
        match variant {
            0 => config.backend = RuntimeBackend::Strata,
            1 => config.stop_command.clear(),
            2 => config.start_command.clear(),
            3 => config.stop_command = vec!["pkill".into(), "vllm".into()],
            4 => config.stop_command[1] = "--pid={pid}".into(),
            5 => config.startup_timeout_seconds = 0,
            _ => unreachable!(),
        }
        assert!(config.validate(Backend::Vllm).is_err());
    }
    assert!(toml::from_str::<RuntimeConfig>("backend = 'vllm'\nunknown = 'secret'").is_err());
    let mut config = scripts.config;
    config.start_command.push(SECRET.into());
    assert!(!format!("{config:?}").contains(SECRET));
}

#[test]
fn healthy_existing_endpoint_is_never_owned_or_stopped_and_alias_is_exact() {
    let scripts = Scripts::normal();
    let mock = Mock::new(|_| Reply::models("canonical-model"));
    let mut ai = ai_config(&mock.url);
    ai.protocol.as_mut().unwrap().response_model = Some("canonical-model".into());
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
    runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap();
    runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap();
    let completion = runtime.finish(Err::<(), _>(anyhow!("inference body failed")));
    assert_eq!(
        completion.result.unwrap_err().to_string(),
        "inference body failed"
    );
    assert!(completion.cleanup_error.is_none());
    assert!(!scripts.marker.exists());
    assert!(!scripts.stopped.exists());
    let requests = mock.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].starts_with("GET /custom/models HTTP/1.1"));
    assert!(
        requests[0]
            .to_lowercase()
            .contains(&format!("authorization: bearer {}", SECRET.to_lowercase()))
    );
}

#[test]
fn live_wrong_model_auth_failure_and_invalid_response_do_not_launch_commands() {
    for variant in 0..5 {
        let scripts = Scripts::normal();
        let mock = Mock::new(move |_| match variant {
            0 => Reply::models("fixture-model-other"),
            1 => Reply {
                status: 401,
                body: SECRET.into(),
                ..Reply::models("fixture-model")
            },
            2 => Reply {
                body: "invalid JSON".into(),
                ..Reply::models("fixture-model")
            },
            3 => Reply {
                body: "x".repeat(2048),
                ..Reply::models("fixture-model")
            },
            4 => Reply::models(SECRET),
            _ => unreachable!(),
        });
        let ai = ai_config(&mock.url);
        let client = OpenAiClient::new(&ai).unwrap();
        let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
        let error = runtime
            .ensure_ready(&client, &AtomicBool::new(false))
            .unwrap_err();
        assert!(!format!("{error:#}\n{error:?}").contains(SECRET));
        let completion = runtime.finish(Err::<(), _>(error));
        assert!(completion.result.is_err());
        assert!(completion.cleanup_error.is_none());
        assert!(!scripts.marker.exists());
        assert!(!scripts.stopped.exists());
    }
}

#[test]
fn starts_only_on_miss_and_stops_the_exact_owned_pid_and_session() {
    let scripts = Scripts::normal();
    let mock = scripts.mock();
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
    assert!(!scripts.marker.exists()); // construction represents a warm cache run
    runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap();
    let requests = mock.requests().len();
    runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(mock.requests().len(), requests);
    let completion = runtime.finish(Ok(42));
    assert_eq!(completion.result.unwrap(), 42);
    assert!(completion.cleanup_error.is_none());
    scripts.assert_stopped();
    assert!(runtime.finish(Ok(())).cleanup_error.is_none());
}

#[test]
fn partial_start_is_cleaned_and_never_reported_ready() {
    let scripts = Scripts::new("exit 7", "kill -TERM \"$1\"");
    let mock = Mock::new(|_| Reply::unavailable());
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
    let error = runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap_err();
    assert!(error.to_string().contains("foreground"));
    let completion = runtime.finish(Err::<(), _>(error));
    assert!(completion.cleanup_error.is_none());
    scripts.assert_stopped();
}

#[test]
fn startup_timeout_is_bounded_and_cleans_the_partial_attempt() {
    let scripts = Scripts::normal();
    let mock = Mock::new(|_| Reply::unavailable());
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
    let started = Instant::now();
    let error = runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap_err();
    assert!(error.to_string().contains("deadline"));
    assert!(started.elapsed() < Duration::from_secs(3));
    let completion = runtime.finish(Err::<(), _>(error));
    assert!(completion.cleanup_error.is_none());
    scripts.assert_stopped();
}

#[test]
fn cancellation_before_start_is_zero_operations_and_during_start_cleans() {
    let scripts = Scripts::normal();
    let mock = Mock::new(|_| Reply::unavailable());
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
    assert!(
        runtime
            .ensure_ready(&client, &AtomicBool::new(true))
            .is_err()
    );
    assert!(mock.requests().is_empty());
    assert!(!scripts.marker.exists());
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancellation = cancelled.clone();
    let marker = scripts.marker.clone();
    let worker = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(3);
        while !marker.exists() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        cancellation.store(true, Ordering::Relaxed);
    });
    let error = runtime.ensure_ready(&client, &cancelled).unwrap_err();
    worker.join().unwrap();
    assert!(error.is::<game_screen_pick::media::Interrupted>());
    let completion = runtime.finish(Err::<(), _>(error));
    assert!(completion.cleanup_error.is_none());
    scripts.assert_stopped();
}

#[test]
fn cancellation_during_existing_health_check_does_not_continue_to_inference() {
    let scripts = Scripts::normal();
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancellation = cancelled.clone();
    let mock = Mock::new(move |_| {
        cancellation.store(true, Ordering::Relaxed);
        Reply::models("fixture-model")
    });
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
    let error = runtime.ensure_ready(&client, &cancelled).unwrap_err();
    assert!(error.is::<game_screen_pick::media::Interrupted>());
    assert!(runtime.finish(Err::<(), _>(error)).cleanup_error.is_none());
    assert!(!scripts.marker.exists());
    assert!(!scripts.stopped.exists());
}

#[test]
fn failed_and_timed_out_stop_preserve_body_error_and_reap_owned_children() {
    for stop in [format!("echo '{SECRET}' >&2\nexit 8"), "sleep 10".into()] {
        let scripts = Scripts::new("while :; do sleep 0.1; done", &stop);
        let mock = scripts.mock();
        let ai = ai_config(&mock.url);
        let client = OpenAiClient::new(&ai).unwrap();
        let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
        runtime
            .ensure_ready(&client, &AtomicBool::new(false))
            .unwrap();
        let started = Instant::now();
        let completion = runtime.finish(Err::<(), _>(anyhow!("primary inference error")));
        assert_eq!(
            completion.result.unwrap_err().to_string(),
            "primary inference error"
        );
        let cleanup = completion.cleanup_error.unwrap();
        assert!(!format!("{cleanup:#}\n{cleanup:?}").contains(SECRET));
        assert!(started.elapsed() < Duration::from_secs(3));
        scripts.assert_stopped();
    }
}

fn recorded_pid(path: &Path) -> libc::pid_t {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if let Ok(text) = fs::read_to_string(path)
            && let Ok(pid) = text.trim().parse()
        {
            return pid;
        }
        assert!(Instant::now() < deadline, "command must record its PID");
        thread::sleep(Duration::from_millis(5));
    }
}

fn assert_process_group_gone(pid: libc::pid_t) {
    assert_eq!(
        unsafe { libc::kill(pid, 0) },
        -1,
        "owned process must be reaped"
    );
    assert_eq!(
        unsafe { libc::kill(-pid, 0) },
        -1,
        "owned process group must be gone"
    );
}

#[test]
fn cancelled_cleanup_bounds_a_hung_stop_with_300_second_timeout_and_reaps_both_groups() {
    for already_cancelled in [true, false] {
        let mut scripts = Scripts::normal();
        scripts.config.command_timeout_seconds = 300;
        scripts.config.startup_timeout_seconds = 3;
        let stop_pid = scripts._directory.path().join("stop-pid");
        scripts.config.start_command[0] = script(
            scripts._directory.path(),
            "foreground-start",
            &format!(
                "printf '%s\\n' \"$GAME_SCREEN_PICK_RUNTIME_SESSION\" > {}\nexec sleep 60",
                quote(&scripts.marker)
            ),
        );
        scripts.config.stop_command[0] = script(
            scripts._directory.path(),
            "hung-stop",
            &format!(
                "test \"$2\" = \"$GAME_SCREEN_PICK_RUNTIME_SESSION\"\nprintf '%s\\n%s\\n' \"$1\" \"$2\" > {}\nprintf '%s\\n' \"$$\" > {}\nexec sleep 60",
                quote(&scripts.stopped),
                quote(&stop_pid)
            ),
        );
        let mock = scripts.mock();
        let ai = ai_config(&mock.url);
        let client = OpenAiClient::new(&ai).unwrap();
        let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
        runtime
            .ensure_ready(&client, &AtomicBool::new(false))
            .unwrap();
        let cancelled = Arc::new(AtomicBool::new(already_cancelled));
        let worker = if already_cancelled {
            None
        } else {
            let cancellation = cancelled.clone();
            let stop_pid = stop_pid.clone();
            Some(thread::spawn(move || {
                recorded_pid(&stop_pid);
                thread::sleep(Duration::from_millis(100));
                cancellation.store(true, Ordering::Relaxed);
            }))
        };
        let primary = if already_cancelled {
            anyhow::Error::from(game_screen_pick::media::Interrupted)
        } else {
            anyhow!("primary inference failure")
        };
        let started = Instant::now();
        let completion = runtime.finish_cancellable(Err::<(), _>(primary), &cancelled);
        assert!(started.elapsed() < Duration::from_secs(2));
        if let Some(worker) = worker {
            worker.join().unwrap();
        }
        let main_error = completion.result.unwrap_err();
        if already_cancelled {
            assert!(main_error.is::<game_screen_pick::media::Interrupted>());
        } else {
            assert_eq!(main_error.to_string(), "primary inference failure");
        }
        let cleanup = completion.cleanup_error.unwrap();
        assert!(cleanup.is::<game_screen_pick::media::Interrupted>());
        assert!(cleanup.to_string().contains("cancellation grace"));
        scripts.assert_stopped();
        let runtime_pid: libc::pid_t = fs::read_to_string(&scripts.stopped)
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert_process_group_gone(runtime_pid);
        assert_process_group_gone(recorded_pid(&stop_pid));
        assert!(
            runtime
                .finish_cancellable(Ok(()), &cancelled)
                .cleanup_error
                .is_none()
        );
    }
}

#[test]
fn already_cancelled_cleanup_attempts_normal_stop_and_preserves_stop_marker() {
    let mut scripts = Scripts::normal();
    scripts.config.command_timeout_seconds = 300;
    let mock = scripts.mock();
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
    runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap();
    let completion = runtime.finish_cancellable(
        Err::<(), _>(game_screen_pick::media::Interrupted.into()),
        &AtomicBool::new(true),
    );
    assert!(
        completion
            .result
            .unwrap_err()
            .is::<game_screen_pick::media::Interrupted>()
    );
    assert!(completion.cleanup_error.is_none());
    scripts.assert_stopped();
}

#[test]
fn cancellation_after_stop_command_exit_bounds_wait_for_its_runtime() {
    let mut scripts = Scripts::new("exec sleep 60", "exit 0");
    scripts.config.command_timeout_seconds = 300;
    let mock = scripts.mock();
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
    runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap();
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancellation = cancelled.clone();
    let stopped = scripts.stopped.clone();
    let worker = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !stopped.exists() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        thread::sleep(Duration::from_millis(100));
        cancellation.store(true, Ordering::Relaxed);
    });
    let started = Instant::now();
    let completion =
        runtime.finish_cancellable(Err::<(), _>(anyhow!("original failure")), &cancelled);
    worker.join().unwrap();
    assert!(started.elapsed() < Duration::from_secs(2));
    assert_eq!(
        completion.result.unwrap_err().to_string(),
        "original failure"
    );
    assert!(
        completion
            .cleanup_error
            .unwrap()
            .is::<game_screen_pick::media::Interrupted>()
    );
    scripts.assert_stopped();
}

#[test]
fn uncancelled_finish_keeps_normal_stop_timeout_beyond_cancellation_grace() {
    let mut scripts = Scripts::new("exec sleep 60", "sleep 0.7\nkill -TERM \"$1\"");
    scripts.config.command_timeout_seconds = 3;
    let mock = scripts.mock();
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
    runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap();
    let started = Instant::now();
    let completion = runtime.finish_cancellable(Ok(42), &AtomicBool::new(false));
    assert_eq!(completion.result.unwrap(), 42);
    assert!(completion.cleanup_error.is_none());
    assert!(started.elapsed() >= Duration::from_millis(700));
    scripts.assert_stopped();
}

#[test]
fn drop_cleans_started_runtime_on_early_return() {
    let scripts = Scripts::normal();
    let mock = scripts.mock();
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    {
        let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
        runtime
            .ensure_ready(&client, &AtomicBool::new(false))
            .unwrap();
    }
    scripts.assert_stopped();
}

#[test]
fn readiness_redirects_stay_authenticated_only_on_same_origin() {
    let same = Mock::new(|request| {
        if request.starts_with("GET /custom/models ") {
            Reply {
                status: 302,
                location: Some("/ready-models".into()),
                ..Reply::models("fixture-model")
            }
        } else {
            Reply::models("fixture-model")
        }
    });
    let client = OpenAiClient::new(&ai_config(&same.url)).unwrap();
    assert!(
        client
            .is_ready(Instant::now() + Duration::from_secs(2))
            .unwrap()
    );
    assert_eq!(same.requests().len(), 2);
    assert!(
        same.requests()
            .iter()
            .all(|request| request.contains(SECRET))
    );
    let other = Mock::new(|_| panic!("cross-origin must not receive credentials"));
    let target = other.url.clone();
    let cross = Mock::new(move |_| Reply {
        status: 307,
        location: Some(target.clone()),
        ..Reply::models("fixture-model")
    });
    let client = OpenAiClient::new(&ai_config(&cross.url)).unwrap();
    let error = client
        .is_ready(Instant::now() + Duration::from_secs(2))
        .unwrap_err();
    assert!(error.to_string().contains("origin"));
    assert!(other.requests().is_empty());
    assert!(!format!("{error:#}\n{error:?}").contains(SECRET));
}

#[test]
fn refused_connection_can_start_but_readiness_timeout_is_uncertain_and_never_starts() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    drop(listener);
    let client = OpenAiClient::new(&ai_config(&url)).unwrap();
    assert!(
        !client
            .is_ready(Instant::now() + Duration::from_secs(1))
            .unwrap()
    );
    let scripts = Scripts::normal();
    let mock = Mock::new(|_| Reply {
        delay: Duration::from_millis(1200),
        ..Reply::models("fixture-model")
    });
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
    let error = runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap_err();
    assert!(error.to_string().contains("deadline"));
    assert!(!scripts.marker.exists());
    assert!(!scripts.stopped.exists());
}

#[test]
fn owned_model_loading_can_stall_beyond_initial_probe_timeout_and_become_ready() {
    let mut scripts = Scripts::normal();
    scripts.config.startup_timeout_seconds = 4;
    let marker = scripts.marker.clone();
    let mut first_owned_probe = true;
    let mock = Mock::new(move |_| {
        if !marker.exists() {
            return Reply::unavailable();
        }
        let delay = if first_owned_probe {
            first_owned_probe = false;
            Duration::from_millis(1400)
        } else {
            Duration::ZERO
        };
        Reply {
            delay,
            ..Reply::models("fixture-model")
        }
    });
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
    let started = Instant::now();
    runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap();
    assert!(started.elapsed() >= Duration::from_millis(1400));
    assert!(started.elapsed() < Duration::from_secs(3));
    assert!(
        !scripts.stopped.exists(),
        "loading must not tear down its session"
    );
    let completion = runtime.finish(Ok(()));
    assert!(completion.cleanup_error.is_none());
    scripts.assert_stopped();
}

#[test]
fn owned_loading_retries_typed_ai_http_timeout_within_startup_budget() {
    let mut scripts = Scripts::normal();
    scripts.config.startup_timeout_seconds = 4;
    let marker = scripts.marker.clone();
    let owned_probes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let probes = owned_probes.clone();
    let mock = Mock::new(move |_| {
        if !marker.exists() {
            return Reply::unavailable();
        }
        let first = probes.fetch_add(1, Ordering::Relaxed) == 0;
        Reply {
            delay: if first {
                Duration::from_millis(1300)
            } else {
                Duration::ZERO
            },
            ..Reply::models("fixture-model")
        }
    });
    let mut ai = ai_config(&mock.url);
    ai.timeout_seconds = 1;
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
    runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(owned_probes.load(Ordering::Relaxed), 2);
    assert!(!scripts.stopped.exists());
    let completion = runtime.finish(Ok(()));
    assert!(completion.cleanup_error.is_none());
    scripts.assert_stopped();
}

#[test]
fn owned_readiness_rejects_non_timeout_failures_immediately_and_cleans() {
    for variant in 0..3 {
        let mut scripts = Scripts::normal();
        scripts.config.startup_timeout_seconds = 4;
        let marker = scripts.marker.clone();
        let owned_probes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let probes = owned_probes.clone();
        let mock = Mock::new(move |_| {
            if !marker.exists() {
                return Reply::unavailable();
            }
            probes.fetch_add(1, Ordering::Relaxed);
            match variant {
                0 => Reply {
                    status: 401,
                    body: SECRET.into(),
                    ..Reply::models("fixture-model")
                },
                1 => Reply {
                    body: "malformed JSON".into(),
                    ..Reply::models("fixture-model")
                },
                2 => Reply::models("wrong-model"),
                _ => unreachable!(),
            }
        });
        let ai = ai_config(&mock.url);
        let client = OpenAiClient::new(&ai).unwrap();
        let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
        let started = Instant::now();
        let error = runtime
            .ensure_ready(&client, &AtomicBool::new(false))
            .unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(owned_probes.load(Ordering::Relaxed), 1);
        assert!(!format!("{error:#}\n{error:?}").contains(SECRET));
        let completion = runtime.finish(Err::<(), _>(error));
        assert!(completion.result.is_err());
        assert!(completion.cleanup_error.is_none());
        scripts.assert_stopped();
    }
}

#[test]
fn owned_readiness_timeouts_end_at_overall_startup_deadline_and_cleanup() {
    let mut scripts = Scripts::normal();
    scripts.config.startup_timeout_seconds = 2;
    let marker = scripts.marker.clone();
    let mock = Mock::new(move |_| {
        if marker.exists() {
            Reply {
                delay: Duration::from_millis(1100),
                ..Reply::models("fixture-model")
            }
        } else {
            Reply::unavailable()
        }
    });
    let mut ai = ai_config(&mock.url);
    ai.timeout_seconds = 1;
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
    let started = Instant::now();
    let error = runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap_err();
    assert_eq!(error.to_string(), "runtime startup deadline exceeded");
    assert!(started.elapsed() >= Duration::from_secs(2));
    assert!(started.elapsed() < Duration::from_secs(3));
    let completion = runtime.finish(Err::<(), _>(error));
    assert!(completion.cleanup_error.is_none());
    scripts.assert_stopped();
}

#[test]
fn cancellation_during_owned_loading_aborts_http_wait_and_cleans_session() {
    let mut scripts = Scripts::normal();
    scripts.config.startup_timeout_seconds = 4;
    let marker = scripts.marker.clone();
    let loading = Arc::new(AtomicBool::new(false));
    let model_loading = loading.clone();
    let mock = Mock::new(move |_| {
        if marker.exists() {
            model_loading.store(true, Ordering::Relaxed);
            Reply {
                delay: Duration::from_millis(1500),
                ..Reply::models("fixture-model")
            }
        } else {
            Reply::unavailable()
        }
    });
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai).unwrap();
    let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai).unwrap();
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancellation = cancelled.clone();
    let worker = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(3);
        while !loading.load(Ordering::Relaxed) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        thread::sleep(Duration::from_millis(100));
        cancellation.store(true, Ordering::Relaxed);
    });
    let started = Instant::now();
    let error = runtime.ensure_ready(&client, &cancelled).unwrap_err();
    worker.join().unwrap();
    assert!(error.is::<game_screen_pick::media::Interrupted>());
    assert!(started.elapsed() < Duration::from_secs(1));
    let completion = runtime.finish(Err::<(), _>(error));
    assert!(completion.cleanup_error.is_none());
    scripts.assert_stopped();
}

#[test]
fn argv_is_literal_and_spawn_failure_never_runs_stop() -> Result<()> {
    let mut scripts = Scripts::normal();
    let literal_path = scripts._directory.path().join("literal");
    let unintended = scripts._directory.path().join("unintended");
    let literal = format!("$(touch {})", unintended.display());
    let start = script(
        scripts._directory.path(),
        "literal-start",
        &format!(
            "printf '%s' \"$1\" > {}\nprintf '%s\\n' \"$GAME_SCREEN_PICK_RUNTIME_SESSION\" > {}\nwhile :; do sleep 0.1; done",
            quote(&literal_path),
            quote(&scripts.marker)
        ),
    );
    scripts.config.start_command = vec![start, literal.clone()];
    let mock = scripts.mock();
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai)?;
    let mut runtime = RuntimeManager::new(Some(&scripts.config), &ai)?;
    runtime.ensure_ready(&client, &AtomicBool::new(false))?;
    assert_eq!(fs::read_to_string(literal_path)?, literal);
    assert!(!unintended.exists());
    let completion = runtime.finish(Ok(()));
    completion.result?;
    assert!(completion.cleanup_error.is_none());
    scripts.assert_stopped();
    let scripts = Scripts::normal();
    let mock = Mock::new(|_| Reply::unavailable());
    let ai = ai_config(&mock.url);
    let client = OpenAiClient::new(&ai)?;
    let mut config = scripts.config.clone();
    config.start_command = vec!["/missing-runtime-command".into(), SECRET.into()];
    let mut runtime = RuntimeManager::new(Some(&config), &ai)?;
    let error = runtime
        .ensure_ready(&client, &AtomicBool::new(false))
        .unwrap_err();
    assert!(!format!("{error:#}\n{error:?}").contains(SECRET));
    assert!(runtime.finish(Err::<(), _>(error)).cleanup_error.is_none());
    assert!(!scripts.stopped.exists());
    Ok(())
}
