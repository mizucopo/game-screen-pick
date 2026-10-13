//! Optional, lazy management of one explicitly configured foreground runtime.
//! Commands are argv vectors; the caller owns all backend-specific launch details.
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use tempfile::{Builder, TempDir};

use crate::ai::OpenAiClient;
use crate::config::{AiConfig, Backend};

const POLL_INTERVAL: Duration = Duration::from_millis(20);
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeBackend {
    Strata,
    Vllm,
}

/// No commands means use the existing endpoint without management or readiness
/// operations. Managed start must remain foreground. Stop must target only the
/// supplied `{pid}` and private `{session}`; both must be complete argv entries.
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    pub backend: RuntimeBackend,
    #[serde(default)]
    pub start_command: Vec<String>,
    #[serde(default)]
    pub stop_command: Vec<String>,
    #[serde(default = "default_command_timeout")]
    pub command_timeout_seconds: u64,
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_seconds: u64,
}

// Command argv may contain operator credentials. Never include it in diagnostics.
impl std::fmt::Debug for RuntimeConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeConfig")
            .field("backend", &self.backend)
            .field("managed", &self.is_managed())
            .field("command_timeout_seconds", &self.command_timeout_seconds)
            .field("startup_timeout_seconds", &self.startup_timeout_seconds)
            .finish()
    }
}

fn default_command_timeout() -> u64 {
    30
}

fn default_startup_timeout() -> u64 {
    300
}

impl RuntimeConfig {
    pub fn validate(&self, backend: Backend) -> Result<()> {
        ensure!(
            matches!(
                (self.backend, backend),
                (RuntimeBackend::Strata, Backend::Strata) | (RuntimeBackend::Vllm, Backend::Vllm)
            ),
            "runtime.backend must match ai.backend"
        );
        ensure!(
            (1..=300).contains(&self.command_timeout_seconds)
                && (1..=3600).contains(&self.startup_timeout_seconds),
            "runtime command/startup timeout must be from 1 to 300/3600 seconds"
        );
        ensure!(
            self.start_command.is_empty() == self.stop_command.is_empty(),
            "runtime start_command and stop_command must both be configured or omitted"
        );
        if !self.is_managed() {
            return Ok(());
        }
        for argv in [&self.start_command, &self.stop_command] {
            ensure!(
                argv.len() <= 128
                    && !argv[0].trim().is_empty()
                    && argv
                        .iter()
                        .all(|arg| arg.len() <= 65536 && !arg.contains('\0')),
                "invalid runtime command argv"
            );
            ensure!(
                argv.iter().all(|arg| {
                    (!arg.contains("{session}") || arg == "{session}")
                        && (!arg.contains("{pid}") || arg == "{pid}")
                }),
                "runtime placeholders must be complete argv entries"
            );
        }
        ensure!(
            !self.start_command.iter().any(|arg| arg == "{pid}")
                && self.stop_command.iter().any(|arg| arg == "{pid}")
                && self.stop_command.iter().any(|arg| arg == "{session}"),
            "runtime stop_command must explicitly target {{pid}} and {{session}}; start cannot use {{pid}}"
        );
        ensure!(
            self.start_command[0] != "{session}"
                && !["{pid}", "{session}"].contains(&self.stop_command[0].as_str()),
            "runtime executable cannot be a session placeholder"
        );
        Ok(())
    }

    fn is_managed(&self) -> bool {
        !self.start_command.is_empty()
    }
}

/// The primary result is retained even if shutdown fails. Callers report the
/// separate cleanup error before returning `result`; successful work does not
/// silently hide cleanup failure either.
pub struct RuntimeCompletion<T> {
    pub result: Result<T>,
    pub cleanup_error: Option<anyhow::Error>,
}

pub struct RuntimeManager {
    config: Option<RuntimeConfig>,
    owned: Option<OwnedSession>,
    ready: bool,
}

impl RuntimeManager {
    /// Validation only: no environment lookup, credentials, filesystem, commands
    /// or HTTP. Keep construction outside cache callbacks and ensure_ready inside.
    pub fn new(config: Option<&RuntimeConfig>, ai: &AiConfig) -> Result<Self> {
        if let Some(config) = config {
            config.validate(ai.backend)?;
        }
        Ok(Self {
            config: config.filter(|config| config.is_managed()).cloned(),
            owned: None,
            ready: false,
        })
    }

    /// Call lazily for inference cache misses, after output/cache preflight.
    /// Existing healthy endpoints are never marked owned and never stopped.
    pub fn ensure_ready(&mut self, client: &OpenAiClient, cancelled: &AtomicBool) -> Result<()> {
        let Some(config) = self.config.clone() else {
            return Ok(());
        };
        check_cancelled(cancelled)?;
        if self.ready {
            return Ok(());
        }
        ensure!(
            self.owned.is_none(),
            "finish the failed runtime session before retrying startup"
        );
        let deadline = Instant::now() + Duration::from_secs(config.startup_timeout_seconds);
        if client.is_ready_cancellable(deadline.min(Instant::now() + PROBE_TIMEOUT), cancelled)? {
            check_cancelled(cancelled)?;
            self.ready = true;
            return Ok(());
        }
        check_cancelled(cancelled)?;
        let directory = Builder::new()
            .prefix("game-screen-pick-runtime-")
            .tempdir()
            .map_err(|_| anyhow!("cannot create private runtime session"))?;
        let token = directory
            .path()
            .to_str()
            .ok_or_else(|| anyhow!("runtime session path is not UTF-8"))?
            .to_owned();
        let child = command(&config.start_command, &token, None)
            .spawn()
            .map_err(|_| anyhow!("cannot start runtime command (command diagnostics withheld)"))?;
        self.owned = Some(OwnedSession {
            child: OwnedChild::new(child),
            token,
            _directory: directory,
        });
        // Keep ownership on every failure. finish() reports cleanup independently
        // and Drop remains a last-resort safety net during unwinding/early returns.
        loop {
            check_cancelled(cancelled)?;
            ensure!(
                Instant::now() < deadline,
                "runtime startup deadline exceeded"
            );
            let session = self.owned.as_mut().expect("runtime session just created");
            ensure!(
                !session.child.has_exited()?,
                "runtime start command exited before readiness; start must remain foreground"
            );
            if client
                .is_ready_cancellable(deadline.min(Instant::now() + PROBE_TIMEOUT), cancelled)?
            {
                check_cancelled(cancelled)?;
                ensure!(
                    !self
                        .owned
                        .as_mut()
                        .expect("owned runtime")
                        .child
                        .has_exited()?,
                    "runtime start command exited before readiness; start must remain foreground"
                );
                self.ready = true;
                return Ok(());
            }
            thread::sleep(POLL_INTERVAL.min(deadline.saturating_duration_since(Instant::now())));
        }
    }

    pub fn finish<T>(&mut self, result: Result<T>) -> RuntimeCompletion<T> {
        RuntimeCompletion {
            result,
            cleanup_error: self.cleanup().err(),
        }
    }

    fn cleanup(&mut self) -> Result<()> {
        let Some(mut session) = self.owned.take() else {
            return Ok(());
        };
        self.ready = false;
        let config = self
            .config
            .as_ref()
            .expect("owned runtime has configuration");
        let deadline = Instant::now() + Duration::from_secs(config.command_timeout_seconds);
        let result = (|| {
            let child = command(
                &config.stop_command,
                &session.token,
                Some(session.child.child.id()),
            )
            .spawn()
            .map_err(|_| {
                anyhow!("cannot start runtime cleanup command (command diagnostics withheld)")
            })?;
            let mut stop = OwnedChild::new(child);
            loop {
                ensure!(
                    Instant::now() < deadline,
                    "runtime cleanup command deadline exceeded"
                );
                if let Some(status) = stop.try_wait()? {
                    ensure!(
                        status.success(),
                        "runtime cleanup command failed (diagnostics withheld)"
                    );
                    break;
                }
                thread::sleep(POLL_INTERVAL);
            }
            loop {
                if session.child.has_exited()? {
                    return Ok(());
                }
                ensure!(
                    Instant::now() < deadline,
                    "runtime cleanup did not stop its foreground process"
                );
                thread::sleep(POLL_INTERVAL);
            }
        })();
        // Only the child/process group we launched is eligible for this fallback.
        // Never scan process lists, signal a pre-existing endpoint or run a global
        // watchdog. OwnedChild drop terminates/reaps even after failed shutdown.
        drop(session);
        result
    }
}

impl Drop for RuntimeManager {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

struct OwnedSession {
    child: OwnedChild,
    token: String,
    _directory: TempDir,
}

struct OwnedChild {
    child: Child,
    reaped: bool,
}

impl OwnedChild {
    fn new(child: Child) -> Self {
        Self {
            child,
            reaped: false,
        }
    }

    fn try_wait(&mut self) -> Result<Option<std::process::ExitStatus>> {
        let status = self
            .child
            .try_wait()
            .map_err(|_| anyhow!("cannot wait for runtime command"))?;
        self.reaped |= status.is_some();
        Ok(status)
    }

    fn has_exited(&mut self) -> Result<bool> {
        #[cfg(unix)]
        {
            // WNOWAIT keeps an exited start process unreaped until cleanup, so
            // its PID cannot be reused while the stop command is targeting it.
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            let result = unsafe {
                libc::waitid(
                    libc::P_PID,
                    self.child.id() as libc::id_t,
                    &mut info,
                    libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                )
            };
            ensure!(result == 0, "cannot inspect runtime foreground process");
            Ok(unsafe { info.si_pid() } != 0)
        }
        #[cfg(not(unix))]
        {
            Ok(self.try_wait()?.is_some())
        }
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        if !self.reaped {
            #[cfg(unix)]
            unsafe {
                libc::kill(-(self.child.id() as libc::pid_t), libc::SIGKILL);
            }
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn command(argv: &[String], token: &str, pid: Option<u32>) -> Command {
    let mut command = Command::new(&argv[0]);
    command.args(argv[1..].iter().map(|arg| {
        match arg.as_str() {
            "{session}" => token.to_owned(),
            "{pid}" => pid
                .expect("validated stop-only pid placeholder")
                .to_string(),
            _ => arg.clone(),
        }
    }));
    command
        .env("GAME_SCREEN_PICK_RUNTIME_SESSION", token)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    command
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<()> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(crate::media::Interrupted.into());
    }
    Ok(())
}
